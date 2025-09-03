"""Defines and trains the Railway Foundation Model (RFM).

This script contains the core implementation of the RFM, a Heterogeneous Graph
Transformer (HGT) designed to simulate railway dynamics. It defines the model
architecture, the loss function, and the complete training and evaluation loop.

The model's primary task is to predict the future state of vehicles in the
railway network, specifically their next track (a classification task) and their
relative position on that track (a regression task).

This module is intended to be executed as the main script for training. It
parses a YAML configuration file for hyperparameters and orchestrates the
data loading, training, and model saving processes.

Usage:
    python src/rfm.py --config path/to/your/config.yaml
"""

import torch
import torch.nn.functional as F
import torch_geometric
from src.data import get_data
from torch_geometric.nn import HGTConv
import argparse
import yaml
from safetensors.torch import save_model


class RFM(torch.nn.Module):
    def __init__(self, hidden_channels, num_heads, num_layers, metadata, use_RTE):
        super().__init__()
        self.convs = torch.nn.ModuleList([
            HGTConv(-1, hidden_channels, metadata, num_heads, use_RTE=use_RTE)
            for _ in range(num_layers)
        ])
        self.scorer = Scorer(2 * hidden_channels, hidden_channels)


    def forward(self, x_dict, edge_index_dict, current, edge_time_diff_dict=None):
        """
        
        Args:
            x_dict (dict): Dictionary of node features.
            edge_index_dict (dict): Dictionary of edge indices.
            current (Tensor): Current vehicle states.
            edge_time_diff_dict (dict, optional): Dictionary of edge time features.
        """
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict, edge_time_diff_dict)
        return self.compute_vehicle_outputs(x_dict, current)

    def compute_vehicle_outputs(self, x_dict, current):
        vehicles_out = []
        for vehicle in x_dict["vehicle"][current[:, 0]]:
            out_dict = self.compute_track_scores(x_dict, vehicle)
            vehicles_out.append(out_dict)
        return vehicles_out

    def compute_track_scores(self, x_dict, vehicle):
        out_dict = {"track_ids": [], "pos": [], "scores": []}
        for i, track_node in enumerate(x_dict["track"]):
            score, pos = self.scorer(track_node, vehicle)
            out_dict["scores"].append(score)
            out_dict["pos"].append(pos)
            out_dict["track_ids"].append(i)
        out_dict["scores"] = torch.stack(out_dict["scores"]).squeeze()
        return out_dict


class Scorer(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.layer_1 = torch.nn.Linear(input_dim, hidden_dim)
        self.activation = torch.nn.ReLU()
        self.layer_2 = torch.nn.Linear(hidden_dim, 2)

    def forward(self, emb_i, emb_j):
        x = torch.cat([emb_i, emb_j], dim=-1)
        x = self.activation(self.layer_1(x))
        score, pos = self.layer_2(x)
        return score, torch.sigmoid(pos)


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch_geometric.is_xpu_available():
        return torch.device('xpu')
    return torch.device('cpu')


def train(data, model, optimizer, pos_weight):
    model.train()
    optimizer.zero_grad()

    num_edges = data['track', 'connects', 'track'].edge_index.size(1)
    data['track', 'connects', 'track'].time_diff = torch.zeros(num_edges, dtype=torch.long)
    num_edges = data['vehicle', 'on', 'track'].edge_index.size(1)
    data['vehicle', 'on', 'track'].time_diff = torch.zeros(num_edges, dtype=torch.long)
    num_edges = data['track', 'hosts', 'vehicle'].edge_index.size(1)
    data['track', 'hosts', 'vehicle'].time_diff = torch.zeros(num_edges, dtype=torch.long)

    vehicles_out = model(data.x_dict, data.edge_index_dict, current=data['vehicle'].current, edge_time_diff_dict=data.time_diff_dict)
    loss = compute_loss(data, vehicles_out, pos_weight=pos_weight)
    loss.backward()
    optimizer.step()
    return loss


def compute_loss(data, vehicles_out, pos_weight=0.1):
    """
    Computes the combined loss for track classification and position regression.

    Args:
        data: The graph data batch.
        vehicles_out (list): The list of output dictionaries from the model.
        pos_weight (float): The weight to apply to the MSE loss to balance it
                            with the cross-entropy loss.
    """
    total_loss = 0.0
    
    # Extract ground truth labels for all vehicles in the current batch for efficiency
    current_vehicle_indices = data["vehicle"].current[:, 0]
    y_pos_true_batch = data["vehicle"].y_pos[current_vehicle_indices]
    y_track_true_batch = data["vehicle"].y_track[current_vehicle_indices]

    # Iterate through the output for each vehicle
    for i, out_dict in enumerate(vehicles_out):
        # Ground truth for the i-th vehicle in the batch
        y_pos_true = y_pos_true_batch[i]
        y_track_true_label = y_track_true_batch[i] # This is a tensor with the class index

        # --- 1. Classification Loss (Cross-Entropy) ---
        # The model's scores (logits) for all possible tracks for this vehicle
        track_scores = out_dict["scores"].unsqueeze(0) # Shape: [1, num_tracks]
        classification_loss = F.cross_entropy(track_scores, y_track_true_label)
        
        # --- 2. Regression Loss (Mean Squared Error) ---
        # Get the index of the TRUE track
        true_track_index = y_track_true_label.item()
        
        # Select the model's position prediction for the TRUE track
        predicted_pos_for_true_track = out_dict["pos"][true_track_index]
        
        # Calculate the regression loss
        regression_loss = F.mse_loss(predicted_pos_for_true_track, y_pos_true.squeeze())

        # --- 3. Combine Losses ---
        # Add the weighted losses for this vehicle to the total
        total_loss += classification_loss + (pos_weight * regression_loss)
        
    return total_loss



@torch.no_grad()
def test(data, model):
    model.eval()

    num_edges = data['track', 'connects', 'track'].edge_index.size(1)
    data['track', 'connects', 'track'].time_diff = torch.zeros(num_edges, dtype=torch.long)
    num_edges = data['vehicle', 'on', 'track'].edge_index.size(1)
    data['vehicle', 'on', 'track'].time_diff = torch.zeros(num_edges, dtype=torch.long)
    num_edges = data['track', 'hosts', 'vehicle'].edge_index.size(1)
    data['track', 'hosts', 'vehicle'].time_diff = torch.zeros(num_edges, dtype=torch.long)

    vehicles_out = model(data.x_dict, data.edge_index_dict, current=data['vehicle'].current, edge_time_diff_dict=data.time_diff_dict)
    correct_track_prediction, pos_error = [], []
    current_vehicle_indices = data["vehicle"].current[:, 0]
    y_pos_true_batch = data["vehicle"].y_pos[current_vehicle_indices]
    y_track_true_batch = data["vehicle"].y_track[current_vehicle_indices]

    for i, out_dict in enumerate(vehicles_out):
        _, predicted_track_index = out_dict["scores"].max(dim=0)
        y_track_true = y_track_true_batch[i]
        correct_track_prediction.append((predicted_track_index == y_track_true).item())

        y_pos_true = y_pos_true_batch[i]
        true_track_index = y_track_true.item()
        predicted_pos_for_true_track = out_dict["pos"][true_track_index]

        pos_error.append(torch.abs(predicted_pos_for_true_track - y_pos_true).item())
    return correct_track_prediction, pos_error


def evaluate_model(data_list, model, device):
    errors = {"track": [], "pos": []}
    for batch in data_list:
        data = batch.to(device)
        correct_track_prediction, pos_error = test(data, model)
        errors["track"] += correct_track_prediction
        errors["pos"] += pos_error
    track_acc = sum(errors["track"]) / len(errors["track"])
    pos_mae = sum(errors["pos"]) / len(errors["pos"])
    return track_acc, pos_mae


def main(data_list: list, config: dict):
    device = get_device()
    data = data_list[0]  # TODO: get metadata from source data

    model_params = config['model']
    model = RFM(
        hidden_channels=model_params['hidden_channels'],
        num_heads=model_params['num_heads'],
        num_layers=model_params['num_layers'],
        metadata=data.metadata(),
        use_RTE=model_params["use_RTE"],
    ).to(device)

    opt_params = config['optimizer']
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=opt_params['lr'],
        weight_decay=opt_params['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode=opt_params["ReduceLROnPlateauScheduler"]["mode"],
        factor=opt_params["ReduceLROnPlateauScheduler"]["factor"],
        patience=opt_params["ReduceLROnPlateauScheduler"]["patience"],
    )

    train_set = data_list[:int(len(data_list) * 0.8)]
    val_set = data_list[int(len(data_list) * 0.8):]

    epochs = config['training']['epochs']
    for epoch in range(1, epochs + 1):
        epoch_loss = 0
        for batch in train_set:
            data = batch.to(device)
            epoch_loss += train(data, model, optimizer, pos_weight=0.1)

        train_track_acc, train_pos_mae = evaluate_model(train_set, model, device)
        val_track_acc, val_pos_mae = evaluate_model(val_set, model, device)

        scheduler.step(val_pos_mae)

        print(f'Epoch: {epoch:03d}')
        print(f'Loss: {epoch_loss:.4f}')
        print(f'Train track: {train_track_acc:.4f}', f'Val track: {val_track_acc:.4f}')
        print(f'Train pos: {train_pos_mae:.4f}', f'Val pos: {val_pos_mae:.4f}')
    return model

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Railway Foundation Model.")
    parser.add_argument('--config', type=str, required=True, help='Path to the config.yaml file.')
    args = parser.parse_args()

    config = load_config(args.config)
    simulation_id = config['simulation_id']

    print(f"Starting training for simulation: {simulation_id}")
    print(f"Using config: {config}")
    data_list, _ = get_data(simulation_id=simulation_id)
    
    model = main(data_list=data_list, config=config)

    model_path = f"simulations/{simulation_id}/model.safetensors"
    save_model(model, model_path)
    print(f"Model saved to {model_path}")
    