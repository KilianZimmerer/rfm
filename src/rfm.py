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
    def __init__(self, hidden_channels, num_heads, num_layers, metadata):
        super().__init__()
        self.convs = torch.nn.ModuleList([
            HGTConv(-1, hidden_channels, metadata, num_heads)
            for _ in range(num_layers)
        ])
        self.scorer = Scorer(2 * hidden_channels, hidden_channels)


    def forward(self, x_dict, edge_index_dict, current):
        """
        
        Args:
            x_dict (dict): Dictionary of node features.
            edge_index_dict (dict): Dictionary of edge indices.
            current (Tensor): Current vehicle states.
            time_dict (dict, optional): Dictionary of time features.
        """
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
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


def train(data, model, optimizer):
    model.train()
    optimizer.zero_grad()
    vehicles_out = model(data.x_dict, data.edge_index_dict, current=data['vehicle'].current)
    loss = compute_loss(data, vehicles_out)
    loss.backward()
    optimizer.step()
    return loss


def compute_loss(data, vehicles_out):
    loss = 0
    for i, out_dict in enumerate(vehicles_out):
        _, max_index = out_dict["scores"].max(dim=0)
        y_pos_true = data["vehicle"].y_pos[data["vehicle"].current[:, 0]][i]
        y_track_true = data["vehicle"].y_track[data["vehicle"].current[:, 0]][i]
        loss += F.cross_entropy(out_dict["scores"], y_track_true[0])
        loss += F.mse_loss(out_dict["pos"][max_index], y_pos_true[0])
    return loss


@torch.no_grad()
def test(data, model):
    model.eval()
    vehicles_out = model(data.x_dict, data.edge_index_dict, current=data['vehicle'].current)
    correct_track_prediction, pos_error = [], []
    for i, out_dict in enumerate(vehicles_out):
        _, max_index = out_dict["scores"].max(dim=0)
        y_pos_true = data["vehicle"].y_pos[data["vehicle"].current[:, 0]][i]
        y_track_true = data["vehicle"].y_track[data["vehicle"].current[:, 0]][i]
        correct_track_prediction.append((max_index == y_track_true))
        pos_error.append(torch.abs(out_dict["pos"][max_index] - y_pos_true))
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
    data = data_list[0]

    model_params = config['model']
    model = RFM(
        hidden_channels=model_params['hidden_channels'],
        num_heads=model_params['num_heads'],
        num_layers=model_params['num_layers'],
        metadata=data.metadata()
    ).to(device)

    opt_params = config['optimizer']
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=opt_params['lr'],
        weight_decay=opt_params['weight_decay']
    )

    train_set = data_list[:int(len(data_list) * 0.8)]
    val_set = data_list[int(len(data_list) * 0.8):]

    epochs = config['training']['epochs']
    for epoch in range(1, epochs + 1):
        epoch_loss = 0
        for batch in train_set:
            data = batch.to(device)
            epoch_loss += train(data, model, optimizer)

        train_track_acc, train_pos_mae = evaluate_model(train_set, model, device)
        val_track_acc, val_pos_mae = evaluate_model(val_set, model, device)

        print(f'Epoch: {epoch:03d}')
        print(f'Loss: {epoch_loss:.4f}')
        print(f'Train track: {train_track_acc.item():.4f}', f'Val track: {val_track_acc.item():.4f}')
        print(f'Train pos: {train_pos_mae.item():.4f}', f'Val pos: {val_pos_mae.item():.4f}')
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
    data_list, _ = get_data(simulation_id=simulation_id)
    
    model = main(data_list=data_list, config=config)

    model_path = f"simulations/{simulation_id}/model.safetensors"
    save_model(model, model_path)
    print(f"Model saved to {model_path}")
    