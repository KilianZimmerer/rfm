import torch
import torch.nn.functional as F
import torch_geometric
from src.data import get_data
from torch_geometric.nn import HGTConv


class HGT(torch.nn.Module):
    def __init__(self, hidden_channels, num_heads, num_layers, node_types, metadata):
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
    return float(loss)


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


def main(data_list: list):
    device = get_device()
    data = data_list[2]
    model = HGT(hidden_channels=60, num_heads=3, num_layers=2,
                node_types=data.node_types, metadata=data.metadata()).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)

    train_set = data_list[:int(len(data_list) * 0.8)]
    val_set = data_list[int(len(data_list) * 0.8):]

    for epoch in range(1, 50):
        epoch_loss = 0
        for batch in train_set:
            data = batch.to(device)
            epoch_loss += train(data, model, optimizer)

        train_track_acc, train_pos_mae = evaluate_model(train_set, model, device)
        val_track_acc, val_pos_mae = evaluate_model(val_set, model, device)

        print(f'Epoch: {epoch:03d}')
        print(f'Loss: {epoch_loss:.4f}')
        print(f'Train track: {train_track_acc[0]:.4f}', f'Val track: {val_track_acc[0]:.4f}')
        print(f'Train pos: {train_pos_mae[0]:.4f}', f'Val pos: {val_pos_mae[0]:.4f}')
    return model
    

if __name__ == "__main__":
    from safetensors.torch import save_model, load_model
    simulation_id = "sim1"
    data_list, _ = get_data(simulation_id=simulation_id)
    model = main(data_list=data_list)

    save_model(model, f"simulations/{simulation_id}/model.safetensors")
    
    # Load the model
    device = get_device()
    model_test = HGT(
        hidden_channels=60, num_heads=3, num_layers=2,
        node_types=data_list[0].node_types, metadata=data_list[0].metadata()).to(device)
    # Perform a dummy forward pass to initialize lazy modules
    model_test(data_list[0].x_dict, data_list[0].edge_index_dict, current=data_list[0]['vehicle'].current)
    load_model(model_test, f"simulations/{simulation_id}/model.safetensors")
    