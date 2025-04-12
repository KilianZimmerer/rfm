import torch
import torch.nn.functional as F

import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.nn import HGTConv, Linear
from torch_geometric.data import HeteroData

from data import get_data

DATA_LIST, NODE_MAPPING = get_data(simulation_id="sim2")


class HGT(torch.nn.Module):
    def __init__(self, hidden_channels, num_heads, num_layers, node_types, metadata):
        super().__init__()

        self.lin_dict = torch.nn.ModuleDict()
        for node_type in node_types:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, metadata,
                           num_heads)
            self.convs.append(conv)
        
        self.scorer = Scorer(int(2 * hidden_channels), hidden_channels)

    def forward(self, x_dict, edge_index_dict, current):
        x_dict = {
            node_type: self.lin_dict[node_type](x).relu_()
            for node_type, x in x_dict.items()
        }

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        # TODO: make this as a tensor instead of a list
        vehicles_out = []
        for vehicle in x_dict["vehicle"][current[:,0]]:
            # TODO: We only need to iterate through candidate nodes
            out_dict = {"track_ids": [], "pos": [], "scores": []}
            for node_type, nodes in x_dict.items():
                if node_type == "track":
                    for i in range(len(nodes)):
                        # TODO: this does only work if there is one vechile node
                        score, pos = self.scorer(nodes[i], vehicle)
                        out_dict["scores"].append(score)
                        out_dict["pos"].append(pos)
                        # TODO: i is only the original track id if we iterate through all nodes
                        out_dict["track_ids"].append(i)
            out_dict["scores"] = torch.stack(out_dict["scores"]).squeeze()
            vehicles_out.append(out_dict)

        return vehicles_out


class Scorer(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.layer_1 = torch.nn.Linear(input_dim, hidden_dim)
        self.activation = torch.nn.ReLU() # Or other non-linearity
        self.layer_2 = torch.nn.Linear(hidden_dim, 2) # Output layer, size 2 for the score + position

    def forward(self, emb_i, emb_j):
        x = torch.cat([emb_i, emb_j], dim=-1)
        x = self.layer_1(x)
        x = self.activation(x)
        score, pos = self.layer_2(x) # Raw logit score
        return score, torch.sigmoid(pos) # Sigmoid for position


if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch_geometric.is_xpu_available():
    device = torch.device('xpu')
else:
    device = torch.device('cpu')


DATA = DATA_LIST[2]
model = HGT(hidden_channels=60, num_heads=3, num_layers=2, 
            node_types=DATA.node_types, metadata=DATA.metadata())

data, model = DATA.to(device), model.to(device)


with torch.no_grad():  # Initialize lazy modules.
    out = model(data.x_dict, data.edge_index_dict, current=data['vehicle'].current)

optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)


def train(data: HeteroData):
    model.train()
    optimizer.zero_grad()
    vehicles_out = model(data.x_dict, data.edge_index_dict, current=data['vehicle'].current)
    # get max value and id of out["scores"]
    loss = 0
    for i, out_dict in enumerate(vehicles_out):
        _, max_index = out_dict["scores"].max(dim=0)
        # TODO: enable several vehicles so max_index becomes a 1-D tensor
        # TODO: weight the loss by the number of track nodes
        y_pos_true = data["vehicle"].y_pos[data["vehicle"].current[:,0]][i]
        y_track_true = data["vehicle"].y_track[data["vehicle"].current[:,0]][i]
        loss += F.cross_entropy(out_dict["scores"], y_track_true[0])
        loss += F.mse_loss(out_dict["pos"][max_index], y_pos_true[0])
    
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test(data: HeteroData):
    model.eval()
    vehicles_out = model(data.x_dict, data.edge_index_dict, current=data['vehicle'].current)
    
    correct_track_prediction = []
    pos_error = []
    for i, out_dict in enumerate(vehicles_out):
        _, max_index = out_dict["scores"].max(dim=0)
    
        y_pos_true = data["vehicle"].y_pos[data["vehicle"].current[:,0]][i]
        y_track_true = data["vehicle"].y_track[data["vehicle"].current[:,0]][i]
        correct_track_prediction.append((max_index == y_track_true))
        pos_error.append(torch.abs(out_dict["pos"][max_index] - y_pos_true))
    return correct_track_prediction, pos_error


# train set split of DATA_LIST by slizing the list
train_set = DATA_LIST[:int(len(DATA_LIST) * 0.8)]
val_set = DATA_LIST[int(len(DATA_LIST) * 0.8):]

for epoch in range(1, 50):
    train_errors = {"track": [], "pos": []}
    epoch_loss = 0
    for batch in train_set:
        data = batch.to(device)
        epoch_loss += train(data)
        correct_track_prediction, pos_error = test(data)
        train_errors["track"] += correct_track_prediction
        train_errors["pos"] += pos_error
    
    val_errors = {"track": [], "pos": []}
    for batch in val_set:
        data = batch.to(device)
        correct_track_prediction, pos_error = test(data)
        val_errors["track"] += correct_track_prediction
        val_errors["pos"] += pos_error
    
    train_track_acc = sum(train_errors["track"]) / len(train_errors["track"])
    train_pos_mae = sum(train_errors["pos"]) / len(train_errors["pos"])
    val_track_acc = sum(val_errors["track"]) / len(val_errors["track"])
    val_pos_mae = sum(val_errors["pos"]) / len(val_errors["pos"])
    
    print(f'Epoch: {epoch:03d}')
    print(f'Loss: {epoch_loss:.4f}')
    print(f'Train track: {train_track_acc[0]:.4f}', f'Val track: {val_track_acc[0]:.4f}')
    print(f'Train pos: {train_pos_mae[0]:.4f}', f'Val pos: {val_pos_mae[0]:.4f}')
