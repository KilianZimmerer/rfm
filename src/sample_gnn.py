import torch
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.lin = Linear(in_channels, out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, edge_index, index, x_j, x_i):
        import pdb;pdb.set_trace()
        return x_j


if __name__ == "__main__":
    conv = GCNConv(4, 32)
    # Sample input
    x = torch.randn(3, 4)  # 3 nodes with 4 features each
    edge_index = torch.tensor([
        [0, 1, 2],
        [1, 0, 1], 
    ], dtype=torch.long)  # 2 edges

    # Apply the convolution
    x = conv(x, edge_index)