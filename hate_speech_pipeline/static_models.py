import torch
from torch import nn
from torch_geometric.nn import GCNConv


class StaticGCN(nn.Module):
    def __init__(
        self, num_node_features, hidden_dim=128, output_dimension=1, dropout=0.3
    ):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.bn1 = nn.LayerNorm(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.LayerNorm(hidden_dim // 2)

        self.dropout = nn.Dropout(dropout)
        self.readout = nn.Linear(hidden_dim // 2, output_dimension)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index):
        # Layer 1
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Layer 2
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Output with sigmoid for 0-1 range
        x = self.readout(x)
        x = torch.sigmoid(x)
        return x.squeeze(0)
