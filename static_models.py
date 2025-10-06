from torch import nn
from torch_geometric.nn import GCNConv
import torch

class StaticGCN(nn.Module):
    def __init__(
        self, num_node_features, hidden_dim=128, output_dimension=1, dropout=0.3
    ):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim // 2)
        self.bn3 = nn.BatchNorm1d(hidden_dim // 2)

        self.dropout = nn.Dropout(dropout)
        self.readout = nn.Linear(hidden_dim // 2, output_dimension)
        self.relu = nn.ReLU()

        # Attention layer parameters
        # TODO: See if this produces the same results if removed.
        self.attn_fc = nn.Linear(hidden_dim // 2, 1)

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

        # Layer 3
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Attention mechanism (global node attention)
        attn_weights = torch.softmax(self.attn_fc(x), dim=0)  # (N, 1)
        x_attn = torch.sum(attn_weights * x, dim=0, keepdim=True)  # (1, F)

        # Output with sigmoid for 0-1 range
        x = self.readout(x_attn)
        x = torch.sigmoid(x)
        return x.squeeze(0)