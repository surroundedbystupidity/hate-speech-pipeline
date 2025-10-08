import torch
from torch import nn
from torch_geometric_temporal.nn import A3TGCN, DCRNN


# TODO: Implement a superclass for common functionality, here and for static_models.py.
class AttentionLayer(nn.Module):
    """Self-attention layer for graph nodes"""

    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: [num_nodes, hidden_dim]
        x_unsqueezed = x.unsqueeze(0)  # [1, num_nodes, hidden_dim]
        attended, _ = self.attention(x_unsqueezed, x_unsqueezed, x_unsqueezed)
        attended = attended.squeeze(0)  # [num_nodes, hidden_dim]
        return self.norm(x + self.dropout(attended))


class BasicRecurrentGCN(nn.Module):
    def __init__(self, node_features, hidden_dim=128, dropout=0.1, num_heads=8):
        super().__init__()
        self.recurrent = DCRNN(node_features, hidden_dim, K=1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.attention = AttentionLayer(
            hidden_dim, num_heads=num_heads, dropout=dropout
        )
        # self.multi_attention = nn.MultiheadAttention(
        #     embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        # )

    def forward(self, x, edge_index):
        h = self.recurrent(x, edge_index)
        attn_out = self.attention(h)
        h = h + attn_out
        out = self.fc(h)
        return out


# TODO: Remove the NNs that are not used.


class BasicAttentionGCN(nn.Module):

    def __init__(self, node_features, periods=4, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.recurrent = A3TGCN(node_features, hidden_dim, periods=periods)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, edge_index):
        """
        x: (batch_size, num_nodes, num_timesteps, node_features)
        edge_index: graph connectivity
        """
        # Get node embeddings from A3TGCN
        # h = self.recurrent(x, edge_index)  # (batch_size, num_nodes, hidden_dim)
        h = self.recurrent(x.view(x.shape[0], 1, x.shape[1]), edge_index)
        # Map hidden_dim to output per node
        out = self.fc(h)  # (batch_size, num_nodes, 1)

        return out


class DeepRecurrentGCN(nn.Module):
    def __init__(self, node_features, hidden_dim=256, dropout=0.15, num_gnn_layers=2):
        super().__init__()

        # Input transformation
        self.input_transform = nn.Sequential(
            nn.Linear(node_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Multiple GNN layers
        self.gnn_layers = nn.ModuleList()
        for i in range(num_gnn_layers):
            input_dim = hidden_dim if i > 0 else hidden_dim
            self.gnn_layers.append(DCRNN(input_dim, hidden_dim, K=2))

        # Attention mechanism
        self.attention = AttentionLayer(hidden_dim, num_heads=4, dropout=dropout)

        # Skip connection
        self.skip_transform = nn.Linear(node_features, hidden_dim)

        # Output layers with more capacity
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Layer normalization for stability
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(num_gnn_layers)]
        )

    def forward(self, x, edge_index):
        # Transform input
        h = self.input_transform(x)
        skip = self.skip_transform(x)

        # Process through GNN layers with residual connections
        for i, (gnn_layer, norm) in enumerate(zip(self.gnn_layers, self.layer_norms)):
            h_new = gnn_layer(h, edge_index)
            h_new = nn.functional.relu(h_new)
            h = norm(h + h_new * 0.5)  # Residual with scaling

        # Apply attention
        h = self.attention(h)

        # Combine with skip connection
        h_combined = torch.cat([h, skip], dim=-1)

        # Generate output
        out = self.output_layers(h_combined)
        return out
