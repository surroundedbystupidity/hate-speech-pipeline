from torch import nn
from torch_geometric_temporal.nn import DCRNN


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

    def forward(self, x, edge_index):
        h = self.recurrent(x, edge_index)
        attn_out = self.attention(h)
        h = h + attn_out
        out = self.fc(h)
        return out
