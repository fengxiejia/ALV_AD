import torch
import torch.nn as nn


class FeatureAttention(nn.Module):
    def __init__(self, n_features: int, win_size: int, dropout: float = 0.1):
        super().__init__()
        self.n_features = n_features
        self.win_size = win_size
        self.proj = nn.Linear(2 * win_size, win_size)
        self.score = nn.Linear(win_size, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        nodes = x.transpose(1, 2)
        left = nodes.unsqueeze(2).expand(-1, -1, self.n_features, -1)
        right = nodes.unsqueeze(1).expand(-1, self.n_features, -1, -1)
        e = self.score(torch.tanh(self.proj(torch.cat([left, right], dim=-1)))).squeeze(-1)
        attn = self.dropout(torch.softmax(e, dim=-1))
        return torch.matmul(attn, nodes).transpose(1, 2)


class TemporalAttention(nn.Module):
    def __init__(self, n_features: int, win_size: int, dropout: float = 0.1):
        super().__init__()
        self.win_size = win_size
        self.proj = nn.Linear(2 * n_features, n_features)
        self.score = nn.Linear(n_features, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        left = x.unsqueeze(2).expand(-1, -1, self.win_size, -1)
        right = x.unsqueeze(1).expand(-1, self.win_size, -1, -1)
        e = self.score(torch.tanh(self.proj(torch.cat([left, right], dim=-1)))).squeeze(-1)
        attn = self.dropout(torch.softmax(e, dim=-1))
        return torch.matmul(attn, x)
