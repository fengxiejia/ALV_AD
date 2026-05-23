import torch
import torch.nn as nn


class USADModel(nn.Module):
    def __init__(self, win_size: int, n_features: int, latent_dim: int = 32, hidden_dim: int = 128):
        super().__init__()
        dim = win_size * n_features
        hidden_dim = max(16, min(hidden_dim, max(16, dim // 2)))
        latent_dim = max(4, min(latent_dim, hidden_dim))
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )
        self.decoder1 = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, dim))
        self.decoder2 = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, dim))
        self.win_size = win_size
        self.n_features = n_features

    def forward(self, x):
        z = self.encoder(x)
        w1 = self.decoder1(z)
        w2 = self.decoder2(z)
        w3 = self.decoder2(self.encoder(w1.view(-1, self.win_size, self.n_features)))
        shape = (-1, self.win_size, self.n_features)
        return w1.view(shape), w2.view(shape), w3.view(shape)
