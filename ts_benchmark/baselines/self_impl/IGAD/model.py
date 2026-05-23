import torch.nn as nn


class IGADReconstructionModel(nn.Module):
    def __init__(self, win_size: int, n_features: int, latent_dim: int = 64, hidden_dim: int = 128):
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
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
        )
        self.win_size = win_size
        self.n_features = n_features

    def forward(self, x):
        out = self.decoder(self.encoder(x))
        return out.view(-1, self.win_size, self.n_features)
