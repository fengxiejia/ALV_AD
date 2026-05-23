import torch
import torch.nn as nn


class OmniAnomalyModel(nn.Module):
    """PyTorch OmniAnomaly implementation from the TranAD baseline repository."""

    def __init__(self, n_features: int, hidden_dim: int = 32, latent_dim: int = 8):
        super().__init__()
        self.name = "OmniAnomaly"
        self.lr = 0.002
        self.beta = 0.01
        self.n_feats = n_features
        self.n_hidden = hidden_dim
        self.n_latent = latent_dim
        self.lstm = nn.GRU(n_features, self.n_hidden, 2, batch_first=True)
        self.encoder = nn.Sequential(
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.PReLU(),
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.PReLU(),
            nn.Linear(self.n_hidden, 2 * self.n_latent),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.n_latent, self.n_hidden),
            nn.PReLU(),
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.PReLU(),
            nn.Linear(self.n_hidden, self.n_feats),
            nn.Sigmoid(),
        )

    def forward(self, x, hidden=None):
        out, hidden = self.lstm(x, hidden)
        enc = self.encoder(out)
        mu, logvar = torch.split(enc, [self.n_latent, self.n_latent], dim=-1)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        recon = self.decoder(z)
        return recon, mu, logvar, hidden
