import torch
import torch.nn.functional as F

from .model import OmniAnomalyModel
from ts_benchmark.baselines.self_impl._detection_adapter import DetectionAdapter


class OmniAnomaly(DetectionAdapter):
    def _make_model(self, n_features: int):
        return OmniAnomalyModel(
            n_features,
            hidden_dim=int(self.config.get("hidden_dim", 32)),
            latent_dim=int(self.config.get("latent_dim", 8)),
        )

    def _loss(self, batch: torch.Tensor, epoch: int) -> torch.Tensor:
        del epoch
        recon, mu, logvar, _ = self.model(batch)
        recon_loss = F.mse_loss(recon, batch)
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + float(self.config.get("beta", 0.01)) * kl

    def _score_batch(self, batch: torch.Tensor) -> torch.Tensor:
        recon, _, _, _ = self.model(batch)
        return (recon - batch).pow(2).mean(dim=-1)
