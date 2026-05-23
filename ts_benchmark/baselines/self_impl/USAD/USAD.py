import torch
import torch.nn.functional as F

from .model import USADModel
from ts_benchmark.baselines.self_impl._detection_adapter import DetectionAdapter


class USAD(DetectionAdapter):
    def _make_model(self, n_features: int):
        return USADModel(
            int(self.config["seq_len"]),
            n_features,
            latent_dim=int(self.config["latent_dim"]),
            hidden_dim=int(self.config["hidden_dim"]),
        )

    def _loss(self, batch: torch.Tensor, epoch: int) -> torch.Tensor:
        w1, w2, w3 = self.model(batch)
        return (1 / epoch) * F.mse_loss(w1, batch) + (1 - 1 / epoch) * F.mse_loss(w3, batch) + F.mse_loss(w2, batch)

    def _score_batch(self, batch: torch.Tensor) -> torch.Tensor:
        w1, _, w3 = self.model(batch)
        return 0.5 * (batch - w1).pow(2).mean(dim=-1) + 0.5 * (batch - w3).pow(2).mean(dim=-1)
