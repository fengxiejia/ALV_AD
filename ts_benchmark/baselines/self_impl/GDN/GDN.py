import torch
import torch.nn.functional as F

from .model import GDNModel
from ts_benchmark.baselines.self_impl._detection_adapter import DetectionAdapter


class GDN(DetectionAdapter):
    def _make_model(self, n_features: int):
        return GDNModel(
            int(self.config["seq_len"]),
            n_features,
            hidden_dim=int(self.config["hidden_dim"]),
            topk=int(self.config["topk"]),
        )

    def _loss(self, batch: torch.Tensor, epoch: int) -> torch.Tensor:
        del epoch
        pred = self.model(batch)
        return F.mse_loss(pred, batch[:, -1, :])

    def _score_batch(self, batch: torch.Tensor) -> torch.Tensor:
        pred = self.model(batch)
        score = torch.zeros(batch.size(0), batch.size(1), device=batch.device)
        score[:, -1] = (pred - batch[:, -1, :]).pow(2).mean(dim=-1)
        return score
