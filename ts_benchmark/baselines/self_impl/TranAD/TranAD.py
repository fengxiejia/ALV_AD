import torch
import torch.nn.functional as F

from .model import TranADModel
from ts_benchmark.baselines.self_impl._detection_adapter import DetectionAdapter


class TranAD(DetectionAdapter):
    def _make_model(self, n_features: int):
        return TranADModel(
            n_features,
            win_size=int(self.config["seq_len"]),
            dropout=float(self.config["dropout"]),
        )

    def _loss(self, batch: torch.Tensor, epoch: int) -> torch.Tensor:
        del epoch
        x1, x2 = self.model(batch)
        return 0.5 * F.mse_loss(x1, batch) + 0.5 * F.mse_loss(x2, batch)

    def _score_batch(self, batch: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.model(batch)
        return 0.5 * (x1 - batch).pow(2).mean(dim=-1) + 0.5 * (x2 - batch).pow(2).mean(dim=-1)
