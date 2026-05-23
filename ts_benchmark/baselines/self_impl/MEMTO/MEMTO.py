import torch
import torch.nn.functional as F

from .Transformer import TransformerVar
from ts_benchmark.baselines.self_impl._detection_adapter import DetectionAdapter


class MEMTO(DetectionAdapter):
    def _make_model(self, n_features: int):
        return TransformerVar(
            win_size=int(self.config["seq_len"]),
            enc_in=n_features,
            c_out=n_features,
            n_memory=int(self.config["n_memory"]),
            d_model=int(self.config["d_model"]),
            n_heads=int(self.config["n_heads"]),
            e_layers=int(self.config.get("e_layers", 1)),
            d_ff=int(self.config.get("d_ff", max(64, 2 * int(self.config["d_model"])))),
            dropout=float(self.config["dropout"]),
            device=self.device,
        )

    def _loss(self, batch: torch.Tensor, epoch: int) -> torch.Tensor:
        del epoch
        outputs = self.model(batch)
        recon = outputs["out"]
        attn = outputs["attn"]
        entropy = -(attn * torch.log(attn + 1e-8)).sum(dim=-1).mean()
        return F.mse_loss(recon, batch) + 1e-4 * entropy

    def _score_batch(self, batch: torch.Tensor) -> torch.Tensor:
        outputs = self.model(batch)
        return (outputs["out"] - batch).pow(2).mean(dim=-1)
