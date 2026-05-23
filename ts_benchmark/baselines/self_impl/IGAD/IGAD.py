import torch
import torch.nn.functional as F

from .model import IGADReconstructionModel
from .util.igad_module import IdempotentLoss
from ts_benchmark.baselines.self_impl._detection_adapter import DetectionAdapter


class IGAD(DetectionAdapter):
    def _make_model(self, n_features: int):
        model = IGADReconstructionModel(
            int(self.config["seq_len"]),
            n_features,
            latent_dim=int(self.config["latent_dim"]),
            hidden_dim=int(self.config["hidden_dim"]),
        )
        model_copy = IGADReconstructionModel(
            int(self.config["seq_len"]),
            n_features,
            latent_dim=int(self.config["latent_dim"]),
            hidden_dim=int(self.config["hidden_dim"]),
        ).to(self.device)
        model_copy.requires_grad_(False)
        self.ign = IdempotentLoss(
            model=model_copy,
            idem_weight=float(self.config.get("idem_weight", 0.1)),
            tight_weight=float(self.config.get("tight_weight", 0.1)),
            loss_tight_clamp_ratio=float(self.config.get("loss_tight_clamp_ratio", 1.1)),
        )
        return model

    def _loss(self, batch: torch.Tensor, epoch: int) -> torch.Tensor:
        del epoch
        output = self.model(batch)
        main_loss = F.mse_loss(output, batch)
        igad_loss = self.ign(
            input_data=batch.clone(),
            output_data=output,
            training_model=self.model,
        )
        return main_loss + igad_loss

    def _score_batch(self, batch: torch.Tensor) -> torch.Tensor:
        output = self.model(batch)
        return (output - batch).pow(2).mean(dim=-1)
