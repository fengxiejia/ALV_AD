import torch
import torch.nn as nn
import torch.nn.functional as F

from .model.Transformer import TransformerVar
from .model.loss_functions import EntropyLoss, GatheringLoss
from .utils.utils import harmonic_loss_compute
from ts_benchmark.baselines.self_impl._detection_adapter import DetectionAdapter


class MtsCID(DetectionAdapter):
    default_hyper_params = dict(DetectionAdapter.default_hyper_params)
    default_hyper_params.update(
        {
            "seq_len": 100,
            "batch_size": 128,
            "num_epochs": 5,
            "patience": 3,
            "lr": 0.002,
            "weight_decay": 5e-5,
            "dropout": 0.3,
            "temperature": 0.1,
            "alpha": 1.0,
            "encoder_layers": 1,
            "decoder_layers": 1,
            "branches_group_embedding": "False_False",
            "multiscale_kernel_size": [5],
            "multiscale_patch_size": [10, 20],
            "branch1_networks": ["fc_linear", "intra_fc_transformer", "multiscale_ts_attention"],
            "branch1_match_dimension": "first",
            "branch2_networks": ["multiscale_conv1d", "inter_fc_transformer"],
            "branch2_match_dimension": "first",
            "decoder_networks": ["linear"],
            "decoder_group_embedding": "False",
            "embedding_init": "normal",
            "memory_guided": "sinusoid",
            "aggregation": "normal_mean",
        }
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.entropy_loss = EntropyLoss()
        self.recon_loss = nn.MSELoss(reduction="none")
        self.gathering_loss = GatheringLoss(reduction="none", memto_framework=True)

    def _make_config(self, n_features: int):
        d_model = int(self.config.get("d_model", n_features))
        if d_model <= 0 or self.config.get("d_model", None) == 128:
            d_model = n_features
        return {
            "win_size": int(self.config["seq_len"]),
            "input_c": n_features,
            "output_c": n_features,
            "d_model": d_model,
            "temperature": float(self.config["temperature"]),
            "encoder_layers": int(self.config["encoder_layers"]),
            "branches_group_embedding": self.config["branches_group_embedding"],
            "multiscale_kernel_size": list(self.config["multiscale_kernel_size"]),
            "multiscale_patch_size": list(self.config["multiscale_patch_size"]),
            "branch1_networks": list(self.config["branch1_networks"]),
            "branch1_match_dimension": self.config["branch1_match_dimension"],
            "branch2_networks": list(self.config["branch2_networks"]),
            "branch2_match_dimension": self.config["branch2_match_dimension"],
            "decoder_networks": list(self.config["decoder_networks"]),
            "decoder_layers": int(self.config["decoder_layers"]),
            "decoder_group_embedding": self.config["decoder_group_embedding"],
            "embedding_init": self.config["embedding_init"],
            "memory_guided": self.config["memory_guided"],
            "aggregation": self.config["aggregation"],
            "device": self.device,
        }

    def _make_model(self, n_features: int):
        return TransformerVar(
            self._make_config(n_features),
            dropout=float(self.config["dropout"]),
        )

    def _loss(self, batch: torch.Tensor, epoch: int) -> torch.Tensor:
        del epoch
        outputs = self.model(batch)
        recon = outputs["out"]
        attn = outputs["attn"]
        rec_loss = F.mse_loss(recon, batch)
        attn_loss = torch.zeros_like(rec_loss) if attn is None else self.entropy_loss(attn)
        return rec_loss + float(self.config["alpha"]) * attn_loss

    def _score_batch(self, batch: torch.Tensor) -> torch.Tensor:
        outputs = self.model(batch, mode="test")
        recon = outputs["out"]
        queries = outputs["queries"]
        mem_items = outputs["mem"]
        rec_loss = self.recon_loss(batch, recon)
        latent_score = torch.softmax(
            self.gathering_loss(queries, mem_items) / float(self.config["temperature"]),
            dim=-1,
        )
        return harmonic_loss_compute(rec_loss, latent_score, self.config["aggregation"])
