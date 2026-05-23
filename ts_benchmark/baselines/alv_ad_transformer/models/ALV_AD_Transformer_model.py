import torch
import torch.nn as nn

from ts_benchmark.baselines.alv_ad_transformer.layers.RevIN import RevIN
from ts_benchmark.baselines.alv_ad_transformer.layers.alv_ad_block import ALV_ADBlock
from ts_benchmark.baselines.alv_ad_transformer.layers.quantizer import (
    ResidualVectorQuantizer,
)


RVQ_STAGE_DECAY = 0.7


class _ALV_AD_TransformerBase(nn.Module):
    """
    Standard time-token ALV_AD Transformer for reconstruction-based anomaly detection.
    This variant keeps time steps as tokens and embeds the channel dimension at
    each step.
    """

    def __init__(self, configs):
        super().__init__()
        self.task_name = getattr(configs, "task_name", "anomaly_detection")
        self.seq_len = configs.seq_len
        self.pred_len = getattr(configs, "pred_len", self.seq_len)
        self.input_dim = configs.enc_in
        self.d_model = getattr(configs, "d_model", 256)
        self.d_ff = getattr(configs, "d_ff", self.d_model * 4)
        self.n_heads = getattr(configs, "n_heads", 8)
        self.n_layers = getattr(configs, "e_layers", 3)
        self.n_streams = getattr(configs, "n_streams", 4)
        self.sinkhorn_iterations = getattr(configs, "sinkhorn_iterations", 8)
        self.rvq_num_embeddings = getattr(configs, "rvq_num_embeddings", 128)
        self.rvq_grad_scale = getattr(configs, "rvq_grad_scale", 0.25)
        self.rvq_gate_floor = getattr(configs, "rvq_gate_floor", 0.2)
        self.rvq_gate_temperature = getattr(configs, "rvq_gate_temperature", 4.0)
        self.latest_quantizer_token_dist = None
        self.latest_quantizer_stage_token_dist = None

        self.revin = RevIN(self.input_dim, affine=True)
        self.enc_embedding = nn.Linear(self.input_dim, self.d_model)
        self.layers = nn.ModuleList(
            [
                ALV_ADBlock(
                    d_model=self.d_model,
                    d_ff=self.d_ff,
                    n_heads=self.n_heads,
                    n_streams=self.n_streams,
                    sinkhorn_iterations=self.sinkhorn_iterations,
                )
                for _ in range(self.n_layers)
            ]
        )
        self.final_stream_mix = nn.Parameter(torch.ones(self.n_streams) / self.n_streams)
        self.rvq = ResidualVectorQuantizer(
            num_embeddings=self.rvq_num_embeddings,
            embedding_dim=self.d_model,
            stage_decay=RVQ_STAGE_DECAY,
            grad_scale=self.rvq_grad_scale,
        )
        self.reconstruction_head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.input_dim),
        )

    def _apply_rvq(
        self,
        latent: torch.Tensor,
        rvq_module: nn.Module,
    ):
        if rvq_module is None:
            return latent, None, None

        quantized_latent, token_dist, stage_token_dist = rvq_module(latent)
        centered_dist = token_dist - token_dist.mean(dim=1, keepdim=True)
        normalized_dist = centered_dist / (token_dist.std(dim=1, keepdim=True) + 1e-6)
        gate = torch.sigmoid(normalized_dist * self.rvq_gate_temperature)
        gate = self.rvq_gate_floor + (1.0 - self.rvq_gate_floor) * gate
        return (
            latent + gate.unsqueeze(-1) * (quantized_latent - latent),
            token_dist,
            stage_token_dist,
        )

    def _encode_branch(
        self,
        tokens: torch.Tensor,
        embedding: nn.Module,
        layers: nn.ModuleList,
        stream_mix: nn.Parameter,
        rvq_module: nn.Module = None,
    ):
        encoded = embedding(tokens)
        streams = encoded.unsqueeze(1).repeat(1, self.n_streams, 1, 1)
        for layer in layers:
            streams = layer(streams)

        weights = torch.softmax(stream_mix, dim=0)
        latent = torch.einsum("n,bnld->bld", weights, streams)
        latent, token_dist, stage_token_dist = self._apply_rvq(latent, rvq_module)
        return latent, token_dist, stage_token_dist

    def _encode(self, x_enc: torch.Tensor) -> torch.Tensor:
        latent, token_dist, stage_token_dist = self._encode_branch(
            tokens=x_enc,
            embedding=self.enc_embedding,
            layers=self.layers,
            stream_mix=self.final_stream_mix,
            rvq_module=self.rvq,
        )
        self.latest_quantizer_token_dist = token_dist
        self.latest_quantizer_stage_token_dist = stage_token_dist
        return latent

    def anomaly_detection(self, x_enc: torch.Tensor) -> torch.Tensor:
        x_enc = self.revin(x_enc, "norm")
        latent = self._encode(x_enc)
        reconstructed = self.reconstruction_head(latent)
        return self.revin(reconstructed, "denorm")

    def forecast(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor,
        x_dec: torch.Tensor,
        x_mark_dec: torch.Tensor,
    ) -> torch.Tensor:
        reconstructed = self.anomaly_detection(x_enc)
        return reconstructed[:, -self.pred_len :, :]

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor,
        x_dec: torch.Tensor,
        x_mark_dec: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        del mask
        if self.task_name == "anomaly_detection":
            return self.anomaly_detection(x_enc)
        if self.task_name in {"long_term_forecast", "short_term_forecast"}:
            return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        raise ValueError(f"Unsupported task_name for ALV_AD_Transformer: {self.task_name}")


class ALV_AD_Transformer(_ALV_AD_TransformerBase):
    pass
