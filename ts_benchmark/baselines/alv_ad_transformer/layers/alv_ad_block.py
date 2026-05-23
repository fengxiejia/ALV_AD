import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight


class SinkhornProjection(nn.Module):
    def __init__(self, iterations: int = 8):
        super().__init__()
        self.iterations = iterations

    def forward(self, affinity: torch.Tensor) -> torch.Tensor:
        transport = torch.exp(affinity)
        for _ in range(self.iterations):
            transport = transport / (transport.sum(dim=-1, keepdim=True) + 1e-6)
            transport = transport / (transport.sum(dim=-2, keepdim=True) + 1e-6)
        return transport


class DynamicResidualRouter(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_streams: int,
        sinkhorn_iterations: int = 8,
    ):
        super().__init__()
        self.n_streams = int(n_streams)

        router_dim = self.n_streams * d_model
        out_dim = self.n_streams * self.n_streams + 2 * self.n_streams
        self.router = nn.Linear(router_dim, out_dim)
        self.norm = RMSNorm(router_dim)
        self.sinkhorn = SinkhornProjection(iterations=sinkhorn_iterations)
        self.register_buffer(
            "res_identity",
            torch.eye(self.n_streams).reshape(1, 1, self.n_streams, self.n_streams),
            persistent=False,
        )

    def forward(self, x_stream: torch.Tensor):
        batch_size, n_streams, n_tokens, d_model = x_stream.shape
        if n_streams != self.n_streams:
            raise ValueError(
                f"Expected {self.n_streams} streams, got {n_streams}"
            )

        flat_state = (
            x_stream.permute(0, 2, 1, 3)
            .reshape(batch_size, n_tokens, n_streams * d_model)
        )
        logits = self.router(self.norm(flat_state))

        pre_end = self.n_streams
        post_end = pre_end + self.n_streams
        pre_logits = logits[..., :pre_end]
        post_logits = logits[..., pre_end:post_end]
        res_logits = logits[..., post_end:].view(
            batch_size, n_tokens, self.n_streams, self.n_streams
        )

        h_pre = F.softmax(pre_logits, dim=-1)
        h_post = F.softmax(post_logits, dim=-1)
        h_res = self.sinkhorn(res_logits + self.res_identity * 2.0)
        return h_pre, h_post, h_res


class ALV_ADBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_heads: int,
        n_streams: int,
        sinkhorn_iterations: int = 3,
    ):
        super().__init__()
        self.n_streams = n_streams
        self.attention = nn.MultiheadAttention(
            d_model,
            n_heads,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dynamic_router = DynamicResidualRouter(
            d_model=d_model,
            n_streams=n_streams,
            sinkhorn_iterations=sinkhorn_iterations,
        )

    def forward(self, x_stream: torch.Tensor) -> torch.Tensor:
        h_pre, h_post, h_res = self.dynamic_router(x_stream)

        layer_input = torch.einsum("bln,bnld->bld", h_pre, x_stream)
        attn_input = self.norm1(layer_input)
        attn_out, _ = self.attention(attn_input, attn_input, attn_input)
        ffn_out = self.ffn(self.norm2(layer_input + attn_out))
        layer_update = attn_out + ffn_out

        residual_state = torch.einsum("blmn,bnld->bmld", h_res, x_stream)
        stream_update = h_post.permute(0, 2, 1).unsqueeze(-1) * layer_update.unsqueeze(1)
        return residual_state + stream_update
