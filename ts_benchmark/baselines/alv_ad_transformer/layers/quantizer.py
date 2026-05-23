import torch
import torch.nn as nn
import torch.nn.functional as F


def _compute_distance(
    flat: torch.Tensor,
    codebook: torch.Tensor,
) -> torch.Tensor:
    flat_norm = F.normalize(flat, dim=-1)
    codebook_norm = F.normalize(codebook, dim=-1)
    return 1.0 - flat_norm @ codebook_norm.t()


class VectorQuantizer(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)

        self.codebook = nn.Parameter(torch.empty(self.num_embeddings, self.embedding_dim))
        nn.init.trunc_normal_(self.codebook, std=0.02)

    def forward(self, z: torch.Tensor):
        bsz, n_vars, dim = z.shape
        flat = z.reshape(-1, dim)
        dist = _compute_distance(flat, self.codebook)

        indices = dist.argmin(dim=-1)
        quantized = F.embedding(indices, self.codebook).view(bsz, n_vars, dim)
        token_dist = (z.detach() - quantized.detach()).pow(2).mean(dim=-1)
        return quantized, token_dist


class ResidualVectorQuantizer(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        stage_decay: float = 0.7,
        grad_scale: float = 0.25,
    ):
        super().__init__()
        self.stage_decay = stage_decay
        self.grad_scale = grad_scale
        self.num_quantizers = 2
        self.quantizers = nn.ModuleList(
            [
                VectorQuantizer(
                    num_embeddings=num_embeddings,
                    embedding_dim=embedding_dim,
                )
                for _ in range(self.num_quantizers)
            ]
        )

    def forward(self, z: torch.Tensor):
        residual = z
        quantized_sum = torch.zeros_like(z)
        token_dists = []
        for quantizer in self.quantizers:
            quantized, token_dist = quantizer(residual)
            quantized_sum = quantized_sum + quantized
            residual = residual - quantized.detach()
            token_dists.append(token_dist)

        weights = z.new_tensor(
            [self.stage_decay ** idx for idx in range(len(self.quantizers))]
        )
        weights = weights / weights.sum()
        stacked_dists = torch.stack(token_dists, dim=0)
        token_dist = torch.einsum("q,qbn->bn", weights, stacked_dists)

        quantized_st = (
            z
            + (quantized_sum - z).detach()
            + self.grad_scale * (quantized_sum - quantized_sum.detach())
        )
        return quantized_st, token_dist, stacked_dists
