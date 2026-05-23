import numpy as np
import torch


def zscore(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    return (values - mean) / max(float(std), 1e-6)


def normalize_stage_token_dist(token_dist: torch.Tensor, stage_token_dist: torch.Tensor):
    if stage_token_dist is None:
        return None
    if stage_token_dist.dim() != 3:
        return stage_token_dist
    if stage_token_dist.shape[0] <= 8 and stage_token_dist.shape[1] == token_dist.shape[0]:
        return stage_token_dist.permute(1, 0, 2).contiguous()
    return stage_token_dist


def derive_quantizer_time_scores(model, rec_loss: torch.Tensor):
    token_dist = getattr(model, "latest_quantizer_token_dist", None)
    stage_token_dist = getattr(model, "latest_quantizer_stage_token_dist", None)
    if token_dist is None or stage_token_dist is None:
        return None, []

    stage_token_dist = normalize_stage_token_dist(token_dist, stage_token_dist)
    seq_len = rec_loss.shape[1]
    input_dim = rec_loss.shape[2]

    if token_dist.dim() == 2 and token_dist.shape[1] == input_dim:
        weights = torch.softmax(token_dist, dim=-1)
        weighted_score = (rec_loss * weights.unsqueeze(1)).sum(dim=-1)
        stage_scores = []
        for stage_idx in range(stage_token_dist.shape[1]):
            stage_weights = torch.softmax(stage_token_dist[:, stage_idx], dim=-1)
            stage_scores.append((rec_loss * stage_weights.unsqueeze(1)).sum(dim=-1))
        return weighted_score, stage_scores

    if token_dist.dim() == 2 and token_dist.shape[1] == seq_len:
        return token_dist, [
            stage_token_dist[:, stage_idx] for stage_idx in range(stage_token_dist.shape[1])
        ]

    return None, []
