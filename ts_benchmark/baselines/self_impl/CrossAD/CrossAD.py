from types import SimpleNamespace

import torch
import torch.nn.functional as F

from .Basic_CrossAD import Basic_CrossAD
from ts_benchmark.baselines.self_impl._detection_adapter import DetectionAdapter


class CrossAD(DetectionAdapter):
    def _crossad_config(self):
        return SimpleNamespace(
            seq_len=int(self.config["seq_len"]),
            patch_len=int(self.config.get("patch_len", 8)),
            d_model=int(self.config.get("d_model", 128)),
            ms_kernels=list(self.config.get("ms_kernels", [2, 4])),
            ms_method=self.config.get("ms_method", "interval_sampling"),
            norm=self.config.get("norm", "layer"),
            attn_dropout=float(self.config.get("attn_dropout", self.config.get("dropout", 0.1))),
            proj_dropout=float(self.config.get("proj_dropout", self.config.get("dropout", 0.1))),
            n_heads=int(self.config.get("n_heads", 4)),
            d_ff=int(self.config.get("d_ff", 256)),
            ff_dropout=float(self.config.get("ff_dropout", self.config.get("dropout", 0.1))),
            activation=self.config.get("activation", "gelu"),
            e_layers=int(self.config.get("e_layers", 1)),
            d_layers=int(self.config.get("d_layers", 1)),
            m_layers=int(self.config.get("m_layers", 1)),
            n_query=int(self.config.get("n_query", 4)),
            query_len=int(self.config.get("query_len", 4)),
            bank_size=int(self.config.get("bank_size", 64)),
            topk=int(self.config.get("topk", 4)),
            decay=float(self.config.get("decay", 0.99)),
            epsilon=float(self.config.get("epsilon", 1e-5)),
        )

    def _make_model(self, n_features: int):
        del n_features
        return Basic_CrossAD(self._crossad_config())

    def _loss(self, batch: torch.Tensor, epoch: int) -> torch.Tensor:
        del epoch
        recon_loss, query_loss = self.model(batch, None, None, None)
        return recon_loss + float(self.config.get("query_loss_weight", 0.1)) * query_loss

    def _score_batch(self, batch: torch.Tensor) -> torch.Tensor:
        score, query_score = self.model.infer(batch, None, None, None)
        score = score.mean(dim=-1)
        return score + float(self.config.get("query_score_weight", 0.0)) * query_score.mean(dim=-1)
