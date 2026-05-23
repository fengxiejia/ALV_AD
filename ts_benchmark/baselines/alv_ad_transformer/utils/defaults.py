DEFAULT_HYBRID_AD_HYPER_PARAMS = {
    "batch_size": 16,
    "lr": 0.0001,
    "num_epochs": 10,
    "patience": 3,
    "seq_len": 100,
    "horizon": 0,
    "lradj": "type1",
    "pct_start": 0.3,
    "d_model": 256,
    "d_ff": 1024,
    "e_layers": 2,
    "n_heads": 8,
    "n_streams": 4,
    "anomaly_ratio": [0.1, 0.5, 1.0, 2, 3, 5.0, 10.0, 15, 20, 25],
    "score_modes": "recon_stage1",
    "hybrid_score_lambda": 1.0,
    "rvq_num_embeddings": 128,
    "rvq_grad_scale": 0.25,
    "rvq_gate_floor": 0.2,
    "rvq_gate_temperature": 4.0,
}


class HybridADConfig:
    def __init__(self, **kwargs):
        for key, value in DEFAULT_HYBRID_AD_HYPER_PARAMS.items():
            setattr(self, key, value)
        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def pred_len(self):
        return self.seq_len
