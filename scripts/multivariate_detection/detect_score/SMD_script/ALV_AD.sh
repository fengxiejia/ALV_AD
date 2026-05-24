#!/usr/bin/env bash
set -euo pipefail
GPU=${GPU:-0}
python ./scripts/run_benchmark.py \
  --config-path "unfixed_detect_score_multi_config.json" \
  --data-name-list "SMD.csv" \
  --model-name "alv_ad_transformer.ALV_AD_Transformer" \
  --model-hyper-params '{"anomaly_ratio": 5.0, "batch_size": 64, "d_ff": 512, "d_model": 128, "e_layers": 2, "hybrid_score_lambda": 0.001, "lr": 0.0001, "n_heads": 8, "n_streams": 6, "num_epochs": 10, "patience": 8, "rvq_gate_floor": 0.3, "rvq_gate_temperature": 8.0, "rvq_grad_scale": 0.2, "rvq_num_embeddings": 96, "score_modes": "recon", "seq_len": 100, "sinkhorn_iterations": 10}' \
  --gpus "${GPU}" \
  --num-workers 1 \
  --timeout 60000 \
  --seed 2021 \
  --save-path "score/ALV_AD_Transformer_remaining_smd_score_mode_recon"
