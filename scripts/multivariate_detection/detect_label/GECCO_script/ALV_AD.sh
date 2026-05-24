#!/usr/bin/env bash
set -euo pipefail
GPU=${GPU:-0}
python ./scripts/run_benchmark.py \
  --config-path "unfixed_detect_label_multi_config.json" \
  --data-name-list "GECCO.csv" \
  --model-name "alv_ad_transformer.ALV_AD_Transformer" \
  --model-hyper-params '{"anomaly_ratio": 2.0, "batch_size": 256, "d_ff": 64, "d_model": 16, "e_layers": 2, "hybrid_score_lambda": 0.001, "lr": 4.265e-05, "n_heads": 8, "n_streams": 6, "num_epochs": 10, "patience": 10, "rvq_gate_floor": 0.3, "rvq_gate_temperature": 7.0, "rvq_grad_scale": 0.2, "rvq_num_embeddings": 32, "score_modes": "recon_stage1", "seq_len": 100, "sinkhorn_iterations": 8}' \
  --gpus "${GPU}" \
  --num-workers 1 \
  --timeout 60000 \
  --seed 2021 \
  --save-path "label/ALV_AD_GECCO_opt_label_seq110_lr4265_lam001"
