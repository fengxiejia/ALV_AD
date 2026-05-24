#!/usr/bin/env bash
set -euo pipefail
GPU=${GPU:-0}
python ./scripts/run_benchmark.py \
  --config-path "unfixed_detect_label_multi_config.json" \
  --data-name-list "Genesis.csv" \
  --model-name "alv_ad_transformer.ALV_AD_Transformer" \
  --model-hyper-params '{"anomaly_ratio": 0.5, "batch_size": 128, "d_ff": 96, "d_model": 24, "e_layers": 2, "hybrid_score_lambda": 0.06, "lr": 0.0001, "n_heads": 4, "n_streams": 8, "num_epochs": 20, "patience": 10, "rvq_gate_floor": 0.4, "rvq_gate_temperature": 8.0, "rvq_grad_scale": 0.1, "rvq_num_embeddings": 96, "score_modes": "recon_stage1", "seq_len": 112, "sinkhorn_iterations": 3}' \
  --gpus "${GPU}" \
  --num-workers 1 \
  --timeout 60000 \
  --seed 2021 \
  --save-path "label/ALV_AD_Transformer_genesis_label_ar05_seq112_modes_dim24"
