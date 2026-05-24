#!/usr/bin/env bash
set -euo pipefail
GPU=${GPU:-0}
python ./scripts/run_benchmark.py \
  --config-path "unfixed_detect_label_multi_config.json" \
  --data-name-list "NYC.csv" \
  --model-name "alv_ad_transformer.ALV_AD_Transformer" \
  --model-hyper-params '{"anomaly_ratio": 0.12, "batch_size": 128, "d_ff": 32, "d_model": 8, "e_layers": 2, "hybrid_score_lambda": 0.005, "lr": 5e-05, "n_heads": 2, "n_streams": 8, "num_epochs": 10, "patience": 8, "rvq_gate_floor": 0.3, "rvq_gate_temperature": 6.0, "rvq_grad_scale": 0.2, "rvq_num_embeddings": 16, "score_modes": "recon_stage1", "seq_len": 164, "sinkhorn_iterations": 3}' \
  --gpus "${GPU}" \
  --num-workers 1 \
  --timeout 60000 \
  --seed 2021 \
  --save-path "label/ALV_AD_Transformer_dbmargin_nyc_over_ar0p12_lam0p005"
