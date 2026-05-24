#!/usr/bin/env bash
set -euo pipefail
GPU=${GPU:-0}
python ./scripts/run_benchmark.py \
  --config-path "unfixed_detect_score_multi_config.json" \
  --data-name-list "SMAP.csv" \
  --model-name "alv_ad_transformer.ALV_AD_Transformer" \
  --model-hyper-params '{"anomaly_ratio": 25.0, "batch_size": 256, "d_ff": 256, "d_model": 64, "e_layers": 2, "hybrid_score_lambda": 0.2, "lr": 0.0001, "n_heads": 8, "n_streams": 6, "num_epochs": 10, "patience": 10, "rvq_gate_floor": 0.3, "rvq_gate_temperature": 4.0, "rvq_grad_scale": 0.2, "rvq_num_embeddings": 96, "score_modes": "recon_stage1", "seq_len": 100, "sinkhorn_iterations": 8}' \
  --gpus "${GPU}" \
  --num-workers 1 \
  --timeout 60000 \
  --seed 2021 \
  --save-path "score/ALV_AD_Transformer_final_repro_smap_score"
