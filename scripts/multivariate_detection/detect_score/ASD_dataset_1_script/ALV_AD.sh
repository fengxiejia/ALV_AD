#!/usr/bin/env bash
set -euo pipefail
GPU=${GPU:-0}
python ./scripts/run_benchmark.py \
  --config-path "unfixed_detect_score_multi_config.json" \
  --data-name-list "ASD_dataset_1.csv" \
  --model-name "alv_ad_transformer.ALV_AD_Transformer" \
  --model-hyper-params '{"anomaly_ratio": 0.1, "batch_size": 64, "d_ff": 128, "d_model": 256, "e_layers": 3, "hybrid_score_lambda": 500.0, "lr": 0.0001, "n_heads": 16, "n_streams": 6, "num_epochs": 10, "patience": 8, "rvq_gate_floor": 0.1, "rvq_gate_temperature": 4.0, "rvq_grad_scale": 0.3, "rvq_num_embeddings": 64, "score_modes": "recon_rvq", "seq_len": 320, "sinkhorn_iterations": 3}' \
  --gpus "${GPU}" \
  --num-workers 1 \
  --timeout 60000 \
  --seed 2021 \
  --save-path "score/ALV_AD_Transformer_asd_catchd_score/asd_dataset_1_t2_seq320_bs64_lam500p0_rvq"
