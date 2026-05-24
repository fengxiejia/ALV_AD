#!/usr/bin/env bash
set -euo pipefail
GPU=${GPU:-0}
python ./scripts/run_benchmark.py \
  --config-path "unfixed_detect_label_multi_config.json" \
  --data-name-list "SMAP.csv" \
  --model-name "alv_ad_transformer.ALV_AD_Transformer" \
  --model-hyper-params '{"anomaly_ratio": 25.0, "batch_size": 256, "d_ff": 256, "d_model": 64, "e_layers": 2, "hybrid_score_lambda": 0.5, "lr": 0.0001, "n_heads": 8, "n_streams": 8, "num_epochs": 10, "patience": 10, "rvq_gate_floor": 0.3, "rvq_gate_temperature": 6.0, "rvq_grad_scale": 0.1, "rvq_num_embeddings": 48, "score_modes": "recon_stage1", "seq_len": 150, "sinkhorn_iterations": 8}' \
  --gpus "${GPU}" \
  --num-workers 1 \
  --timeout 60000 \
  --seed 2021 \
  --save-path "label/ALV_AD_UnfoldRetune_label_smap_focus1_seq150_emb48_grad01"
