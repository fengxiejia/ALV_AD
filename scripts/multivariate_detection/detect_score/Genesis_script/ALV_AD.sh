#!/usr/bin/env bash
set -euo pipefail
GPU=${GPU:-0}
python ./scripts/run_benchmark.py \
  --config-path "unfixed_detect_score_multi_config.json" \
  --data-name-list "Genesis.csv" \
  --model-name "alv_ad_transformer.ALV_AD_Transformer" \
  --model-hyper-params '{"anomaly_ratio": 0.5, "batch_size": 249, "d_ff": 256, "d_model": 64, "e_layers": 2, "hybrid_score_lambda": 3000, "lr": 0.000574, "n_heads": 2, "n_streams": 6, "num_epochs": 5, "patience": 2, "rvq_gate_floor": 0.045, "rvq_gate_temperature": 12.5, "rvq_grad_scale": 0.2, "rvq_num_embeddings": 32, "score_modes": "recon_stage1", "seq_len": 184, "sinkhorn_iterations": 4}' \
  --gpus "${GPU}" \
  --num-workers 1 \
  --timeout 60000 \
  --seed 2021 \
  --save-path "score/ALV_AD_Transformer_genesis_score_ar05_floor005_fine_f0p045_t12p5_lr0p000574"
