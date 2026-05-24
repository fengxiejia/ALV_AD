python ./scripts/run_benchmark.py \
  --config-path "unfixed_detect_label_multi_config.json" \
  --data-name-list "Creditcard.csv" \
  --model-name "alv_ad_transformer.ALV_AD_Transformer" \
  --model-hyper-params '{"anomaly_ratio": 2.0, "batch_size": 128, "d_ff": 64, "d_model": 30, "e_layers": 2, "hybrid_score_lambda": 1e-08, "lr": 6e-05, "n_heads": 5, "n_streams": 6, "num_epochs": 12, "patience": 8, "rvq_gate_floor": 0.3, "rvq_gate_temperature": 1.7, "rvq_grad_scale": 0.2, "rvq_num_embeddings": 32, "score_modes": "recon_stage1", "seq_len": 192, "sinkhorn_iterations": 3}' \
  --gpus 0 \
  --num-workers 1 \
  --timeout 60000 \
  --seed 2021 \
  --save-path "label/ALV_AD_Transformer_final_repro_credit_label"
