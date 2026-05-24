python ./scripts/run_benchmark.py \
  --config-path "unfixed_detect_label_multi_config.json" \
  --data-name-list "SMD.csv" \
  --model-name "alv_ad_transformer.ALV_AD_Transformer" \
  --model-hyper-params '{"anomaly_ratio": 1.8, "batch_size": 64, "d_ff": 64, "d_model": 32, "e_layers": 1, "hybrid_score_lambda": 1e-07, "lr": 0.0001, "n_heads": 2, "n_streams": 6, "num_epochs": 15, "patience": 5, "rvq_gate_floor": 0.3, "rvq_gate_temperature": 6.0, "rvq_grad_scale": 0.4, "rvq_num_embeddings": 24, "score_modes": "recon_stage1", "seq_len": 100, "sinkhorn_iterations": 10}' \
  --gpus 0 \
  --num-workers 1 \
  --timeout 60000 \
  --seed 2021 \
  --save-path "label/ALV_AD_Transformer_final_repro_smd_label_smd_ar1_8_lam1e_07"
