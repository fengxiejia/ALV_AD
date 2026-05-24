python ./scripts/run_benchmark.py \
  --config-path "unfixed_detect_label_multi_config.json" \
  --data-name-list "CalIt2.csv" \
  --model-name "alv_ad_transformer.ALV_AD_Transformer" \
  --model-hyper-params '{"anomaly_ratio": 5.5, "batch_size": 128, "d_ff": 16, "d_model": 4, "e_layers": 1, "hybrid_score_lambda": 1.0, "lr": 0.0001, "n_heads": 4, "n_streams": 10, "num_epochs": 20, "patience": 5, "rvq_gate_floor": 0.3, "rvq_gate_temperature": 8.0, "rvq_grad_scale": 0.1, "rvq_num_embeddings": 8, "score_modes": "recon_stage1", "seq_len": 186, "sinkhorn_iterations": 10}' \
  --gpus 0 \
  --num-workers 1 \
  --timeout 60000 \
  --seed 2021 \
  --save-path "label/ALV_AD_Transformer_final_repro_callt2_label"
