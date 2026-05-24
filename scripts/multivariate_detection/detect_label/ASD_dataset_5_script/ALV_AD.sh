python ./scripts/run_benchmark.py \
  --config-path "unfixed_detect_label_multi_config.json" \
  --data-name-list "ASD_dataset_5.csv" \
  --model-name "alv_ad_transformer.ALV_AD_Transformer" \
  --model-hyper-params '{"anomaly_ratio": 1.2, "batch_size": 128, "d_ff": 128, "d_model": 32, "e_layers": 2, "hybrid_score_lambda": 0.25, "lr": 5e-05, "n_heads": 4, "n_streams": 6, "num_epochs": 12, "patience": 8, "rvq_gate_floor": 0.3, "rvq_gate_temperature": 6.0, "rvq_grad_scale": 0.2, "rvq_num_embeddings": 64, "score_modes": "recon_stage1", "seq_len": 164, "sinkhorn_iterations": 3}' \
  --gpus 0 \
  --num-workers 1 \
  --timeout 60000 \
  --seed 2021 \
  --save-path "label/ALV_AD_Transformer_remaining_asd_label_asd_dataset_5_label_ar1p2"
