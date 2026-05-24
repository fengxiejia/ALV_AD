python ./scripts/run_benchmark.py \
  --config-path "unfixed_detect_label_multi_config.json" \
  --data-name-list "ASD_dataset_9.csv" \
  --model-name "alv_ad_transformer.ALV_AD_Transformer" \
  --model-hyper-params '{"anomaly_ratio": 2.0, "batch_size": 128, "d_ff": 256, "d_model": 64, "e_layers": 2, "hybrid_score_lambda": 0.5, "lr": 0.0001, "n_heads": 8, "n_streams": 6, "num_epochs": 5, "patience": 5, "rvq_gate_floor": 0.3, "rvq_gate_temperature": 6.0, "rvq_grad_scale": 0.2, "rvq_num_embeddings": 96, "score_modes": "recon_stage1", "seq_len": 192, "sinkhorn_iterations": 3}' \
  --gpus 0 \
  --num-workers 1 \
  --timeout 60000 \
  --seed 2021 \
  --save-path "label/ALV_AD_Transformer_asd_dataset_9_catchar2p0_seq192"
