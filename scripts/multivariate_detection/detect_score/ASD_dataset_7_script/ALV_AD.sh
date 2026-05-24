python ./scripts/run_benchmark.py \
  --config-path "unfixed_detect_score_multi_config.json" \
  --data-name-list "ASD_dataset_7.csv" \
  --model-name "alv_ad_transformer.ALV_AD_Transformer" \
  --model-hyper-params '{"batch_size": 128, "d_ff": 128, "d_model": 128, "e_layers": 2, "hybrid_score_lambda": 1000.0, "lr": 0.0001, "n_heads": 8, "n_streams": 6, "num_epochs": 10, "patience": 8, "rvq_gate_floor": 0.3, "rvq_gate_temperature": 6.0, "rvq_grad_scale": 0.2, "rvq_num_embeddings": 96, "score_modes": "recon_rvq", "seq_len": 128, "sinkhorn_iterations": 3}' \
  --gpus 0 \
  --num-workers 1 \
  --timeout 60000 \
  --seed 2021 \
  --save-path "score/ALV_AD_Transformer_asd_catchd_score/asd_dataset_7_t2_seq128_temp6p0_s6_rvq"
