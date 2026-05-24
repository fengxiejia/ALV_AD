python ./scripts/run_benchmark.py \
  --config-path "unfixed_detect_score_multi_config.json" \
  --data-name-list "ASD_dataset_4.csv" \
  --model-name "alv_ad_transformer.ALV_AD_Transformer" \
  --model-hyper-params '{"batch_size": 64, "d_ff": 256, "d_model": 128, "e_layers": 3, "hybrid_score_lambda": 1e-08, "lr": 0.0001, "n_heads": 8, "n_streams": 6, "num_epochs": 10, "patience": 8, "rvq_gate_floor": 0.3, "rvq_gate_temperature": 6.0, "rvq_grad_scale": 0.2, "rvq_num_embeddings": 64, "score_modes": "recon_stage1", "seq_len": 112, "sinkhorn_iterations": 3}' \
  --gpus 0 \
  --num-workers 1 \
  --timeout 60000 \
  --seed 2021 \
  --save-path "score/ALV_AD_Transformer_asd_catchd_score/asd_dataset_4_t2_seq112_ff256_l3_stage1"
