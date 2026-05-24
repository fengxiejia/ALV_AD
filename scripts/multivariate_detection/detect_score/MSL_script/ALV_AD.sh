python ./scripts/run_benchmark.py \
  --config-path "unfixed_detect_score_multi_config.json" \
  --data-name-list "MSL.csv" \
  --model-name "alv_ad_transformer.ALV_AD_Transformer" \
  --model-hyper-params '{"batch_size": 256, "d_ff": 256, "d_model": 128, "e_layers": 2, "hybrid_score_lambda": 1e-12, "lr": 0.000025, "n_heads": 8, "n_streams": 4, "num_epochs": 5, "patience": 5, "rvq_gate_floor": 0.3, "rvq_gate_temperature": 6.8, "rvq_grad_scale": 0.3, "rvq_num_embeddings": 16, "score_modes": "recon_stage1", "seq_len": 122, "sinkhorn_iterations": 8}' \
  --gpus 0 \
  --num-workers 1 \
  --timeout 60000 \
  --seed 2021 \
  --save-path "score/ALV_AD_Transformer_msl_score_dm128_stage1_ep5_seq122_lr25e6"
