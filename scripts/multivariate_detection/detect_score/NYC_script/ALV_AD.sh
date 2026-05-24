python ./scripts/run_benchmark.py \
  --config-path "unfixed_detect_score_multi_config.json" \
  --data-name-list "NYC.csv" \
  --model-name "alv_ad_transformer.ALV_AD_Transformer" \
  --model-hyper-params '{"batch_size": 128, "d_ff": 96, "d_model": 24, "e_layers": 2, "hybrid_score_lambda": 1e-06, "lr": 0.0001, "n_heads": 8, "n_streams": 6, "num_epochs": 8, "patience": 5, "rvq_gate_floor": 0.3, "rvq_gate_temperature": 2.0, "rvq_grad_scale": 0.2, "rvq_num_embeddings": 48, "score_modes": "recon_stage1", "seq_len": 100, "sinkhorn_iterations": 3}' \
  --gpus 0 \
  --num-workers 1 \
  --timeout 60000 \
  --seed 2021 \
  --save-path "score/ALV_AD_Transformer_final_repro_nyc_score_nyc_lam1e_6"
