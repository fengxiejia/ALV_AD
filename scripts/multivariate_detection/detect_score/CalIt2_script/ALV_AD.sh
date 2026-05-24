python ./scripts/run_benchmark.py \
  --config-path "unfixed_detect_score_multi_config.json" \
  --data-name-list "CalIt2.csv" \
  --model-name "alv_ad_transformer.ALV_AD_Transformer" \
  --model-hyper-params '{"batch_size": 255, "d_ff": 512, "d_model": 128, "e_layers": 2, "hybrid_score_lambda": 0.1168, "lr": 0.000178, "n_heads": 32, "n_streams": 7, "num_epochs": 2, "patience": 2, "rvq_gate_floor": 0.881, "rvq_gate_temperature": 1.2, "rvq_grad_scale": 0.5, "rvq_num_embeddings": 384, "score_modes": "recon_rvq", "seq_len": 192, "sinkhorn_iterations": 12}' \
  --gpus 0 \
  --num-workers 1 \
  --timeout 60000 \
  --seed 2021 \
  --save-path "score/ALV_AD_Transformer_callt2_score_512x128_e2_peak54_floor0881_temp12_rvq"
