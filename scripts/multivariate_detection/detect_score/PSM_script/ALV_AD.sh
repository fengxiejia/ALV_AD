python ./scripts/run_benchmark.py \
  --config-path "unfixed_detect_score_multi_config.json" \
  --data-name-list "PSM.csv" \
  --model-name "alv_ad_transformer.ALV_AD_Transformer" \
  --model-hyper-params '{"batch_size": 40, "d_ff": 192, "d_model": 128, "e_layers": 2, "hybrid_score_lambda": 0.0097, "lr": 0.000156, "n_heads": 8, "n_streams": 4, "num_epochs": 5, "patience": 5, "rvq_gate_floor": 0.0, "rvq_gate_temperature": 4.5, "rvq_grad_scale": 0.3, "rvq_num_embeddings": 32, "score_modes": "recon_stage1", "seq_len": 100, "sinkhorn_iterations": 3}' \
  --gpus 0 \
  --num-workers 1 \
  --timeout 60000 \
  --seed 2021 \
  --save-path "score/ALV_AD_Transformer_psm_score_gatebest_b40_lr156e6"
