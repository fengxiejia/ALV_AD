python ./scripts/run_benchmark.py \
  --config-path "unfixed_detect_score_multi_config.json" \
  --data-name-list "CICIDS.csv" \
  --model-name "alv_ad_transformer.ALV_AD_Transformer" \
  --model-hyper-params '{"batch_size": 128, "d_ff": 128, "d_model": 128, "e_layers": 3, "hybrid_score_lambda": 163840.0, "lr": 5e-05, "n_heads": 16, "n_streams": 6, "num_epochs": 5, "patience": 5, "rvq_gate_floor": 0.3, "rvq_gate_temperature": 4.0, "rvq_grad_scale": 0.14, "rvq_num_embeddings": 96, "score_modes": "recon_stage1", "seq_len": 192, "sinkhorn_iterations": 3}' \
  --gpus 0 \
  --num-workers 1 \
  --timeout 60000 \
  --seed 2021 \
  --save-path "score/ALV_AD_Transformer_cicids_dmodel128/catchshape_seq192_lr5em05"
