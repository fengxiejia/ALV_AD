python ./scripts/run_benchmark.py \
  --config-path "unfixed_detect_score_multi_config.json" \
  --data-name-list "GECCO.csv" \
  --model-name "alv_ad_transformer.ALV_AD_Transformer" \
  --model-hyper-params '{"batch_size": 256, "d_ff": 256, "d_model": 64, "e_layers": 2, "hybrid_score_lambda": 1e-08, "lr": 3.5e-05, "n_heads": 8, "n_streams": 6, "num_epochs": 20, "patience": 10, "rvq_gate_floor": 0.18, "rvq_gate_temperature": 7.25, "rvq_grad_scale": 0.2, "rvq_num_embeddings": 96, "score_modes": "recon_stage1", "seq_len": 150, "sinkhorn_iterations": 10}' \
  --gpus 0 \
  --num-workers 1 \
  --timeout 60000 \
  --seed 2021 \
  --save-path "score/ALV_AD_Transformer_final_repro_gecco_score_gecco_pos_lam1e_08_seq150"
