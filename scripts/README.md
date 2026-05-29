# Scripts

This directory contains curated run scripts for real-world multivariate time-series anomaly detection benchmarks.

## Scope

- Included tasks: `detect_label` and `detect_score`.
- Included datasets: real-world benchmark folders under `scripts/multivariate_detection`.
- Excluded datasets: all `synthetic_*` simulation folders.
- Included methods: `ALV_AD`, `CATCH`, `CrossAD`, `MtsCID`, `iTransformer`, `DCdetector`, `TimesNet`, `PatchTST`, `AnomalyTransformer`, `OmniAnomaly`, `TranAD`, `USAD`, `GDN`, `MTAD_GAT`, `DLinear`, `MEMTO`, and `IsolationForest`.

The ALV_AD run scripts use the following model entry:

```text
alv_ad_transformer.ALV_AD_Transformer
```

## Notes

These shell files are benchmark run configurations. The benchmark runner is included as `scripts/run_benchmark.py`.

## Label-Free Threshold Figure

`scripts/build_label_free_threshold_figure.py` builds the label-free threshold analysis figure used for the paper's raw-score and thresholding study. It combines a synthetic-global score distribution panel with dataset-level Aff-F comparisons under `Best`, `TopK`, `POT`, and `SD` threshold rules.

This corresponds to the PVLDB manuscript figure `label_free_threshold_analysis_synthetic` in the RQ3 thresholding discussion.

Example:

```bash
python scripts/build_label_free_threshold_figure.py --single-column
```

This plotting utility uses `numpy`, `pandas`, and `matplotlib`, which are included in `requirements.txt`.

Expected default inputs:

```text
final_ex/synthetic_threshold_score_groups_20260528/global_synthetic_threshold_scores.npz
final_ex/threshold_selection_multimodel_native_window_msl_cicids_swat_nyc_20260529/threshold_multimodel_raw.csv
```

You can override them with `--score-npz` and `--threshold-csv`. Use `--no-copy-paper` if you only want to write the generated figure to `--out-dir` without updating a local paper directory.

## Grouped Ablation Runner

`scripts/run_benchmark_grouped_ablations.py` launches grouped ALV-AD ablation benchmarks across datasets and GPUs. It reads each dataset's tuned `ALV_AD.sh` configuration, swaps in the requested `ablation_variant`, and writes one log plus one JSON metadata file per dataset-variant job.

The default ablation variants are:

```text
full
lsr_only
rvq_only
lsr_no_sinkhorn
one_stage_vq
uniform_rw
```

Example score-protocol run:

```bash
python scripts/run_benchmark_grouped_ablations.py \
  --protocol detect_score \
  --datasets CalIt2 MSL SWAT SMAP \
  --variants full lsr_only rvq_only lsr_no_sinkhorn one_stage_vq uniform_rw \
  --gpus 0 1 2 \
  --seed 2021 \
  --max-per-gpu 1 \
  --save-root grouped_benchmark_score_alv_ad_final \
  --log-dir final_ex/grouped_benchmark_score_alv_ad_final/logs
```

## Robustness Scripts

Additional robustness experiment utilities are available under `scripts/robustness/`.

- `run_final_afff_robustness_msl_smap_asd1_calit2_4seeds.sh` launches the four-seed Aff-F1 training-pollution robustness experiment for MSL, SMAP, ASD-1, and CalIt2. This corresponds to the PVLDB manuscript RQ5 figure `robustness_aff_f_msl_smap_asd_dataset_1_calit2`.
- `plot_aucroc_score_full_msl_calit2_4seeds.py` plots four-seed AUC-ROC score robustness curves for MSL and CalIt2, together with an ALV-AD-versus-best-baseline margin heatmap.

See `scripts/robustness/README.md` for required inputs, environment variables, and full usage notes.
