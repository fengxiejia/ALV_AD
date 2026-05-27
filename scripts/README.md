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

## Robustness Scripts

Additional robustness experiment utilities are available under `scripts/robustness/`.

- `run_final_afff_robustness_msl_smap_asd1_calit2_4seeds.sh` launches the four-seed Aff-F1 training-pollution robustness experiment for MSL, SMAP, ASD-1, and CalIt2. This corresponds to the PVLDB manuscript RQ5 figure `robustness_aff_f_msl_smap_asd_dataset_1_calit2`.
- `plot_aucroc_score_full_msl_calit2_4seeds.py` plots four-seed AUC-ROC score robustness curves for MSL and CalIt2, together with an ALV-AD-versus-best-baseline margin heatmap.

See `scripts/robustness/README.md` for required inputs, environment variables, and full usage notes.
