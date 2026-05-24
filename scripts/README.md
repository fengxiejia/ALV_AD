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

These shell files are benchmark run configurations. They assume the benchmark runner is available as `scripts/run_benchmark.py` in the full experiment environment.
