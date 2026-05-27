# Robustness Scripts

This directory contains auxiliary scripts for robustness experiments and figures used in the ALV-AD paper.

## `run_final_afff_robustness_msl_smap_asd1_calit2_4seeds.sh`

Launches the four-seed Aff-F1 training-pollution robustness experiment on `MSL`, `SMAP`, `ASD_dataset_1`, and `CalIt2`.

This script corresponds to the PVLDB manuscript robustness study in `pvldbstyle-master/sections/04_experiments.tex` under `RQ5: Robustness to Training-Set Pollution`, and to the figure:

```text
pvldbstyle-master/figures/robustness_aff_f_msl_smap_asd_dataset_1_calit2.{pdf,png}
```

The experiment compares `ALV-AD`, `CrossAD`, `MtsCID`, `MTAD-GAT`, `USAD`, and `TranAD` under two contamination protocols:

- simulated training pollution at `0%`, `2%`, `5%`, `10%`, and `15%`;
- repeated insertion of real anomaly segments from the test split at `1x`, `2x`, `3x`, and `4x`.

It expects the helper scripts below to be available, because the runner and summarizer were part of the full experiment environment:

```text
final_ex/run_robustness_requested_baselines.py
final_ex/summarize_multiseed_metric.py
final_ex/plot_final_afff_robustness_msl_smap_asd1_calit2.py
```

By default, the script activates the `mhc` conda environment if conda is available. Override this behavior with:

```bash
CONDA_ENV=my_env sh scripts/robustness/run_final_afff_robustness_msl_smap_asd1_calit2_4seeds.sh
ACTIVATE_CONDA=0 sh scripts/robustness/run_final_afff_robustness_msl_smap_asd1_calit2_4seeds.sh
```

## `plot_aucroc_score_full_msl_calit2_4seeds.py`

Plots the four-seed AUC-ROC score robustness curves for `MSL` and `CalIt2`, plus a heatmap showing the margin between `ALV-AD` and the strongest baseline.

Expected input files:

```text
robustness_multiseed_mean_std.csv
robustness_multiseed_auc_roc_alv_vs_best.csv
```

The input and output paths can be configured with environment variables:

```bash
ROBUSTNESS_COMBINED_DIR=/path/to/combined \
ROBUSTNESS_FIGURE_DIR=/path/to/figures \
python scripts/robustness/plot_aucroc_score_full_msl_calit2_4seeds.py
```

The script writes both PNG and PDF figures.
