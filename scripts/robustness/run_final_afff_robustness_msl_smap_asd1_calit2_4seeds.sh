#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${REPO_ROOT}"

if [[ "${ACTIVATE_CONDA:-1}" == "1" ]]; then
  CONDA_BASE="${CONDA_BASE:-$(conda info --base 2>/dev/null || true)}"
  if [[ -n "${CONDA_BASE}" && -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV:-mhc}"
  else
    echo "Warning: conda was not found; continuing with the current Python environment." >&2
  fi
fi

DATASETS=(MSL SMAP ASD_dataset_1 CalIt2)
METHODS=(CrossAD MtsCID MTAD-GAT USAD TranAD ALV-AD)
SEEDS=(2021 2022 2023 2024)
GPUS=(0 1 2)

BASE_OUT="${BASE_OUT:-final_ex/robustness_afff_final_msl_smap_asd1_calit2_4seeds}"
RUNNER="${RUNNER:-final_ex/run_robustness_requested_baselines.py}"
SUMMARIZER="${SUMMARIZER:-final_ex/summarize_multiseed_metric.py}"
PLOTTER="${PLOTTER:-final_ex/plot_final_afff_robustness_msl_smap_asd1_calit2.py}"

mkdir -p "${BASE_OUT}"
exec > >(tee -a "${BASE_OUT}/launcher.log") 2>&1

cat > "${BASE_OUT}/experiment_note.txt" <<'NOTE'
Final Aff-F1 robustness experiment.

Datasets:
  MSL
  SMAP
  ASD_dataset_1 (shown as ASD-1 in the paper)
  CalIt2

ASD-1 subset:
  data file: dataset/anomaly_detect/data/ASD_dataset_1.csv
  variables: 19
  train length: 8640
  test length: 4320
  test anomaly points: 441
  test anomaly ratio: 10.21%

Methods:
  CrossAD, MtsCID, MTAD-GAT, USAD, TranAD, ALV-AD

Protocol:
  label profile, Aff-F1, four seeds.
  Simulated training pollution ratios: 0%, 2%, 5%, 10%, 15%.
  Real anomaly insertion repeats: 1x, 2x, 3x, 4x.

Hyperparameters:
  The runner loads tuned configs from scripts/multivariate_detection/detect_label/<dataset>_script/<method>.sh.
  ALV-AD therefore uses each dataset's tuned ALV_AD.sh:
    MSL_script/ALV_AD.sh
    SMAP_script/ALV_AD.sh
    ASD_dataset_1_script/ALV_AD.sh
    CalIt2_script/ALV_AD.sh
NOTE

echo "=== final Aff-F1 robustness start $(date -Is) ==="
echo "base_out=${BASE_OUT}"
echo "datasets=${DATASETS[*]}"
echo "methods=${METHODS[*]}"
echo "seeds=${SEEDS[*]}"

for required_file in "${RUNNER}" "${SUMMARIZER}" "${PLOTTER}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "Missing helper script: ${required_file}" >&2
    echo "Set RUNNER, SUMMARIZER, or PLOTTER to the correct path if the helper lives elsewhere." >&2
    exit 1
  fi
done

for dataset in "${DATASETS[@]}"; do
  for method in "${METHODS[@]}"; do
    case "${method}" in
      MTAD-GAT) script_name="MTAD_GAT.sh" ;;
      ALV-AD) script_name="ALV_AD.sh" ;;
      *) script_name="${method}.sh" ;;
    esac
    script_path="scripts/multivariate_detection/detect_label/${dataset}_script/${script_name}"
    if [[ ! -f "${script_path}" ]]; then
      echo "Missing tuned script: ${script_path}" >&2
      exit 1
    fi
  done
done

for seed in "${SEEDS[@]}"; do
  OUT="${BASE_OUT}/seed_${seed}"
  mkdir -p "${OUT}"
  echo "=== seed ${seed} start $(date -Is) ==="
  ROBUSTNESS_PLOT_METRIC=aff_f python "${RUNNER}" \
    --profile label \
    --out-dir "${OUT}" \
    --datasets "${DATASETS[@]}" \
    --methods "${METHODS[@]}" \
    --gpus "${GPUS[@]}" \
    --poll-seconds 5 \
    --seed "${seed}"
  echo "=== seed ${seed} done $(date -Is) ==="
done

python "${SUMMARIZER}" \
  --base-out "${BASE_OUT}" \
  --out-dir "${BASE_OUT}/combined" \
  --metric aff_f

ROBUSTNESS_RAW="${BASE_OUT}/combined/robustness_multiseed_raw_metrics.csv" \
ROBUSTNESS_FIGURE_DIR="${BASE_OUT}/figures" \
python "${PLOTTER}"

echo "=== final Aff-F1 robustness done $(date -Is) ==="
echo "combined=${BASE_OUT}/combined"
echo "figures=${BASE_OUT}/figures"
