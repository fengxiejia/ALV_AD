#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
import time
from collections import deque
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

SCRIPT_DIRS = {
    "MSL": "MSL_script",
    "CalIt2": "CalIt2_script",
    "GECCO": "GECCO_script",
    "NYC": "NYC_script",
    "SWAT": "SWAT_script",
    "SMAP": "SMAP_script",
    "Genesis": "Genesis_script",
}

DATASET_ALIASES = {
    "Callt2": "CalIt2",
    "Calt2": "CalIt2",
    "Calit2": "CalIt2",
    "SWaT": "SWAT",
    "Swat": "SWAT",
}


def canonical_dataset(dataset: str) -> str:
    return DATASET_ALIASES.get(dataset, dataset)


def load_alvad_params(dataset: str, protocol: str):
    dataset = canonical_dataset(dataset)
    script = (
        ROOT
        / "scripts"
        / "multivariate_detection"
        / protocol
        / SCRIPT_DIRS[dataset]
        / "ALV_AD.sh"
    )
    text = script.read_text()
    match = re.search(r"--model-hyper-params\s+'([^']+)'", text)
    if not match:
        raise ValueError(f"Cannot parse --model-hyper-params from {script}")
    config_match = re.search(r'--config-path\s+"([^"]+)"', text)
    if not config_match:
        raise ValueError(f"Cannot parse --config-path from {script}")
    return json.loads(match.group(1)), config_match.group(1), str(script.relative_to(ROOT))


def make_command(dataset: str, protocol: str, variant: str, seed: int, save_root: str):
    dataset = canonical_dataset(dataset)
    params, config_path, source = load_alvad_params(dataset, protocol)
    if variant != "full":
        params["ablation_variant"] = variant
    save_path = f"{protocol.replace('detect_', '')}/{save_root}/{dataset.lower()}_{variant}"
    cmd = [
        "python",
        "./scripts/run_benchmark.py",
        "--config-path",
        config_path,
        "--data-name-list",
        f"{dataset}.csv",
        "--model-name",
        "alv_ad_transformer.ALV_AD_Transformer",
        "--model-hyper-params",
        json.dumps(params, sort_keys=True),
        "--gpus",
        "{GPU}",
        "--num-workers",
        "1",
        "--timeout",
        "60000",
        "--seed",
        str(seed),
        "--save-path",
        save_path,
    ]
    return cmd, params, source, save_path


def run_job(job, gpu: str, log_dir: Path):
    dataset, protocol, variant, seed, save_root = job
    cmd, params, source, save_path = make_command(dataset, protocol, variant, seed, save_root)
    cmd = [gpu if part == "{GPU}" else part for part in cmd]
    log_path = log_dir / f"{dataset.lower()}_{variant}.log"
    meta_path = log_dir / f"{dataset.lower()}_{variant}.json"
    meta_path.write_text(
        json.dumps(
            {
                "dataset": dataset,
                "protocol": protocol,
                "variant": variant,
                "seed": seed,
                "gpu": gpu,
                "param_source": source,
                "params": params,
                "save_path": save_path,
                "cmd": cmd,
            },
            indent=2,
        )
    )
    log = log_path.open("w")
    env = os.environ.copy()
    env["MPLCONFIGDIR"] = "/tmp/alv_ad_matplotlib"
    proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    return proc, log, log_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", choices=["detect_label", "detect_score"], required=True)
    parser.add_argument("--datasets", nargs="+", default=["MSL", "CalIt2", "GECCO", "SWAT", "SMAP", "Genesis"])
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["full", "lsr_only", "rvq_only", "lsr_no_sinkhorn", "one_stage_vq", "uniform_rw"],
    )
    parser.add_argument("--gpus", nargs="+", default=["0"])
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--max-per-gpu", type=int, default=1)
    parser.add_argument("--save-root", default=None)
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--poll-seconds", type=int, default=20)
    args = parser.parse_args()
    args.datasets = [canonical_dataset(dataset) for dataset in args.datasets]

    if args.save_root is None:
        tag = "label" if args.protocol == "detect_label" else "score"
        args.save_root = f"grouped_benchmark_{tag}_alv_ad_20260527"
    if args.log_dir is None:
        args.log_dir = f"final_ex/{args.save_root}/logs"

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    jobs = deque(
        (dataset, args.protocol, variant, args.seed, args.save_root)
        for dataset in args.datasets
        for variant in args.variants
    )
    running = {gpu: [] for gpu in args.gpus}
    completed = []
    failed = []

    while jobs or any(running.values()):
        for gpu in args.gpus:
            alive = []
            for proc, log, log_path, job in running[gpu]:
                if proc.poll() is None:
                    alive.append((proc, log, log_path, job))
                else:
                    log.close()
                    rc = proc.returncode
                    completed.append((job, rc, str(log_path)))
                    if rc != 0:
                        failed.append((job, rc, str(log_path)))
                    print(
                        json.dumps(
                            {
                                "event": "finished",
                                "job": job,
                                "gpu": gpu,
                                "returncode": rc,
                                "log": str(log_path),
                                "completed": len(completed),
                                "remaining": len(jobs),
                            }
                        ),
                        flush=True,
                    )
            running[gpu] = alive
            while jobs and len(running[gpu]) < args.max_per_gpu:
                job = jobs.popleft()
                proc, log, log_path = run_job(job, gpu, log_dir)
                running[gpu].append((proc, log, log_path, job))
                print(
                    json.dumps(
                        {
                            "event": "started",
                            "job": job,
                            "gpu": gpu,
                            "pid": proc.pid,
                            "log": str(log_path),
                            "remaining": len(jobs),
                        }
                    ),
                    flush=True,
                )
        if jobs or any(running.values()):
            time.sleep(args.poll_seconds)

    summary = {
        "completed": len(completed),
        "failed": failed,
        "save_root": args.save_root,
        "log_dir": str(log_dir),
    }
    (log_dir.parent / "run_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
