#!/usr/bin/env python3
from pathlib import Path
import os

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("PAPER_ROOT", REPO_ROOT))
DATA_DIR = Path(
    os.environ.get(
        "ROBUSTNESS_COMBINED_DIR",
        ROOT / "final_ex/robustness_aucroc_score_full_msl_calit2_20260526/combined",
    )
)
OUT_DIR = Path(os.environ.get("ROBUSTNESS_FIGURE_DIR", ROOT / "figures_preview"))


METHODS = ["ALV-AD", "CrossAD", "MtsCID", "MTAD-GAT", "USAD", "TranAD"]
COLORS = {
    "ALV-AD": "#c9252d",
    "CrossAD": "#2f5fb3",
    "MtsCID": "#4aa564",
    "MTAD-GAT": "#df7f3f",
    "USAD": "#7b68b8",
    "TranAD": "#12aeb0",
}
MARKERS = {
    "ALV-AD": "s",
    "CrossAD": "o",
    "MtsCID": "^",
    "MTAD-GAT": "D",
    "USAD": "*",
    "TranAD": "v",
}


def set_style():
    matplotlib.rcParams.update(
        {
            "font.size": 10,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_overview(mean_df: pd.DataFrame) -> None:
    datasets = ["MSL", "CalIt2"]
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 5.7), sharey=False)
    plt.subplots_adjust(left=0.08, right=0.99, top=0.88, bottom=0.13, wspace=0.22, hspace=0.34)

    for col, dataset in enumerate(datasets):
        for row, protocol in enumerate(["simulated", "real"]):
            ax = axes[row, col]
            if protocol == "simulated":
                sub = mean_df[(mean_df.dataset == dataset) & (mean_df.protocol == "simulated")]
                x_label = "Simulated pollution (%)"
                xticks = [0, 2, 5, 10, 15]
                xticklabels = ["0", "2", "5", "10", "15"]
                x_scale = 100
            else:
                clean = mean_df[
                    (mean_df.dataset == dataset)
                    & (mean_df.protocol == "simulated")
                    & np.isclose(mean_df.level, 0.0)
                ].copy()
                clean.loc[:, "protocol"] = "real"
                clean.loc[:, "level"] = 0.0
                real = mean_df[(mean_df.dataset == dataset) & (mean_df.protocol == "real")]
                sub = pd.concat([clean, real], ignore_index=True)
                x_label = "Real anomaly repeat count"
                xticks = [0, 1, 2, 3, 4]
                xticklabels = ["0", "1x", "2x", "3x", "4x"]
                x_scale = 1

            for method in METHODS:
                g = sub[sub.method == method].sort_values("level")
                if g.empty:
                    continue
                x = g["level"].to_numpy(dtype=float) * x_scale
                y = g["auc_roc_mean"].to_numpy(dtype=float) * 100
                yerr = g["auc_roc_std"].fillna(0).to_numpy(dtype=float) * 100
                is_ours = method == "ALV-AD"
                ax.plot(
                    x,
                    y,
                    color=COLORS[method],
                    marker=MARKERS[method],
                    markersize=5.4 if method != "USAD" else 7.2,
                    linewidth=2.25 if is_ours else 1.45,
                    alpha=1.0 if is_ours else 0.82,
                    zorder=5 if is_ours else 2,
                )
                if is_ours:
                    ax.fill_between(x, y - yerr, y + yerr, color=COLORS[method], alpha=0.14, lw=0)

            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels)
            ax.set_xlabel(x_label)
            if col == 0:
                ax.set_ylabel("AUC-ROC (%)")
            ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.35)
            if row == 0:
                ax.set_title(dataset, pad=6)

    for col in range(2):
        vals = []
        for ax in axes[:, col]:
            for line in ax.lines:
                vals.extend(line.get_ydata())
        lo = max(0, np.nanmin(vals) - 4)
        hi = min(100, np.nanmax(vals) + 4)
        for ax in axes[:, col]:
            ax.set_ylim(lo, hi)

    handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[method],
            marker=MARKERS[method],
            lw=2.25 if method == "ALV-AD" else 1.5,
            markersize=5.7 if method != "USAD" else 7.5,
            label=method,
        )
        for method in METHODS
    ]
    fig.legend(handles=handles, loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 0.995))
    fig.text(
        0.5,
        0.035,
        "Mean over 4 seeds. Shaded band shows ALV-AD standard deviation.",
        ha="center",
        color="#444444",
        fontsize=9,
    )
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"aucroc_score_full_msl_calit2_4seeds.{ext}")
    plt.close(fig)


def plot_delta(alv_vs_best: pd.DataFrame) -> None:
    cond_order = [
        ("simulated", 0.00),
        ("simulated", 0.02),
        ("simulated", 0.05),
        ("simulated", 0.10),
        ("simulated", 0.15),
        ("real", 1.00),
        ("real", 2.00),
        ("real", 3.00),
        ("real", 4.00),
    ]
    cond_labels = ["S0", "S2", "S5", "S10", "S15", "R1", "R2", "R3", "R4"]
    datasets = ["MSL", "CalIt2"]
    mat = np.full((len(datasets), len(cond_order)), np.nan, dtype=float)
    labels = [["" for _ in cond_order] for _ in datasets]

    for i, dataset in enumerate(datasets):
        for j, (protocol, level) in enumerate(cond_order):
            row = alv_vs_best[
                (alv_vs_best.dataset == dataset)
                & (alv_vs_best.protocol == protocol)
                & np.isclose(alv_vs_best.level.astype(float), level)
            ]
            if row.empty:
                continue
            delta = float(row.iloc[0].delta_mean) * 100
            mat[i, j] = delta
            labels[i][j] = f"{delta:+.1f}"

    fig, ax = plt.subplots(figsize=(8.4, 2.55))
    lim = max(2.0, float(np.nanmax(np.abs(mat))))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_yticks(np.arange(len(datasets)))
    ax.set_yticklabels(datasets)
    ax.set_xticks(np.arange(len(cond_labels)))
    ax.set_xticklabels(cond_labels)
    ax.set_xlabel("S*: simulated pollution %, R*: real anomaly repeat count")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if not np.isfinite(mat[i, j]):
                continue
            color = "white" if abs(mat[i, j]) > lim * 0.45 else "#222222"
            ax.text(j, i, labels[i][j], ha="center", va="center", fontsize=9, color=color)
    ax.set_xticks(np.arange(-0.5, len(cond_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(datasets), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.4)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("ALV-AD - best baseline (pp)")
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"aucroc_score_full_msl_calit2_delta_4seeds.{ext}")
    plt.close(fig)


def main():
    set_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mean_df = pd.read_csv(DATA_DIR / "robustness_multiseed_mean_std.csv")
    alv_vs_best = pd.read_csv(DATA_DIR / "robustness_multiseed_auc_roc_alv_vs_best.csv")
    plot_overview(mean_df)
    plot_delta(alv_vs_best)
    print(OUT_DIR / "aucroc_score_full_msl_calit2_4seeds.png")
    print(OUT_DIR / "aucroc_score_full_msl_calit2_delta_4seeds.png")


if __name__ == "__main__":
    main()
