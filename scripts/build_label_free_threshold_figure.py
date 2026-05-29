#!/usr/bin/env python3
"""Build the label-free threshold analysis figure from local result files.

This script is intended for paper reproduction. It does not contact the
authors' server unless --sync-remote is explicitly set.

Required inputs:
- --score-npz: a NumPy archive with train_normal, test_normal, test_anomaly,
  threshold_names, and thresholds arrays for panel (a).
- --threshold-csv: a CSV with dataset, method, rule, and aff_f columns for
  panels (b)--(e).

Examples:
    python scripts/build_label_free_threshold_figure.py --single-column
    python scripts/build_label_free_threshold_figure.py --single-column --compile
    python scripts/build_label_free_threshold_figure.py --score-npz path/to/global_synthetic_threshold_scores.npz --threshold-csv path/to/threshold_multimodel_raw.csv
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch, Patch, Rectangle
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = PROJECT_ROOT / "pvldbstyle-master"
PAPER_FIGURE_DIR = PAPER_DIR / "figures"

SYNTHETIC_REL = (
    Path("final_ex")
    / "synthetic_threshold_score_groups_20260528"
    / "global_synthetic_threshold_scores.npz"
)
RAW_REL = (
    Path("final_ex")
    / "threshold_selection_multimodel_native_window_msl_cicids_swat_nyc_20260529"
    / "threshold_multimodel_raw.csv"
)
DEFAULT_OUT_DIR = PROJECT_ROOT / "final_ex" / "paper_figures" / "native_window_msl_cicids_swat_nyc"
PREVIEW_DIR = PROJECT_ROOT / "final_ex" / "paper_figures"
FIGURE_STEM = "label_free_threshold_analysis_synthetic"

RULE_ORDER = ["Best", "TopK", "POT", "SD"]
RULE_DISPLAY = {"Best": "Best", "TopK": "TopK", "POT": "POT", "SD": "SD"}
TOP_THRESHOLD_ORDER = ["SD", "POT", "TopK", "Best"]
DATASET_ORDER = ["MSL", "CICIDS", "SWaT", "NYC"]
METHOD_ORDER = ["CATCH", "CrossAD", "USAD", "ALV-AD"]

METHOD_COLORS = {
    "CATCH": "#EAF3FF",
    "CrossAD": "#B9DDF4",
    "USAD": "#2B97D3",
    "ALV-AD": "#0857B8",
}
METHOD_HATCHES = {
    "CATCH": "//",
    "CrossAD": "\\\\",
    "USAD": "---",
    "ALV-AD": "",
}
THRESHOLD_STYLES = {
    "SD": ("#111111", (0, (3, 2))),
    "POT": ("#D62728", (0, (5, 2))),
    "TopK": ("#1F77B4", (0, (2, 1.5))),
    "Best": ("#FF7F0E", (0, (1, 1.2))),
}
SCORE_COLORS = {
    "train": "#6C70FF",
    "normal": "#4CAF50",
    "anomaly": "#FF4D4D",
}


@dataclass(frozen=True)
class FigureInputs:
    score_path: Path
    raw_path: Path


@dataclass(frozen=True)
class FigureLayout:
    single_column: bool
    figsize: tuple[float, float]
    height_ratios: tuple[float, float, float, float]
    hspace: float
    wspace: float
    base_font: float
    axis_font: float
    title_font: float
    tick_font: float
    legend_font: float
    top_legend_ncol: int
    top_legend_y: float
    top_legend_handle: float
    top_legend_columnspace: float
    method_legend_handle: float
    method_legend_columnspace: float
    inset_width: str
    inset_height: str
    inset_borderpad: float
    save_pad: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--score-npz", "--score-path", dest="score_path", type=Path, default=PROJECT_ROOT / SYNTHETIC_REL)
    parser.add_argument("--threshold-csv", "--raw-path", dest="raw_path", type=Path, default=PROJECT_ROOT / RAW_REL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--paper-dir", type=Path, default=PAPER_DIR)
    parser.add_argument("--single-column", action="store_true", help="Use compact dimensions for a one-column figure.")
    parser.add_argument("--no-copy-paper", action="store_true", help="Only write to --out-dir; do not update paper figures.")
    parser.add_argument("--compile", action="store_true", help="Run latexmk after replacing the paper figure.")
    parser.add_argument("--sync-remote", action="store_true", help="Author-only convenience: fetch missing default inputs over SSH.")
    parser.add_argument("--remote", default="fxj2", help="SSH host used only with --sync-remote.")
    parser.add_argument("--remote-root", default=None, help="Remote project root used only with --sync-remote.")
    return parser.parse_args()


def make_layout(single_column: bool) -> FigureLayout:
    if single_column:
        return FigureLayout(
            single_column=True,
            figsize=(3.35, 3.78),
            height_ratios=(1.28, 0.1, 0.95, 0.95),
            hspace=0.54,
            wspace=0.48,
            base_font=5.8,
            axis_font=6.0,
            title_font=6.3,
            tick_font=5.5,
            legend_font=5.5,
            top_legend_ncol=4,
            top_legend_y=1.42,
            top_legend_handle=1.0,
            top_legend_columnspace=0.35,
            method_legend_handle=1.0,
            method_legend_columnspace=0.45,
            inset_width="39%",
            inset_height="50%",
            inset_borderpad=0.45,
            save_pad=0.018,
        )

    return FigureLayout(
        single_column=False,
        figsize=(7.15, 4.85),
        height_ratios=(1.55, 0.18, 1.0, 1.0),
        hspace=0.78,
        wspace=0.44,
        base_font=7.8,
        axis_font=7.8,
        title_font=8.2,
        tick_font=7.0,
        legend_font=7.0,
        top_legend_ncol=7,
        top_legend_y=1.34,
        top_legend_handle=1.45,
        top_legend_columnspace=0.55,
        method_legend_handle=1.45,
        method_legend_columnspace=0.85,
        inset_width="36%",
        inset_height="56%",
        inset_borderpad=0.75,
        save_pad=0.035,
    )


def configure_matplotlib(layout: FigureLayout) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": layout.base_font,
            "axes.labelsize": layout.axis_font,
            "axes.titlesize": layout.title_font,
            "xtick.labelsize": layout.tick_font,
            "ytick.labelsize": layout.tick_font,
            "legend.fontsize": layout.legend_font,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def fetch_if_missing(local_path: Path, remote_rel: Path, args: argparse.Namespace) -> None:
    if local_path.exists():
        return
    if not args.sync_remote:
        raise FileNotFoundError(
            "Missing required input:\n"
            f"  {local_path}\n"
            "For reproduction, provide it with --score-npz/--threshold-csv or place it at the default path.\n"
            "Authors can additionally pass --sync-remote to fetch the default files from the internal server."
        )
    if args.remote_root is None:
        raise ValueError("--sync-remote requires --remote-root")

    local_path.parent.mkdir(parents=True, exist_ok=True)
    remote_path = f"{args.remote}:{args.remote_root}/{remote_rel.as_posix()}"
    print(f"[sync] {remote_path} -> {local_path}")
    subprocess.run(["scp", remote_path, str(local_path)], check=True)


def ensure_inputs(args: argparse.Namespace) -> FigureInputs:
    score_path = args.score_path.resolve()
    raw_path = args.raw_path.resolve()
    fetch_if_missing(score_path, SYNTHETIC_REL, args)
    fetch_if_missing(raw_path, RAW_REL, args)
    return FigureInputs(score_path=score_path, raw_path=raw_path)


def percent_hist(ax, values: np.ndarray, bins: np.ndarray, color: str, label: str, alpha: float) -> None:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    weights = np.full(values.shape, 100.0 / max(len(values), 1))
    ax.hist(
        values,
        bins=bins,
        weights=weights,
        color=color,
        alpha=alpha,
        edgecolor=color,
        linewidth=0.25,
        label=label,
    )


def percent_hist_max(arrays: list[np.ndarray], bins: np.ndarray) -> float:
    best = 0.0
    for values in arrays:
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        weights = np.full(values.shape, 100.0 / max(len(values), 1))
        counts, _ = np.histogram(values, bins=bins, weights=weights)
        best = max(best, float(np.max(counts)) if counts.size else 0.0)
    return best


def threshold_legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color=THRESHOLD_STYLES[name][0],
            linestyle=THRESHOLD_STYLES[name][1],
            lw=1.25,
            label=name,
        )
        for name in ["Best", "TopK", "POT", "SD"]
    ]


def method_legend_handles() -> list[Patch]:
    return [
        Patch(
            facecolor=METHOD_COLORS[method],
            edgecolor="black",
            linewidth=0.4,
            hatch=METHOD_HATCHES[method],
            label=method,
        )
        for method in METHOD_ORDER
    ]


def combined_top_legend(ax, layout: FigureLayout) -> None:
    hist_handles, hist_labels = ax.get_legend_handles_labels()
    handles = threshold_legend_handles() + hist_handles
    labels = [handle.get_label() for handle in threshold_legend_handles()] + hist_labels
    ax.legend(
        handles=handles,
        labels=labels,
        loc="upper center",
        bbox_to_anchor=(0.5, layout.top_legend_y),
        frameon=True,
        ncol=layout.top_legend_ncol,
        handlelength=layout.top_legend_handle,
        columnspacing=layout.top_legend_columnspace,
        borderpad=0.16 if layout.single_column else 0.22,
        fontsize=layout.legend_font,
    )


def draw_zoom_region(ax, x_left: float, x_right: float, y_top: float) -> None:
    rect = Rectangle(
        (x_left, 0.0),
        x_right - x_left,
        y_top,
        fill=False,
        edgecolor="#333333",
        linewidth=0.95,
        linestyle=(0, (3, 2)),
        zorder=5,
    )
    ax.add_patch(rect)
    ax.vlines(
        [x_left, x_right],
        0.0,
        y_top,
        colors="#333333",
        linestyles=(0, (3, 2)),
        linewidth=0.95,
        zorder=5,
    )


def draw_zoom_connectors(ax, ax_inset, x_left: float, x_right: float, y_top: float) -> None:
    for x_value, inset_x in ((x_left, 0.0), (x_right, 1.0)):
        connector = ConnectionPatch(
            xyA=(x_value, y_top),
            coordsA=ax.transData,
            xyB=(inset_x, 0.0),
            coordsB=ax_inset.transAxes,
            axesA=ax,
            axesB=ax_inset,
            color="#333333",
            linewidth=0.85,
            linestyle=(0, (3, 2)),
            zorder=4,
        )
        ax.add_artist(connector)


def draw_threshold_lines(ax, thresholds: dict[str, float], y_text: float) -> None:
    for rule in TOP_THRESHOLD_ORDER:
        value = thresholds.get(rule)
        if value is None:
            continue
        color, dash = THRESHOLD_STYLES[rule]
        ax.axvline(value, color=color, linestyle=dash, linewidth=1.05)


def adaptive_bar_ylim(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return 0.0, 0.82

    y_min = float(np.min(values))
    y_max = float(np.max(values))
    lower = 0.0 if y_min < 0.2 else max(0.0, np.floor((y_min - 0.035) * 20.0) / 20.0)
    upper = min(1.02, np.ceil((y_max + 0.035) * 20.0) / 20.0)
    if upper - lower < 0.25:
        upper = min(1.02, lower + 0.25)
    return lower, upper


def bar_tick_steps(y_low: float, y_high: float) -> tuple[float, float]:
    if y_high - y_low > 0.6:
        return 0.2, 0.1
    return 0.1, 0.05


def load_threshold_results(raw_path: Path) -> pd.DataFrame:
    raw = pd.read_csv(raw_path)
    raw["dataset"] = raw["dataset"].replace({"SWAT": "SWaT"})
    raw["method"] = raw["method"].replace({"ALV_AD": "ALV-AD"})
    raw = raw[raw["rule"].isin(RULE_ORDER) & raw["method"].isin(METHOD_ORDER)].copy()
    raw = raw.drop_duplicates(subset=["dataset", "method", "rule"], keep="last")

    expected = {(dataset, method, rule) for dataset in DATASET_ORDER for method in METHOD_ORDER for rule in RULE_ORDER}
    found = set(zip(raw["dataset"], raw["method"], raw["rule"], strict=False))
    missing = sorted(expected - found)
    if missing:
        preview = ", ".join(f"{dataset}/{method}/{rule}" for dataset, method, rule in missing[:8])
        raise ValueError(f"Missing {len(missing)} threshold rows. First missing rows: {preview}")
    return raw


def plot_distribution_panel(fig, grid, scores, layout: FigureLayout) -> None:
    ax_dist = fig.add_subplot(grid[0, :])
    dist_arrays = [scores["train_normal"], scores["test_normal"], scores["test_anomaly"]]
    combined = np.concatenate([np.asarray(array, dtype=float) for array in dist_arrays])
    thresholds = dict(zip([str(item) for item in scores["threshold_names"]], [float(item) for item in scores["thresholds"]], strict=False))

    x_lo = float(np.floor(np.nanpercentile(combined, 0.2) * 2.0) / 2.0)
    x_hi = float(np.ceil(max(np.nanpercentile(combined, 99.5), max(thresholds.values()) * 1.08) * 2.0) / 2.0)
    bins = np.linspace(x_lo, x_hi, 76)
    y_hi = max(8.0, percent_hist_max(dist_arrays, bins) * 1.18)

    percent_hist(ax_dist, scores["train_normal"], bins, SCORE_COLORS["train"], "Train normal", 0.45)
    percent_hist(ax_dist, scores["test_normal"], bins, SCORE_COLORS["normal"], "Test normal", 0.48)
    percent_hist(ax_dist, scores["test_anomaly"], bins, SCORE_COLORS["anomaly"], "Test anomalous", 0.48)
    ax_dist.set_xlim(x_lo, x_hi)
    ax_dist.set_ylim(0.0, y_hi)
    ax_dist.set_ylabel("Percent (%)")
    ax_dist.set_xlabel("" if layout.single_column else "Shifted anomaly score")
    title = "(a) Score distribution with threshold rules" if layout.single_column else "(a) Synthetic global score distribution with threshold rules"
    ax_dist.set_title(title, loc="left", pad=1 if layout.single_column else 2)
    ax_dist.grid(axis="y", color="#D7D7D7", linewidth=0.5, alpha=0.8)
    combined_top_legend(ax_dist, layout)

    ax_inset = inset_axes(
        ax_dist,
        width=layout.inset_width,
        height=layout.inset_height,
        loc="upper center",
        borderpad=layout.inset_borderpad,
    )
    zoom_lo = max(x_lo, min(thresholds.values()) * 0.72)
    zoom_hi = min(x_hi, max(thresholds.values()) * 1.10)
    zoom_bins = np.linspace(zoom_lo, zoom_hi, 48)
    inset_y_hi = max(2.0, percent_hist_max(dist_arrays, zoom_bins) * 1.18)
    zoom_box_top = max(y_hi * 0.20, percent_hist_max(dist_arrays, zoom_bins) * 1.15)

    draw_zoom_region(ax_dist, zoom_lo, zoom_hi, zoom_box_top)
    draw_zoom_connectors(ax_dist, ax_inset, zoom_lo, zoom_hi, zoom_box_top)
    percent_hist(ax_inset, scores["train_normal"], zoom_bins, SCORE_COLORS["train"], "Train normal", 0.45)
    percent_hist(ax_inset, scores["test_normal"], zoom_bins, SCORE_COLORS["normal"], "Test normal", 0.48)
    percent_hist(ax_inset, scores["test_anomaly"], zoom_bins, SCORE_COLORS["anomaly"], "Test anomalous", 0.48)
    draw_threshold_lines(ax_inset, thresholds, inset_y_hi * 0.82)
    ax_inset.set_xlim(zoom_lo, zoom_hi)
    ax_inset.set_ylim(0.0, inset_y_hi)
    ax_inset.set_xticks(np.round(np.linspace(zoom_lo, zoom_hi, 3), 1))
    ax_inset.grid(axis="y", color="#D7D7D7", linewidth=0.45, alpha=0.7)


def plot_method_legend(fig, grid, layout: FigureLayout) -> None:
    ax_method_legend = fig.add_subplot(grid[1, :])
    ax_method_legend.axis("off")
    ax_method_legend.legend(
        handles=method_legend_handles(),
        loc="center",
        ncol=4,
        frameon=True,
        borderpad=0.18 if layout.single_column else 0.25,
        columnspacing=layout.method_legend_columnspace,
        handlelength=layout.method_legend_handle,
        fontsize=layout.legend_font,
    )


def plot_bar_panels(fig, grid, raw: pd.DataFrame, layout: FigureLayout) -> None:
    for idx, dataset in enumerate(DATASET_ORDER):
        ax = fig.add_subplot(grid[2 + idx // 2, (idx % 2) * 2 : (idx % 2) * 2 + 2])
        subset = (
            raw[raw["dataset"] == dataset]
            .pivot_table(index="rule", columns="method", values="aff_f", aggfunc="mean")
            .reindex(RULE_ORDER)
            .reindex(columns=METHOD_ORDER)
        )
        x = np.arange(len(RULE_ORDER))
        width = 0.18
        offsets = (np.arange(len(METHOD_ORDER)) - (len(METHOD_ORDER) - 1) / 2.0) * width

        for method_idx, method in enumerate(METHOD_ORDER):
            ax.bar(
                x + offsets[method_idx],
                subset[method].to_numpy(dtype=float),
                color=METHOD_COLORS[method],
                edgecolor="black",
                linewidth=0.35,
                hatch=METHOD_HATCHES[method],
                width=width * 0.92,
                label=method,
            )

        finite_values = subset.to_numpy(dtype=float)
        finite_values = finite_values[np.isfinite(finite_values)]
        y_low, y_high = adaptive_bar_ylim(finite_values)
        major_step, minor_step = bar_tick_steps(y_low, y_high)

        ax.set_ylim(y_low, y_high)
        ax.set_xticks(x)
        ax.set_xticklabels([RULE_DISPLAY[rule] for rule in RULE_ORDER])
        ax.set_ylabel("Aff-F" if (not layout.single_column or idx % 2 == 0) else "")
        ax.set_title(f"({chr(ord('b') + idx)}) {dataset}", loc="left", pad=1)
        ax.set_axisbelow(True)
        ax.yaxis.set_major_locator(MultipleLocator(major_step))
        ax.yaxis.set_minor_locator(MultipleLocator(minor_step))
        ax.grid(axis="y", which="major", color="#BEBEBE", linestyle=(0, (3, 3)), linewidth=0.55, alpha=0.85)
        ax.grid(axis="y", which="minor", color="#D5D5D5", linestyle=(0, (2, 3)), linewidth=0.4, alpha=0.72)
        for sep in np.arange(len(RULE_ORDER) - 1) + 0.5:
            ax.axvline(sep, color="#BEBEBE", linestyle=(0, (3, 3)), linewidth=0.55, alpha=0.85, zorder=0)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)


def build_figure(inputs: FigureInputs, out_dir: Path, layout: FigureLayout) -> list[Path]:
    configure_matplotlib(layout)
    scores = np.load(inputs.score_path, allow_pickle=True)
    raw = load_threshold_results(inputs.raw_path)

    fig = plt.figure(figsize=layout.figsize)
    grid = fig.add_gridspec(
        4,
        4,
        height_ratios=layout.height_ratios,
        hspace=layout.hspace,
        wspace=layout.wspace,
    )
    plot_distribution_panel(fig, grid, scores, layout)
    plot_method_legend(fig, grid, layout)
    plot_bar_panels(fig, grid, raw, layout)

    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for ext in ("pdf", "png"):
        out_path = out_dir / f"{FIGURE_STEM}.{ext}"
        fig.savefig(out_path, bbox_inches="tight", pad_inches=layout.save_pad)
        outputs.append(out_path)
        print(f"[figure] wrote {out_path}")
    plt.close(fig)
    return outputs


def copy_to_paper(outputs: list[Path], paper_dir: Path) -> None:
    figure_dir = paper_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    for source in outputs:
        for directory in (figure_dir, PREVIEW_DIR):
            shutil.copy2(source, directory / source.name)

        # Keep the old alias available because older paper versions referenced it.
        alias = figure_dir / f"label_free_threshold_analysis{source.suffix}"
        shutil.copy2(source, alias)
        print(f"[copy] updated {figure_dir / source.name}")


def compile_paper(paper_dir: Path) -> None:
    print(f"[compile] latexmk in {paper_dir}")
    subprocess.run(
        ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
        cwd=paper_dir,
        check=True,
    )


def main() -> None:
    args = parse_args()
    inputs = ensure_inputs(args)
    layout = make_layout(args.single_column)
    outputs = build_figure(inputs, args.out_dir.resolve(), layout)
    if not args.no_copy_paper:
        paper_dir = args.paper_dir.resolve()
        if paper_dir.exists():
            copy_to_paper(outputs, paper_dir)
        else:
            print(f"[copy] skipped because paper directory does not exist: {paper_dir}")
    if args.compile:
        compile_paper(args.paper_dir.resolve())


if __name__ == "__main__":
    main()
