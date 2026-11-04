#!/usr/bin/env python
"""
Post-hoc noise-condition breakdown for a finished optuna_search.py study.

Not part of the optimization itself -- purely reads already-saved output.
`best_{split}_raw.csv` (written by optuna_search.py's evaluate_train/
evaluate_validation, and by evaluate_test.py for test) has one row per seed
for the single winning config, in the same order as `seed_idx = np.arange(n)`
-- which matches train_test_split_data/{split}_manifest.csv's row order
exactly (verified by comparing per-row n_events between the .npz and the
manifest). So the per-seed F1 can be joined against each seed's noise
`color`/`level`/`source` by row position, with no seed identifier needing to
be added to optuna_search.py's own output.

Usage (after scp-ing results/ back from CARC):
    python analyze_noise_effect.py --results-dir path/to/results
    python analyze_noise_effect.py --results-dir path/to/results --split test
    python analyze_noise_effect.py --results-dir path/to/results --metric f1_all
"""
import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.normpath(os.path.join(HERE, "..", "train_test_split_data"))
# Noisier split (levels k0p25/k0p5/k1, 30 validation + 30 test seeds) the newer Optuna runs were trained on.
NOISIER_DATA_DIR = os.path.normpath(os.path.join(HERE, "..", "noisier_data", "train_test_split_data"))

# Noisiest -> cleanest. Old SNR grid: snr2 noisiest. Noisier k grid: noise amplitude = 0.352 / k, so k0p25 is noisiest.
LEVEL_ORDER = ["snr2", "snr3", "snr4", "k0p25", "k0p5", "k1"]


def load_merged(results_dir, data_dir, split):
    raw_path = os.path.join(results_dir, f"best_{split}_raw.csv")
    manifest_path = os.path.join(data_dir, f"{split}_manifest.csv")
    if not os.path.exists(raw_path):
        return None
    raw = pd.read_csv(raw_path).reset_index(drop=True)
    manifest = pd.read_csv(manifest_path).reset_index(drop=True)
    if len(raw) != len(manifest):
        raise ValueError(
            f"{raw_path} has {len(raw)} rows but {manifest_path} has {len(manifest)} -- "
            f"row-position join is unsafe unless these match exactly."
        )
    return pd.concat([raw, manifest[["source", "color", "level", "orig_seed"]]], axis=1)


def summarize(merged, metric):
    summary = (merged.groupby(["color", "level"])[metric]
                      .agg(["mean", "std", "count"])
                      .sort_values("mean", ascending=False))
    return summary


def summarize_all_splits(merged_by_split, metric, splits=("train", "validation", "test")):
    """Side-by-side version of `summarize` -- one table with each available split's
    (color, level) mean/std/count as its own column group (MultiIndex columns:
    split -> stat), so the same noise condition's numbers across train/validation/test
    sit in one row instead of three separate tables. Sorted by the first available
    split's mean, descending."""
    available = [s for s in splits if merged_by_split.get(s) is not None]
    if not available:
        return pd.DataFrame()
    frames = []
    for split in available:
        s = summarize(merged_by_split[split], metric)
        s.columns = pd.MultiIndex.from_product([[split], s.columns])
        frames.append(s)
    combined = pd.concat(frames, axis=1)
    return combined.sort_values((available[0], "mean"), ascending=False)


def _padded_range(values, pad=0.05, step=0.05, floor=0.0, ceil=1.0):
    """Round the data's own [min, max] outward to the nearest `step`, plus a small
    `pad` -- enough to give real differences visual contrast without hardcoding the
    full [0, 1] F1 range (which flattens everything when actual values only span a
    narrow band) or zooming so tight that sampling noise looks like a dramatic
    pattern. Falls back to (floor, ceil) if there's no usable spread."""
    values = values[~pd.isna(values)]
    if len(values) == 0:
        return floor, ceil
    lo, hi = float(values.min()), float(values.max())
    if hi - lo < 1e-9:
        lo, hi = lo - step, hi + step
    lo = max(floor, step * ((lo - pad) // step))
    hi = min(ceil, step * -((-(hi + pad)) // step))  # ceil-round hi up to nearest step
    return lo, hi


def plot_heatmap_row(merged_by_split, metric, outdir, splits=("train", "validation", "test")):
    """Same per-split heatmap as `plot_heatmap`, but every available split side by side
    in ONE figure -- shared vmin/vmax (data-driven, see `_padded_range`) and one shared
    colorbar, so a noise-condition pattern (e.g. red/snr2 being worst) is directly
    comparable across splits at a glance, instead of scrolling between separately-saved
    PNGs."""
    available = [s for s in splits if merged_by_split.get(s) is not None]
    if not available:
        return None

    level_order_all = LEVEL_ORDER
    color_order_all = ["white", "pink", "red"]

    pivots = {}
    for split in available:
        merged = merged_by_split[split]
        pivot = merged.pivot_table(index="color", columns="level", values=metric, aggfunc="mean")
        level_order = [c for c in level_order_all if c in pivot.columns]
        color_order = [c for c in color_order_all if c in pivot.index]
        pivots[split] = pivot.reindex(index=color_order, columns=level_order)

    all_values = pd.concat([p.stack() for p in pivots.values()])
    vmin, vmax = _padded_range(all_values)

    fig, axes = plt.subplots(1, len(available), figsize=(4.2 * len(available), 3.6), squeeze=False)
    axes = axes[0]
    im = None
    for ax, split in zip(axes, available):
        pivot = pivots[split]

        im = ax.imshow(pivot.values, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xlabel("noise level (SNR)")
        if ax is axes[0]:
            ax.set_ylabel("noise color")
        ax.set_title(split)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.values[i, j]
                if pd.notna(val):
                    frac = (val - vmin) / (vmax - vmin) if vmax > vmin else 1.0
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            color="white" if frac < 0.5 else "black")

    fig.suptitle(f"{metric} by noise condition -- {' vs '.join(available)}", y=1.03)
    fig.colorbar(im, ax=axes.tolist(), label=metric, shrink=0.85)
    out_path = os.path.join(outdir, f"noise_heatmap_row_{metric}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_heatmap(merged, metric, split, outdir):
    pivot = merged.pivot_table(index="color", columns="level", values=metric, aggfunc="mean")
    level_order = [c for c in LEVEL_ORDER if c in pivot.columns]
    color_order = [c for c in ["white", "pink", "red"] if c in pivot.index]
    pivot = pivot.reindex(index=color_order, columns=level_order)

    vmin, vmax = _padded_range(pivot.stack())

    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    im = ax.imshow(pivot.values, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("noise level (SNR)")
    ax.set_ylabel("noise color")
    ax.set_title(f"{metric} by noise condition -- {split}")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if pd.notna(val):
                frac = (val - vmin) / (vmax - vmin) if vmax > vmin else 1.0
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        color="white" if frac < 0.5 else "black")
    fig.colorbar(im, ax=ax, label=metric)
    fig.tight_layout()
    out_path = os.path.join(outdir, f"noise_heatmap_{split}_{metric}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", required=True,
                         help="directory holding best_train_raw.csv / best_validation_raw.csv / "
                              "best_test_raw.csv (e.g. the results/ folder downloaded from CARC)")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                         help="directory holding {train,validation,test}_manifest.csv "
                              f"(default: {DEFAULT_DATA_DIR})")
    parser.add_argument("--split", choices=["train", "validation", "test", "all"], default="all")
    parser.add_argument("--metric", choices=["f1_all", "f1_warming", "f1_cooling"], default="f1_warming")
    args = parser.parse_args()

    splits = ["train", "validation", "test"] if args.split == "all" else [args.split]

    for split in splits:
        merged = load_merged(args.results_dir, args.data_dir, split)
        if merged is None:
            print(f"[{split}] best_{split}_raw.csv not found in {args.results_dir} -- skipping")
            continue

        merged_path = os.path.join(args.results_dir, f"{split}_raw_with_noise.csv")
        merged.to_csv(merged_path, index=False)

        summary = summarize(merged, args.metric)
        print(f"\n=== {split} -- {args.metric} by noise condition (best to worst) ===")
        print(summary.to_string(float_format=lambda x: f"{x:.3f}"))

        heatmap_path = plot_heatmap(merged, args.metric, split, args.results_dir)
        print(f"Saved -> {merged_path}")
        print(f"Saved -> {heatmap_path}")


if __name__ == "__main__":
    main()
