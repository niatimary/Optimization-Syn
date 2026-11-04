#!/usr/bin/env python
"""
Non-interactive grid-search runner: ruptures method x cost function, on the
calibrated red/snr3 noise condition (ngrip_calibrated_noise_sensitivity_data/
noise_red_snr3.npz). Ports binseg_grid_red_snr3.ipynb / pelt_grid_red_snr3.ipynb /
bottomup_grid_red_snr3.ipynb / kernelcpd_grid_red_snr3.ipynb into a single script
for a batch SLURM run -- same scoring / train-test-split / selection logic as
those notebooks, results saved to disk instead of shown inline (matplotlib Agg
backend, fig.savefig, no plt.show()).

Window is intentionally excluded: it underperformed every other method across
every cost function tested (see Window/results/*_conclusion.csv), so it was
dropped from method_configs.py's METHODS.

Dynp (dynamic programming) and the KS test are intentionally excluded: Dynp
needs a separate n_bkps-based calibration scheme rather than a pen grid, and
this pass is the ruptures penalty-based methods only.

Usage:
    python run_grid_experiment.py --method binseg
    python run_grid_experiment.py --method pelt --data-path /path/to/other.npz

Run one method per invocation (see run_grid_experiments.slurm for a SLURM
array job that runs the remaining methods in parallel).
"""
import argparse
import os
import sys
import time

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from grid_common import GridRunner, plot_train_surface, plot_test_pr, plot_test_pr_grid, save_csvs
from method_configs import METHODS, HEAVY_COSTS, LIGHT_BUDGET, HEAVY_BUDGET

TOLERANCE   = 0.5   # +/- 0.5 kyr matching tolerance, matches cv_noise_sensitivity.py
RANDOM_SEED = 42
DATASET_TAG = "calibrated_snr3"
DEFAULT_DATA_PATH = os.path.normpath(
    os.path.join(HERE, "..", "ngrip_calibrated_noise_sensitivity_data", "noise_red_snr3.npz")
)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", required=True, choices=sorted(METHODS.keys()),
                         help="which ruptures method to run")
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH,
                         help=f"default: {DEFAULT_DATA_PATH}")
    parser.add_argument("--n-jobs", type=int, default=-1,
                         help="joblib n_jobs; -1 = use all allocated cores")
    parser.add_argument("--outdir", default=None,
                         help="default: <Methods>/<Algo>/results_calibrated_snr3")
    args = parser.parse_args()

    method_key = args.method
    spec = METHODS[method_key]
    algo = spec['algo']
    costs = spec['costs']

    outdir = args.outdir or os.path.join(HERE, algo, f"results_{DATASET_TAG}")
    os.makedirs(outdir, exist_ok=True)
    out_prefix = os.path.join(outdir, f"{method_key}_grid_{DATASET_TAG}")

    print(f"=== {algo} ({method_key}) on {DATASET_TAG} ===", flush=True)
    print(f"data:   {args.data_path}", flush=True)
    print(f"outdir: {outdir}", flush=True)
    print(f"costs:  {list(costs.keys())}", flush=True)

    t_start = time.perf_counter()
    runner = GridRunner(algo=algo, data_path=args.data_path, tolerance=TOLERANCE,
                         random_seed=RANDOM_SEED, n_jobs=args.n_jobs,
                         heavy_costs=HEAVY_COSTS, light_budget=LIGHT_BUDGET,
                         heavy_budget=HEAVY_BUDGET)
    n_transitions = [len(g) for g in runner.gt_times]
    print(f"Loaded {args.data_path}: {runner.n_seeds} seeds, {len(runner.time_arr)} points/series", flush=True)
    print(f"Ground-truth transitions per seed: min={min(n_transitions)}, max={max(n_transitions)}, "
          f"mean={sum(n_transitions) / len(n_transitions):.1f}", flush=True)

    for cost, grid in costs.items():
        t0 = time.perf_counter()
        result = runner.run_one_cost(cost, grid['pens'], grid['min_sizes'], widths=grid.get('widths'))
        plot_train_surface(cost, result, out_prefix)
        plot_test_pr(cost, result, out_prefix)
        print(f"  -> {cost} done in {time.perf_counter() - t0:.0f}s (surface + pr plots saved)", flush=True)

    plot_test_pr_grid(runner.results, out_prefix)
    raw_path, summary_path, conclusion_path, conclusion_df = save_csvs(runner.results, out_prefix)
    print(f"Saved {len(runner.results)} costs -> {raw_path}", flush=True)
    print(f"Saved {summary_path}", flush=True)
    print(f"Saved {conclusion_path}", flush=True)

    # Sanity check: distinguish a genuine all-zero result (e.g. 'ar'/'linear',
    # documented in the notebooks as legitimately detecting nothing on pure
    # mean-shift data) from a silent wiring bug that would look identical in
    # test_f1 alone. Every _score_one failure is tagged with its exception in
    # the 'error' column of *_raw.csv -- read that first if either fires.
    raw_df = pd.concat([r['train_raw'] for r in runner.results.values()], ignore_index=True, sort=False)
    error_counts = raw_df['error'].dropna().value_counts()
    if len(error_counts):
        print(f"\nWARNING: {raw_df['error'].notna().sum()}/{len(raw_df)} train-grid calls raised an "
              f"exception (caught, scored as 0.0) -- top errors:", flush=True)
        for msg, n in error_counts.head(5).items():
            print(f"  [{n:>5d}x] {msg}", flush=True)
        print("  -> inspect the 'error' column in *_raw.csv before trusting any cost with nonzero "
              "error counts.", flush=True)

    all_zero = conclusion_df['test_f1'].eq(0).all()
    if all_zero:
        print(f"\nWARNING: EVERY cost function scored test_f1=0.0 for {algo}. Known costs that can "
              f"legitimately do this (ar, linear) never do so for ALL costs at once in past runs -- "
              f"this looks like a systemic failure (wrong data path, env mismatch, ammonyte API "
              f"change), not a real result. Check the 'error' column above before trusting anything "
              f"else in this run.", flush=True)

    print("=" * 92)
    print(f"CONCLUSION: {algo} on {DATASET_TAG} -- train/test split, optimized independently per cost function")
    print("=" * 92)
    for _, row in conclusion_df.iterrows():
        edge_note = "  <-- ON GRID EDGE: widen this cost's range and rerun" if row['on_edge'] else ""
        gap = row['train_f1'] - row['test_f1']
        print(f"{row['cost']:>12s} [{row['seed_budget']:5s}]: pen={row['best_pen']:>8.3g}  "
              f"min_size={row['best_min_size']:>3.0f}  train_f1={row['train_f1']:.3f}  "
              f"test_f1={row['test_f1']:.3f}  (gap={gap:+.3f}){edge_note}")

    n_edge = int(conclusion_df['on_edge'].sum())
    if n_edge:
        print(f"\n{n_edge} cost function(s) did NOT converge on the train grid (optimum on the tested "
              f"boundary) -- widen those ranges and rerun before trusting their numbers.")
    else:
        print(f"\nAll {len(conclusion_df)} cost functions converged (train optimum interior to their tested range).")

    best_overall = conclusion_df.iloc[0]
    print(f"\nBest cost function by TEST f1: '{best_overall['cost']}' "
          f"(pen={best_overall['best_pen']:.3g}, min_size={best_overall['best_min_size']:.0f}, "
          f"test_f1={best_overall['test_f1']:.3f})")
    print(f"\nTotal wall time: {time.perf_counter() - t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
