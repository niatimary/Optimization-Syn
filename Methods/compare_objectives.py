#!/usr/bin/env python
"""
Compares two winning configs -- one from an `optuna_search.py --objective all` run, one
from `--objective warming` -- side by side on the same split, so "does optimizing on the
full warming+cooling mix cost anything on the actual target (NGRIP DO events, which are
warming-only)?" has a number attached instead of a guess.

Scores both configs' best_params.json on VALIDATION (safe to run anytime, unlike TEST --
see optuna_search.py's data-discipline note in its module docstring) and prints a table
with f1_all, f1_warming, f1_cooling for each, plus the f1_warming delta between them.

Usage:
    python optuna_search.py --objective all     --outdir optuna_results_all     --study-name search_all
    python optuna_search.py --objective warming --outdir optuna_results_warming --study-name search_warming
    python compare_objectives.py \
        --all-params optuna_results_all/best_params.json \
        --warming-params optuna_results_warming/best_params.json
"""
import argparse
import json
import os

import pandas as pd

from optuna_search import DEFAULT_VAL_PATH, TOLERANCE, evaluate_final_test


def _score(params_json_path, val_path, n_jobs):
    with open(params_json_path) as f:
        saved = json.load(f)
    params = dict(method=saved['method'], cost=saved['cost'], pen=saved['pen'],
                  n_bkps=saved['n_bkps'], min_size=saved['min_size'], jump=saved['jump'])
    df = evaluate_final_test(val_path, TOLERANCE, n_jobs, params)
    return saved, df


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--all-params", required=True, help="best_params.json from --objective all")
    parser.add_argument("--warming-params", required=True, help="best_params.json from --objective warming")
    parser.add_argument("--val-path", default=DEFAULT_VAL_PATH)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--out-csv", default=None, help="optional path to save the comparison table")
    args = parser.parse_args()

    rows = []
    for label, path in (("optimized_on_all", args.all_params), ("optimized_on_warming", args.warming_params)):
        saved, df = _score(path, args.val_path, args.n_jobs)
        rows.append(dict(
            objective=label,
            method=saved['method'], cost=saved['cost'],
            pen=saved['pen'], n_bkps=saved['n_bkps'], min_size=saved['min_size'],
            val_f1_all=df['f1_all'].mean(),
            val_f1_warming=df['f1_warming'].mean(),
            val_f1_cooling=df['f1_cooling'].mean(),
        ))

    table = pd.DataFrame(rows)
    print(table.to_string(index=False))

    f1w_all_obj = table.loc[table.objective == 'optimized_on_all', 'val_f1_warming'].iloc[0]
    f1w_warm_obj = table.loc[table.objective == 'optimized_on_warming', 'val_f1_warming'].iloc[0]
    delta = f1w_warm_obj - f1w_all_obj
    print(f"\nf1_warming: optimized_on_all={f1w_all_obj:.3f}  optimized_on_warming={f1w_warm_obj:.3f}  "
          f"delta={delta:+.3f}")
    if abs(delta) < 0.01:
        print("Negligible difference -- optimizing on the combined transition set transfers fine "
              "to the warming-only target; the simpler --objective all config is safe to use.")
    else:
        print("Non-negligible difference -- the config chosen by --objective warming is the one "
              "that should carry over to the warming-only (NGRIP DO) application.")

    if args.out_csv:
        table.to_csv(args.out_csv, index=False)
        print(f"Saved -> {args.out_csv}")


if __name__ == "__main__":
    main()
