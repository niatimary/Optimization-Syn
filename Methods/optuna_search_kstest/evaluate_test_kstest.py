#!/usr/bin/env python
"""
Standalone, one-shot scoring of the winning augmented-KS-test config
(w_min, w_max, n_w, d_c, n_c, s_c; x_c always auto) against the held-out TEST split.

KS-test sibling of ../optuna_search/evaluate_test.py, using
optuna_search_kstest.evaluate_final_test (which standardizes test series with the same
load_split as train/validation). Only reads <outdir>/best_params.json and
train_test_split_data/test.npz -- touching test never requires re-running the Optuna study.

Refuses to overwrite an existing best_test_raw.csv unless --force is passed: TEST is meant
to be scored exactly once per finalized config.

Usage (on CARC, from ~/Syntheticdata/optimization_warming):
    python evaluate_test_kstest.py --outdir results_ks --n-jobs 24
Then copy results_ks/best_test_raw.csv into optuna_search_kstest/results/.
"""
import argparse
import json
import os

import numpy as np

from optuna_search_kstest import DEFAULT_TRAIN_PATH, TOLERANCE, evaluate_final_test

DEFAULT_TEST_PATH = os.path.join(os.path.dirname(DEFAULT_TRAIN_PATH), "test.npz")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outdir", default="results_ks",
                        help="directory holding best_params.json; best_test_raw.csv is written here too")
    parser.add_argument("--params-json", default=None)
    parser.add_argument("--test-path", default=DEFAULT_TEST_PATH)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing best_test_raw.csv (re-spends TEST)")
    args = parser.parse_args()

    params_json_path = args.params_json or os.path.join(args.outdir, "best_params.json")
    test_csv = os.path.join(args.outdir, "best_test_raw.csv")
    if os.path.exists(test_csv) and not args.force:
        raise SystemExit(f"{test_csv} already exists -- TEST has already been scored for this config.\n"
                         f"Pass --force only if you deliberately mean to overwrite it.")

    with open(params_json_path) as f:
        saved = json.load(f)
    params = {k: saved[k] for k in ("w_min", "w_max", "n_w", "d_c", "n_c", "s_c")}
    print("Scoring on held-out TEST (one-time evaluation): "
          + "  ".join(f"{k}={v:.4g}" for k, v in params.items()), flush=True)

    test_time_arr = np.load(args.test_path)['time']
    test_df = evaluate_final_test(args.test_path, test_time_arr, TOLERANCE, args.n_jobs, params)
    test_df.to_csv(test_csv, index=False)
    print(f"test_f1_warming={test_df['f1_warming'].mean():.3f}  "
          f"test_precision_warming={test_df['precision_warming'].mean():.3f}  "
          f"test_recall_warming={test_df['recall_warming'].mean():.3f}")
    print(f"test_f1_all={test_df['f1_all'].mean():.3f}  test_f1_cooling={test_df['f1_cooling'].mean():.3f}  "
          f"mean_call_s={test_df['elapsed_s'].mean():.1f}")
    print(f"Saved -> {test_csv}")


if __name__ == "__main__":
    main()
