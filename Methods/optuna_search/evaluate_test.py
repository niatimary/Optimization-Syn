#!/usr/bin/env python
"""
Standalone, one-shot scoring of a previously chosen (method, cost, pen/n_bkps,
min_size) config against the held-out TEST split.

Works with a best_params.json from EITHER optuna_search.py (patience-based early
stopping) or optuna_search_pruning.py (WilcoxonPruner-based early stopping) -- both
write the same method/cost/pen/n_bkps/min_size/jump fields. Their load_split
functions are NOT identical though: optuna_search_pruning.py's standardizes every
series (zero mean, unit std, from its own stats) before anything else touches it;
optuna_search.py's does not. So which module's evaluate_final_test scores test.npz
matters now -- using the wrong one would score test at the wrong scale relative to
whatever train/validation were optimized against. --source picks the right one
(default: inferred from --outdir, per this file's own --outdir convention below --
override with --source if your --outdir doesn't follow that naming).

Deliberately kept separate from optuna_search.py/optuna_search_pruning.py: touching
test.npz should never require re-running (or being capable of re-running) the Optuna
study. This script only reads <outdir>/best_params.json (written once a study
finishes and its winning config is validated) and train_test_split_data/test.npz --
nothing else needs to happen first.

Refuses to overwrite an existing best_test_raw.csv unless --force is passed: TEST is
meant to be scored exactly once per finalized config, after every method/cost/
pen-range/sampler decision is already locked in via validation.

Usage:
    python evaluate_test.py --outdir results            # score the patience run's winner
    python evaluate_test.py --outdir results_pruning     # score the pruning run's winner
    python evaluate_test.py --outdir results --n-jobs 8
    python evaluate_test.py --outdir results --force   # deliberately re-score, overwriting the previous result
"""
import argparse
import json
import os
import sys

import numpy as np

from optuna_search import HERE, DEFAULT_TEST_PATH, TOLERANCE
from optuna_search import evaluate_final_test as _evaluate_final_test_patience

# optuna_search.py (imported above) inserts Methods/ onto sys.path so it can find
# method_configs.py -- but that makes `import optuna_search_pruning` resolve to the
# Methods/optuna_search_pruning/ FOLDER as an implicit namespace package (it has no
# __init__.py) instead of descending into it to find optuna_search_pruning.py, since
# the folder and the .py file inside it share the same name. Put that folder itself
# on sys.path first so the file resolves as the top-level module.
sys.path.insert(0, os.path.join(HERE, "..", "optuna_search_pruning"))
from optuna_search_pruning import evaluate_final_test as _evaluate_final_test_pruning


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outdir", default=os.path.join(HERE, "optuna_results"),
                         help="directory holding best_params.json; best_test_raw.csv is written here too")
    parser.add_argument("--params-json", default=None,
                         help="override path to the winning-config JSON (default: <outdir>/best_params.json)")
    parser.add_argument("--test-path", default=DEFAULT_TEST_PATH)
    parser.add_argument("--source", choices=["patience", "pruning"], default=None,
                         help="which module's load_split/evaluate_final_test scores test.npz -- "
                              "'pruning' standardizes each series first (optuna_search_pruning.py), "
                              "'patience' does not (optuna_search.py). Must match whichever module's "
                              "search actually produced --outdir's best_params.json, or test.npz gets "
                              "scored at the wrong scale relative to what was optimized. Default: "
                              "inferred from --outdir ('pruning' in the name -> pruning, else "
                              "patience) -- pass this explicitly if your --outdir doesn't follow "
                              "that naming.")
    parser.add_argument("--n-jobs", type=int, default=-1,
                         help="joblib n_jobs; -1 = all allocated cores. Mind Dynp's memory use if the "
                              "winning config is Dynp -- see optuna_search.py's module docstring")
    parser.add_argument("--force", action="store_true",
                         help="overwrite an existing best_test_raw.csv -- TEST is meant to be scored once, "
                              "so only pass this when you deliberately mean to re-spend it")
    args = parser.parse_args()

    params_json_path = args.params_json or os.path.join(args.outdir, "best_params.json")
    test_csv = os.path.join(args.outdir, "best_test_raw.csv")

    if os.path.exists(test_csv) and not args.force:
        raise SystemExit(
            f"{test_csv} already exists -- TEST has already been scored for this config.\n"
            f"Pass --force only if you deliberately mean to overwrite it."
        )

    with open(params_json_path) as f:
        saved = json.load(f)
    best_full_params = dict(method=saved['method'], cost=saved['cost'], pen=saved['pen'],
                             n_bkps=saved['n_bkps'], min_size=saved['min_size'], jump=saved['jump'])
    param_desc = (f"method={best_full_params['method']}  cost={best_full_params['cost']}  "
                  f"min_size={best_full_params['min_size']}  "
                  f"{'pen=' + format(best_full_params['pen'], '.3g') if best_full_params['pen'] is not None else 'n_bkps=' + str(best_full_params['n_bkps'])}")

    source = args.source or ("pruning" if "pruning" in args.outdir.lower() else "patience")
    print(f"Scoring on held-out TEST (one-time evaluation): {param_desc}", flush=True)
    print(f"source={source} ({'standardized' if source == 'pruning' else 'raw, unstandardized'} "
          f"test data, via {'optuna_search_pruning' if source == 'pruning' else 'optuna_search'}.py)",
          flush=True)

    if source == "pruning":
        # optuna_search_pruning.py's evaluate_final_test loads test's time array itself
        # (via its own, standardizing load_split) -- no separate time_arr argument.
        test_df = _evaluate_final_test_pruning(args.test_path, TOLERANCE, args.n_jobs, best_full_params)
    else:
        # optuna_search.py's evaluate_final_test takes test's own time array as an explicit
        # argument rather than loading it itself -- load it directly here (cheap: it's one
        # small array out of the .npz, not the full detect-and-score pipeline) rather than
        # assuming it happens to match some other split's time array.
        test_time_arr = np.load(args.test_path)['time']
        test_df = _evaluate_final_test_patience(args.test_path, test_time_arr, TOLERANCE, args.n_jobs,
                                                  best_full_params)
    test_df.to_csv(test_csv, index=False)
    print(f"test_f1_warming={test_df['f1_warming'].mean():.3f}  "
          f"test_precision_warming={test_df['precision_warming'].mean():.3f}  "
          f"test_recall_warming={test_df['recall_warming'].mean():.3f}")
    print(f"test_f1_all={test_df['f1_all'].mean():.3f}  test_f1_cooling={test_df['f1_cooling'].mean():.3f}  "
          f"mean_call_s={test_df['elapsed_s'].mean():.3f}")
    print(f"Saved -> {test_csv}")


if __name__ == "__main__":
    main()
