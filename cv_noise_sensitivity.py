#!/usr/bin/env python
"""
Cross-Validation for Detection Methods on Noise-Sensitivity Synthetic Data
Full run across all noise levels (snr2, snr3, snr4) and colors (white, pink, red)

Methods: ruptures (Pelt, Binseg, BottomUp, Window, KernelCPD) and KS test.
Note: ruptures 'Dynp' is excluded — like in cv_penalty_3000.py, it needs a
      separate n_bkps-based calibration scheme rather than a penalty grid.
      LERM is excluded for this pass — its epsilon-search + recurrence-matrix
      pipeline is too slow to include alongside these.
Note: cost='rbf' is excluded everywhere it would otherwise apply (Pelt,
      Binseg, BottomUp, Window, KernelCPD all support it). Benchmarking on
      this exact 6001-point series showed rbf
      is ~100-1000x slower than l1/l2 (24-116s/call vs 0.1-1.8s/call)
      regardless of algorithm — including it here would make the grid below
      computationally infeasible. KernelCPD keeps 'linear' since rbf/linear
      are its only supported costs.

Cross-validation is over seeds, not time: each (color, level) condition is 20
independent realizations of the same dynamics, so folds split over seed index
(sklearn KFold) rather than TimeSeriesSplit — see Section 4 of the smoke-test
notebook for the full rationale.
"""

import ammonyte as amt
from ammonyte.utils.metrics import evaluate_detection
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from joblib import Parallel, delayed
from joblib.externals.loky import get_reusable_executor
import warnings
import logging
import os
import json
import signal
from datetime import datetime

warnings.filterwarnings('ignore')

# ── Configuration ──────────────────────────────────────────────────────────────
DATA_DIR     = 'noise_sensitivity_data'   # run from the Syntheticdata directory
OUTPUT_DIR   = os.path.join(DATA_DIR, 'cv_results_full')
LEVELS       = ['snr2', 'snr3', 'snr4']
COLORS       = ['white', 'pink', 'red']
N_FOLDS      = 5                          # real run (smoke test used 3)
TOLERANCE    = 0.5                        # +/- 0.5 kyr matching tolerance
RANDOM_SEED  = 42
FOLD_TIMEOUT = 6 * 3600                   # 6-hour per-fold safety limit

# ── Parameter grids ──────────────────────────────────────────────────────────
# Widened from the smoke test per its own notes: ruptures reintroduces more
# algos (Pelt, BottomUp, Window, KernelCPD alongside Binseg) with more pens
# and min_sizes. Pelt and KernelCPD are still comparatively slow per-call
# (~22-46s and likely similar order, respectively, vs ~0.1-1.8s for
# Binseg/BottomUp/Window with l1/l2) — they're included because the goal now
# is broader method coverage, but they will dominate the wall-clock budget.
# If a checkpoint shows they're too slow in practice, trim them down and
# resume (this script skips configs already in checkpoint.csv).
PENS      = [4, 8, 12, 16, 20, 24, 28, 32, 40, 48]
MIN_SIZES = [10, 15, 20, 30]
WIDTHS    = [20, 30, 40]

RUPTURES_PARAM_GRID = (
    [{'algo': algo, 'cost': cost, 'pen': pen, 'min_size': ms}
     for algo in ['Pelt', 'Binseg', 'BottomUp']
     for cost in ['l1', 'l2']
     for pen in PENS
     for ms in MIN_SIZES]
    + [{'algo': 'Window', 'cost': cost, 'pen': pen, 'min_size': ms, 'width': w}
       for cost in ['l1', 'l2']
       for pen in PENS
       for ms in MIN_SIZES
       for w in WIDTHS]
    + [{'algo': 'KernelCPD', 'cost': 'linear', 'pen': pen, 'min_size': ms}
       for pen in PENS
       for ms in MIN_SIZES]
)

# KS test: every scalar knob that plausibly trades off precision/recall is
# now searched, for the same reason as ruptures' pen/min_size — w_min/w_max
# (window-size range), n_w (number of window lengths), n_c (min samples per
# window, the direct analog of ruptures' min_size), d_c (KS-statistic cutoff),
# and s_c (std-ratio cutoff). x_c is the one exception, left at its default
# None: that's not a scalar value to grid-search, it's a switch that tells
# the algorithm to auto-calculate the change threshold per series — pinning
# it to fixed numbers would remove that per-series adaptivity and, unlike
# w_min/w_max (anchored to the package's own NGRIP example), there's no
# principled anchor for what fixed values to try.
# n_w is the dominant per-call cost driver (~linear in n_w), so it's kept to
# two values (the smoke-test value and the package default) rather than a
# finer sweep, to keep this grid's cost from swamping the ruptures grid.
W_MINS = [0.08, 0.12, 0.2]
W_MAXS = [1.5, 2.5, 4.0]
N_WS   = [7, 15]
N_CS   = [3, 5]

KSTEST_PARAM_GRID = [
    {'w_min': w_min, 'w_max': w_max, 'n_w': n_w, 'd_c': d_c, 'n_c': n_c, 's_c': s_c, 'x_c': None}
    for w_min in W_MINS
    for w_max in W_MAXS
    for n_w in N_WS
    for n_c in N_CS
    for d_c in [0.5, 0.6, 0.7, 0.8, 0.9]
    for s_c in [1.0, 1.5, 2.0, 2.5]
]

METHODS = {
    'ruptures': RUPTURES_PARAM_GRID,
    #'kstest':   KSTEST_PARAM_GRID,
}
# ───────────────────────────────────────────────────────────────────────────────

np.random.seed(RANDOM_SEED)
N_JOBS = int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count()))
os.makedirs(OUTPUT_DIR, exist_ok=True)

log_file = os.path.join(OUTPUT_DIR, f'run_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


class FoldTimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise FoldTimeoutError()


# ── Ground truth ──────────────────────────────────────────────────────────────
def transition_times_from_states(time, states):
    """True transition times (both directions) from a state-label array.

    Mirrors Noise_Sensitivity_Synthetic_Data.ipynb / the smoke-test notebook:
    any stadial/interstadial change counts as ground truth, regardless of
    direction (this synthetic model's transitions are symmetric, unlike the
    D-O catalog which only lists warming events).
    """
    change_idx = np.where(np.diff(states) != 0)[0] + 1
    return time[change_idx]


# ── Method wrappers ────────────────────────────────────────────────────────────
def _clean_times(times):
    times = np.asarray(times, dtype=float)
    return times[~np.isnan(times)]


def detect_ruptures(time, value, algo, cost, pen, min_size, width=None):
    series = amt.Series(time=time, value=value, label='cv', verbose=False)
    kwargs = {'width': width} if width is not None else {}
    transitions = series.ruptures(algo=algo, cost=cost, pen=pen, min_size=min_size, **kwargs)
    return _clean_times(transitions.jump_times)


def detect_kstest(time, value, w_min, w_max, n_w, d_c, n_c, s_c, x_c):
    series = amt.Series(time=time, value=value, label='cv', verbose=False)
    transitions = series.kstest(w_min=w_min, w_max=w_max, n_w=n_w, d_c=d_c, n_c=n_c, s_c=s_c, x_c=x_c)
    return _clean_times(transitions.jump_times)


DETECT_FNS = {
    'ruptures': detect_ruptures,
    'kstest':   detect_kstest,
}


# ── CV framework (seed-based, method-agnostic) ─────────────────────────────────
def kfold_splits(n_seeds, n_folds=N_FOLDS, seed=RANDOM_SEED):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return list(kf.split(np.arange(n_seeds)))


def _score_one(detect_fn, params, time, value, gt, tolerance):
    try:
        detected = detect_fn(time, value, **params)
    except Exception:
        return 0.0, 0.0, 0.0
    m = evaluate_detection(detected=detected, ground_truth=gt, tolerance=tolerance)
    return m.precision, m.recall, m.f1_score


def calibrate(detect_fn, param_grid, time, db_noisy, gt_times, train_idx,
              tolerance=TOLERANCE, n_jobs=N_JOBS):
    jobs = [(pi, s) for pi in range(len(param_grid)) for s in train_idx]
    results = Parallel(n_jobs=n_jobs)(
        delayed(_score_one)(detect_fn, param_grid[pi], time, db_noisy[s], gt_times[s], tolerance)
        for pi, s in jobs
    )
    f1_by_param = [[] for _ in param_grid]
    for (pi, _), (_, _, f1) in zip(jobs, results):
        f1_by_param[pi].append(f1)
    mean_f1 = [float(np.mean(v)) for v in f1_by_param]
    best_i = int(np.argmax(mean_f1))
    return param_grid[best_i], mean_f1[best_i]


def score_on_seeds(detect_fn, params, time, db_noisy, gt_times, seed_idx,
                    tolerance=TOLERANCE, n_jobs=N_JOBS):
    results = Parallel(n_jobs=n_jobs)(
        delayed(_score_one)(detect_fn, params, time, db_noisy[s], gt_times[s], tolerance)
        for s in seed_idx
    )
    precisions, recalls, f1s = zip(*results)
    return float(np.mean(precisions)), float(np.mean(recalls)), float(np.mean(f1s))


def run_cv(method_name, detect_fn, param_grid, time, db_noisy, gt_times, n_folds=N_FOLDS,
           done_folds=frozenset(), on_fold_complete=None):
    n_seeds = db_noisy.shape[0]
    fold_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(kfold_splits(n_seeds, n_folds)):
        if fold_idx in done_folds:
            logger.info(f"    fold {fold_idx}: already done, skipping")
            continue

        fold_start = datetime.now()
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(FOLD_TIMEOUT)

        try:
            best_params, train_f1 = calibrate(detect_fn, param_grid, time, db_noisy, gt_times, train_idx)
            test_precision, test_recall, test_f1 = score_on_seeds(
                detect_fn, best_params, time, db_noisy, gt_times, test_idx
            )
            fold_time = (datetime.now() - fold_start).total_seconds()

            row = dict(
                method=method_name, fold=fold_idx,
                train_f1=train_f1,
                test_precision=test_precision,
                test_recall=test_recall,
                test_f1=test_f1,
                best_params=json.dumps(best_params),
                fold_time_seconds=fold_time,
            )
            fold_results.append(row)
            if on_fold_complete is not None:
                on_fold_complete(row)
            logger.info(f"    fold {fold_idx}: train_f1={train_f1:.3f} test_f1={test_f1:.3f} "
                        f"({fold_time:.0f}s) params={best_params}")

        except FoldTimeoutError:
            logger.warning(f"    fold {fold_idx}: TIMEOUT ({FOLD_TIMEOUT // 3600}h limit)")
            # SIGALRM interrupts the main process's blocking Parallel() call, but the
            # loky worker processes it was waiting on keep running in the background
            # (they were mid-calibrate() on the full param grid). Left alone, they'd
            # still be consuming CPUs when the next fold's Parallel() call spins up a
            # fresh batch of workers, oversubscribing the 24 allocated cores and
            # slowing down everything after a timeout, not just the fold that hit it.
            get_reusable_executor().shutdown(kill_workers=True)
        except Exception as e:
            logger.error(f"    fold {fold_idx}: ERROR — {type(e).__name__}: {e}")
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

    return fold_results


# ── Data loading ───────────────────────────────────────────────────────────────
with open(os.path.join(DATA_DIR, "noise_grid_config.json")) as f:
    grid_config = json.load(f)
logger.info(f"Noise grid config: {grid_config}")
logger.info(f"Parallel workers: {N_JOBS}")


def load_condition(level, color):
    path = os.path.join(DATA_DIR, f"noise_{color}_{level}.npz")
    d = np.load(path)
    time = d['time']
    db_noisy = d['db_noisy']
    state_true = d['state_true']
    n_seeds = db_noisy.shape[0]
    gt_times = [transition_times_from_states(time, state_true[s]) for s in range(n_seeds)]
    return time, db_noisy, gt_times


# ── Main loop with checkpoint/resume ────────────────────────────────────────────
configs = [
    {'level': level, 'color': color, 'method': method}
    for level in LEVELS
    for color in COLORS
    for method in METHODS
]

checkpoint_file = os.path.join(OUTPUT_DIR, 'checkpoint.csv')
all_fold_results = []
done_folds_by_config = {}   # (level, color, method) -> set of completed fold indices

if os.path.exists(checkpoint_file):
    try:
        prev = pd.read_csv(checkpoint_file)
        all_fold_results = prev.to_dict('records')
        for _, row in prev.iterrows():
            key = (row['level'], row['color'], row['method'])
            done_folds_by_config.setdefault(key, set()).add(int(row['fold']))
        logger.info(f"Resuming: {len(all_fold_results)} fold results already recorded "
                    f"across {len(done_folds_by_config)} configs")
    except Exception as e:
        logger.warning(f"Could not load checkpoint: {e}")

logger.info("=" * 60)
logger.info(f"NOISE-SENSITIVITY CV — levels={LEVELS}, colors={COLORS}")
logger.info(f"Configs: {len(configs)}, Folds: {N_FOLDS}, Tolerance: +/-{TOLERANCE} kyr")
logger.info(f"Ruptures grid: {len(RUPTURES_PARAM_GRID)} combos, "
            f"KS test grid: {len(KSTEST_PARAM_GRID)} combos")
logger.info("=" * 60)

for i, cfg in enumerate(configs):
    level, color, method = cfg['level'], cfg['color'], cfg['method']
    key = (level, color, method)
    done_folds = done_folds_by_config.get(key, set())
    if len(done_folds) >= N_FOLDS:
        logger.info(f"[{i+1}/{len(configs)}] {level}/{color}/{method} — already done, skipping")
        continue

    logger.info(f"[{i+1}/{len(configs)}] {level}/{color}/{method} "
                f"({len(done_folds)}/{N_FOLDS} folds already done)")
    time, db_noisy, gt_times = load_condition(level, color)

    # Checkpoint after every fold (not just after the whole config finishes) —
    # some configs run for hours, so per-config checkpointing would lose an
    # entire config's progress if the job hits its walltime mid-config.
    def _save_fold(row):
        row = dict(row)
        row['level'] = level
        row['color'] = color
        all_fold_results.append(row)
        pd.DataFrame(all_fold_results).to_csv(checkpoint_file, index=False)

    new_fold_results = run_cv(method, DETECT_FNS[method], METHODS[method], time, db_noisy, gt_times,
                               done_folds=done_folds, on_fold_complete=_save_fold)

    total_done = len(done_folds) + len(new_fold_results)
    if total_done:
        config_f1s = [r['test_f1'] for r in all_fold_results
                      if r['level'] == level and r['color'] == color and r['method'] == method]
        logger.info(f"  -> F1={np.mean(config_f1s):.4f} +/- {np.std(config_f1s):.4f} "
                    f"({total_done}/{N_FOLDS} folds)")
    else:
        logger.warning(f"  -> no folds completed for {level}/{color}/{method}")

df = pd.DataFrame(all_fold_results)
df.to_csv(os.path.join(OUTPUT_DIR, 'cv_results.csv'), index=False)

summary = (df.groupby(['level', 'color', 'method'])['test_f1']
           .agg(['mean', 'std']).round(3).sort_values('mean', ascending=False))
logger.info(f"\nDone. Results saved to {OUTPUT_DIR}/cv_results.csv")
logger.info(f"\nSummary (test F1 by level/color/method):\n{summary}")
