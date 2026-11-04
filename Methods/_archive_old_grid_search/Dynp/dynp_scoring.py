"""Cached per-(params, seed) scoring for dynp_grid_red_snr3.ipynb.

Kept in a real module (not defined inline in the notebook) so joblib.Memory's cache
path is stable across kernels/SLURM jobs. A function defined live in a Jupyter cell
gets tagged with that session's temp file path (SLURM job ID + kernel PID baked in),
so a new day's job would build a different cache path and never find yesterday's
results. This module has a real __file__, so the path never changes.
"""
import functools
import time as _time

import numpy as np
import ammonyte as amt
from ammonyte.utils.metrics import evaluate_detection
from joblib import Memory

_memory = Memory("dynp_score_cache", verbose=0)


@functools.lru_cache(maxsize=4)
def _load_data(data_path):
    d = np.load(data_path)
    time_arr = d['time']
    db_noisy = d['db_noisy']
    state_true = d['state_true']
    gt_times = []
    for s in range(len(state_true)):
        change_idx = np.where(np.diff(state_true[s]) != 0)[0] + 1
        gt_times.append(time_arr[change_idx])
    return time_arr, db_noisy, gt_times


def _clean_times(times):
    times = np.asarray(times, dtype=float)
    return times[~np.isnan(times)]


def _detect_dynp(time, value, algo, cost, n_bkps, min_size, jump=1):
    series = amt.Series(time=time, value=value, label='grid', verbose=False)
    t0 = _time.perf_counter()
    transitions = series.ruptures(algo=algo, cost=cost, n_bkps=n_bkps, min_size=min_size, jump=jump)
    elapsed_s = _time.perf_counter() - t0
    return _clean_times(transitions.jump_times), elapsed_s


@_memory.cache
def score_one(data_path, algo, tolerance, params, seed):
    time_arr, db_noisy, gt_times = _load_data(data_path)
    try:
        detected, elapsed_s = _detect_dynp(time_arr, db_noisy[seed], algo, **params)
    except Exception:
        return dict(precision=0.0, recall=0.0, f1=0.0, false_positives=np.nan, elapsed_s=np.nan)
    m = evaluate_detection(detected=detected, ground_truth=gt_times[seed], tolerance=tolerance)
    return dict(precision=m.precision, recall=m.recall, f1=m.f1_score, false_positives=m.false_positives,
                elapsed_s=elapsed_s)
