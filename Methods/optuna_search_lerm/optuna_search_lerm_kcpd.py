#!/usr/bin/env python
"""
VERSION 2 of this search (the earlier version used auto tau, a fixed penalty and a swept block size).
Everything further down describes that earlier version; these changes OVERRIDE it:
  - tau is TUNED (log int [2, 48], one value shared by every series), not auto.
  - m in [2, 12]; target_density in [0.01, 0.15] (log); w_size in [10, 60];
    w_incre = max(1, round(w_incre_frac * w_size)) with w_incre_frac in [0.1, 0.5].
  - block_size is FIXED at 2 (no longer swept).
  - The KernelCPD penalty is no longer the fixed 8.3475. It is rescaled to the FI series length,
        beta = k * 8.3475 * log(n_FI) / log(6001),
    n_FI = number of points of the FI series KernelCPD sees, 6001 = raw series length the 8.3475
    was tuned on. k is SWEPT inside every trial (one FI series reused), k in
    {0.05, 0.1, 0.15, 0.25, 0.35, 0.5, 0.75, 1}; the trial score is the best k's series-mean f1
    and that k is recorded.
  - find_epsilon failures ("Runaway operation") are retried (start eps 0.5, then 2, then tolerance 0.004) before a series counts as failed.
  - Still fixed: linear kernel, min_size = 20 raw samples (converted to FI points), jump = 1.

Joint Optuna (TPE) search over LERM (Laplacian Eigenmaps of Recurrence Matrices,
James et al. 2024 -- ammonyte.TimeEmbeddedSeries / RecurrenceMatrix / RQARes) with
KernelCPD as the transition detector, on the NOISIER synthetic data. Sibling of
optuna_search_lerm.py (which detected transitions with bootstrap upper/lower percentile
bounds): same train/validation/test discipline, same direction-stratified scoring, same
resumable sqlite study, same patience-based study-level early stopping, same output
file shapes.

Pipeline per series:  standardize -> TimeEmbeddedSeries(m, tau auto) -> find_epsilon
(target_density) -> laplacian_eigenmaps(w_size, w_incre) = Fisher Information (FI)
series -> smooth_series(block_size) -> standardize FI -> KernelCPD -> detections ->
warming/cooling label from the ORIGINAL signal before/after each detection.

What changed vs. optuna_search_lerm.py:
  - The upper/lower percentile bounds (and the bootstrap w / n_samples) are GONE: they
    are neither tuned nor used. KernelCPD replaces lerm_transitions.
  - KernelCPD is NOT tuned. Its settings are inherited, held fixed, from the winner of the
    ruptures search on the same noisier data (optuna_search/results/best_params.json):
    linear kernel, pen=8.3475, min_size=20 samples, jump=1. They were tuned on the raw
    series, so they are not claimed to be optimal for FI (see transfer_check_kcpd.py for how
    sensitive the result is to pen). Two details of reusing them on the FI series:
      * the smoothed FI is standardized (zero mean, unit std) before KernelCPD, as the raw
        signal was when pen was tuned, so the series enters on the same scale. pen itself
        has no exact conversion between sampling rates and is carried over unchanged.
      * min_size=20 raw samples (0.4 kyr) is a duration, so it is converted to the FI
        series' own sampling (FI has one point per w_incre raw samples) to keep the same
        0.4 kyr: min_size_fi = max(2, round(20 * dt_raw / dt_fi)). The floor of 2 only
        binds for w_incre >= 10.
  - Direction: warming/cooling is classified from the original signal (mean before vs.
    after each detection, same classifier as the ruptures/KS searches), NOT from the
    sign of any FI jump -- an FI jump says the dynamics changed, not which way the signal went.
  - Ranges widened because the first search piled up at its edges (m=5 lower bound,
    target_density near its lower bound, w_incre near its upper bound, block_size=1,
    loosest bounds). block_size is swept over multiples of the window overlap r (below), not an arbitrary grid.

Five parameters are tuned: four by TPE, block_size by an exhaustive sweep inside each trial:

  Tuned (the eigendecomposition in laplacian_eigenmaps is essentially all of the cost):
            m              int    [2, 10]
            target_density float  [0.005, 0.15] (log)   recurrence-matrix density
            w_size         int    [10, 60]             eigenmap window length
            w_incre        int    [2, min(20, w_size//3)]   window stride (r >= 3); sampled
                                                       after w_size, which bounds it
  block_size (swept exhaustively inside every trial, reusing the FI series just computed):
          block_size = max(2, round(c * r)) with r = w_size / w_incre >= 3 and
          c in {0.25, 0.5, 0.75, 1, 1.5}.
          r is the number of FI points one eigenmap window spans: each FI point comes
          from a window of w_size raw samples moved by w_incre, so neighbouring FI points
          overlap heavily, and a block of r points smooths over one window (w_size * dt of
          time, at most 1.2 kyr over the allowed ranges). The largest multiplier, 1.5r,
          covers at most 1.8 kyr (w_size=60); the median gap between true events is
          ~1.5 kyr, so longer blocks would smooth events away and are not offered.
          For small r several c collapse to the same block_size; identical cells just
          repeat (harmless; ties go to the smallest c). Both c and block_size are recorded.
          r >= 3 (MIN_R) keeps the previous search's winning region (r ~ 3-4) searchable.
          smooth_series (ammonyte) is NON-overlapping block averaging, so the smoothed FI is
          piecewise constant and KernelCPD can only place a change at a block edge: detection
          times are quantized to the block duration c * w_size * dt.
          Each FI point is stamped with the END time of its window (no centering), so detections
          can lag a true transition by up to ~w_size * dt; NGRIP gets the same treatment.

A trial's score: each series is run end-to-end separately; the 5 cells are averaged over
the train series; the best cell's mean f1 (--objective: warming (default) or all) is the
trial value, and its c / block_size are recorded as trial attributes. The best cell is
always picked from the SERIES-MEAN, never per series. Taking the best of 5 cells makes a
trial's train score slightly optimistic, equally for every trial; validation is the
honest number.

If the best trials pile up at a bound, widen it and start a new --study-name; the
search-space fingerprint blocks resuming otherwise.

Fixed (not tuned): tau (auto per series), epsilon (derived by find_epsilon), the
find_epsilon start eps=1, density tolerance 0.002, amp 10, and the KernelCPD settings above.

Failures: any series failing makes the whole trial FAIL (never scored as 0). A series
with zero detections is a legitimate f1=0, not a failure. --max-call-seconds is a very
generous per-series safety net against a hang (e.g. find_epsilon never converging),
NOT a cost constraint -- slow configs (small w_incre, large m) must stay searchable.

Data: train_test_split_data/{train,validation,test}.npz = 60/30/30
series. Test is never opened here; evaluate_final_test() is ready for a one-time
final evaluation.

Outputs under --outdir: optuna_trials.csv, best_params.json, best_train_raw.csv /
best_validation_raw.csv (per-series, winning cell), best_inner_surface_train.csv /
best_inner_surface_validation.csv (mean metrics for every c of the winning config),
optuna_convergence.csv/.png.

Usage:
    python optuna_search_lerm_kcpd.py --n-trials 250 --min-trials 100 --patience 50 --n-jobs 24
    # interrupted? rerun the same command -- it resumes from the sqlite study.
"""
import argparse
import copy
import hashlib
import json
import os
import resource
import signal
import sys
import time

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from optuna.trial import TrialState
from joblib import Parallel, delayed

import ammonyte as amt
from pyleoclim.utils.tsutils import standardize as pyleo_standardize
from ammonyte.utils.metrics import evaluate_detection

HERE = os.path.dirname(os.path.abspath(__file__))
METHODS_DIR = os.path.dirname(HERE)
SYNTHETICDATA_DIR = os.path.dirname(METHODS_DIR)


class SeriesEvaluationError(Exception):
    """Any series (or the whole Parallel call) failing during the live search; caught
    in study.optimize(catch=...) so that ONE trial is marked FAIL and the study goes on,
    instead of scoring the failure as f1=0.0."""


class CallTimeoutError(Exception):
    """A single series' LERM call exceeded --max-call-seconds (SIGALRM, Unix-only; valid
    because each call runs as the sole task of a joblib/loky worker process)."""


def _alarm_handler(signum, frame):
    raise CallTimeoutError("LERM call exceeded --max-call-seconds")


TOLERANCE = 0.5            # +/- 0.5 kyr matching tolerance
RANDOM_SEED = 42
DEFAULT_MAX_CALL_SECONDS = 7200   # safety net against a hang only -- normal calls take 20-90 s
DIRECTION_MIN_WINDOW = 10  # raw samples each side of a detection used to classify warming/cooling (as in the ruptures/KS searches)

_SPLIT_DIR = os.path.join(SYNTHETICDATA_DIR, "train_test_split_data")
DEFAULT_TRAIN_PATH = os.path.normpath(os.path.join(_SPLIT_DIR, "train.npz"))
DEFAULT_VAL_PATH = os.path.normpath(os.path.join(_SPLIT_DIR, "validation.npz"))
DEFAULT_TEST_PATH = os.path.normpath(os.path.join(_SPLIT_DIR, "test.npz"))

# Phase 1 (sampled by TPE)
M_BOUNDS = (2, 12)
TAU_BOUNDS = (2, 48)                 # samples, log scale; one value shared by every series
TARGET_DENSITY_BOUNDS = (0.01, 0.15)
W_SIZE_BOUNDS = (10, 60)
W_INCRE_FRAC_BOUNDS = (0.1, 0.5)     # w_incre = max(1, round(frac * w_size))
# Phase 2 (swept exhaustively inside every trial, reusing the FI series): penalty multiplier k
BLOCK_SIZE = 2                       # fixed
K_GRID = (0.05, 0.1, 0.15, 0.25, 0.35, 0.5, 0.75, 1.0)
CELLS = K_GRID
# Fixed settings
EPS = 1                     # find_epsilon starting value (only a start; the search converges to target_density)
DENSITY_TOLERANCE = 0.002
# find_epsilon sometimes fails to converge ("Runaway operation", seen on CARC at m >= 13). Instead of failing the
# whole trial for one series, retry with other starting eps values, then once with a looser tolerance; only if
# every attempt fails does the series (and so the trial) fail, as before.
EPS_RETRY = ((EPS, DENSITY_TOLERANCE), (0.5, DENSITY_TOLERANCE), (2, DENSITY_TOLERANCE), (EPS, 0.004))
AMP = 10
# KernelCPD: winner of the ruptures search on the same noisier data (optuna_search/results/best_params.json)
KCPD_COST = 'linear'        # kernel
KCPD_PEN = 8.34753880040116  # beta_full, tuned on raw series of KCPD_N_FULL points; rescaled per series, see _kcpd_penalty
KCPD_N_FULL = 6001
KCPD_MIN_SIZE_RAW = 20      # raw samples; converted to FI points per series (see module docstring)
KCPD_JUMP = 1


CELL_COLS = ['k']


def _search_space_fingerprint():
    """Hash of everything that defines what a trial can be: every bound, the
    penalty-multiplier grid, and the fixed settings -- used to refuse resuming a study whose space changed."""
    payload = json.dumps(dict(m=M_BOUNDS, tau=TAU_BOUNDS, td=TARGET_DENSITY_BOUNDS, w_size=W_SIZE_BOUNDS,
                               w_incre_frac=W_INCRE_FRAC_BOUNDS, block_size=BLOCK_SIZE, k_grid=K_GRID, eps=EPS,
                               density_tol=DENSITY_TOLERANCE, amp=AMP,
                               kcpd=(KCPD_COST, KCPD_PEN, KCPD_N_FULL, KCPD_MIN_SIZE_RAW, KCPD_JUMP), eps_retry=EPS_RETRY),
                         sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _hash_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]



def transitions_from_states(time_arr, states):
    """True transition times AND directions from a binary stadial(1)/interstadial(0)
    state-label array. A 1->0 transition is warming (stadial->interstadial, matching
    Rasmussen et al. DO onsets and build_train_test_split.ipynb's
    warming_transition_times); 0->1 is cooling. Direction always resolves -- diff is
    always exactly +-1 for a binary state array, never 0 at a change_idx by
    construction."""
    diffs = np.diff(states)
    change_idx = np.where(diffs != 0)[0]
    times = time_arr[change_idx + 1]
    directions = np.array(['warming' if d == -1 else 'cooling' for d in diffs[change_idx]])
    return times, directions


def _matched_timing_errors(detected, ground_truth, tolerance):
    """Same one-to-one closest-match rule as ammonyte's evaluate_detection (each
    ground-truth point matches at most one detection, within tolerance), but returns
    the matched |detected - ground_truth| distances themselves instead of collapsing
    them to precision/recall/f1. Empty array if nothing matched (or either input is
    empty)."""
    detected = np.asarray(detected, dtype=float)
    ground_truth = np.asarray(ground_truth, dtype=float)
    if len(detected) == 0 or len(ground_truth) == 0:
        return np.array([], dtype=float)

    detected = np.sort(detected)
    ground_truth = np.sort(ground_truth)
    matched_gt = set()
    errors = []
    for det in detected:
        best_j, best_d = None, np.inf
        for j, gt in enumerate(ground_truth):
            if j in matched_gt:
                continue
            d = abs(gt - det)
            if d <= tolerance and d < best_d:
                best_j, best_d = j, d
        if best_j is not None:
            matched_gt.add(best_j)
            errors.append(best_d)
    return np.array(errors, dtype=float)


def evaluate_detection_by_direction(detected, det_directions, gt_times, gt_directions, tolerance):
    """f1_all (direction-agnostic) plus f1_warming/f1_cooling: BOTH detections and ground
    truth are filtered to one direction before calling ammonyte's unmodified
    evaluate_detection on that subset. `det_directions` are LERM's own labels (see
    _classify_detected_directions). Also returns the matched timing errors (see _matched_timing_errors)
    for the same three (all/warming/cooling) subsets."""
    detected = np.asarray(detected, dtype=float)
    det_directions = np.asarray(det_directions)
    gt_times = np.asarray(gt_times, dtype=float)
    gt_directions = np.asarray(gt_directions)

    metrics = {'all': evaluate_detection(detected=detected, ground_truth=gt_times, tolerance=tolerance)}
    timing_errors = {'all': _matched_timing_errors(detected, gt_times, tolerance)}
    for label in ('warming', 'cooling'):
        det_l = detected[det_directions == label]
        gt_l = gt_times[gt_directions == label]
        metrics[label] = evaluate_detection(detected=det_l, ground_truth=gt_l, tolerance=tolerance)
        timing_errors[label] = _matched_timing_errors(det_l, gt_l, tolerance)
    return metrics, timing_errors


def _direction_label(delta):
    if delta > 0:
        return 'warming'
    elif delta < 0:
        return 'cooling'
    return 'ambiguous'   # exact tie -- effectively never happens with continuous noisy data


def _classify_detected_directions(value, time_arr, detected_times):
    """Classify each detection as 'warming' or 'cooling' by comparing the local mean of the
    ORIGINAL signal just before vs. just after it, in plain forward time -- the same
    classifier the ruptures and KS searches use (a warming 1->0 transition always shows the
    value increasing forward through the array). DIRECTION_MIN_WINDOW samples each side,
    clipped at series edges and at neighboring detections; an empty window -> 'ambiguous'
    (excluded from the warming/cooling F1s, still scored in f1_all)."""
    detected_times = np.asarray(detected_times, dtype=float)
    if len(detected_times) == 0:
        return np.array([], dtype=object)

    order = np.argsort(detected_times)
    idxs = np.array([int(np.argmin(np.abs(time_arr - t))) for t in detected_times[order]])

    directions = np.empty(len(idxs), dtype=object)
    for k, idx in enumerate(idxs):
        before_start = max(0, idx - DIRECTION_MIN_WINDOW)
        after_end = min(len(value), idx + DIRECTION_MIN_WINDOW)
        if k > 0:
            before_start = max(before_start, idxs[k - 1])
        if k < len(idxs) - 1:
            after_end = min(after_end, idxs[k + 1])

        before_vals = value[before_start:idx]
        after_vals = value[idx:after_end]
        if len(before_vals) == 0 or len(after_vals) == 0:
            directions[k] = 'ambiguous'
        else:
            directions[k] = _direction_label(after_vals.mean() - before_vals.mean())

    out = np.empty(len(idxs), dtype=object)
    out[order] = directions
    return out


def _kcpd_penalty(k, n_fi):
    """beta = k * beta_full * log(n_FI) / log(n_full): the ruptures-search penalty, rescaled
    from the raw series length it was tuned on to the (much shorter) FI series."""
    return k * KCPD_PEN * np.log(n_fi) / np.log(KCPD_N_FULL)


def _standardized_fi(fi_smoothed):
    return amt.Series(time=np.asarray(fi_smoothed.time), value=np.asarray(fi_smoothed.value),
                      label='fi', verbose=False).standardize()


def _detect_kcpd(fi_series, raw_dt, k):
    """KernelCPD (linear kernel, min_size converted from raw samples to FI points, jump=1)
    with the length-rescaled penalty for multiplier k, on the standardized smoothed FI
    series. Returns (detection times on the FI series' time axis, the penalty used)."""
    dt_fi = float(np.median(np.diff(fi_series.time)))
    min_size = max(2, int(round(KCPD_MIN_SIZE_RAW * raw_dt / dt_fi)))
    pen = _kcpd_penalty(k, len(fi_series.time))
    transitions = fi_series.ruptures(algo='KernelCPD', cost=KCPD_COST, pen=pen,
                                      min_size=min_size, jump=KCPD_JUMP)
    times = np.asarray(transitions.jump_times, dtype=float)
    return times[~np.isnan(times)], pen


def load_split(npz_path):
    d = np.load(npz_path)
    time_arr = d['time']
    db_noisy = d['db_noisy']
    state_true = d['state_true']
    n = db_noisy.shape[0]

    # Standardize each series independently (zero mean, unit std, using that series' OWN
    # mean/std), the same call as optuna_search.py's load_split.
    db_noisy, _, _ = pyleo_standardize(db_noisy, axis=1)

    gt_times, gt_directions = [], []
    for i in range(n):
        t, dirs = transitions_from_states(time_arr, state_true[i])
        gt_times.append(t)
        gt_directions.append(dirs)
    return time_arr, db_noisy, gt_times, gt_directions, n

def _find_epsilon_with_retry(embedded, target_density):
    """find_epsilon with the EPS_RETRY fallbacks (start eps, tolerance). Re-raises the last
    RuntimeError if every attempt fails."""
    for attempt, (eps0, tol) in enumerate(EPS_RETRY):
        try:
            return embedded.find_epsilon(eps=eps0, target_density=target_density, tolerance=tol,
                                          amp=AMP, parallelize=False, verbose=False)
        except RuntimeError:
            if attempt == len(EPS_RETRY) - 1:
                raise


def _fisher_series(time_arr, value, m, tau, target_density, w_size, w_incre):
    """Phase 1: embed -> find_epsilon (with retries) -> recurrence matrix -> Laplacian eigenmaps."""
    series = amt.Series(time=time_arr, value=value, label='optuna', verbose=False)
    embedded = amt.TimeEmbeddedSeries(series, m=m, tau=tau)
    eps_result = _find_epsilon_with_retry(embedded, target_density)
    return eps_result['Output'].laplacian_eigenmaps(w_size=w_size, w_incre=w_incre)


def _score_one(time_arr, value, gt_times, gt_directions, m, tau, target_density, w_size, w_incre,
               cells, tolerance, max_call_seconds):
    """One series, end to end: phase 1 once, smoothing once (block_size fixed), then KernelCPD
    for every penalty multiplier k in `cells` on that same FI series. Returns dict(cells=[row per cell], error, elapsed_s,
    cpu_time_s, peak_rss_gb). peak_rss_gb is the whole worker process's high-water mark,
    not this call alone."""
    ru_before = resource.getrusage(resource.RUSAGE_SELF)
    t0 = time.perf_counter()
    rows, error = [], None
    raw_dt = float(np.median(np.diff(time_arr)))
    old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(max_call_seconds)
    try:
        fi = _fisher_series(time_arr, value, m, tau, target_density, w_size, w_incre)
        smoothed = amt.utils.fisher.smooth_series(copy.deepcopy(fi), block_size=BLOCK_SIZE)
        fi_std = _standardized_fi(smoothed)
        for k in cells:
            detected, pen = _detect_kcpd(fi_std, raw_dt, k)
            det_directions = _classify_detected_directions(value, time_arr, detected)
            metrics, timing_errors = evaluate_detection_by_direction(
                detected, det_directions, gt_times, gt_directions, tolerance)
            row = dict(k=k, pen=float(pen), n_fi=int(len(fi_std.time)), n_detected=int(len(detected)),
                       false_positives_all=metrics['all'].false_positives)
            for label, mt in metrics.items():
                row[f'precision_{label}'] = mt.precision
                row[f'recall_{label}'] = mt.recall
                row[f'f1_{label}'] = mt.f1_score
                errs = timing_errors[label]
                row[f'timing_mae_{label}'] = float(np.mean(errs)) if len(errs) else np.nan
                row[f'timing_n_matched_{label}'] = int(len(errs))
            rows.append(row)
    except Exception as e:
        rows, error = [], f"{type(e).__name__}: {e}"[:300]
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

    ru_after = resource.getrusage(resource.RUSAGE_SELF)
    return dict(cells=rows, error=error, elapsed_s=time.perf_counter() - t0,
                cpu_time_s=(ru_after.ru_utime - ru_before.ru_utime) + (ru_after.ru_stime - ru_before.ru_stime),
                peak_rss_gb=ru_after.ru_maxrss / (1024 ** 2 if sys.platform != 'darwin' else 1024 ** 3))


def _cell_means(cells_df):
    """Mean over series of every metric, per penalty multiplier k (pen and n_fi are means over series)."""
    return cells_df.groupby(CELL_COLS).mean(numeric_only=True).drop(columns='series').reset_index()


class OptunaSearch:
    """Pooled train/validation data and evaluation logic. train: what `objective` scores
    every trial against. validation: checked freely, once per candidate config. test: not
    loaded here at all (see evaluate_final_test)."""

    def __init__(self, train_path, val_path, tolerance, n_jobs, objective_metric='warming',
                 max_call_seconds=DEFAULT_MAX_CALL_SECONDS):
        (self.time_arr, self.db_train, self.gt_times_train, self.gt_dirs_train,
         self.n_train) = load_split(train_path)
        val_time_arr, self.db_val, self.gt_times_val, self.gt_dirs_val, self.n_val = load_split(val_path)
        # evaluate_validation reuses train's time axis -- make that assumption loud
        if not np.array_equal(self.time_arr, val_time_arr):
            raise ValueError("train and validation time arrays differ; evaluate_validation would "
                             "score validation series on train's time axis")
        self.tolerance = tolerance
        self.n_jobs = n_jobs
        self.max_call_seconds = max_call_seconds
        self.objective_metric = objective_metric

    def _sample_params(self, trial):
        m = trial.suggest_int('m', *M_BOUNDS)
        tau = trial.suggest_int('tau', *TAU_BOUNDS, log=True)
        target_density = trial.suggest_float('target_density', *TARGET_DENSITY_BOUNDS, log=True)
        w_size = trial.suggest_int('w_size', *W_SIZE_BOUNDS)
        w_incre_frac = trial.suggest_float('w_incre_frac', *W_INCRE_FRAC_BOUNDS)
        w_incre = max(1, round(w_incre_frac * w_size))
        trial.set_user_attr('w_incre', int(w_incre))
        return dict(m=m, tau=tau, target_density=target_density, w_size=w_size, w_incre=w_incre)

    def _evaluate(self, params, db, gt_times_list, gt_dirs_list, n_series, cells):
        """Returns (cells_df, info_df, err): cells_df has one row per (series, cell),
        info_df one row per series (error/elapsed/cpu/rss)."""
        try:
            scored = Parallel(n_jobs=self.n_jobs)(
                delayed(_score_one)(self.time_arr, db[s], gt_times_list[s], gt_dirs_list[s],
                                     params['m'], params['tau'], params['target_density'],
                                     params['w_size'], params['w_incre'], cells, self.tolerance, self.max_call_seconds)
                for s in range(n_series))
        except Exception as e:   # any worker crash
            return None, None, f"{type(e).__name__}: {e}"[:300]
        info_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'cells'} for r in scored])
        cells_df = pd.DataFrame([dict(row, series=s) for s, r in enumerate(scored) for row in r['cells']])
        return cells_df, info_df, None

    def objective(self, trial):
        params = self._sample_params(trial)
        cells_df, info_df, err = self._evaluate(params, self.db_train, self.gt_times_train,
                                                 self.gt_dirs_train, self.n_train, CELLS)
        if cells_df is None:
            trial.set_user_attr('error', err)
            raise SeriesEvaluationError(f"train evaluation crashed: {err}")
        errored = info_df[info_df['error'].notna()]
        if len(errored):
            first_error = errored['error'].iloc[0]
            trial.set_user_attr('error', first_error)
            trial.set_user_attr('n_errors', int(len(errored)))
            raise SeriesEvaluationError(
                f"{len(errored)}/{len(info_df)} train series failed to score; first error: {first_error}")

        surface = _cell_means(cells_df)
        best = surface.loc[surface[f'f1_{self.objective_metric}'].idxmax()]
        trial.set_user_attr('best_k', float(best['k']))
        trial.set_user_attr('best_pen_mean', float(best['pen']))
        trial.set_user_attr('n_fi_mean', float(best['n_fi']))
        trial.set_user_attr('mean_elapsed_s', float(info_df['elapsed_s'].mean()))
        trial.set_user_attr('mean_cpu_time_s', float(info_df['cpu_time_s'].mean()))
        trial.set_user_attr('max_peak_rss_gb', float(info_df['peak_rss_gb'].max()))
        trial.set_user_attr('mean_n_detected', float(best['n_detected']))
        for label in ('all', 'warming', 'cooling'):
            for stat in ('precision', 'recall', 'f1', 'timing_mae'):
                trial.set_user_attr(f'mean_{stat}_{label}', float(best[f'{stat}_{label}']))
        keep = CELL_COLS + ['pen', 'n_fi'] + [f'{s}_{l}' for l in ('all', 'warming', 'cooling') for s in ('precision', 'recall', 'f1')]
        trial.set_user_attr('inner_surface', surface[keep].round(5).to_dict(orient='records'))
        return float(best[f'f1_{self.objective_metric}'])

    def _evaluate_split(self, params, split, cells):
        db, gtt, gtd, n = ((self.db_train, self.gt_times_train, self.gt_dirs_train, self.n_train)
                           if split == 'train' else
                           (self.db_val, self.gt_times_val, self.gt_dirs_val, self.n_val))
        cells_df, info_df, err = self._evaluate(params, db, gtt, gtd, n, cells)
        if cells_df is None:
            raise RuntimeError(f"{split} evaluation failed: {err}")
        errored = info_df[info_df['error'].notna()]
        if len(errored):
            raise RuntimeError(f"{len(errored)} {split} series failed; first error: {errored['error'].iloc[0]}")
        return cells_df, info_df

    def evaluate_train(self, params, cells=CELLS):
        return self._evaluate_split(params, 'train', cells)

    def evaluate_validation(self, params, cells=CELLS):
        return self._evaluate_split(params, 'validation', cells)


def evaluate_final_test(test_path, time_arr, tolerance, n_jobs, params, max_call_seconds=DEFAULT_MAX_CALL_SECONDS):
    """Loads test data fresh and scores exactly one config (params must include
    tau and k). Call once, after everything is locked in via validation."""
    _, db_test, gt_times_test, gt_dirs_test, n_test = load_split(test_path)
    cell = (params['k'],)
    scored = Parallel(n_jobs=n_jobs)(
        delayed(_score_one)(time_arr, db_test[s], gt_times_test[s], gt_dirs_test[s], params['m'], params['tau'],
                             params['target_density'], params['w_size'], params['w_incre'],
                             cell, tolerance, max_call_seconds)
        for s in range(n_test))
    return pd.DataFrame([dict(r['cells'][0], series=s, error=r['error'], elapsed_s=r['elapsed_s'])
                         if r['cells'] else dict(series=s, error=r['error'], elapsed_s=r['elapsed_s'])
                         for s, r in enumerate(scored)])


def _make_early_stopping_callback(patience, min_trials, min_delta, stop_info):
    """Stops the STUDY (not any individual trial) once `patience` COMPLETE trials
    have passed with no improvement greater than `min_delta` over the running best
    -- see module docstring's 'Early stopping' section. Two guards on top of plain
    patience-counting:

      - min_trials: the callback is a no-op until this many trials have completed,
        so the study can't stop during TPE's own random-startup phase
        (n_startup_trials=20).
      - min_delta: an "improvement" only resets the patience counter if it clears
        this margin, so the study doesn't keep running indefinitely chasing
        float-noise-sized F1 gains.

    Recomputes best-so-far from the full study.trials history (filtered to
    TrialState.COMPLETE) on every call rather than tracking running state, so it
    also does the right thing on a resumed study that already plateaued in an
    earlier run. O(n) per call / O(n^2) over a whole study -- irrelevant next to
    per-trial evaluation cost here.

    `stop_info` is a dict the caller owns (mutated in place, not returned) --
    records whether *this* callback is what ended the study and at which trial,
    since study.optimize() otherwise gives no way to tell "stopped early" apart
    from "ran out of --timeout/--n-trials" after the fact."""
    def callback(study, trial):
        completed = sorted(
            (t for t in study.trials if t.state == TrialState.COMPLETE),
            key=lambda t: t.number,
        )

        if len(completed) < min_trials:
            return

        best_value = float("-inf")
        last_meaningful_improvement = 0
        for position, completed_trial in enumerate(completed):
            if completed_trial.value > best_value + min_delta:
                best_value = completed_trial.value
                last_meaningful_improvement = position

        trials_since_improvement = len(completed) - 1 - last_meaningful_improvement
        if trials_since_improvement >= patience:
            print(f"No improvement greater than {min_delta} during the last "
                  f"{trials_since_improvement} completed trials -- stopping study early.",
                  flush=True)
            stop_info['stopped_early'] = True
            stop_info['stopped_at_trial'] = trial.number
            stop_info['trials_since_improvement'] = trials_since_improvement
            study.stop()
    return callback


def _make_fail_streak_callback(max_consecutive_fails):
    """Stops the study once this many trials in a row have FAILED -- a systematic fault
    (e.g. missing OpenCL runtime) would otherwise burn through every trial in seconds."""
    def callback(study, trial):
        recent = [t for t in sorted(study.trials, key=lambda t: t.number)
                  if t.state in (TrialState.COMPLETE, TrialState.FAIL)][-max_consecutive_fails:]
        if len(recent) == max_consecutive_fails and all(t.state == TrialState.FAIL for t in recent):
            print(f"{max_consecutive_fails} consecutive FAILED trials (last error: "
                  f"{trial.user_attrs.get('error')}) -- stopping study.", flush=True)
            study.stop()
    return callback


def _save_convergence(trials_df, outdir, objective_metric):
    """Saves cumulative-best objective value vs. trial number, as both a CSV
    (optuna_convergence.csv) and a quick PNG (optuna_convergence.png) so the shape
    is visible without opening a notebook. Only COMPLETE trials are considered
    (matches what `_make_early_stopping_callback` itself watches)."""
    complete = trials_df[trials_df['state'] == 'COMPLETE'].sort_values('number').copy()
    complete['best_value_so_far'] = complete['value'].cummax()

    conv_csv = os.path.join(outdir, "optuna_convergence.csv")
    complete[['number', 'value', 'best_value_so_far']].to_csv(conv_csv, index=False)
    print(f"Saved convergence data ({len(complete)} completed trials) -> {conv_csv}")

    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(complete['number'], complete['value'], s=10, alpha=0.4, label=f'f1_{objective_metric} (per trial)')
        ax.plot(complete['number'], complete['best_value_so_far'], color='C1', linewidth=2, label='best so far')
        ax.set_xlabel('trial number')
        ax.set_ylabel(f'f1_{objective_metric}')
        ax.set_title('Optuna convergence (train)')
        ax.legend()
        fig.tight_layout()
        conv_png = os.path.join(outdir, "optuna_convergence.png")
        fig.savefig(conv_png, dpi=150)
        plt.close(fig)
        print(f"Saved convergence plot -> {conv_png}")
    except ImportError:
        print("matplotlib not available -- skipped optuna_convergence.png (optuna_convergence.csv still saved)")


def _best_full_params(best_trial):
    """Reconstruct every tuned parameter (plus the swept k and the fixed block_size) from the best trial."""
    p, a = best_trial.params, best_trial.user_attrs
    return dict(m=p['m'], tau=p['tau'], target_density=p['target_density'], w_size=p['w_size'],
                w_incre=a['w_incre'], w_incre_frac=p['w_incre_frac'], block_size=BLOCK_SIZE, k=a['best_k'])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-path", default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--val-path", default=DEFAULT_VAL_PATH)
    parser.add_argument("--n-trials", type=int, default=250,
                         help="hard cap on the TOTAL number of trials across all resumed invocations")
    parser.add_argument("--timeout", type=int, default=None,
                         help="optional wall-clock stop in seconds (e.g. just under the SLURM limit)")
    parser.add_argument("--n-jobs", type=int, default=-1, help="joblib n_jobs; -1 = all cores")
    parser.add_argument("--objective", choices=["all", "warming"], default="warming",
                         help="which F1 the sampler chases (f1_all/warming/cooling are always saved)")
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--storage", default=None, help="Optuna storage URL (default: sqlite under --outdir)")
    parser.add_argument("--study-name", default=None, help="default 'optuna_lerm_search_<objective>'")
    parser.add_argument("--patience", type=int, default=40,
                         help="stop once this many COMPLETE trials pass without an improvement > "
                              "--min-delta; 0 disables")
    parser.add_argument("--min-trials", type=int, default=100,
                         help="early stopping cannot trigger before this many completed trials")
    parser.add_argument("--min-delta", type=float, default=0.01)
    parser.add_argument("--max-call-seconds", type=int, default=DEFAULT_MAX_CALL_SECONDS,
                         help="per-series safety net against a hang, NOT a cost limit "
                             f"(default {DEFAULT_MAX_CALL_SECONDS}s)")
    args = parser.parse_args()

    if args.max_call_seconds <= 0:
        parser.error(f"--max-call-seconds must be > 0, got {args.max_call_seconds}")
    if args.n_jobs == 0:
        parser.error("--n-jobs cannot be 0 (use -1 for all cores)")
    if args.n_trials is not None and args.n_trials < 0:
        parser.error(f"--n-trials must be >= 0, got {args.n_trials}")
    if args.timeout is not None and args.timeout <= 0:
        parser.error(f"--timeout must be > 0, got {args.timeout}")

    outdir = args.outdir or os.path.join(HERE, f"optuna_results_{args.objective}")
    os.makedirs(outdir, exist_ok=True)
    storage = args.storage or f"sqlite:///{os.path.join(outdir, 'study.db')}"
    study_name = args.study_name or f"optuna_lerm_kcpd_search_{args.objective}"

    search = OptunaSearch(args.train_path, args.val_path, TOLERANCE, args.n_jobs,
                           objective_metric=args.objective, max_call_seconds=args.max_call_seconds)
    print(f"train seeds: {search.n_train}   validation seeds: {search.n_val}", flush=True)
    print(f"inner grid: {len(CELLS)} penalty multipliers k = {K_GRID}; block_size fixed at {BLOCK_SIZE}", flush=True)
    print(f"max_call_seconds={args.max_call_seconds}s (safety net)   n_jobs={args.n_jobs}", flush=True)

    # Flat 5-parameter TPE space (w_incre is derived from w_size via w_incre_frac, so no conditional bound)
    sampler = TPESampler(seed=RANDOM_SEED, n_startup_trials=20)
    study = optuna.create_study(direction="maximize", sampler=sampler, storage=storage,
                                 study_name=study_name, load_if_exists=True)
    n_trials_before = len(study.trials)
    print(f"study {study_name!r} @ {storage!r}: {n_trials_before} trial(s) already recorded", flush=True)

    study_config = dict(n_train=int(search.n_train), random_seed=RANDOM_SEED,
                        train_data_hash=_hash_file(args.train_path), objective=args.objective,
                        tolerance=TOLERANCE, search_space_hash=_search_space_fingerprint())
    if n_trials_before == 0:
        for key, value in study_config.items():
            study.set_user_attr(f"config_{key}", value)
    else:
        mismatches = [f"  {k}: recorded={study.user_attrs.get(f'config_{k}')!r}  this run={v!r}"
                      for k, v in study_config.items() if study.user_attrs.get(f"config_{k}") != v]
        if mismatches:
            raise RuntimeError(f"Resumed study {study_name!r} was created under different conditions:\n"
                               + "\n".join(mismatches) + "\nUse a new --study-name if this is intentional.")

    n_trials_to_run = max(0, args.n_trials - n_trials_before)
    print(f"--n-trials cap is {args.n_trials} total -- running up to {n_trials_to_run} more", flush=True)

    stop_info = dict(stopped_early=False, stopped_at_trial=None, trials_since_improvement=None)
    callbacks = ([_make_early_stopping_callback(args.patience, args.min_trials, args.min_delta, stop_info)]
                 if args.patience > 0 else [])
    callbacks.append(_make_fail_streak_callback(5))

    t0 = time.perf_counter()
    if n_trials_to_run > 0:
        study.optimize(search.objective, n_trials=n_trials_to_run, timeout=args.timeout,
                        callbacks=callbacks, catch=(SeriesEvaluationError,), show_progress_bar=False)
    wall = time.perf_counter() - t0
    n_trials_total = len(study.trials)
    n_trials_this_run = n_trials_total - n_trials_before
    n_failed_total = sum(1 for t in study.trials if t.state == TrialState.FAIL)

    trials_df = study.trials_dataframe()
    trials_path = os.path.join(outdir, "optuna_trials.csv")
    trials_df.drop(columns=[c for c in trials_df.columns if c.endswith('inner_surface')]).to_csv(trials_path, index=False)
    print(f"\nSaved {len(trials_df)} trials -> {trials_path}", flush=True)
    print(f"Study wall time: {wall:.0f}s   trials this run: {n_trials_this_run}   "
          f"trials total: {n_trials_total}   failed total: {n_failed_total}", flush=True)
    if stop_info['stopped_early']:
        print(f"Stopped early by the patience callback at trial #{stop_info['stopped_at_trial']}.", flush=True)

    _save_convergence(trials_df, outdir, args.objective)

    if sum(1 for t in study.trials if t.state == TrialState.COMPLETE) == 0:
        print("No trials reached COMPLETE -- check FAIL reasons in optuna_trials.csv. Stopping.", flush=True)
        sys.exit(1)

    best = study.best_trial
    print(f"\nBest trial #{best.number} (optimized on f1_{args.objective}): "
          f"train f1_all={best.user_attrs['mean_f1_all']:.3f}  f1_warming={best.user_attrs['mean_f1_warming']:.3f}  "
          f"f1_cooling={best.user_attrs['mean_f1_cooling']:.3f}")
    full = _best_full_params(best)
    print(f"  params: {full}")

    with open(os.path.join(outdir, "best_params.json"), "w") as f:
        json.dump(dict(**full, trial_number=best.number, objective=args.objective,
                        fixed=dict(tau='shared across series (tuned)', eps=EPS,
                                   density_tolerance=DENSITY_TOLERANCE, amp=AMP,
                                   kernelcpd=dict(cost=KCPD_COST, beta_full=KCPD_PEN, n_full=KCPD_N_FULL,
                                                  pen_rule='k * beta_full * log(n_FI) / log(n_full)',
                                                  min_size_raw_samples=KCPD_MIN_SIZE_RAW, jump=KCPD_JUMP)),
                        train_f1_all=best.user_attrs['mean_f1_all'],
                        train_f1_warming=best.user_attrs['mean_f1_warming'],
                        train_f1_cooling=best.user_attrs['mean_f1_cooling'],
                        train_mean_elapsed_s=best.user_attrs['mean_elapsed_s'],
                        run_metadata=dict(wall_time_s=wall, n_trials_this_run=n_trials_this_run,
                                          n_trials_total=n_trials_total, n_failed_total=n_failed_total,
                                          patience=args.patience, min_trials=args.min_trials,
                                          min_delta=args.min_delta, stopped_early=stop_info['stopped_early'],
                                          stopped_at_trial=stop_info['stopped_at_trial'])), f, indent=2)
    print(f"Saved winning config -> {os.path.join(outdir, 'best_params.json')}")

    phase1 = {key: full[key] for key in ('m', 'tau', 'target_density', 'w_size', 'w_incre')}
    win = full['k']
    for split, evaluate in (('train', search.evaluate_train), ('validation', search.evaluate_validation)):
        cells_df, _ = evaluate(phase1)    # every multiplier, one FI computation per series
        _cell_means(cells_df).to_csv(os.path.join(outdir, f"best_inner_surface_{split}.csv"), index=False)
        raw = cells_df[np.isclose(cells_df['k'], win)]
        raw.to_csv(os.path.join(outdir, f"best_{split}_raw.csv"), index=False)
        if split == 'validation':
            print(f"\nVALIDATION (winning k={win}, block_size={full['block_size']}):")
            print(f"  val_f1_warming={raw['f1_warming'].mean():.3f}  "
                  f"precision_warming={raw['precision_warming'].mean():.3f}  "
                  f"recall_warming={raw['recall_warming'].mean():.3f}")
            print(f"  val_f1_all={raw['f1_all'].mean():.3f}  val_f1_cooling={raw['f1_cooling'].mean():.3f}")
    print("\ntest.npz was never opened this run (evaluate_final_test() is ready for the one-time final check).")


if __name__ == "__main__":
    main()
