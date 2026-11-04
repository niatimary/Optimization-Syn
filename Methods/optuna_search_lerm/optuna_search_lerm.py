#!/usr/bin/env python
"""
Joint Optuna (TPE) search over LERM (Laplacian Eigenmaps of Recurrence Matrices,
James et al. 2024 -- ammonyte.TimeEmbeddedSeries / RecurrenceMatrix /
RQARes.lerm_transitions) on the NOISIER synthetic data. This is the LERM sibling of
optuna_search_kstest.py: same train/validation/test discipline, same direction-
stratified scoring, same resumable sqlite study, same patience-based study-level
early stopping, same output file shapes. Only the thing being tuned differs.

Pipeline per series:  standardize -> TimeEmbeddedSeries(m, tau) -> find_epsilon
(target_density) -> laplacian_eigenmaps(w_size, w_incre) = Fisher Information (FI)
series -> smooth_series(block_size) -> lerm_transitions(95, 5) -> detections.

Five parameters are tuned JOINTLY (nothing is frozen before the others are scored),
but computed in two computational phases so nothing is recomputed needlessly:

  Phase 1 (EXPENSIVE, essentially all of it the eigen-decomposition in
          laplacian_eigenmaps) depends only on (m, tau, target_density, w_size,
          w_incre_frac) -- sampled by TPE:
            m              int    [2, 20]
            tau            int    [2, 48] (log)   embedding lag in samples, ONE value shared
                                                  by every series in a trial (no tau_search)
            target_density float  [0.01, 0.15] (log)   recurrence-matrix density
            w_size         int    [10, 60]             eigenmap window length
            w_incre_frac   float  [0.1, 0.5]           w_incre = max(1, round(frac * w_size)),
                                                       so the stride is always valid for w_size
  Phase 2 (CHEAP) depends on the FI series plus block_size (bounds are fixed at 95/5):
            block_size     {1, 2, 3, 5, 8}
          Every trial sweeps these 5 cells exhaustively on the FI series each worker has
          just computed (held in memory for that one series, dropped afterwards), so a
          trial costs ONE eigendecomposition per series yet tests 5 smoothing settings.

A trial's score: each series is run end-to-end separately (never averaging FI series
across series); the 5 cells are averaged over the train series; the best cell's
mean f1 (--objective: warming (default) or all) is the trial value, and its
block_size is recorded as a trial attribute. The best cell is always picked
from the SERIES-MEAN, never per series. Taking the best of 5 cells makes a trial's
train score slightly optimistic, equally for every trial; validation is the honest
number.

Ranges bracket the defaults of ammonyte's notebooks/LERM_validation.ipynb (m=12,
target_density=0.05, w_size=20, w_incre=4, block_size=2, bounds 95/5). tau is no longer
computed per series (tau_search): a single shared value is tuned, since an auto tau varied
3-18 samples across series and made m and the embedding window (m-1)*tau hard to separate. If the best
trials pile up at a bound, widen it (see the edge-case procedure in the project notes)
and start a new --study-name; the search-space fingerprint blocks resuming otherwise.

Fixed (not tuned): bounds 95/5, epsilon (derived by find_epsilon), the
find_epsilon start eps=1 (as in LERM_validation.ipynb), density tolerance
0.002, amp 10, bootstrap size w=50 and n_samples=10000 (lerm_transitions defaults).

Direction: warming/cooling labels come from LERM's own jump_values (as in the
LERM_validation notebook), mapped warming <=> -1 because ammonyte's sign convention
assumes an ascending age axis and this data runs forward in time -- see _lerm_directions.

Failures: any series failing makes the whole trial FAIL (never scored as 0). A series
with zero detections is a legitimate f1=0, not a failure. --max-call-seconds is a very
generous per-series safety net against a hang (e.g. find_epsilon never converging),
NOT a cost constraint -- slow configs (small w_incre, large m, large tau) must stay searchable.

Data: train_test_split_data/{train,validation,test}.npz = 60/30/30
series. Test is never opened here; evaluate_final_test() is ready for a one-time
final evaluation.

Outputs under --outdir: optuna_trials.csv, best_params.json, best_train_raw.csv /
best_validation_raw.csv (per-series, winning cell), best_inner_surface_train.csv /
best_inner_surface_validation.csv (mean metrics for all 5 cells of the winning
phase-1 config), optuna_convergence.csv/.png.

Usage:
    python optuna_search_lerm.py --n-trials 250 --min-trials 100 --patience 50 --n-jobs 24
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
DEFAULT_MAX_CALL_SECONDS = 7200   # safety net against a hang only -- normal calls take 20-90 s at the old ranges
SCORING_VERSION = "2"      # bump when scoring/detection/search space meaningfully change

_SPLIT_DIR = os.path.join(SYNTHETICDATA_DIR, "train_test_split_data")
DEFAULT_TRAIN_PATH = os.path.normpath(os.path.join(_SPLIT_DIR, "train.npz"))
DEFAULT_VAL_PATH = os.path.normpath(os.path.join(_SPLIT_DIR, "validation.npz"))
DEFAULT_TEST_PATH = os.path.normpath(os.path.join(_SPLIT_DIR, "test.npz"))

# Phase 1 (sampled by TPE)
M_BOUNDS = (2, 20)
TAU_BOUNDS = (2, 48)                 # samples, log scale; one value shared by every series
TARGET_DENSITY_BOUNDS = (0.01, 0.15)
W_SIZE_BOUNDS = (10, 60)
W_INCRE_FRAC_BOUNDS = (0.1, 0.5)     # w_incre = max(1, round(frac * w_size))
# Phase 2 (swept exhaustively inside every trial)
BLOCK_SIZE_GRID = (1, 2, 3, 5, 8)
BOUNDS_GRID = ((95, 5),)    # fixed (upper, lower) percentiles, not tuned
CELLS = tuple((bs, up, lo) for bs in BLOCK_SIZE_GRID for up, lo in BOUNDS_GRID)
# Fixed settings
EPS = 1                     # find_epsilon starting value (only a start; the search converges to target_density)
DENSITY_TOLERANCE = 0.002
AMP = 10
BOOT_W = 50
BOOT_N_SAMPLES = 10000

CELL_COLS = ['block_size', 'upper', 'lower']


def _search_space_fingerprint():
    """Hash of everything that defines what a trial can be: every bound, both inner
    grids, and the fixed settings -- used to refuse resuming a study whose space changed."""
    payload = json.dumps(dict(m=M_BOUNDS, tau=TAU_BOUNDS, td=TARGET_DENSITY_BOUNDS, w_size=W_SIZE_BOUNDS,
                               w_incre_frac=W_INCRE_FRAC_BOUNDS, blocks=BLOCK_SIZE_GRID,
                               bounds=BOUNDS_GRID, eps=EPS,
                               density_tol=DENSITY_TOLERANCE, amp=AMP, boot_w=BOOT_W,
                               boot_n=BOOT_N_SAMPLES), sort_keys=True)
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
    _lerm_directions). Also returns the matched timing errors (see _matched_timing_errors)
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


def _lerm_directions(transitions):
    """Detected times and warming/cooling labels from LERM's OWN jump_values (as the
    LERM_validation notebook does), NaN times dropped together with their values.
    ammonyte flips the sign of jump_values whenever the time axis ascends, which is right
    for NGRIP's age axis (kyr b2k, where +1 = DO warming) but inverted for this dataset's
    forward simulation time. Checked against ground truth (3 train series, notebook
    default config, 35 matched detections): -1 matched a warming event 15/18 times and
    +1 a cooling event 15/17 times -- so here warming <=> jump_values == -1."""
    times = np.asarray(transitions.jump_times, dtype=float)
    values = np.asarray(transitions.jump_values)
    keep = ~np.isnan(times)
    return times[keep], np.where(values[keep] == -1, 'warming', 'cooling')


def load_split(npz_path):
    d = np.load(npz_path)
    time_arr = d['time']
    db_noisy = d['db_noisy']
    state_true = d['state_true']
    n = db_noisy.shape[0]

    # Standardize each series independently (zero mean, unit std, using that
    # series' OWN mean/std) -- kept for consistency with optuna_search.py's
    # load_split (byte-for-byte the same standardization call), NOT because KS
    # test's own correctness depends on it -- see module docstring's
    # 'Standardization' section for the invariance argument (d_c/s_c/auto-x_c are
    # all provably unaffected by any per-series affine transform).
    db_noisy, _, _ = pyleo_standardize(db_noisy, axis=1)

    gt_times, gt_directions = [], []
    for i in range(n):
        t, dirs = transitions_from_states(time_arr, state_true[i])
        gt_times.append(t)
        gt_directions.append(dirs)
    return time_arr, db_noisy, gt_times, gt_directions, n

def _fisher_series(time_arr, value, m, tau, target_density, w_size, w_incre):
    """Phase 1: embed -> find_epsilon -> recurrence matrix -> Laplacian eigenmaps."""
    series = amt.Series(time=time_arr, value=value, label='optuna', verbose=False)
    embedded = amt.TimeEmbeddedSeries(series, m=m, tau=tau)
    eps_result = embedded.find_epsilon(eps=EPS,
                                        target_density=target_density, tolerance=DENSITY_TOLERANCE,
                                        amp=AMP, parallelize=False, verbose=False)
    return eps_result['Output'].laplacian_eigenmaps(w_size=w_size, w_incre=w_incre)


def _score_one(time_arr, value, gt_times, gt_directions, m, tau, target_density, w_size, w_incre,
               cells, tolerance, max_call_seconds):
    """One series, end to end: phase 1 once, then phase 2 for every (block_size, upper,
    lower) in `cells` on that same FI series. Returns dict(cells=[row per cell], error,
    elapsed_s, cpu_time_s, peak_rss_gb). peak_rss_gb is the whole worker process's
    high-water mark, not this call alone."""
    ru_before = resource.getrusage(resource.RUSAGE_SELF)
    t0 = time.perf_counter()
    rows, error = [], None
    old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(max_call_seconds)
    try:
        fi = _fisher_series(time_arr, value, m, tau, target_density, w_size, w_incre)
        for block_size in sorted({c[0] for c in cells}):
            # smooth_series modifies its input in place -- copy so cells don't contaminate each other
            smoothed = amt.utils.fisher.smooth_series(copy.deepcopy(fi), block_size=block_size)
            for bs, upper, lower in cells:
                if bs != block_size:
                    continue
                transitions = smoothed.lerm_transitions(upper=upper, lower=lower, w=BOOT_W,
                                                         n_samples=BOOT_N_SAMPLES)
                detected, det_directions = _lerm_directions(transitions)
                metrics, timing_errors = evaluate_detection_by_direction(
                    detected, det_directions, gt_times, gt_directions, tolerance)
                row = dict(block_size=bs, upper=upper, lower=lower, n_detected=int(len(detected)),
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
    """Mean over series of every metric, per (block_size, upper, lower) cell."""
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
        trial.set_user_attr('best_block_size', int(best['block_size']))
        trial.set_user_attr('best_upper', float(best['upper']))
        trial.set_user_attr('best_lower', float(best['lower']))
        trial.set_user_attr('mean_elapsed_s', float(info_df['elapsed_s'].mean()))
        trial.set_user_attr('mean_cpu_time_s', float(info_df['cpu_time_s'].mean()))
        trial.set_user_attr('max_peak_rss_gb', float(info_df['peak_rss_gb'].max()))
        trial.set_user_attr('mean_n_detected', float(best['n_detected']))
        for label in ('all', 'warming', 'cooling'):
            for stat in ('precision', 'recall', 'f1', 'timing_mae'):
                trial.set_user_attr(f'mean_{stat}_{label}', float(best[f'{stat}_{label}']))
        # the whole inner surface, for later sensitivity analysis (no NaN-prone timing columns)
        keep = CELL_COLS + [f'{s}_{l}' for l in ('all', 'warming', 'cooling') for s in ('precision', 'recall', 'f1')]
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
    block_size/upper/lower). Call once, after everything is locked in via validation."""
    _, db_test, gt_times_test, gt_dirs_test, n_test = load_split(test_path)
    cell = ((params['block_size'], params['upper'], params['lower']),)
    scored = Parallel(n_jobs=n_jobs)(
        delayed(_score_one)(time_arr, db_test[s], gt_times_test[s], gt_dirs_test[s], params['m'],
                             params['tau'], params['target_density'], params['w_size'], params['w_incre'],
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
    """Reconstruct all tuned parameters (w_incre derived from w_incre_frac) from the best trial."""
    p, a = best_trial.params, best_trial.user_attrs
    return dict(m=p['m'], tau=p['tau'], target_density=p['target_density'], w_size=p['w_size'],
                w_incre=a['w_incre'], w_incre_frac=p['w_incre_frac'],
                block_size=a['best_block_size'], upper=a['best_upper'], lower=a['best_lower'])



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
    study_name = args.study_name or f"optuna_lerm_search_{args.objective}"

    search = OptunaSearch(args.train_path, args.val_path, TOLERANCE, args.n_jobs,
                           objective_metric=args.objective, max_call_seconds=args.max_call_seconds)
    print(f"train seeds: {search.n_train}   validation seeds: {search.n_val}", flush=True)
    print(f"inner grid: {len(CELLS)} cells ({len(BLOCK_SIZE_GRID)} block sizes x {len(BOUNDS_GRID)} fixed bounds)", flush=True)
    print(f"max_call_seconds={args.max_call_seconds}s (safety net)   n_jobs={args.n_jobs}", flush=True)

    # Flat 5-parameter TPE space (w_incre is derived from w_size via w_incre_frac, so no conditional bound)
    sampler = TPESampler(seed=RANDOM_SEED, n_startup_trials=20)
    study = optuna.create_study(direction="maximize", sampler=sampler, storage=storage,
                                 study_name=study_name, load_if_exists=True)
    n_trials_before = len(study.trials)
    print(f"study {study_name!r} @ {storage!r}: {n_trials_before} trial(s) already recorded", flush=True)

    study_config = dict(n_train=int(search.n_train), random_seed=RANDOM_SEED,
                        train_data_hash=_hash_file(args.train_path), objective=args.objective,
                        tolerance=TOLERANCE, search_space_hash=_search_space_fingerprint(),
                        scoring_version=SCORING_VERSION)
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
                                   density_tolerance=DENSITY_TOLERANCE, amp=AMP, w=BOOT_W,
                                   n_samples=BOOT_N_SAMPLES),
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

    phase1 = {k: full[k] for k in ('m', 'tau', 'target_density', 'w_size', 'w_incre')}
    win = (full['block_size'], full['upper'], full['lower'])
    for split, evaluate in (('train', search.evaluate_train), ('validation', search.evaluate_validation)):
        cells_df, _ = evaluate(phase1)    # all 5 cells, one FI computation per series
        _cell_means(cells_df).to_csv(os.path.join(outdir, f"best_inner_surface_{split}.csv"), index=False)
        raw = cells_df[(cells_df['block_size'] == win[0]) & (cells_df['upper'] == win[1])
                       & (cells_df['lower'] == win[2])]
        raw.to_csv(os.path.join(outdir, f"best_{split}_raw.csv"), index=False)
        if split == 'validation':
            print(f"\nVALIDATION (winning cell {win}):")
            print(f"  val_f1_warming={raw['f1_warming'].mean():.3f}  "
                  f"precision_warming={raw['precision_warming'].mean():.3f}  "
                  f"recall_warming={raw['recall_warming'].mean():.3f}")
            print(f"  val_f1_all={raw['f1_all'].mean():.3f}  val_f1_cooling={raw['f1_cooling'].mean():.3f}")
    print("\ntest.npz was never opened this run (evaluate_final_test() is ready for the one-time final check).")


if __name__ == "__main__":
    main()
