#!/usr/bin/env python
"""
Joint Optuna (TPE) search over the augmented KS test's own parameters
(w_min, w_max, n_w, n_c, d_c, s_c), pruning clearly-losing trials mid-evaluation
via Optuna's WilcoxonPruner. This is the KS-test sibling of
optuna_search_pruning.py (the ruptures analog -- see that file's module
docstring for the fuller design rationale reused verbatim below), and the
pruning counterpart of optuna_search_kstest.py (patience-based study-level
early stopping, stays as-is, untouched). Run this as a completely separate
study (its own --outdir/--study-name, defaulted below) so it never touches
optuna_search_kstest.py's results.

KS test has no categorical branching the way ruptures' method x cost did --
every trial samples all 6 parameters unconditionally, so there is no grouped/
conditional search space to reconstruct here (contrast
optuna_search_pruning.py's _sample_params/_best_full_params, which are
method/cost-namespaced).

Search space -- identical to optuna_search_kstest.py's (see that file's module
docstring for the full rationale behind each bound, reproduced only in brief
here):

  n_c   in [3, 15]     (int)   minimum sample size required in EACH compared
                                window. KS_test's std calculation needs ddof=1,
                                i.e. >=2 points -- 2 remains the hard
                                mathematical floor, but the search's own lower
                                bound is set to 3 here.
  w_min in [max(0.05, n_c*dt), 0.4] (float, log)  smallest window width; lower
                                bound depends on the sampled n_c and this
                                split's own dt so a window too short to ever
                                clear the n_c gate is never sampled.
  w_max in [w_min*1.5, 4.0] (float, log)  largest window width; lower bound
                                enforces w_max > w_min with real margin.
  n_w   in [5, 25]     (int)   number of window widths tested -- primarily a
                                coverage-vs-cost dial (cost scales close to
                                linearly with n_w).
  d_c   in [0.5, 0.95] (float) KS-statistic cutoff, the main sensitivity axis.
  s_c   in [0.5, 3.0]  (float) effect-size gate (mean difference >= s_c times
                                each window's own std).

x_c is NEVER tuned -- always x_c=None, auto-computed fresh per series by
KS_test itself. See optuna_search_kstest.py's module docstring for why (per-
seed adaptivity would otherwise be broken by a fixed gridded value).

Standardization: load_split below standardizes every series independently
(zero mean, unit std, from its OWN stats), matching both
optuna_search_kstest.py's and the ruptures scripts' load_split exactly, for
pipeline consistency -- NOT because KS test's own correctness depends on it.
See optuna_search_kstest.py's module docstring for the full invariance
argument (KS D-statistic, s_c's ratio, and the auto x_c default are all
provably unaffected by any per-series affine standardization).

Sampler: TPESampler(seed=RANDOM_SEED, n_startup_trials=30) -- deliberately
WITHOUT multivariate=True, group=True, for the same reason as
optuna_search_kstest.py: those flags exist in the ruptures scripts to handle a
method -> cost -> pen grouped/conditional search space that doesn't exist
here (KS test's space is flat, every trial samples every dimension).
n_startup_trials=30 (vs. the patience script's 10) matches
optuna_search_pruning.py's own asymmetry with optuna_search.py -- a pruning-
enabled study benefits from a longer unpruned random-exploration phase before
TPE starts modeling the space, since early pruning decisions are only as good
as the trials they're compared against.

Per-call cost is dramatically higher than ruptures': measured 12-22s per
KS_test call on this project's actual train.npz (60 series, 6001 points each),
versus ruptures' ~0.02-0.03s -- roughly 500-1000x more expensive per call.
This makes WilcoxonPruner's per-trial savings matter proportionally MORE here
than they did for ruptures: killing a clearly-bad trial after 1-2 of its
--pruning-batch-size batches instead of paying for the whole train pool saves
real, substantial wall-clock time, not a rounding error on an already-cheap
call. It also raises the stakes on --max-call-seconds (CallTimeoutError) --
see optuna_search_kstest.py's module docstring.

Pruning mechanics (WilcoxonPruner instead of study-level patience) -- reused
unchanged from optuna_search_pruning.py's design, only the thing being scored
per-series differs:
  `OptunaSearch._evaluate_staged` scores train in batches of `n_jobs` series
  at a time (full within-batch parallelism), reports each series' individual
  objective-metric score to Optuna after its batch finishes, and checks
  `trial.should_prune()` once per batch. `optuna.pruners.WilcoxonPruner`
  performs a paired Wilcoxon signed-rank test between the current trial's
  per-series scores so far and the best completed trial's scores on those
  SAME series indices (pairing cancels out per-series noise/difficulty) --
  once evidence crosses --p-threshold, the trial is pruned (raises
  TrialPruned) and its later batches are skipped entirely.

  Two things this depends on (identical to optuna_search_pruning.py):
    - Trials are evaluated sequentially in this script (study.optimize is
      called without its own n_jobs), so time saved pruning one trial is
      immediately available to start the next trial sooner.
    - Step i must map to the SAME train series for every trial and every
      resumed invocation -- self.train_eval_order (a RANDOM_SEED-seeded
      permutation of train's indices, never re-shuffled per trial) guarantees
      this, same as the ruptures pruning script.

  `evaluate_validation`/`evaluate_train` are untouched -- full, unbatched,
  no pruning, exactly as in optuna_search_kstest.py. Pruning only ever
  applies to trials still being compared against each other during the live
  search.

  --p-threshold controls how much evidence WilcoxonPruner needs before
  pruning (default 0.05, matching optuna_search_pruning.py's own deliberately-
  conservative-vs-Optuna's-0.1-default choice, for the same reason: this
  dataset's series vary in noise level/color and transition character, so a
  stricter threshold requires stronger evidence before pruning a trial that
  might just have hit a hard subset).

Everything else (direction-stratified scoring, why TEST is scored on more
than f1, train/validation/test data discipline) is unchanged from
optuna_search_kstest.py / the ruptures scripts -- see those docstrings for
the full reasoning, not repeated here.

Usage:
    python optuna_search_kstest_pruning.py --n-trials 300 --n-jobs 8
    python optuna_search_kstest_pruning.py --n-trials 300 --p-threshold 0.05 --n-jobs 24
    # interrupted (SLURM time limit, OOM, ctrl-c)? just rerun the same command -- it resumes
    # from the persistent study at --storage and only runs enough MORE trials to reach the
    # --n-trials cap (a TOTAL across every resumed invocation, not additional trials each time)
"""
import argparse
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
from optuna.pruners import WilcoxonPruner
from optuna.exceptions import TrialPruned
from joblib import Parallel, delayed

import ammonyte as amt
from pyleoclim.utils.tsutils import standardize as pyleo_standardize
from ammonyte.utils.metrics import evaluate_detection

HERE = os.path.dirname(os.path.abspath(__file__))
METHODS_DIR = os.path.dirname(HERE)                # this script lives one level below Methods/
SYNTHETICDATA_DIR = os.path.dirname(METHODS_DIR)   # train_test_split_data/ is a Methods/ sibling, not a child


class SeriesEvaluationError(Exception):
    """Raised by OptunaSearch._evaluate_staged when any series (or the whole batch
    Parallel call) fails during the live search -- caught by name in main()'s
    study.optimize(..., catch=(SeriesEvaluationError,)) so Optuna marks that ONE
    trial as FAIL and moves on to the next trial. See _evaluate_staged's docstring."""


class CallTimeoutError(Exception):
    """Raised when a single _detect call exceeds --max-call-seconds (see _detect's
    signal.alarm wrapper). Caught by _score_one's existing except-Exception block
    exactly like any other detection failure, becoming that series' 'error' entry,
    which _evaluate_staged then escalates to SeriesEvaluationError -> trial FAIL.
    Matters more here than for ruptures -- KS_test calls are ~500-1000x more
    expensive per call (measured 12-22s vs. ruptures' ~0.02-0.03s).

    Relies on SIGALRM, so Unix-only, and only fires correctly because _detect
    always runs as the sole thread of a single-task joblib (loky) worker process
    here."""


def _alarm_handler(signum, frame):
    raise CallTimeoutError("_detect exceeded --max-call-seconds")


TOLERANCE = 0.5           # +/- 0.5 kyr matching tolerance, same as ruptures/prior grid search
RANDOM_SEED = 42
DEFAULT_MAX_CALL_SECONDS = 7200   # 2h hard cap on a SINGLE series' _detect call (see
                                   # CallTimeoutError). Measured KS_test calls are 12-22s on
                                   # this project's data, so this is generous headroom.
SCORING_VERSION = "2"     # bump this any time _score_one/_detect/evaluate_detection_by_direction
                           # or the search-space bounds below meaningfully change -- included in
                           # the resume-compatibility check (see main()).
DIRECTION_MIN_WINDOW = 10  # samples each side of a detected changepoint used to classify
                            # it warming/cooling; FIXED for every config (was max(this, n_c*2)).

DEFAULT_TRAIN_PATH = os.path.normpath(os.path.join(SYNTHETICDATA_DIR, "train_test_split_data", "train.npz"))
DEFAULT_VAL_PATH = os.path.normpath(os.path.join(SYNTHETICDATA_DIR, "train_test_split_data", "validation.npz"))
DEFAULT_TEST_PATH = os.path.normpath(os.path.join(SYNTHETICDATA_DIR, "train_test_split_data", "test.npz"))

# Search-space bounds -- identical to optuna_search_kstest.py, see that file's module
# docstring for the rationale behind each. w_min/w_max's lower bounds are computed at
# runtime (depend on n_c and this split's own dt), not listed here as fixed numbers.
N_C_BOUNDS = (3, 15)
W_MIN_UPPER = 0.4
W_MAX_RATIO_MIN = 1.5   # w_max's lower bound is sampled_w_min * this ratio
W_MAX_UPPER = 4.0
N_W_BOUNDS = (5, 25)
D_C_BOUNDS = (0.5, 0.95)
S_C_BOUNDS = (0.5, 3.0)


def _search_space_fingerprint():
    """Short hash of everything that defines what a trial's params dict can even be
    -- used by main() to detect a resumed study whose search space no longer
    matches what its already-completed trials were sampled from."""
    payload = json.dumps(dict(n_c_bounds=N_C_BOUNDS, w_min_upper=W_MIN_UPPER,
                               w_max_ratio_min=W_MAX_RATIO_MIN, w_max_upper=W_MAX_UPPER,
                               n_w_bounds=N_W_BOUNDS, d_c_bounds=D_C_BOUNDS, s_c_bounds=S_C_BOUNDS),
                          sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _hash_file(path):
    """Short content hash of an entire file's bytes -- used to detect train.npz being
    swapped for a different file with the same series COUNT between resumed
    invocations of the same study. Hashes the whole file, not just one array."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def transitions_from_states(time_arr, states):
    """True transition times AND directions from a binary stadial(1)/interstadial(0)
    state-label array. A 1->0 transition is warming; 0->1 is cooling."""
    diffs = np.diff(states)
    change_idx = np.where(diffs != 0)[0]
    times = time_arr[change_idx + 1]
    directions = np.array(['warming' if d == -1 else 'cooling' for d in diffs[change_idx]])
    return times, directions


def _clean_times(times):
    times = np.asarray(times, dtype=float)
    return times[~np.isnan(times)]


def _direction_label(delta):
    if delta > 0:
        return 'warming'
    elif delta < 0:
        return 'cooling'
    return 'ambiguous'   # exact tie -- effectively never happens with continuous noisy data


def _classify_detected_directions(value, time_arr, detected_times, min_size):
    """Classify each detected transition as 'warming' or 'cooling' by comparing the
    local mean value just before vs. just after it, in plain forward array/time
    order. Method-agnostic (operates only on the detected times/value/time array).
    `min_size` is unused (the window is fixed); kept in the signature for callers."""
    detected_times = np.asarray(detected_times, dtype=float)
    if len(detected_times) == 0:
        return np.array([], dtype=object)

    window = DIRECTION_MIN_WINDOW   # fixed for every config; min_size is no longer used here (kept in the signature for callers)
    order = np.argsort(detected_times)
    idxs = np.array([int(np.argmin(np.abs(time_arr - t))) for t in detected_times[order]])

    directions = np.empty(len(idxs), dtype=object)
    for k, idx in enumerate(idxs):
        before_start = max(0, idx - window)
        after_end = min(len(value), idx + window)
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


def _matched_timing_errors(detected, ground_truth, tolerance):
    """Same one-to-one closest-match rule as ammonyte's evaluate_detection, but
    returns the matched |detected - ground_truth| distances themselves."""
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


def evaluate_detection_by_direction(time_arr, value, detected, gt_times, gt_directions, tolerance, min_size):
    """f1_all plus f1_warming/f1_cooling: detections are classified warming/cooling
    first, then BOTH detections and ground truth are filtered to one direction
    before calling the existing, unmodified evaluate_detection on that subset."""
    det_directions = _classify_detected_directions(value, time_arr, detected, min_size)
    detected = np.asarray(detected, dtype=float)
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


def load_split(npz_path):
    d = np.load(npz_path)
    time_arr = d['time']
    db_noisy = d['db_noisy']
    state_true = d['state_true']
    n = db_noisy.shape[0]

    # Standardize each series independently -- kept for pipeline consistency, NOT
    # because KS test's own correctness depends on it -- see module docstring's
    # 'Standardization' section (and optuna_search_kstest.py's, the full argument).
    db_noisy, _, _ = pyleo_standardize(db_noisy, axis=1)

    gt_times, gt_directions = [], []
    for i in range(n):
        t, dirs = transitions_from_states(time_arr, state_true[i])
        gt_times.append(t)
        gt_directions.append(dirs)
    return time_arr, db_noisy, gt_times, gt_directions, n


def _detect(time_arr, value, w_min, w_max, n_w, d_c, n_c, s_c, max_call_seconds):
    series = amt.Series(time=time_arr, value=value, label='optuna', verbose=False)
    t0 = time.perf_counter()

    old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(max_call_seconds)
    try:
        # x_c always None -- see module docstring, never tuned.
        transitions = series.kstest(w_min=w_min, w_max=w_max, n_w=n_w, d_c=d_c, n_c=n_c, s_c=s_c, x_c=None)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

    elapsed_s = time.perf_counter() - t0
    return _clean_times(transitions.jump_times), elapsed_s


def _score_one(time_arr, value, gt_times, gt_directions, w_min, w_max, n_w, d_c, n_c, s_c, tolerance,
                max_call_seconds):
    ru_before = resource.getrusage(resource.RUSAGE_SELF)
    try:
        detected, elapsed_s = _detect(time_arr, value, w_min, w_max, n_w, d_c, n_c, s_c, max_call_seconds)
        metrics, timing_errors = evaluate_detection_by_direction(time_arr, value, detected, gt_times,
                                                                   gt_directions, tolerance, n_c)
    except Exception as e:
        ru_after = resource.getrusage(resource.RUSAGE_SELF)
        cpu_time_s = (ru_after.ru_utime - ru_before.ru_utime) + (ru_after.ru_stime - ru_before.ru_stime)
        peak_rss_gb = ru_after.ru_maxrss / (1024 ** 2 if sys.platform != 'darwin' else 1024 ** 3)
        out = dict(false_positives_all=np.nan, error=f"{type(e).__name__}: {e}"[:300], elapsed_s=np.nan,
                    cpu_time_s=cpu_time_s, peak_rss_gb=peak_rss_gb)
        for label in ('all', 'warming', 'cooling'):
            out[f'precision_{label}'] = 0.0
            out[f'recall_{label}'] = 0.0
            out[f'f1_{label}'] = 0.0
            out[f'timing_mae_{label}'] = np.nan
            out[f'timing_n_matched_{label}'] = 0
        return out

    ru_after = resource.getrusage(resource.RUSAGE_SELF)
    cpu_time_s = (ru_after.ru_utime - ru_before.ru_utime) + (ru_after.ru_stime - ru_before.ru_stime)
    peak_rss_gb = ru_after.ru_maxrss / (1024 ** 2 if sys.platform != 'darwin' else 1024 ** 3)
    out = dict(false_positives_all=metrics['all'].false_positives, error=None, elapsed_s=elapsed_s,
               cpu_time_s=cpu_time_s, peak_rss_gb=peak_rss_gb)
    for label, m in metrics.items():
        out[f'precision_{label}'] = m.precision
        out[f'recall_{label}'] = m.recall
        out[f'f1_{label}'] = m.f1_score
        errs = timing_errors[label]
        out[f'timing_mae_{label}'] = float(np.mean(errs)) if len(errs) else np.nan
        out[f'timing_n_matched_{label}'] = int(len(errs))
    return out


def _resolve_n_jobs(n_jobs):
    """joblib's own -1/-2/... convention (-1 = all cores, -2 = all but one, ...),
    resolved to a concrete worker count -- needed here (unlike a plain Parallel
    call) because _evaluate_staged uses it as an actual batch size."""
    if n_jobs is None or n_jobs == 0:
        return 1
    if n_jobs < 0:
        cpu = os.cpu_count() or 1
        return max(1, cpu + 1 + n_jobs)
    return n_jobs


class OptunaSearch:
    """Bundles the pooled train/validation data and evaluation logic. Three
    disjoint roles, never blurred:

      - train:      what `objective` scores every single trial against (via
                     `_evaluate_staged`, batched + prunable). This is the only
                     thing Optuna's sampler ever sees.
      - validation: what `evaluate_validation` scores a candidate winning config
                     against, via the original, unbatched `_evaluate`.
      - test:       deliberately NOT loaded by this class at all.
    """

    def __init__(self, train_path, val_path, tolerance, n_jobs, objective_metric='warming',
                 pruning_batch_size=None, max_call_seconds=DEFAULT_MAX_CALL_SECONDS):
        (self.time_arr, self.db_train, self.gt_times_train, self.gt_dirs_train,
         self.n_train) = load_split(train_path)
        val_time_arr, self.db_val, self.gt_times_val, self.gt_dirs_val, self.n_val = load_split(val_path)
        if not np.array_equal(self.time_arr, val_time_arr):
            raise ValueError(
                f"train.npz's time array does not match val.npz's time array -- "
                f"evaluate_validation would silently score validation series using train's "
                f"time axis (see OptunaSearch.__init__). Regenerate validation.npz on the "
                f"same time grid as train.npz, or this class needs to be extended to carry "
                f"a separate time array per split."
            )
        self.tolerance = tolerance
        self.n_jobs = n_jobs
        # Batch size for _evaluate_staged's pruning checkpoints -- decoupled from n_jobs so
        # --pruning-batch-size can require more evidence per should_prune() check without
        # changing parallelism. Defaults to n_jobs (one checkpoint per full core-wave).
        self.batch_size = pruning_batch_size if pruning_batch_size is not None else _resolve_n_jobs(n_jobs)
        self.max_call_seconds = max_call_seconds
        self.objective_metric = objective_metric
        # Fixed (seeded, not re-shuffled per trial) permutation of train indices -- step i
        # must map to the SAME series every trial/resume for WilcoxonPruner's paired
        # comparison to be valid. Shuffled (not raw index order) so an early prune decision
        # isn't biased by however train.npz's series happen to be grouped/stored.
        self.train_eval_order = np.random.RandomState(RANDOM_SEED).permutation(self.n_train)
        # This split's own median sample spacing -- used to lower-bound w_min against n_c
        # in _sample_params. Computed once here, not per-trial.
        self.dt = float(np.median(np.diff(self.time_arr)))

    def _sample_params(self, trial):
        # No categorical branching -- every trial samples all 6 parameters
        # unconditionally. n_c sampled first so its value can lower-bound w_min;
        # w_min sampled next so ITS value can lower-bound w_max -- see module
        # docstring for why each bound is what it is.
        n_c = trial.suggest_int('n_c', *N_C_BOUNDS)
        w_min = trial.suggest_float('w_min', max(0.05, n_c * self.dt), W_MIN_UPPER, log=True)
        w_max = trial.suggest_float('w_max', w_min * W_MAX_RATIO_MIN, W_MAX_UPPER, log=True)
        n_w = trial.suggest_int('n_w', *N_W_BOUNDS)
        d_c = trial.suggest_float('d_c', *D_C_BOUNDS)
        s_c = trial.suggest_float('s_c', *S_C_BOUNDS)
        return dict(w_min=w_min, w_max=w_max, n_w=n_w, d_c=d_c, n_c=n_c, s_c=s_c)

    def _evaluate(self, params, db, gt_times_list, gt_dirs_list, seed_idx):
        """Unbatched full evaluation -- used by evaluate_validation/evaluate_train.
        No pruning applies here."""
        try:
            scored = Parallel(n_jobs=self.n_jobs)(
                delayed(_score_one)(self.time_arr, db[s], gt_times_list[s], gt_dirs_list[s],
                                     params['w_min'], params['w_max'], params['n_w'], params['d_c'],
                                     params['n_c'], params['s_c'], self.tolerance, self.max_call_seconds)
                for s in seed_idx
            )
        except Exception as e:   # any worker crash
            return None, f"{type(e).__name__}: {e}"[:300]
        return pd.DataFrame(scored), None

    def _evaluate_staged(self, trial, params):
        """Used only by `objective` (the live search). Scores train in batches of
        self.batch_size series at a time, walking self.train_eval_order (not raw
        index order), reports each series' individual objective-metric score to
        Optuna right after its batch finishes, and checks trial.should_prune()
        once per batch. Lets WilcoxonPruner (wired in main()) kill a trial that's
        already clearly worse than the current best after 1-2 batches instead of
        paying for all n_train series -- full batch_size-way parallelism is
        preserved within each batch; only the between-batch wait is new versus
        optuna_search_kstest.py's single all-at-once Parallel call.

        ANY series failing raises SeriesEvaluationError immediately -- same
        stricter-than-skip-and-continue reasoning as optuna_search_pruning.py's
        version of this method (see that file's docstring): the search space
        here is constructed so every sampled combo is a priori valid, so a
        failure is overwhelmingly more likely a real bug/resource issue than an
        "expected" invalid config, and a fabricated f1=0.0 for a crash could
        wrongly influence a pruning decision."""
        order = self.train_eval_order
        all_scored = []

        for batch_start in range(0, len(order), self.batch_size):
            batch_idx = order[batch_start: batch_start + self.batch_size]
            try:
                batch_scored = Parallel(n_jobs=self.n_jobs)(
                    delayed(_score_one)(self.time_arr, self.db_train[s], self.gt_times_train[s],
                                         self.gt_dirs_train[s], params['w_min'], params['w_max'],
                                         params['n_w'], params['d_c'], params['n_c'], params['s_c'],
                                         self.tolerance, self.max_call_seconds)
                    for s in batch_idx
                )
            except Exception as e:   # any worker crash
                msg = f"{type(e).__name__}: {e}"[:300]
                trial.set_user_attr('error', msg)
                raise SeriesEvaluationError(f"batch starting at train_eval_order position "
                                             f"{batch_start} crashed: {msg}") from e

            errored = [result for result in batch_scored if result['error'] is not None]
            if errored:
                trial.set_user_attr('error', errored[0]['error'])
                trial.set_user_attr('n_errors_in_failing_batch', len(errored))
                raise SeriesEvaluationError(
                    f"{len(errored)}/{len(batch_scored)} series in batch starting at "
                    f"train_eval_order position {batch_start} failed to score; "
                    f"first error: {errored[0]['error']}"
                )

            all_scored.extend(batch_scored)
            for local_i, result in enumerate(batch_scored):
                # step = position in the FIXED, seeded train_eval_order permutation --
                # e.g. step=37 always means "the 38th series in train_eval_order", the
                # exact same physical series for every trial. main() saves
                # train_eval_order to <outdir>/train_eval_order.csv if you need to map
                # a step number back to its raw series index.
                trial.report(result[f'f1_{self.objective_metric}'], step=batch_start + local_i)

            if trial.should_prune():
                raise TrialPruned()

        return pd.DataFrame(all_scored)

    def objective(self, trial):
        params = self._sample_params(trial)
        df = self._evaluate_staged(trial, params)   # raises SeriesEvaluationError/TrialPruned on
                                                      # failure/pruning -- both caught by main()'s
                                                      # study.optimize(catch=...)

        trial.set_user_attr('mean_elapsed_s', float(df['elapsed_s'].mean()))
        trial.set_user_attr('mean_cpu_time_s', float(df['cpu_time_s'].mean()))
        trial.set_user_attr('max_peak_rss_gb', float(df['peak_rss_gb'].max()))
        for label in ('all', 'warming', 'cooling'):
            trial.set_user_attr(f'mean_precision_{label}', float(df[f'precision_{label}'].mean()))
            trial.set_user_attr(f'mean_recall_{label}', float(df[f'recall_{label}'].mean()))
            trial.set_user_attr(f'mean_f1_{label}', float(df[f'f1_{label}'].mean()))
            timing = df[f'timing_mae_{label}'].dropna()
            trial.set_user_attr(f'mean_timing_mae_{label}', float(timing.mean()) if len(timing) else float('nan'))
        return float(df[f'f1_{self.objective_metric}'].mean())

    def evaluate_validation(self, params):
        seed_idx = np.arange(self.n_val)
        df, err = self._evaluate(params, self.db_val, self.gt_times_val, self.gt_dirs_val, seed_idx)
        if df is None:
            raise RuntimeError(f"validation evaluation failed: {err}")
        return df

    def evaluate_train(self, params):
        """Per-seed raw scores for one config on train (full, unbatched) -- used
        once, after the study finishes, to save the winning config's full
        per-seed train breakdown."""
        seed_idx = np.arange(self.n_train)
        df, err = self._evaluate(params, self.db_train, self.gt_times_train, self.gt_dirs_train, seed_idx)
        if df is None:
            raise RuntimeError(f"train evaluation failed: {err}")
        return df


def evaluate_final_test(test_path, tolerance, n_jobs, params, max_call_seconds=DEFAULT_MAX_CALL_SECONDS):
    """Loads test data fresh (never held on OptunaSearch) and scores exactly one
    config against it. Call this once, after every design decision is finalized
    via validation. Uses test.npz's OWN time array (loaded fresh here), matching
    optuna_search_pruning.py's version of this function.

    Not yet wired into evaluate_test.py (that file currently only knows the
    ruptures patience/pruning scripts' --source dispatch) -- deliberate follow-up."""
    test_time_arr, db_test, gt_times_test, gt_dirs_test, n_test = load_split(test_path)
    seed_idx = np.arange(n_test)
    scored = Parallel(n_jobs=n_jobs)(
        delayed(_score_one)(test_time_arr, db_test[s], gt_times_test[s], gt_dirs_test[s],
                             params['w_min'], params['w_max'], params['n_w'], params['d_c'],
                             params['n_c'], params['s_c'], tolerance, max_call_seconds)
        for s in seed_idx
    )
    return pd.DataFrame(scored)


def _save_convergence(trials_df, outdir, objective_metric):
    """Saves cumulative-best objective value vs. trial number, as both a CSV and
    a quick PNG. Only COMPLETE trials are considered -- PRUNED trials are
    automatically excluded here since their state isn't 'COMPLETE'."""
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
    """Reconstruct the {w_min, w_max, n_w, d_c, n_c, s_c} dict _evaluate expects
    from Optuna's flat best_trial.params -- no method/cost namespacing needed."""
    p = best_trial.params
    return dict(w_min=p['w_min'], w_max=p['w_max'], n_w=p['n_w'], d_c=p['d_c'], n_c=p['n_c'], s_c=p['s_c'])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-path", default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--val-path", default=DEFAULT_VAL_PATH)
    parser.add_argument("--n-trials", type=int, default=300,
                         help="hard cap on the TOTAL number of trials this study will ever run, "
                              "counted across every resumed invocation via --storage (default 300).")
    parser.add_argument("--timeout", type=int, default=None,
                         help="optional wall-clock safety net in seconds (default: disabled).")
    parser.add_argument("--n-jobs", type=int, default=-1,
                         help="joblib n_jobs; -1 = all allocated cores. Also used as the batch size "
                              "for pruning-enabled staged evaluation.")
    parser.add_argument("--objective", choices=["all", "warming"], default="warming",
                         help="which F1 the Optuna sampler chases (also what gets reported to "
                              "WilcoxonPruner for pruning decisions).")
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--storage", default=None,
                         help="Optuna storage URL for a persistent, resumable study (default: a "
                              "sqlite file under --outdir).")
    parser.add_argument("--study-name", default=None,
                         help="must match across runs you want to resume into the same study. "
                              "Default: 'optuna_kstest_search_pruning_<objective>'.")
    parser.add_argument("--p-threshold", type=float, default=0.05,
                         help="WilcoxonPruner's significance threshold (default 0.05 -- "
                              "deliberately more conservative than Optuna's own 0.1 default, "
                              "matching optuna_search_pruning.py's reasoning: this dataset's "
                              "series vary in noise level/color and transition character, so a "
                              "stricter threshold requires stronger evidence before pruning a "
                              "trial that might just have hit a hard subset). Lower = more "
                              "conservative.")
    parser.add_argument("--n-startup-steps", type=int, default=None,
                         help="minimum number of this trial's series scores that must be reported "
                              "before WilcoxonPruner will even attempt to prune it (default: one "
                              "full pruning batch -- see --pruning-batch-size). should_prune() is "
                              "only checked once per batch, so setting this below "
                              "--pruning-batch-size has no effect.")
    parser.add_argument("--max-call-seconds", type=int, default=DEFAULT_MAX_CALL_SECONDS,
                         help="hard wall-time cap (seconds) on a SINGLE series' _detect call "
                             f"(default {DEFAULT_MAX_CALL_SECONDS}s = 2h).")
    parser.add_argument("--pruning-batch-size", type=int, default=None,
                         help="how many train series are scored (in parallel, across --n-jobs) "
                              "between each should_prune() check (default: same as the resolved "
                              "--n-jobs). Must be > 0.")
    args = parser.parse_args()

    if args.max_call_seconds <= 0:
        parser.error(f"--max-call-seconds must be > 0, got {args.max_call_seconds}")
    if args.pruning_batch_size is not None and args.pruning_batch_size <= 0:
        parser.error(f"--pruning-batch-size must be > 0, got {args.pruning_batch_size}")
    if args.n_startup_steps is not None and args.n_startup_steps < 0:
        parser.error(f"--n-startup-steps must be >= 0, got {args.n_startup_steps}")
    if args.n_jobs == 0:
        parser.error("--n-jobs cannot be 0 (joblib's Parallel rejects it outright; use -1 for all "
                      "cores, or a positive core count)")
    if args.n_trials is not None and args.n_trials < 0:
        parser.error(f"--n-trials must be >= 0, got {args.n_trials}")
    if args.timeout is not None and args.timeout <= 0:
        parser.error(f"--timeout must be > 0, got {args.timeout}")

    outdir = args.outdir or os.path.join(HERE, f"optuna_results_pruning_{args.objective}")
    os.makedirs(outdir, exist_ok=True)
    storage = args.storage or f"sqlite:///{os.path.join(outdir, 'study.db')}"
    study_name = args.study_name or f"optuna_kstest_search_pruning_{args.objective}"

    search = OptunaSearch(args.train_path, args.val_path, TOLERANCE, args.n_jobs,
                           objective_metric=args.objective, pruning_batch_size=args.pruning_batch_size,
                           max_call_seconds=args.max_call_seconds)
    print(f"train seeds: {search.n_train}   validation seeds: {search.n_val}   dt={search.dt:.4f}", flush=True)
    print(f"max_call_seconds={args.max_call_seconds}s (hard per-series _detect timeout)", flush=True)

    order_path = os.path.join(outdir, "train_eval_order.csv")
    pd.DataFrame({'step': np.arange(search.n_train), 'train_series_index': search.train_eval_order}) \
        .to_csv(order_path, index=False)
    print(f"Saved step -> train series index mapping -> {order_path}", flush=True)

    budget_desc = (f"n_trials cap={args.n_trials} (total across all resumed invocations)"
                   if args.n_trials is not None else f"timeout={args.timeout}s")
    print(f"budget: {budget_desc}   n_jobs={args.n_jobs}", flush=True)

    n_startup_steps = args.n_startup_steps if args.n_startup_steps is not None else search.batch_size
    print(f"pruning: WilcoxonPruner(p_threshold={args.p_threshold}, n_startup_steps={n_startup_steps}) "
          f"-- trials scored in batches of {search.batch_size} train series, checked for pruning "
          f"after each batch", flush=True)

    # No multivariate=True/group=True -- see module docstring: those flags exist in the
    # ruptures scripts to handle a method -> cost -> pen conditional search space that
    # doesn't exist here.
    sampler = TPESampler(seed=RANDOM_SEED, n_startup_trials=30)
    pruner = WilcoxonPruner(p_threshold=args.p_threshold, n_startup_steps=n_startup_steps)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner, storage=storage,
                                 study_name=study_name, load_if_exists=True)
    n_trials_before = len(study.trials)
    print(f"study {study_name!r} @ {storage!r}: {n_trials_before} trial(s) already recorded "
          f"before this run", flush=True)

    study_config = dict(
        n_train=int(search.n_train),
        random_seed=RANDOM_SEED,
        train_data_hash=_hash_file(args.train_path),
        objective=args.objective,
        tolerance=TOLERANCE,
        search_space_hash=_search_space_fingerprint(),
        scoring_version=SCORING_VERSION,
    )
    if n_trials_before == 0:
        for key, value in study_config.items():
            study.set_user_attr(f"config_{key}", value)
    else:
        mismatches = []
        for key, value in study_config.items():
            recorded = study.user_attrs.get(f"config_{key}")
            if recorded != value:
                mismatches.append(f"  {key}: recorded={recorded!r}  this run={value!r}")
        if mismatches:
            raise RuntimeError(
                f"Resumed study {study_name!r} was created under different conditions than this "
                f"invocation (or is missing this compatibility metadata entirely) -- resuming "
                f"would silently mix scores/pruning decisions from two eras of the study that "
                f"aren't actually comparable:\n" + "\n".join(mismatches) +
                f"\nUse a new --study-name if this change is intentional."
            )

    n_trials_to_run = None
    if args.n_trials is not None:
        n_trials_to_run = max(0, args.n_trials - n_trials_before)
        print(f"--n-trials cap is {args.n_trials} total -- running up to {n_trials_to_run} more "
              f"trial(s) this invocation", flush=True)

    t0 = time.perf_counter()
    if n_trials_to_run == 0:
        print(f"study already has {n_trials_before} trials >= --n-trials cap of {args.n_trials} -- "
              f"nothing to run this invocation; re-reporting the existing best.", flush=True)
    else:
        study.optimize(search.objective, n_trials=n_trials_to_run, timeout=args.timeout,
                        catch=(SeriesEvaluationError,), show_progress_bar=False)
    wall = time.perf_counter() - t0
    n_trials_total = len(study.trials)
    n_trials_this_run = n_trials_total - n_trials_before
    n_pruned_total = sum(1 for t in study.trials if t.state == TrialState.PRUNED)
    n_failed_total = sum(1 for t in study.trials if t.state == TrialState.FAIL)

    trials_df = study.trials_dataframe()
    trials_path = os.path.join(outdir, "optuna_trials.csv")
    trials_df.to_csv(trials_path, index=False)
    print(f"\nSaved {len(trials_df)} trials -> {trials_path}", flush=True)
    print(f"Study wall time: {wall:.0f}s   trials this run: {n_trials_this_run}   "
          f"trials total: {n_trials_total}   pruned total: {n_pruned_total}   "
          f"failed total: {n_failed_total}", flush=True)

    _save_convergence(trials_df, outdir, args.objective)

    n_complete = sum(1 for t in study.trials if t.state == TrialState.COMPLETE)
    if n_complete == 0:
        print(f"\nNo trials reached COMPLETE this study (trials_total={n_trials_total}, "
              f"pruned={n_pruned_total}, failed={n_failed_total}) -- nothing to report. Check "
              f"{trials_path} for FAIL/PRUNED reasons; if everything is getting pruned, "
              f"--p-threshold and/or --n-startup-steps are probably too aggressive for this "
              f"data. Stopping here without writing best_params.json/best_train_raw.csv/"
              f"best_validation_raw.csv.", flush=True)
        return

    best = study.best_trial
    print(f"\nBest trial #{best.number} (optimized on f1_{args.objective}): "
          f"train mean_f1_all={best.user_attrs.get('mean_f1_all'):.3f}  "
          f"mean_f1_warming={best.user_attrs.get('mean_f1_warming'):.3f}  "
          f"mean_f1_cooling={best.user_attrs.get('mean_f1_cooling'):.3f}")
    print(f"  params: {best.params}")
    print(f"  mean call time (train): {best.user_attrs.get('mean_elapsed_s'):.3f}s")

    best_full_params = _best_full_params(best)
    param_desc = (f"w_min={best_full_params['w_min']:.3g}  w_max={best_full_params['w_max']:.3g}  "
                  f"n_w={best_full_params['n_w']}  d_c={best_full_params['d_c']:.3g}  "
                  f"n_c={best_full_params['n_c']}  s_c={best_full_params['s_c']:.3g}")

    params_json_path = os.path.join(outdir, "best_params.json")
    with open(params_json_path, "w") as f:
        json.dump(dict(**best_full_params, x_c=None, trial_number=best.number,
                        objective=args.objective,
                        train_f1_all=best.user_attrs.get('mean_f1_all'),
                        train_f1_warming=best.user_attrs.get('mean_f1_warming'),
                        train_f1_cooling=best.user_attrs.get('mean_f1_cooling'),
                        train_mean_elapsed_s=best.user_attrs.get('mean_elapsed_s'),
                        run_metadata=dict(
                            wall_time_s=wall,
                            n_trials_this_run=n_trials_this_run,
                            n_trials_total=n_trials_total,
                            n_pruned_total=n_pruned_total,
                            n_failed_total=n_failed_total,
                            p_threshold=args.p_threshold,
                            batch_size=search.batch_size,
                        )), f, indent=2)
    print(f"Saved winning config -> {params_json_path}")

    train_df = search.evaluate_train(best_full_params)
    train_csv = os.path.join(outdir, "best_train_raw.csv")
    train_df.to_csv(train_csv, index=False)
    print(f"Saved per-seed TRAIN breakdown -> {train_csv}")

    val_df = search.evaluate_validation(best_full_params)
    val_csv = os.path.join(outdir, "best_validation_raw.csv")
    val_df.to_csv(val_csv, index=False)
    print(f"\nVALIDATION (safe to check every run -- use this to compare search-procedure choices):")
    print(f"  {param_desc}")
    print(f"  val_f1_warming={val_df['f1_warming'].mean():.3f}  "
          f"val_precision_warming={val_df['precision_warming'].mean():.3f}  "
          f"val_recall_warming={val_df['recall_warming'].mean():.3f}")
    print(f"  val_f1_all={val_df['f1_all'].mean():.3f}  val_f1_cooling={val_df['f1_cooling'].mean():.3f}  "
          f"mean_call_s={val_df['elapsed_s'].mean():.3f}")
    print(f"Saved -> {val_csv}")

    print(f"\ntest.npz was never opened this run. evaluate_final_test() in this file is ready to "
          f"score it once validation results are locked in and evaluate_test.py is wired up for "
          f"KS test (not yet done -- deliberate follow-up).")


if __name__ == "__main__":
    main()
