#!/usr/bin/env python
"""
Joint Optuna (TPE) search over the augmented KS test's own parameters
(w_min, w_max, n_w, n_c, d_c, s_c), applying the same study-level patience-based
early stopping as optuna_search.py (the ruptures analog of this file -- see
that file's module docstring for the fuller design rationale this one reuses
verbatim where noted below). This is the KS-test sibling of that pipeline, not
a replacement for it: same train/validation/test discipline, same direction-
stratified scoring, same resumable/checkpointed study design, same output file
shapes -- only the thing being tuned (ammonyte.Series.kstest / KS_test, the
augmented Kolmogorov-Smirnov test of Bagniewski et al. 2021) differs.

KS test has no categorical branching the way ruptures' method x cost did --
every trial samples all 6 parameters unconditionally, so there is no grouped/
conditional search space to reconstruct here (contrast optuna_search.py's
_sample_params/_best_full_params, which are method/cost-namespaced).

Search space (all 6 parameters tuned jointly, x_c always left at its per-series
auto default -- see 'Standardization' below):

  n_c   in [3, 15]     (int)   minimum sample size required in EACH of the two
                                compared windows. KS_test computes each window's
                                own std with ddof=1, which needs at least 2
                                points to be defined at all -- 2 remains the
                                hard mathematical floor, but the search's own
                                lower bound is set to 3 here.
  w_min in [max(0.05, n_c*dt), 0.4] (float, log-spaced)
                                smallest sliding window width (time units).
                                Lower-bounded by n_c*dt (dt = this split's own
                                median sample spacing, computed at runtime, not
                                hardcoded) so a sampled w_min can never be too
                                short to contain n_c points -- a window that
                                fails that gate is unconditionally zeroed out by
                                KS_test's own th_si threshold regardless of any
                                other parameter, making that trial pay full
                                compute cost for a window that could never
                                contribute anything. n_c is sampled first so its
                                value is available to compute this bound.
  w_max in [w_min*1.5, 4.0] (float, log-spaced)
                                largest sliding window width. Lower-bounded at
                                1.5x the sampled w_min (not just w_min itself)
                                for two reasons: (1) enforces w_max > w_min,
                                which KS_test's own window-sequence construction
                                assumes but does not itself validate; (2) a
                                w_max only marginally above w_min buys almost
                                none of the multi-scale averaging KS_test relies
                                on (see ksstat2's cross-window blending in
                                ammonyte/utils/ks.py), so trials in that
                                near-degenerate region are mostly wasted compute
                                for no real coverage benefit. 4.0 upper bound
                                comfortably spans this dataset's ground-truth
                                transition-gap distribution (median 1.8 kyr).
  n_w   in [5, 25]     (int)   number of window widths tested between w_min and
                                w_max (log-spaced within KS_test itself). This is
                                primarily a coverage-vs-cost dial, not a
                                detection-quality one -- cost scales close to
                                linearly with n_w (measured: ~12s at n_w=15 vs.
                                ~20s at n_w=25 on this project's train.npz, one
                                series), since it's literally n_w full passes
                                over every time point.
  d_c   in [0.5, 0.95] (float) KS-statistic cutoff -- the main sensitivity axis.
                                0 = identical window distributions, 1 = fully
                                separated. Below 0.5 risks firing on ordinary
                                noise-level distributional wiggle; above 0.95
                                demands near-total separation, which real noisy
                                transitions may rarely clear. ammonyte's own
                                default is 0.75 (Bagniewski et al. 2021), well
                                inside this range with margin both sides.
  s_c   in [0.5, 3.0]  (float) effect-size gate: requires the two windows' mean
                                difference to be >= s_c times EACH window's own
                                std. ammonyte's own default is 1.5, likewise
                                centered with margin inside this range.

x_c (absolute mean-difference floor, ammonyte's third gate alongside d_c/s_c)
is NEVER tuned -- always passed as x_c=None, so KS_test auto-computes it fresh
per series (std(x)*0.1 + (max(x)-min(x))*0.05, from that series' OWN values).
Gridding a fixed x_c across seeds with different noise realizations would
silently break that per-seed adaptivity and bias results toward whichever
seeds happen to match a fixed value -- same reasoning the prior (never-executed)
KS/ks_grid_red_snr3.ipynb grid search used to justify leaving it out of its own
grid.

Standardization: load_split below standardizes every series independently
(zero mean, unit std, from its OWN stats), matching optuna_search.py's
load_split exactly, for consistency with the rest of this project's data-
loading convention -- NOT because KS test's own correctness depends on it, the
way ruptures' pen does. With x_c left at its auto default, the entire KS test
detection pipeline is provably invariant to any per-series affine
standardization: the KS D-statistic (d_c's gate) is invariant to any common
monotonic transform of the two compared samples; s_c*std <= |mean_diff| has the
scale factor cancel identically on both sides of the inequality; and the auto
x_c default scales in exact proportion to the series' own std/range, so it
moves with the transform too. So standardizing here changes nothing about
which changepoints get detected -- it's kept for pipeline consistency (and so
this file's load_split can be compared byte-for-byte against the ruptures
scripts'), not correctness.

Sampler: TPESampler(seed=RANDOM_SEED, n_startup_trials=20) -- deliberately
WITHOUT multivariate=True, group=True. Those two flags exist in
optuna_search.py specifically to handle its method -> cost -> pen grouped/
conditional search space (different trials activate different parameter
subsets). KS test's search space is flat and unconditional -- every trial
samples all 6 parameters every time -- so there is no grouping problem for
`group=True` to solve, and carrying that complexity over here would be solving
a problem that doesn't exist for this search space.

Per-call cost is dramatically higher than ruptures': measured 12-22s per
KS_test call on this project's actual train.npz (60 series, 6001 points each),
versus ruptures' ~0.02-0.03s -- roughly 500-1000x more expensive per call, and
scaling close to linearly with n_w. This is why --max-call-seconds (the
CallTimeoutError wall-time cap on a single series' call) matters more here
than it did for ruptures, not less, and why this is a CARC/SLURM job, not a
laptop-interactive one -- see optuna_search_kstest_warming.slurm.

Everything below is otherwise unchanged from optuna_search.py -- reproduced
here (not imported) to keep this file independently runnable, matching this
project's existing convention of intentional duplication between the ruptures
patience/pruning script pair:

Direction-stratified scoring: KS_test's jump_values (+1/-1 up/down) are NOT
used for warming/cooling labeling here, same as ruptures -- see
_classify_detected_directions below, which is method-agnostic (operates on the
detected times/value/time array only, no assumptions about how those times
were produced) and reused verbatim. f1_all, f1_warming, and f1_cooling are all
always computed and logged every trial, regardless of which one the sampler is
chasing (--objective).

Cross-validation was deliberately NOT added: nothing here is *fit* the way a
model would be -- KS_test just applies a fixed, Optuna-proposed
(w_min, w_max, n_w, d_c, n_c, s_c) to each series independently. validation
(scored freely, to compare search-procedure choices) already covers the role
CV would otherwise be reached for.

Early stopping is done at the STUDY level (see _make_early_stopping_callback),
not per-trial -- see optuna_search_kstest_pruning.py for the WilcoxonPruner-
based per-trial alternative, run as its own separate study for comparison.

Why TEST is scored on more than f1 (timing_mae_*), and the train/validation/
test data discipline (60/20/20 series, train.npz scored every trial and only
by this script, validation checked freely, test opened exactly once by
evaluate_test.py after every design choice is locked in via validation): both
unchanged from optuna_search.py -- see that file's docstring for the full
reasoning.

Outputs (all under --outdir), each independently re-usable without re-running
anything upstream:
  - optuna_trials.csv:       one row per trial tried -- every (n_c, w_min,
                              w_max, n_w, d_c, s_c) combo with its train mean
                              f1_all/f1_warming/f1_cooling/precision_*/
                              recall_*/elapsed_s/cpu_time_s/peak_rss_gb.
  - best_params.json:        the single winning config (w_min, w_max, n_w,
                              d_c, n_c, s_c, x_c) plus which trial it was and
                              its train f1_warming/f1_all/f1_cooling. This is
                              the only file evaluate_test.py needs (once wired
                              up for KS test -- not yet done, deliberate
                              follow-up, see conversation).
  - best_train_raw.csv:      per-series (per-seed) precision/recall/f1 (all
                              three directions) for the winning config on train.
  - best_validation_raw.csv: same, on validation.
  - (best_test_raw.csv is written by evaluate_test.py, not this script.)

Usage:
    python optuna_search_kstest.py --n-trials 250 --n-jobs 8
    python optuna_search_kstest.py --n-trials 250 --min-trials 100 --patience 40 --min-delta 0.01 --n-jobs 8
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
from joblib import Parallel, delayed

import ammonyte as amt
from pyleoclim.utils.tsutils import standardize as pyleo_standardize
from ammonyte.utils.metrics import evaluate_detection

HERE = os.path.dirname(os.path.abspath(__file__))
METHODS_DIR = os.path.dirname(HERE)                # this script lives one level below Methods/
SYNTHETICDATA_DIR = os.path.dirname(METHODS_DIR)   # train_test_split_data/ is a Methods/ sibling, not a child


class SeriesEvaluationError(Exception):
    """Raised by OptunaSearch.objective when any series (or the whole batch's
    Parallel call) fails during the live search -- caught by name in main()'s
    study.optimize(..., catch=(SeriesEvaluationError,)) so Optuna marks that ONE
    trial as FAIL and moves on to the next trial, instead of either (a) silently
    scoring the failure as f1=0.0 (indistinguishable from a real bad result, and
    able to influence which config gets selected as best) or (b) propagating
    uncaught and aborting the entire study."""


class CallTimeoutError(Exception):
    """Raised when a single _detect call exceeds --max-call-seconds (see _detect's
    signal.alarm wrapper). Caught by _score_one's existing except-Exception block
    exactly like any other detection failure, becoming that series' 'error' entry
    (readable as "CallTimeoutError: ..." in trials.csv/user_attrs), which
    `objective` then escalates to SeriesEvaluationError -> trial FAIL, same path as
    a crash -- an unacceptably-slow config is now a labeled, queryable outcome
    instead of the study stalling on it indefinitely. Matters more here than for
    ruptures -- KS_test calls are ~500-1000x more expensive per call (measured
    12-22s vs. ruptures' ~0.02-0.03s), so a pathological (e.g. very large n_w on a
    long series) config is a real risk, not a theoretical one.

    Relies on SIGALRM, so Unix-only, and only fires correctly because _detect
    always runs as the sole thread of a single-task joblib (loky) worker process
    here -- signal handlers/alarms are process- and main-thread-scoped in Python,
    which is exactly the execution model of every call site in this file."""


def _alarm_handler(signum, frame):
    raise CallTimeoutError("_detect exceeded --max-call-seconds")


TOLERANCE = 0.5           # +/- 0.5 kyr matching tolerance, same as ruptures/prior grid search
RANDOM_SEED = 42
DEFAULT_MAX_CALL_SECONDS = 7200   # 2h hard cap on a SINGLE series' _detect call (see
                                   # CallTimeoutError). Measured KS_test calls are 12-22s on
                                   # this project's data, so this is generous headroom, not a
                                   # tight budget -- a genuine safety net against a
                                   # pathological config (e.g. large n_w on unusually long
                                   # data), not something normal trials should approach.
SCORING_VERSION = "2"     # bump this any time _score_one/_detect/evaluate_detection_by_direction
                           # or the search-space bounds below meaningfully change -- included in
                           # the resume-compatibility check (see main()) so a study resumed after
                           # such a change fails loudly instead of silently mixing scores from two
                           # different scoring definitions across its trial history.
DIRECTION_MIN_WINDOW = 10  # samples each side of a detected changepoint used to classify it
                            # warming/cooling. FIXED for every config/n_c (was max(this, n_c*2))
                            # so all configs are treated identically.

DEFAULT_TRAIN_PATH = os.path.normpath(os.path.join(SYNTHETICDATA_DIR, "train_test_split_data", "train.npz"))
DEFAULT_VAL_PATH = os.path.normpath(os.path.join(SYNTHETICDATA_DIR, "train_test_split_data", "validation.npz"))
DEFAULT_TEST_PATH = os.path.normpath(os.path.join(SYNTHETICDATA_DIR, "train_test_split_data", "test.npz"))

# Search-space bounds -- see module docstring for the rationale behind each. w_min/w_max's
# lower bounds are computed at runtime (depend on n_c and this split's own dt), not listed
# here as fixed numbers.
N_C_BOUNDS = (3, 15)
W_MIN_UPPER = 0.4
W_MAX_RATIO_MIN = 1.5   # w_max's lower bound is sampled_w_min * this ratio
W_MAX_UPPER = 4.0
N_W_BOUNDS = (5, 25)
D_C_BOUNDS = (0.5, 0.95)
S_C_BOUNDS = (0.5, 3.0)


def _search_space_fingerprint():
    """Short hash of everything that defines what a trial's params dict can even be
    (every search-space bound above) -- used by main() to detect a resumed study
    whose search space no longer matches what its already-completed trials were
    sampled from (e.g. these bounds edited between resumed invocations)."""
    payload = json.dumps(dict(n_c_bounds=N_C_BOUNDS, w_min_upper=W_MIN_UPPER,
                               w_max_ratio_min=W_MAX_RATIO_MIN, w_max_upper=W_MAX_UPPER,
                               n_w_bounds=N_W_BOUNDS, d_c_bounds=D_C_BOUNDS, s_c_bounds=S_C_BOUNDS),
                          sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _hash_file(path):
    """Short content hash of an entire file's bytes -- used to detect train.npz being
    swapped for a different file with the same series COUNT (n_train alone wouldn't
    catch that) between resumed invocations of the same study. Hashes the whole file
    (time/db_noisy/state_true/everything in the .npz), not just one array, since
    scores depend on time (used to build each amt.Series and for tolerance matching)
    and state_true (via the derived ground-truth transition times/directions) just as
    much as db_noisy -- see load_split."""
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
    local mean value just before vs. just after it, in plain forward array/time order
    -- verified against this dataset's own ground truth (see transitions_from_states):
    a warming (1->0) transition always shows the value increasing forward through the
    array, matching the documented Melcher25 convention (db higher = warmer).
    Method-agnostic -- operates only on the detected times/value/time array, with no
    assumption about how those times were produced (ruptures vs. KS test vs.
    anything else), and deliberately NOT ammonyte's ruptures_transitions.py
    jump_values / KS test's own jump_values sign convention, both of which assume a
    paleo age axis (index ascending = more recent -> older) that is backwards for
    this dataset's plain forward-simulation-time array.

    Window is DIRECTION_MIN_WINDOW (fixed) samples each side, clipped at series edges
    and at neighboring detected transitions so it never blends across an adjacent,
    possibly oppositely-signed transition. A window left empty by that clipping (e.g.
    two detections one sample apart) is labeled 'ambiguous' -- excluded from both the
    warming- and cooling-filtered F1s, but still scored normally in f1_all.

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


def evaluate_detection_by_direction(time_arr, value, detected, gt_times, gt_directions, tolerance, min_size):
    """f1_all (direction-agnostic, exactly the old single-F1 behavior) plus
    f1_warming/f1_cooling: detections are classified warming/cooling first (see
    _classify_detected_directions), then BOTH detections and ground truth are
    filtered to one direction before calling the existing, unmodified
    evaluate_detection on that subset. Also returns the matched timing errors (see
    _matched_timing_errors) for the same three (all/warming/cooling) subsets."""
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


def _detect(time_arr, value, w_min, w_max, n_w, d_c, n_c, s_c, max_call_seconds):
    series = amt.Series(time=time_arr, value=value, label='optuna', verbose=False)
    t0 = time.perf_counter()

    # Hard wall-time cap on this ONE series' detection call -- see CallTimeoutError.
    # Previous handler/alarm are restored in `finally` so a nested/outer alarm (there
    # isn't one today, but a future caller might add one) is never clobbered.
    old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(max_call_seconds)
    try:
        # x_c always None -- see module docstring, never tuned, auto-computed fresh
        # per series by KS_test itself.
        transitions = series.kstest(w_min=w_min, w_max=w_max, n_w=n_w, d_c=d_c, n_c=n_c, s_c=s_c, x_c=None)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

    elapsed_s = time.perf_counter() - t0
    return _clean_times(transitions.jump_times), elapsed_s


def _score_one(time_arr, value, gt_times, gt_directions, w_min, w_max, n_w, d_c, n_c, s_c, tolerance,
                max_call_seconds):
    # cpu_time_s: accurate per-call CPU (user+sys) seconds, via before/after delta --
    # isolates this one call regardless of what else this worker process has done.
    # peak_rss_gb: NOT isolated the same way -- ru_maxrss is a whole-process
    # high-water mark that never resets, so under joblib's reused worker processes
    # this is "this worker's peak RSS across every call it's handled so far", not
    # this call alone. Still useful as a rough signal for which param combos
    # coincide with memory jumps.
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


class OptunaSearch:
    """Bundles the pooled train/validation data and evaluation logic. Three
    disjoint roles, never blurred:

      - train:      what `objective` scores every single trial against. This is
                     the only thing Optuna's sampler ever sees, and it's fine to
                     look at train scores/diagnostics as often as you want.
      - validation: what `evaluate_validation` scores a candidate winning config
                     against, once per config you're considering. Use this to
                     compare experimental choices (narrower/wider bounds, a
                     different sampler, a different --n-trials budget) and
                     decide which search *procedure* to finalize. Safe to check
                     repeatedly while iterating -- that's its whole job.
      - test:       deliberately NOT loaded by this class at all -- see
                     module docstring's data-discipline section.
    """

    def __init__(self, train_path, val_path, tolerance, n_jobs, objective_metric='warming',
                 max_call_seconds=DEFAULT_MAX_CALL_SECONDS):
        (self.time_arr, self.db_train, self.gt_times_train, self.gt_dirs_train,
         self.n_train) = load_split(train_path)
        val_time_arr, self.db_val, self.gt_times_val, self.gt_dirs_val, self.n_val = load_split(val_path)
        # evaluate_validation (via _evaluate) reuses self.time_arr (TRAIN's) for every
        # series it scores, including validation ones -- val_time_arr itself is discarded
        # above. This assertion makes the assumption that makes it safe -- train and
        # validation share one time axis -- loud and immediate if it's ever violated,
        # rather than silently scoring validation series against the wrong time axis.
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
        self.max_call_seconds = max_call_seconds   # hard per-series _detect timeout -- see CallTimeoutError
        self.objective_metric = objective_metric   # 'all' or 'warming' -- which f1_* column
                                                     # `objective` hands the sampler; see module docstring
        # This split's own median sample spacing -- used to lower-bound w_min against n_c
        # in _sample_params (see module docstring). Computed once here, not per-trial,
        # since it never changes for a given train_path.
        self.dt = float(np.median(np.diff(self.time_arr)))

    def _sample_params(self, trial):
        # No categorical branching -- every trial samples all 6 parameters
        # unconditionally (contrast optuna_search.py's method/cost-namespaced
        # sampling). n_c sampled first so its value can lower-bound w_min; w_min
        # sampled next so ITS value can lower-bound w_max -- see module docstring
        # for why each bound is what it is.
        n_c = trial.suggest_int('n_c', *N_C_BOUNDS)
        w_min = trial.suggest_float('w_min', max(0.05, n_c * self.dt), W_MIN_UPPER, log=True)
        w_max = trial.suggest_float('w_max', w_min * W_MAX_RATIO_MIN, W_MAX_UPPER, log=True)
        n_w = trial.suggest_int('n_w', *N_W_BOUNDS)
        d_c = trial.suggest_float('d_c', *D_C_BOUNDS)
        s_c = trial.suggest_float('s_c', *S_C_BOUNDS)
        return dict(w_min=w_min, w_max=w_max, n_w=n_w, d_c=d_c, n_c=n_c, s_c=s_c)

    def _evaluate(self, params, db, gt_times_list, gt_dirs_list, seed_idx):
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

    def objective(self, trial):
        params = self._sample_params(trial)
        seed_idx = np.arange(self.n_train)
        df, err = self._evaluate(params, self.db_train, self.gt_times_train, self.gt_dirs_train, seed_idx)

        if df is None:   # the whole Parallel dispatch crashed (e.g. a worker process died)
            trial.set_user_attr('error', err)
            raise SeriesEvaluationError(f"train evaluation crashed: {err}")

        errored = df[df['error'].notna()]
        if len(errored):
            # Any series failing to score is treated as this whole trial failing,
            # instead of silently averaging a fabricated f1=0.0 into the trial's
            # score -- a crash-caused zero is indistinguishable from a genuinely
            # bad result once averaged in, and could wrongly tank (or wrongly help
            # select) a config that's actually fine. See module docstring.
            first_error = errored['error'].iloc[0]
            trial.set_user_attr('error', first_error)
            trial.set_user_attr('n_errors', int(len(errored)))
            raise SeriesEvaluationError(
                f"{len(errored)}/{len(df)} train series failed to score; first error: {first_error}"
            )

        trial.set_user_attr('mean_elapsed_s', float(df['elapsed_s'].mean()))
        trial.set_user_attr('mean_cpu_time_s', float(df['cpu_time_s'].mean()))
        # max, not mean -- peak_rss_gb is a high-water mark (see _score_one), so the
        # largest value seen in this trial's series is the meaningful number, not an average.
        trial.set_user_attr('max_peak_rss_gb', float(df['peak_rss_gb'].max()))
        for label in ('all', 'warming', 'cooling'):
            trial.set_user_attr(f'mean_precision_{label}', float(df[f'precision_{label}'].mean()))
            trial.set_user_attr(f'mean_recall_{label}', float(df[f'recall_{label}'].mean()))
            trial.set_user_attr(f'mean_f1_{label}', float(df[f'f1_{label}'].mean()))
            timing = df[f'timing_mae_{label}'].dropna()
            trial.set_user_attr(f'mean_timing_mae_{label}', float(timing.mean()) if len(timing) else float('nan'))
        # which column the sampler chases is a run-level choice (--objective); f1_all,
        # f1_warming, f1_cooling are all still computed/logged above either way
        return float(df[f'f1_{self.objective_metric}'].mean())

    def evaluate_validation(self, params):
        seed_idx = np.arange(self.n_val)
        df, err = self._evaluate(params, self.db_val, self.gt_times_val, self.gt_dirs_val, seed_idx)
        if df is None:
            raise RuntimeError(f"validation evaluation failed: {err}")
        return df

    def evaluate_train(self, params):
        """Per-seed raw scores for one config on train -- separate from `objective`,
        which only ever keeps the mean (see class docstring). Used once, after the
        study finishes, to save the winning config's full per-seed train breakdown."""
        seed_idx = np.arange(self.n_train)
        df, err = self._evaluate(params, self.db_train, self.gt_times_train, self.gt_dirs_train, seed_idx)
        if df is None:
            raise RuntimeError(f"train evaluation failed: {err}")
        return df


def evaluate_final_test(test_path, time_arr, tolerance, n_jobs, params, max_call_seconds=DEFAULT_MAX_CALL_SECONDS):
    """Loads test data fresh (never held on OptunaSearch) and scores exactly one
    config against it. Call this once, after every design decision -- search-space
    bounds, sampler settings -- is already finalized via validation. Do not change
    anything and call it again.

    Not yet wired into evaluate_test.py (that file currently only knows the
    ruptures patience/pruning scripts' --source dispatch) -- deliberate follow-up,
    kept here so it exists and is ready when that wiring happens."""
    _, db_test, gt_times_test, gt_dirs_test, n_test = load_split(test_path)
    seed_idx = np.arange(n_test)
    scored = Parallel(n_jobs=n_jobs)(
        delayed(_score_one)(time_arr, db_test[s], gt_times_test[s], gt_dirs_test[s],
                             params['w_min'], params['w_max'], params['n_w'], params['d_c'],
                             params['n_c'], params['s_c'], tolerance, max_call_seconds)
        for s in seed_idx
    )
    return pd.DataFrame(scored)


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
    """Reconstruct the {w_min, w_max, n_w, d_c, n_c, s_c} dict _evaluate expects
    from Optuna's flat best_trial.params -- no method/cost namespacing needed
    here (contrast optuna_search.py's _best_full_params), since there's no
    conditional branching in this search space."""
    p = best_trial.params
    return dict(w_min=p['w_min'], w_max=p['w_max'], n_w=p['n_w'], d_c=p['d_c'], n_c=p['n_c'], s_c=p['s_c'])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-path", default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--val-path", default=DEFAULT_VAL_PATH)
    parser.add_argument("--n-trials", type=int, default=250,
                         help="hard cap on the TOTAL number of trials this study will ever run, "
                              "counted across every resumed invocation via --storage -- not trials "
                              "added by this invocation alone. If the study already has >= this "
                              "many trials when the run starts, this invocation runs zero more and "
                              "just re-reports the existing best/validation.")
    parser.add_argument("--timeout", type=int, default=None,
                         help="optional wall-clock safety net in seconds (default: disabled -- "
                              "--n-trials is the budget mechanism). Set this only if you also want a "
                              "hard stop under e.g. a SLURM time limit regardless of trial count.")
    parser.add_argument("--n-jobs", type=int, default=-1,
                         help="joblib n_jobs; -1 = all allocated cores.")
    parser.add_argument("--objective", choices=["all", "warming"], default="warming",
                         help="which F1 the Optuna sampler chases: 'warming' (default -- the "
                              "eventual application, NGRIP DO events, is warming-only) or 'all' "
                              "(direction-agnostic). f1_all/f1_warming/f1_cooling are always "
                              "computed and saved regardless.")
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--storage", default=None,
                         help="Optuna storage URL for a persistent, resumable study (default: a "
                              "sqlite file under --outdir). Rerunning with the same --storage and "
                              "--study-name reconnects to the existing study and keeps sampling on "
                              "top of its trial history -- safe to interrupt (SLURM time limit, OOM, "
                              "manual kill) and resume later with no lost trials.")
    parser.add_argument("--study-name", default=None,
                         help="must match across runs you want to resume into the same study. "
                              "Default: 'optuna_kstest_search_<objective>'.")
    parser.add_argument("--patience", type=int, default=40,
                         help="stop the whole study early once this many COMPLETE trials have passed "
                              "with no improvement greater than --min-delta over the running best "
                              "(default 40), instead of always running until the --n-trials cap is "
                              "reached. Set to 0 to disable and always run the full budget.")
    parser.add_argument("--min-trials", type=int, default=100,
                         help="early stopping cannot trigger until at least this many trials have "
                              "completed (default 100 -- higher than the ruptures search's 50: unlike "
                              "ruptures' method/cost branching, every KS-test trial refines the same "
                              "flat 6-parameter space, so there's no branch-exploration cost being "
                              "traded away by running longer here). Only matters when --patience > 0.")
    parser.add_argument("--min-delta", type=float, default=0.01,
                         help="an improvement only resets the --patience counter if it exceeds this "
                              "margin over the running best f1 (default 0.01). Only matters when "
                              "--patience > 0.")
    parser.add_argument("--max-call-seconds", type=int, default=DEFAULT_MAX_CALL_SECONDS,
                         help="hard wall-time cap (seconds) on a SINGLE series' _detect call "
                             f"(default {DEFAULT_MAX_CALL_SECONDS}s = 2h -- see DEFAULT_MAX_CALL_SECONDS). "
                             "A call that exceeds this raises CallTimeoutError, which fails just that "
                             "one series, escalating to that trial being marked FAIL.")
    args = parser.parse_args()

    if args.max_call_seconds <= 0:
        parser.error(f"--max-call-seconds must be > 0, got {args.max_call_seconds}")
    if args.n_jobs == 0:
        parser.error("--n-jobs cannot be 0 (joblib's Parallel rejects it outright; use -1 for all "
                      "cores, or a positive core count)")
    if args.n_trials is not None and args.n_trials < 0:
        parser.error(f"--n-trials must be >= 0, got {args.n_trials}")
    if args.timeout is not None and args.timeout <= 0:
        parser.error(f"--timeout must be > 0, got {args.timeout}")

    outdir = args.outdir or os.path.join(HERE, f"optuna_results_{args.objective}")
    os.makedirs(outdir, exist_ok=True)
    storage = args.storage or f"sqlite:///{os.path.join(outdir, 'study.db')}"
    study_name = args.study_name or f"optuna_kstest_search_{args.objective}"

    search = OptunaSearch(args.train_path, args.val_path, TOLERANCE, args.n_jobs,
                           objective_metric=args.objective, max_call_seconds=args.max_call_seconds)
    print(f"train seeds: {search.n_train}   validation seeds: {search.n_val}   dt={search.dt:.4f}", flush=True)
    print(f"max_call_seconds={args.max_call_seconds}s (hard per-series _detect timeout)", flush=True)
    budget_desc = (f"n_trials cap={args.n_trials} (total across all resumed invocations)"
                   if args.n_trials is not None else f"timeout={args.timeout}s")
    print(f"budget: {budget_desc}   n_jobs={args.n_jobs}", flush=True)

    # No multivariate=True/group=True -- those exist in optuna_search.py specifically to
    # handle its method -> cost -> pen conditional/grouped search space. KS test's search
    # space is flat (every trial samples all 6 parameters unconditionally), so there is no
    # grouping problem for those flags to solve here -- see module docstring.
    sampler = TPESampler(seed=RANDOM_SEED, n_startup_trials=20)
    study = optuna.create_study(direction="maximize", sampler=sampler, storage=storage,
                                 study_name=study_name, load_if_exists=True)
    n_trials_before = len(study.trials)
    print(f"study {study_name!r} @ {storage!r}: {n_trials_before} trial(s) already recorded "
          f"before this run", flush=True)

    # Guards against resuming this study under conditions that no longer match what its
    # already-completed trials were scored against -- see optuna_search.py's identical
    # mechanism for the fuller rationale (reused here unchanged).
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
                f"would silently mix scores from two eras of the study that aren't actually "
                f"comparable:\n" + "\n".join(mismatches) +
                f"\nUse a new --study-name if this change is intentional."
            )

    n_trials_to_run = None
    if args.n_trials is not None:
        n_trials_to_run = max(0, args.n_trials - n_trials_before)
        print(f"--n-trials cap is {args.n_trials} total -- running up to {n_trials_to_run} more "
              f"trial(s) this invocation", flush=True)

    stop_info = dict(stopped_early=False, stopped_at_trial=None, trials_since_improvement=None)
    callbacks = ([_make_early_stopping_callback(args.patience, args.min_trials, args.min_delta, stop_info)]
                 if args.patience > 0 else [])
    if args.patience > 0:
        print(f"early stopping: will halt the study once {args.min_trials}+ trials have completed and "
              f"{args.patience} of them pass with no improvement > {args.min_delta}", flush=True)
    else:
        print("early stopping: disabled (--patience 0) -- will run the full --n-trials budget.", flush=True)

    t0 = time.perf_counter()
    if n_trials_to_run == 0:
        print(f"study already has {n_trials_before} trials >= --n-trials cap of {args.n_trials} -- "
              f"nothing to run this invocation; re-reporting the existing best.", flush=True)
    else:
        study.optimize(search.objective, n_trials=n_trials_to_run, timeout=args.timeout,
                        callbacks=callbacks, catch=(SeriesEvaluationError,), show_progress_bar=False)
    wall = time.perf_counter() - t0
    n_trials_total = len(study.trials)
    n_trials_this_run = n_trials_total - n_trials_before
    n_failed_total = sum(1 for t in study.trials if t.state == TrialState.FAIL)

    trials_df = study.trials_dataframe()
    trials_path = os.path.join(outdir, "optuna_trials.csv")
    trials_df.to_csv(trials_path, index=False)
    print(f"\nSaved {len(trials_df)} trials -> {trials_path}", flush=True)
    print(f"Study wall time: {wall:.0f}s   trials this run: {n_trials_this_run}   "
          f"trials total: {n_trials_total}   failed total: {n_failed_total}", flush=True)
    if stop_info['stopped_early']:
        print(f"Study was stopped early by the patience callback at trial #{stop_info['stopped_at_trial']} "
              f"({stop_info['trials_since_improvement']} trials since the last improvement).", flush=True)

    _save_convergence(trials_df, outdir, args.objective)

    n_complete = sum(1 for t in study.trials if t.state == TrialState.COMPLETE)
    if n_complete == 0:
        print(f"\nNo trials reached COMPLETE this study (trials_total={n_trials_total}, "
              f"failed={n_failed_total}) -- nothing to report. Check {trials_path} for FAIL "
              f"reasons in each trial's user_attrs (an 'error' entry). Stopping here without "
              f"writing best_params.json/best_train_raw.csv/best_validation_raw.csv.", flush=True)
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
                            n_failed_total=n_failed_total,
                            patience=args.patience,
                            min_trials=args.min_trials,
                            min_delta=args.min_delta,
                            stopped_early=stop_info['stopped_early'],
                            stopped_at_trial=stop_info['stopped_at_trial'],
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
