#!/usr/bin/env python
"""
Joint Optuna (TPE) search over ruptures method x cost x pen x min_size.

This is a variant of optuna_search.py (which stays as-is, untouched, patience-based
study-level early stopping): instead of always scoring every trial's full train set,
this version prunes clearly-losing trials mid-evaluation using Optuna's
WilcoxonPruner. Run this as a completely separate study (its own --outdir/
--study-name, defaulted below) so it never touches optuna_search.py's results.

Replaces the brute-force per-(method,cost) grid search in the old grid_common.py /
method_configs.py (old grid engine archived under _archive_old_grid_search/) with a
single Optuna study where `method` and `cost` are themselves optimized categorical
dimensions -- the sampler adaptively spends trials on whatever (method, cost) combo
is actually paying off, rather than a fixed, manually pre-allocated per-branch budget.

Kept methods and costs (decisions from the prior full grid-search round -- see
{Pelt,Binseg,BottomUp,KernelCPD,Window}/results*/*_conclusion.csv):

  - Window:   dropped entirely. Underperforms every other method across every cost
              tested -- a single fixed window width can't match irregularly-spaced
              transitions, and its failure mode shifts unpredictably across costs
              (over-detection under l2/rbf/rank, severe under-detection under
              mahalanobis, near-total non-detection under clinear). That points to a
              limitation of the approach, not something more tuning fixes.
  - clinear:  dropped entirely. Piecewise-linear, continuity-constrained cost -- can't
              represent an abrupt jump, blurs the transition instead. Worst or
              near-worst cost for every method it was tried with.
  - linear:   dropped for Pelt/Binseg/BottomUp (no covariate supplied -> mathematically
              reduces to a plain sum of squares -> f1=0.0000 across the entire prior
              grid, not a tuning issue). KEPT for KernelCPD, where 'linear' selects a
              kernel rather than fitting a regression -- unaffected, tied for the best
              result in the whole prior study (0.857 test_f1).
  - ar:       dropped entirely. Reduces to a degenerate/near-zero-signal cost for
              every method it's kept for here (f1=0.0000 across the prior grid for
              Pelt/Binseg/BottomUp; it was only ever non-degenerate under Dynp, now
              removed -- see below).
  - Dynp:     dropped entirely (this was itself a live method through an earlier
              version of this file; removed after profiling -- see below). Dynp's
              defining feature is taking an exact target n_bkps instead of a penalty,
              which requires knowing the true number of transitions in advance --
              available for the synthetic training set (ground truth) but NOT for the
              real deployment target (NGRIP), where the whole point is not knowing
              that count ahead of time. Pelt solves the same kind of exact-optimal
              segmentation problem via a penalty instead (Killick et al. 2012,
              pruned/near-linear time rather than Dynp's O(n_bkps * n^2)), and already
              matches or beats Dynp's results in the prior grid at a tiny fraction of
              the cost -- so Dynp added compute cost and a deployment-incompatible
              assumption without a performance edge to justify either. (Separately,
              profiling traced Dynp's specific slowness to `ruptures.costs.CostL2`/
              `CostRbf` recomputing variance from raw signal slices via numpy on
              every DP state with no cached cumulative stats -- confirmed by
              `ruptures.KernelCPD`, which solves the identical l2/rbf cost problems
              via compiled code, returning byte-identical breakpoints >1000x faster
              on this project's real data. That's a fixable implementation
              inefficiency in `ruptures`, not evidence Dynp-the-idea is unworkable --
              but combined with the n_bkps deployment mismatch above, still not worth
              keeping as a method here.)

jump is fixed at 1 for every method (deliberate -- so any wall-clock savings this
search finds over a brute grid are attributable to smarter search, not to a jump
shortcut).

min_size is searched over {5, 20, 40} (discrete) for every method -- chosen after
checking both (a) the historical min_size sweep on Pelt [5..20 tie for best f1, 40
already shows a measurable recall drop] and (b) the actual ground-truth
transition-gap distribution in the pooled dataset [~14% of consecutive transitions
are closer together (fewer than 40 samples apart) than min_size=40 allows,
structurally un-detectable at that setting regardless of tuning].

Every series is standardized (zero mean, unit std, from its OWN stats) in
load_split before anything else touches it -- see that function's docstring.
NOTE: PEN_BOUNDS below still reflects the OLD grid search's winning pen values
on RAW (unstandardized) data -- those bounds have not been recalibrated for
the new unit-variance scale and may now be wrong (too wide, too narrow, or
centered on the wrong order of magnitude). Re-derive them before trusting this
search's pen results.

Direction-stratified scoring, why TEST is scored on more than f1, and data
discipline are unchanged from optuna_search.py -- see that file's docstring for
the full reasoning (identical here, not repeated).

Pruning (this file's difference from optuna_search.py) -- WilcoxonPruner instead of
study-level patience:
  optuna_search.py's `objective()` always scores every one of train's series in a
  single Parallel call before returning a value -- a bad trial pays full price, and
  the only savings come from `_make_early_stopping_callback` ending the WHOLE STUDY
  once it detects the search has converged. That saves nothing on any individual
  trial, and only kicks in after --min-trials trials have already run at full cost.

  This file instead prunes at the TRIAL level: `OptunaSearch._evaluate_staged`
  scores train in batches of `n_jobs` series at a time (full within-batch
  parallelism, same speed as a normal Parallel call), reports each series'
  individual objective-metric score to Optuna after its batch finishes, and checks
  `trial.should_prune()` once per batch. `optuna.pruners.WilcoxonPruner` performs a
  paired Wilcoxon signed-rank test between the current trial's per-series scores so
  far and the best completed trial's scores on those SAME series indices -- pairing
  cancels out per-series noise/difficulty (a series that's hard for everyone doesn't
  look like evidence against any one config), so a config only gets pruned once
  there's real, accumulated statistical evidence it's worse, not from one unlucky
  series. Once evidence crosses --p-threshold, the trial is pruned (raises
  TrialPruned) and its later batches are skipped entirely -- e.g. a clearly-bad
  clearly-bad trial can be killed after 1 of ~3 batches instead of paying for all of them.

  Two things this depends on:
    - Trials are evaluated sequentially in this script (study.optimize is called
      without its own n_jobs), so time saved pruning one trial is immediately
      available to start the next trial sooner -- a real wall-clock win here, not
      just a theoretical one.
    - Step i must map to the SAME train series for every trial and every resumed
      invocation, or the paired comparison is meaningless. `OptunaSearch.__init__`
      builds `self.train_eval_order`, a permutation of train's indices seeded with
      RANDOM_SEED (so it's identical across every run/resume, never re-shuffled per
      trial) -- shuffled, not raw index order, specifically so an early prune
      decision isn't biased by however train.npz's series happen to be grouped or
      stored (e.g. by noise level/color or transition regime). If train.npz's
      series are already known to be stored in random order w.r.t. those
      properties this shuffle is redundant but harmless; if they are NOT
      known to be random, this shuffle is required for pruning decisions to be
      trustworthy -- verify before trusting results from this script.

  `evaluate_validation`/`evaluate_train` (the full, final re-scoring of the winning
  config after the search ends) are untouched -- they still use the original
  `_evaluate`, unbatched, no pruning, exactly as in optuna_search.py. Pruning only
  ever applies to trials still being compared against each other during the live
  search.

  --p-threshold controls how much evidence WilcoxonPruner needs before pruning
  (default 0.1, matching Optuna's own default). Lower = more conservative (needs
  stronger evidence, prunes less aggressively).

Usage:
    python optuna_search_pruning.py --n-trials 300 --n-jobs 8
    python optuna_search_pruning.py --n-trials 300 --p-threshold 0.1 --n-jobs 8
    python optuna_search_pruning.py --n-trials 300 --n-jobs 24
    # interrupted (SLURM time limit, OOM, ctrl-c)? just rerun the same command -- it resumes
    # from the persistent study at --storage and only runs enough MORE trials to reach the
    # --n-trials cap (a TOTAL across every resumed invocation, not additional trials each time)
    # then, once validation results are locked in:
    python evaluate_test.py --outdir <this run's --outdir>
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
METHODS_DIR = os.path.dirname(HERE)          # this script now lives one level below Methods/,
sys.path.insert(0, METHODS_DIR)              # where method_configs.py lives
SYNTHETICDATA_DIR = os.path.dirname(METHODS_DIR)   # train_test_split_data/ is a Methods/ sibling, not a child
from method_configs import BINSEG_COSTS, PELT_COSTS, BOTTOMUP_COSTS, KERNELCPD_COSTS

class SeriesEvaluationError(Exception):
    """Raised by OptunaSearch._evaluate_staged when any series (or the whole batch
    Parallel call) fails during the live search -- caught by name in main()'s
    study.optimize(..., catch=(SeriesEvaluationError,)) so Optuna marks that ONE
    trial as FAIL and moves on to the next trial, instead of either (a) silently
    scoring the failure as f1=0.0 (indistinguishable from a real bad result, and
    able to influence which config gets selected as best) or (b) propagating
    uncaught and aborting the entire study. See _evaluate_staged's docstring."""


class CallTimeoutError(Exception):
    """Raised when a single _detect call exceeds --max-call-seconds (see _detect's
    signal.alarm wrapper). Caught by _score_one's existing except-Exception block
    exactly like any other detection failure, becoming that series' 'error' entry
    (readable as "CallTimeoutError: ..." in trials.csv/user_attrs), which
    _evaluate_staged then escalates to SeriesEvaluationError -> trial FAIL, same
    path as a crash (see that docstring) -- an unacceptably-slow config is now a
    labeled, queryable outcome instead of the study stalling on it indefinitely.

    Relies on SIGALRM, so Unix-only, and only fires correctly because _detect
    always runs as the sole thread of a single-task joblib (loky) worker process
    here -- signal handlers/alarms are process- and main-thread-scoped in Python,
    which is exactly the execution model of every call site in this file."""


def _alarm_handler(signum, frame):
    raise CallTimeoutError("_detect exceeded --max-call-seconds")


TOLERANCE = 0.5           # +/- 0.5 kyr matching tolerance, same as the prior grid search
RANDOM_SEED = 42
JUMP = 1                  # fixed for every method, deliberately -- see module docstring
DEFAULT_MAX_CALL_SECONDS = 7200   # 2h hard cap on a SINGLE series' _detect call (see
                                   # CallTimeoutError). Originally sized around Dynp's
                                   # known slow worst case (now removed -- see module
                                   # docstring); kept as a general safety net for every
                                   # remaining method (all penalty-based, all fast in
                                   # practice) so a genuinely hung/pathological call still
                                   # can't stall a trial indefinitely, at essentially zero
                                   # cost since none of the kept methods should approach it.
SCORING_VERSION = "2"     # bump this any time _score_one/_detect/evaluate_detection_by_direction
                           # or method_configs.py's cost grids meaningfully change -- included in
                           # the resume-compatibility check (see main()) so a study resumed after
                           # such a change fails loudly instead of silently mixing scores from two
                           # different scoring definitions across its trial history.
DIRECTION_MIN_WINDOW = 10  # samples each side of a detected changepoint used to classify
                            # it warming/cooling; FIXED for every config (was max(this, min_size*2)).

DEFAULT_TRAIN_PATH = os.path.normpath(os.path.join(SYNTHETICDATA_DIR, "train_test_split_data", "train.npz"))
DEFAULT_VAL_PATH = os.path.normpath(os.path.join(SYNTHETICDATA_DIR, "train_test_split_data", "validation.npz"))
DEFAULT_TEST_PATH = os.path.normpath(os.path.join(SYNTHETICDATA_DIR, "train_test_split_data", "test.npz"))

MIN_SIZE_CHOICES = [5, 20, 40]

KEPT_COSTS = {
    'Pelt':      ['l1', 'l2', 'rbf', 'normal', 'rank', 'mahalanobis'],
    'Binseg':    ['l1', 'l2', 'rbf', 'normal', 'rank', 'mahalanobis'],
    'BottomUp':  ['l1', 'l2', 'rbf', 'normal', 'rank', 'mahalanobis'],
    'KernelCPD': ['linear', 'rbf'],
}

METHODS = ['Pelt', 'Binseg', 'BottomUp', 'KernelCPD']   # Dynp removed -- see module docstring


def _pen_bounds(cost_grid, kept):
    return {c: bounds for c, bounds in cost_grid.items() if c in kept}


PEN_BOUNDS = {
    'Pelt':      _pen_bounds(PELT_COSTS, KEPT_COSTS['Pelt']),
    'Binseg':    _pen_bounds(BINSEG_COSTS, KEPT_COSTS['Binseg']),
    'BottomUp':  _pen_bounds(BOTTOMUP_COSTS, KEPT_COSTS['BottomUp']),
    'KernelCPD': _pen_bounds(KERNELCPD_COSTS, KEPT_COSTS['KernelCPD']),
}


def _search_space_fingerprint():
    """Short hash of everything that defines what a trial's params dict can even be
    (methods, kept costs, pen bounds, min_size grid) -- used by main() to detect a
    resumed study whose search space no longer matches what its already-completed
    trials were sampled from (e.g. method_configs.py edited, or METHODS/KEPT_COSTS
    changed -- such as this file's Dynp removal -- between resumed invocations)."""
    payload = json.dumps(dict(methods=METHODS, kept_costs=KEPT_COSTS, pen_bounds=PEN_BOUNDS,
                               min_size_choices=MIN_SIZE_CHOICES),
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
    Deliberately NOT ammonyte's ruptures_transitions.py jump_values -- that assumes a
    paleo age axis (index ascending = more recent -> older) which is backwards for
    this dataset's forward-simulation-time array; see module docstring.

    Window is DIRECTION_MIN_WINDOW (fixed) samples each side, clipped at series edges
    and at neighboring detected transitions so it never blends across an adjacent,
    possibly oppositely-signed transition. A window left empty by that clipping (e.g.
    two detections one sample apart) is labeled 'ambiguous' -- excluded from both the
    warming- and cooling-filtered F1s, but still scored normally in f1_all."""
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
    # series' OWN mean/std) before anything else touches db_noisy. Two reasons:
    # (1) a single tuned `pen` value is only comparable across series if they're
    # on the same scale -- without this, series with different original
    # amplitudes/noise levels would need different pen thresholds, which a
    # single global pen can't provide; (2) this is what lets the optimized
    # params transfer to NGRIP at all -- NGRIP gets self-standardized the same
    # way at application time, so its raw min/max range never needs to match
    # this synthetic pool's. Uses pyleoclim's own standardize (ammonyte.Series
    # subclasses pyleo.Series, so this matches the library the rest of the
    # pipeline is built on) -- axis=1 standardizes each row independently, and
    # a (near-)constant series is left unscaled with a warning rather than
    # raising ZeroDivisionError.
    db_noisy, _, _ = pyleo_standardize(db_noisy, axis=1)

    gt_times, gt_directions = [], []
    for i in range(n):
        t, dirs = transitions_from_states(time_arr, state_true[i])
        gt_times.append(t)
        gt_directions.append(dirs)
    return time_arr, db_noisy, gt_times, gt_directions, n


def _detect(time_arr, value, method, cost, pen, n_bkps, min_size, jump, max_call_seconds):
    series = amt.Series(time=time_arr, value=value, label='optuna', verbose=False)
    t0 = time.perf_counter()
    kwargs = dict(algo=method, cost=cost, min_size=min_size, jump=jump)
    if pen is not None:
        kwargs['pen'] = pen
    else:
        kwargs['n_bkps'] = n_bkps

    # Hard wall-time cap on this ONE series' detection call -- see CallTimeoutError.
    # Previous handler/alarm are restored in `finally` so a nested/outer alarm (there
    # isn't one today, but a future caller might add one) is never clobbered.
    old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(max_call_seconds)
    try:
        transitions = series.ruptures(**kwargs)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

    elapsed_s = time.perf_counter() - t0
    return _clean_times(transitions.jump_times), elapsed_s


def _score_one(time_arr, value, gt_times, gt_directions, method, cost, pen, n_bkps, min_size, jump, tolerance,
                max_call_seconds):
    # cpu_time_s: accurate per-call CPU (user+sys) seconds, via before/after delta --
    # isolates this one call regardless of what else this worker process has done.
    # peak_rss_gb: NOT isolated the same way -- ru_maxrss is a whole-process
    # high-water mark that never resets, so under joblib's reused worker processes
    # this is "this worker's peak RSS across every call it's handled so far", not
    # this call alone. Still useful as a rough signal for which (method, cost,
    # n_bkps) combos coincide with memory jumps; for a clean isolated number, see
    # the old profile_dynp_memory.py approach (one fresh subprocess per call).
    ru_before = resource.getrusage(resource.RUSAGE_SELF)
    try:
        detected, elapsed_s = _detect(time_arr, value, method, cost, pen, n_bkps, min_size, jump, max_call_seconds)
        metrics, timing_errors = evaluate_detection_by_direction(time_arr, value, detected, gt_times,
                                                                   gt_directions, tolerance, min_size)
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
    resolved to a concrete worker count -- needed here (unlike a plain Parallel call)
    because _evaluate_staged uses it as an actual batch size, and a batch size of -1
    doesn't mean anything."""
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
                     `_evaluate_staged`, batched + prunable -- see module
                     docstring). This is the only thing Optuna's sampler ever sees.
      - validation: what `evaluate_validation` scores a candidate winning config
                     against, once per config you're considering -- via the
                     original, unbatched `_evaluate`. Use this to compare
                     experimental choices and decide which search *procedure* to
                     finalize. Safe to check repeatedly while iterating.
      - test:       deliberately NOT loaded by this class at all. Loaded and
                     scored exactly once, by the separate evaluate_test.py script,
                     once every method/cost/range/sampler decision is already
                     locked in via validation.
    """

    def __init__(self, train_path, val_path, tolerance, n_jobs, objective_metric='warming',
                 pruning_batch_size=None, max_call_seconds=DEFAULT_MAX_CALL_SECONDS):
        (self.time_arr, self.db_train, self.gt_times_train, self.gt_dirs_train,
         self.n_train) = load_split(train_path)
        val_time_arr, self.db_val, self.gt_times_val, self.gt_dirs_val, self.n_val = load_split(val_path)
        # evaluate_validation (via _evaluate) reuses self.time_arr (TRAIN's) for every
        # series it scores, including validation ones -- val_time_arr itself is discarded
        # above, not a bug fix here (touching that would mean threading a second time
        # array through _evaluate/_score_one, shared unchanged with optuna_search.py).
        # This assertion instead makes the assumption that makes it safe -- train and
        # validation share one time axis -- loud and immediate if it's ever violated
        # (verified true for the current train/validation.npz: identical (6001,) arrays),
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
        # Batch size for _evaluate_staged's pruning checkpoints -- decoupled from n_jobs (the
        # actual joblib worker count) so --pruning-batch-size can require more evidence per
        # should_prune() check (e.g. 2x n_jobs -> a checkpoint only every other core-wave)
        # without changing parallelism. Defaults to n_jobs (one checkpoint per full
        # core-wave) when not explicitly set.
        self.batch_size = pruning_batch_size if pruning_batch_size is not None else _resolve_n_jobs(n_jobs)
        self.max_call_seconds = max_call_seconds   # hard per-series _detect timeout -- see CallTimeoutError
        self.objective_metric = objective_metric   # 'all' or 'warming' -- which f1_* column
                                                     # `objective` hands the sampler; see module docstring
        # Fixed (seeded, not re-shuffled per trial) permutation of train indices, used only
        # by `_evaluate_staged`. WilcoxonPruner pairs trial A's step-i score against the
        # best trial's step-i score, so step i must map to the SAME series every trial and
        # every resumed invocation, or the pairing is meaningless -- hence seeded, not
        # random each call. Shuffled (not raw 0..n_train-1 order) so an early prune
        # decision isn't biased by however train.npz's series happen to be grouped/stored
        # (e.g. by noise/transition regime) -- see module docstring.
        self.train_eval_order = np.random.RandomState(RANDOM_SEED).permutation(self.n_train)

    def _sample_params(self, trial):
        method = trial.suggest_categorical('method', METHODS)
        cost = trial.suggest_categorical(f'cost_{method}', KEPT_COSTS[method])
        min_size = trial.suggest_categorical('min_size', MIN_SIZE_CHOICES)

        lo, hi = PEN_BOUNDS[method][cost]
        pen = trial.suggest_float(f'pen_{method}_{cost}', lo, hi, log=True)

        # n_bkps stays None for every method now that Dynp (the only method that used
        # it, in place of pen) is removed -- see module docstring. _detect/_score_one
        # still accept it generically rather than hardcoding a pen-only path, in case
        # a future n_bkps-based method is ever added back.
        return dict(method=method, cost=cost, pen=pen, n_bkps=None, min_size=min_size, jump=JUMP)

    def _evaluate(self, params, db, gt_times_list, gt_dirs_list, seed_idx):
        """Unbatched full evaluation -- used by evaluate_validation/evaluate_train,
        which each score one already-chosen config in full, not part of the live
        search. No pruning applies here."""
        try:
            scored = Parallel(n_jobs=self.n_jobs)(
                delayed(_score_one)(self.time_arr, db[s], gt_times_list[s], gt_dirs_list[s],
                                     params['method'], params['cost'], params['pen'], params['n_bkps'],
                                     params['min_size'], params['jump'], self.tolerance, self.max_call_seconds)
                for s in seed_idx
            )
        except Exception as e:   # any worker crash
            return None, f"{type(e).__name__}: {e}"[:300]
        return pd.DataFrame(scored), None

    def _evaluate_staged(self, trial, params):
        """Used only by `objective` (the live search). Scores train in batches of
        self.batch_size series at a time, walking self.train_eval_order (not raw
        index order -- see __init__), reports each series' individual
        objective-metric score to Optuna right after its batch finishes, and checks
        trial.should_prune() once per batch. Lets WilcoxonPruner (wired in main())
        kill a trial that's already clearly worse than the current best after 1-2
        batches instead of paying for all n_train series -- full batch_size-way
        parallelism is preserved within each batch; only the between-batch wait is
        new versus optuna_search.py's single all-at-once Parallel call.

        ANY series failing -- one _score_one call raising (result['error'] is not
        None), or the whole Parallel call crashing (a worker dying) -- now raises
        SeriesEvaluationError immediately instead of continuing. Two reasons this is
        stricter than just skipping the failed series: (1) this dataset's search
        space (KEPT_COSTS/PEN_BOUNDS) is already constructed so every (method, cost)
        combination Optuna can sample is a priori valid, so an exception here is
        overwhelmingly more likely a real bug/resource issue than an "expected"
        invalid config -- there isn't a meaningful "expected failure" category to
        preserve; (2) a fabricated f1=0.0 for a crash is indistinguishable from a
        genuinely bad result once it's averaged into this trial's score, and can
        wrongly tank (or, upstream, wrongly influence a pruning decision toward
        killing) a config that's actually fine. main()'s study.optimize(...,
        catch=(SeriesEvaluationError,)) turns this into a normal FAILED trial
        (skipped, not selectable as best) rather than crashing the whole study.
        Scope note: `_evaluate`/evaluate_validation/evaluate_train (post-hoc,
        shared verbatim with optuna_search.py) are NOT changed by this -- only this
        live-search path, which is what can actually corrupt which config gets
        selected as best.

        Future improvement (not yet implemented -- see module docstring): use the
        raw series index as `step` (instead of batch position) together with a
        per-trial deterministic shuffle of iteration order, so every trial's
        early pruning checkpoints aren't drawn from the identical fixed first
        `batch_size` series every time (reduces fixed-prefix bias). Deferred
        because it requires confirming Optuna's trial.report() tolerates
        non-monotonic step values within a trial, which needs an environment with
        optuna installed to verify."""
        order = self.train_eval_order
        all_scored = []

        for batch_start in range(0, len(order), self.batch_size):
            batch_idx = order[batch_start: batch_start + self.batch_size]
            try:
                batch_scored = Parallel(n_jobs=self.n_jobs)(
                    delayed(_score_one)(self.time_arr, self.db_train[s], self.gt_times_train[s],
                                         self.gt_dirs_train[s], params['method'], params['cost'],
                                         params['pen'], params['n_bkps'], params['min_size'],
                                         params['jump'], self.tolerance, self.max_call_seconds)
                    for s in batch_idx
                )
            except Exception as e:   # any worker crash
                msg = f"{type(e).__name__}: {e}"[:300]
                trial.set_user_attr('error', msg)
                raise SeriesEvaluationError(f"batch starting at train_eval_order position "
                                             f"{batch_start} crashed: {msg}") from e

            errored = [result for result in batch_scored if result['error'] is not None]
            if errored:
                # Any series failing to score is treated as this whole trial failing --
                # see docstring above for why this is stricter than skip-and-continue.
                trial.set_user_attr('error', errored[0]['error'])
                trial.set_user_attr('n_errors_in_failing_batch', len(errored))
                raise SeriesEvaluationError(
                    f"{len(errored)}/{len(batch_scored)} series in batch starting at "
                    f"train_eval_order position {batch_start} failed to score; "
                    f"first error: {errored[0]['error']}"
                )

            all_scored.extend(batch_scored)
            for local_i, result in enumerate(batch_scored):
                # step = position in the FIXED, seeded train_eval_order permutation (not the
                # raw series index, and never re-shuffled per trial/resume) -- e.g. step=37
                # always means "the 38th series in train_eval_order", the exact same
                # physical series for every trial. That fixed identity is what makes
                # WilcoxonPruner's paired comparison below valid: step 37 of this trial vs.
                # step 37 of the best trial are always the same series, never apples-to-
                # oranges. main() saves train_eval_order to <outdir>/train_eval_order.csv if
                # you need to map a step number back to its raw series index.
                trial.report(result[f'f1_{self.objective_metric}'], step=batch_start + local_i)

            if trial.should_prune():
                raise TrialPruned()

        return pd.DataFrame(all_scored)

    def objective(self, trial):
        params = self._sample_params(trial)
        df = self._evaluate_staged(trial, params)   # raises SeriesEvaluationError/TrialPruned on
                                                      # failure/pruning -- see its docstring; both are
                                                      # caught by main()'s study.optimize(catch=...)

        trial.set_user_attr('mean_elapsed_s', float(df['elapsed_s'].mean()))
        trial.set_user_attr('mean_cpu_time_s', float(df['cpu_time_s'].mean()))
        # max, not mean -- peak_rss_gb is a high-water mark (see _score_one), so the
        # largest value seen in this trial's batch is the meaningful number, not an average.
        trial.set_user_attr('max_peak_rss_gb', float(df['peak_rss_gb'].max()))
        for label in ('all', 'warming', 'cooling'):
            trial.set_user_attr(f'mean_precision_{label}', float(df[f'precision_{label}'].mean()))
            trial.set_user_attr(f'mean_recall_{label}', float(df[f'recall_{label}'].mean()))
            trial.set_user_attr(f'mean_f1_{label}', float(df[f'f1_{label}'].mean()))
            timing = df[f'timing_mae_{label}'].dropna()
            trial.set_user_attr(f'mean_timing_mae_{label}', float(timing.mean()) if len(timing) else float('nan'))
        # df is guaranteed error-free here -- any series error already raised
        # SeriesEvaluationError above, so there's no 'n_errors' left to report.
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
        """Per-seed raw scores for one config on train (full, unbatched) -- separate
        from `objective`, which only ever keeps the mean and may run on a partial
        batch if pruned. Used once, after the study finishes, to save the winning
        config's full per-seed train breakdown."""
        seed_idx = np.arange(self.n_train)
        df, err = self._evaluate(params, self.db_train, self.gt_times_train, self.gt_dirs_train, seed_idx)
        if df is None:
            raise RuntimeError(f"train evaluation failed: {err}")
        return df


def evaluate_final_test(test_path, tolerance, n_jobs, params, max_call_seconds=DEFAULT_MAX_CALL_SECONDS):
    """Loads test data fresh (never held on OptunaSearch) and scores exactly one
    config against it. Call this once, after every design decision -- kept
    methods/costs, pen ranges, min_size/n_bkps grids, sampler settings -- is
    already finalized via validation. Do not change anything and call it again.

    Uses test.npz's OWN time array (test_time_arr, loaded fresh right here), not a
    caller-supplied one -- optuna_search.py's version of this function instead takes
    an external `time_arr` parameter that callers have in practice been passing
    train's time array (evaluate_test.py currently calls this with a mismatched
    argument count against that older signature, a separate pre-existing bug -- see
    conversation). Verified empirically that train/validation/test.npz currently all
    share an identical (6001,) time array, so this change has no effect on scores
    today; it removes the latent risk (and the extra, easy-to-get-wrong parameter)
    going forward without relying on that coincidence."""
    test_time_arr, db_test, gt_times_test, gt_dirs_test, n_test = load_split(test_path)
    seed_idx = np.arange(n_test)
    scored = Parallel(n_jobs=n_jobs)(
        delayed(_score_one)(test_time_arr, db_test[s], gt_times_test[s], gt_dirs_test[s],
                             params['method'], params['cost'], params['pen'], params['n_bkps'],
                             params['min_size'], params['jump'], tolerance, max_call_seconds)
        for s in seed_idx
    )
    return pd.DataFrame(scored)


def _save_convergence(trials_df, outdir, objective_metric):
    """Saves cumulative-best objective value vs. trial number, as both a CSV
    (optuna_convergence.csv) and a quick PNG (optuna_convergence.png) so the shape
    is visible without opening a notebook. Only COMPLETE trials are considered --
    PRUNED trials (this file's difference from optuna_search.py) are automatically
    excluded here since their state isn't 'COMPLETE', same as FAIL/RUNNING always
    were. state/number/value columns come straight from study.trials_dataframe(),
    which optuna_trials.csv is also saved from."""
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
    """Reconstruct the {method, cost, pen, n_bkps, min_size, jump} dict _evaluate
    expects from Optuna's flat, method/cost-namespaced best_trial.params."""
    p = best_trial.params
    method = p['method']
    cost = p[f'cost_{method}']
    min_size = p['min_size']
    pen = p[f'pen_{method}_{cost}']
    return dict(method=method, cost=cost, pen=pen, n_bkps=None, min_size=min_size, jump=JUMP)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-path", default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--val-path", default=DEFAULT_VAL_PATH)
    parser.add_argument("--n-trials", type=int, default=300,
                         help="hard cap on the TOTAL number of trials this study will ever run, "
                              "counted across every resumed invocation via --storage -- not trials "
                              "added by this invocation alone (default 300). If the study already has "
                              ">= this many trials when the run starts, this invocation runs zero more "
                              "and just re-reports the existing best/validation.")
    parser.add_argument("--timeout", type=int, default=None,
                         help="optional wall-clock safety net in seconds (default: disabled -- "
                              "--n-trials is the budget mechanism). Set this only if you also want a "
                              "hard stop under e.g. a SLURM time limit regardless of trial count.")
    parser.add_argument("--n-jobs", type=int, default=-1,
                         help="joblib n_jobs; -1 = all allocated cores. Also used as the batch size "
                              "for pruning-enabled staged evaluation (see module docstring).")
    parser.add_argument("--objective", choices=["all", "warming"], default="warming",
                         help="which F1 the Optuna sampler chases: 'warming' (default -- the "
                              "eventual application, NGRIP DO events, is warming-only) or 'all' "
                              "(direction-agnostic, the old single-F1 behavior). Also which per-series "
                              "score gets reported to WilcoxonPruner for pruning decisions.")
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--storage", default=None,
                         help="Optuna storage URL for a persistent, resumable study (default: a "
                              "sqlite file under --outdir). Rerunning with the same --storage and "
                              "--study-name reconnects to the existing study and keeps sampling on "
                              "top of its trial history.")
    parser.add_argument("--study-name", default=None,
                         help="must match across runs you want to resume into the same study. "
                              "Default: 'optuna_ruptures_search_pruning_<objective>' -- distinct from "
                              "optuna_search.py's default study names so the two never collide.")
    parser.add_argument("--p-threshold", type=float, default=0.05,
                         help="WilcoxonPruner's significance threshold (default 0.05 -- more "
                              "conservative than Optuna's own 0.1 default, deliberately: this "
                              "dataset's series vary in noise level/color and transition character, "
                              "so a stricter threshold requires stronger evidence before pruning a "
                              "trial that might just have hit a hard subset). Once a trial's train "
                              "series scored so far look significantly worse than the current best "
                              "trial's scores on those SAME series indices (paired Wilcoxon "
                              "signed-rank test -- see OptunaSearch._evaluate_staged / module "
                              "docstring), the trial is pruned before scoring the rest of train. "
                              "Lower = more conservative.")
    parser.add_argument("--n-startup-steps", type=int, default=None,
                         help="minimum number of this trial's series scores that must be reported "
                              "before WilcoxonPruner will even attempt to prune it (default: one full "
                              "pruning batch -- see --pruning-batch-size). NOTE: should_prune() is "
                              "only ever checked once per batch in this script (not once per series), "
                              "so setting this below --pruning-batch-size has NO effect -- the first "
                              "possible prune check already happens after --pruning-batch-size "
                              "observations have been reported (steps are 0-indexed, so that's "
                              "step=batch_size-1, not step=batch_size). Set this to a multiple of "
                              "--pruning-batch-size (e.g. 2x) to require more than one batch of "
                              "evidence before any trial can be pruned.")
    parser.add_argument("--max-call-seconds", type=int, default=DEFAULT_MAX_CALL_SECONDS,
                         help="hard wall-time cap (seconds) on a SINGLE series' _detect call "
                             f"(default {DEFAULT_MAX_CALL_SECONDS}s = 2h -- see DEFAULT_MAX_CALL_SECONDS; "
                             "a general safety net, not tuned to any specific method). "
                             "A call that exceeds this raises CallTimeoutError, which fails just that "
                             "one series (same path as any other detection crash -- see "
                             "_evaluate_staged), escalating to that trial being marked FAIL with the "
                             "reason recorded in its user_attrs. Protects against an unacceptably slow "
                             "or hung config consuming unbounded wall-clock; does not affect any call "
                             "that already finishes under the cap.")
    parser.add_argument("--pruning-batch-size", type=int, default=None,
                         help="how many train series are scored (in parallel, across --n-jobs) "
                              "between each should_prune() check (default: same as the resolved "
                              "--n-jobs, i.e. one checkpoint per full core-wave). Larger than --n-jobs "
                              "means a checkpoint only comes every N core-waves -- more evidence per "
                              "decision, less frequent pruning opportunities; do not set below "
                              "--n-jobs or cores sit idle within a batch. Must be > 0.")
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
    study_name = args.study_name or f"optuna_ruptures_search_pruning_{args.objective}"

    search = OptunaSearch(args.train_path, args.val_path, TOLERANCE, args.n_jobs,
                           objective_metric=args.objective, pruning_batch_size=args.pruning_batch_size,
                           max_call_seconds=args.max_call_seconds)
    print(f"train seeds: {search.n_train}   validation seeds: {search.n_val}", flush=True)
    print(f"max_call_seconds={args.max_call_seconds}s (hard per-series _detect timeout)", flush=True)

    order_path = os.path.join(outdir, "train_eval_order.csv")
    pd.DataFrame({'step': np.arange(search.n_train), 'train_series_index': search.train_eval_order}) \
        .to_csv(order_path, index=False)
    print(f"Saved step -> train series index mapping -> {order_path}", flush=True)

    budget_desc = (f"n_trials cap={args.n_trials} (total across all resumed invocations)"
                   if args.n_trials is not None else f"timeout={args.timeout}s")
    print(f"budget: {budget_desc}   n_jobs={args.n_jobs}   jump={JUMP}", flush=True)

    n_startup_steps = args.n_startup_steps if args.n_startup_steps is not None else search.batch_size
    print(f"pruning: WilcoxonPruner(p_threshold={args.p_threshold}, n_startup_steps={n_startup_steps}) "
          f"-- trials scored in batches of {search.batch_size} train series, checked for pruning "
          f"after each batch", flush=True)

    sampler = TPESampler(seed=RANDOM_SEED, multivariate=True, group=True, n_startup_trials=30)
    pruner = WilcoxonPruner(p_threshold=args.p_threshold, n_startup_steps=n_startup_steps)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner, storage=storage,
                                 study_name=study_name, load_if_exists=True)
    n_trials_before = len(study.trials)
    print(f"study {study_name!r} @ {storage!r}: {n_trials_before} trial(s) already recorded "
          f"before this run", flush=True)

    # Guards against resuming this study under conditions that no longer match what its
    # already-completed trials were scored/pruned against. n_train alone wouldn't catch
    # train.npz being swapped for a different file with the same series COUNT, so this
    # hashes the actual loaded array; RANDOM_SEED covers train_eval_order's permutation;
    # objective/tolerance/search-space/SCORING_VERSION cover the rest of what defines
    # "what a trial's score means" -- any change here would let two eras of the same
    # study be silently comparing scores that aren't actually comparable, corrupting both
    # WilcoxonPruner's pairing and the final best-trial selection.
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
        # Deliberately NOT `recorded is not None and recorded != value` -- missing
        # metadata (recorded is None, e.g. a study created by an earlier version of
        # this script before this check existed) must also fail, not silently resume
        # unchecked. Use a new --study-name for a study created before this check.
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
        # catch=(SeriesEvaluationError,) -- a series/batch failure (see
        # OptunaSearch._evaluate_staged) marks just that one trial FAIL and moves on
        # to the next trial, instead of the default (any uncaught exception aborts
        # the whole study.optimize() call and kills the job).
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
              f"data; if everything is FAILing, check the FAIL reason in each trial's "
              f"user_attrs (an 'error' entry) -- SeriesEvaluationError means every trial hit a "
              f"genuine execution error, likely a real bug, not a tuning issue. Stopping here "
              f"without writing best_params.json/best_train_raw.csv/best_validation_raw.csv.",
              flush=True)
        return

    best = study.best_trial
    print(f"\nBest trial #{best.number} (optimized on f1_{args.objective}): "
          f"train mean_f1_all={best.user_attrs.get('mean_f1_all'):.3f}  "
          f"mean_f1_warming={best.user_attrs.get('mean_f1_warming'):.3f}  "
          f"mean_f1_cooling={best.user_attrs.get('mean_f1_cooling'):.3f}")
    print(f"  params: {best.params}")
    print(f"  mean call time (train): {best.user_attrs.get('mean_elapsed_s'):.3f}s")

    best_full_params = _best_full_params(best)
    param_desc = (f"method={best_full_params['method']}  cost={best_full_params['cost']}  "
                  f"min_size={best_full_params['min_size']}  "
                  f"pen={format(best_full_params['pen'], '.3g')}")

    params_json_path = os.path.join(outdir, "best_params.json")
    with open(params_json_path, "w") as f:
        json.dump(dict(**best_full_params, trial_number=best.number,
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

    print(f"\ntest.npz was never opened this run. Once validation results are locked in, "
          f"score the winning config on held-out TEST with: "
          f"python evaluate_test.py --outdir {outdir}")


if __name__ == "__main__":
    main()
