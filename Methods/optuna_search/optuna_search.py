#!/usr/bin/env python
"""
Joint Optuna (TPE) search over ruptures method x cost x pen x min_size.

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

Every individual detect-and-score call is timed (`elapsed_s`), and its CPU time
(`cpu_time_s`) and this worker's high-water-mark memory (`peak_rss_gb`) are also
recorded -- all three ride along in the Optuna trial's user attributes and the
saved trial dataframe, right next to precision/recall/f1, so the wall-clock and
resource cost of every (method, cost, params) combo this study actually tries is
visible after the fact, the same way F1 is. A hard per-call wall-time cap
(--max-call-seconds, default 2h -- see DEFAULT_MAX_CALL_SECONDS) protects against
a single genuinely hung/pathological config stalling a trial indefinitely; a call
that exceeds it raises CallTimeoutError, which fails just that one series the same
way any other detection crash does.

Direction-stratified scoring -- does optimizing on the full warming+cooling mix cost
anything on the actual target (NGRIP DO events, which are warming-only), or not?:
  Ammonyte's costs (l1, l2, mahalanobis, rank, rbf, normal) are all direction-blind --
  they flag a shift in the signal's statistics, not its sign -- so a pen/min_size
  tuned on the full mix of warming+cooling synthetic transitions isn't learning
  anything warming-specific in a categorical sense. But it IS implicitly a compromise
  over whatever mix of transition magnitudes/abruptness the training set contains, and
  the eventual target is warming-only. So every score is now broken into three:
  f1_all (direction-agnostic, exactly the old `f1`), f1_warming, and f1_cooling -- all
  three are always computed and logged every trial (user_attrs + the per-config train/
  validation/test CSVs), regardless of which one the sampler is chasing.

  --objective {all,warming} (default warming) picks which of f1_all/f1_warming
  `objective()` returns to the sampler -- i.e. which metric this run's search
  actually chases. Answering "does it matter?" takes two separate runs, one per
  objective (distinct --outdir/--study-name, since each is its own Optuna study):

      python optuna_search.py --objective all     --outdir optuna_results_all     --study-name search_all
      python optuna_search.py --objective warming --outdir optuna_results_warming --study-name search_warming

  then compare_objectives.py scores both saved configs on validation side by side --
  f1_all and f1_warming for each -- so the answer is a number (how much f1_warming is
  left on the table by optimizing on f1_all instead), not a guess.

  Ground truth is directionally unambiguous: `state_true` is a binary stadial(1)/
  interstadial(0) regime label (Melcher25 convention, db = higher is warmer), and a
  1->0 transition is warming, 0->1 is cooling -- see `transitions_from_states`, same
  convention as `build_train_test_split.ipynb`'s `warming_transition_times` and
  DO_events.csv's framing of DO onsets as warming transitions.

  Detections carry no such label -- ruptures only returns a location -- so each
  detected changepoint is classified post-hoc by comparing the local mean value just
  before vs. just after it in plain forward array/time order (see
  `_classify_detected_directions`). This deliberately does NOT reuse ammonyte's
  `ruptures_transitions.py` jump_values: that code's sign convention assumes a
  paleo age axis (index ascending = more recent -> older), which is backwards for
  this dataset's plain forward-simulation-time array -- confirmed empirically against
  state_true (a 1->0/warming transition here always shows the value increasing
  forward through the array). Re-deriving direction locally keeps the convention
  correct for whatever array order the input actually has, instead of silently
  inheriting an assumption that only holds for age-axis proxy data.

  f1_warming/f1_cooling are computed by filtering BOTH detections and ground truth to
  one direction and calling the existing (unmodified) `evaluate_detection` on that
  subset -- see `evaluate_detection_by_direction`. This is what keeps a
  cooling-classified detection from corrupting warming's precision: excluded from the
  warming-filtered detected set, it can't count as a warming false positive; instead
  it shows up as a false negative against the true warming transition it sat near
  (nothing detected on the warming side) and, separately, as a false positive against
  the true cooling transitions (nothing there to match it either) -- exactly the
  failure mode direction-blind scoring would otherwise hide.

Cross-validation was deliberately NOT added here: it protects against a model
overfitting to whatever data it was *fit* on, and nothing here is fit -- `ruptures`
just applies a fixed, Optuna-proposed (method, cost, pen, min_size) to each series
independently, so there's no fold-induced train/held-out split doing any real work,
and it would only add complexity (restructuring the evaluation loop into batches)
and, if the batch/fold size doesn't happen to divide evenly into --n-jobs, real
wall-clock slowdown for no corresponding benefit. validation (scored freely, to
compare search-procedure choices) already covers the actual role CV would otherwise
be reached for.

Early stopping is done at the STUDY level instead of per-trial: every trial still
scores all of train's series in one pass (no per-trial pruning -- see
optuna_search_pruning.py for the WilcoxonPruner-based per-trial alternative, run as
its own separate study for comparison), and a callback (see
`_make_early_stopping_callback`, wired in via --patience/--min-trials/--min-delta)
watches the sequence of COMPLETE trials and calls `study.stop()` once --patience of
them have passed with no improvement greater than --min-delta over the running
best. This targets a real, distinct cost this pipeline has: the primary budget is
`--n-trials` (a hard cap on the TOTAL number of trials the study will ever run,
across every resumed invocation -- see --n-trials below), and once TPE has
converged there is otherwise nothing that tells the study to stop before that cap
is reached, wasting compute on trials that were never going to beat the existing
best. --min-trials is a warm-up floor below which the callback is a no-op, so the
study can't stop during TPE's own random-startup phase or before every (method,
cost) branch of this conditional search space has been sampled a few times, and
--min-delta keeps a float-noise-sized F1 wobble from resetting the patience counter
and running the study indefinitely. Since `study.stop()` only ends the current
process's call to `study.optimize()`, this composes cleanly with the existing
resumable/checkpointed design (--storage/--study-name): the callback recomputes
best-so-far from the WHOLE persisted study history on every call (not just this
invocation), so resuming a study that already plateaued in a prior run will
correctly stop again quickly rather than re-running the rest of the --n-trials cap,
and --patience/--min-trials/--min-delta can simply be raised (or --patience set to
0) if you deliberately want the resumed run to keep exploring past where it
previously stopped. `--timeout` still exists as an optional wall-clock safety net
(e.g. under a SLURM time limit) but is disabled by default -- --n-trials is the
budget mechanism unless you explicitly opt into --timeout as well.

A resumed study is also checked for compatibility with the CURRENT invocation (see
`_search_space_fingerprint`/`_hash_file` and main()'s `study_config` block) --
methods/costs/pen bounds/min_size grid, the train data file's own content hash,
--objective, TOLERANCE, and a SCORING_VERSION constant are all recorded in the
study's own user_attrs the first time it's created, and checked again on every
resume; a mismatch (e.g. resuming under this study name after removing Dynp, or
after swapping train.npz) raises loudly instead of silently mixing scores from two
eras of the study that aren't actually comparable. Use a new --study-name if a
search-space/data change is intentional.

A live-search series failure (a `_score_one` call raising, or a whole batch's
Parallel dispatch crashing) now fails that ONE trial cleanly (raises
SeriesEvaluationError, caught by `study.optimize(..., catch=(SeriesEvaluationError,))`
so Optuna marks it FAIL and moves on) instead of silently averaging a fabricated
f1=0.0 into that trial's score -- a crash-caused zero is indistinguishable from a
genuinely bad result once averaged in, and could wrongly tank a config that's
actually fine.

Why TEST is scored on more than f1:
  f1_all/f1_warming/f1_cooling are all a fixed-tolerance (TOLERANCE=0.5) hit-or-miss
  match, and every series across train/validation/test is a draw from the same
  synthetic generative process -- there's no real distribution shift between splits
  the way there would be with, say, held-out real proxy records. That means a
  validation-vs-test f1 gap is a weak signal either way: a small gap doesn't prove
  the search generalized (there was barely any distribution shift to fail to
  generalize across), and it can't surface a real regression that f1's binary
  hit/miss threshold happens to mask (two configs, or two splits, can land on
  identical f1 while one is consistently detecting right next to the true
  transition and the other is scraping in at the tolerance edge). So `_score_one`
  also computes timing_mae_{all,warming,cooling} -- the mean absolute time offset of
  each matched detection from the true transition it matched, via
  `_matched_timing_errors` (same one-to-one closest-match rule as
  `evaluate_detection`, just returning the matched distances instead of collapsing
  them to precision/recall/f1). It's logged for every trial/split alongside f1 (free
  -- reuses the same detected/ground-truth arrays), but it's TEST's decision-making
  metric: compare validation vs. test timing_mae_warming, not f1_warming, when
  judging whether the chosen config's actual localization accuracy (which is what
  the eventual application -- dating DO onsets -- cares about) holds up on held-out
  data. See evaluate_test.py.

Data discipline -- three disjoint pools (train_test_split_data/{train,validation,test}.npz,
currently 60/30/30 series):
  - train (60):      every Optuna trial's objective is scored against this, and only this.
  - validation (30):  compare experimental choices against this as often as you like --
                      narrower/wider pen ranges, different min_size grid, a different
                      sampler, a different set of kept methods/costs, a different
                      --n-trials budget. Finalize the whole search procedure using
                      validation results only.
  - test (30):        never opened by this script at all. Scored exactly once, by the
                      separate evaluate_test.py, after the procedure above is fully
                      locked in via validation. See that file's docstring.

Outputs (all under --outdir, default optuna_results_<objective>/), each independently
re-usable without re-running anything upstream:
  - optuna_trials.csv:      one row per trial tried -- every (method, cost, pen,
                             min_size) combo with its train mean f1_all/f1_warming/
                             f1_cooling/precision_*/recall_*/elapsed_s/cpu_time_s/
                             peak_rss_gb (f1_warming is what the sampler actually
                             optimizes; the rest ride along as user_attrs). Enough
                             on its own for parameter-surface, heatmap, or
                             precision/recall plots across the whole search.
  - best_params.json:       the single winning config (method, cost, pen, min_size,
                             jump) plus which trial it was and its train
                             f1_warming/f1_all/f1_cooling. This is the only file
                             evaluate_test.py needs.
  - best_train_raw.csv:     per-series (per-seed) precision_{all,warming,cooling}/
                             recall_{all,warming,cooling}/f1_{all,warming,cooling}/
                             elapsed_s for the winning config on train.
  - best_validation_raw.csv: same, on validation.
  - (best_test_raw.csv is written by evaluate_test.py, not this script.)

Usage:
    python optuna_search.py --n-trials 130 --n-jobs 8
    python optuna_search.py --n-trials 130 --min-trials 50 --patience 30 --min-delta 0.01 --n-jobs 8
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
    instead of the study stalling on it indefinitely.

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
                                   # CallTimeoutError). A general safety net for every
                                   # kept method (all penalty-based, all fast in practice)
                                   # so a genuinely hung/pathological call still can't
                                   # stall a trial indefinitely, at essentially zero cost
                                   # since none of the kept methods should approach it.
SCORING_VERSION = "2"     # bump this any time _score_one/_detect/evaluate_detection_by_direction
                           # or method_configs.py's cost grids meaningfully change -- included in
                           # the resume-compatibility check (see main()) so a study resumed after
                           # such a change fails loudly instead of silently mixing scores from two
                           # different scoring definitions across its trial history.
DIRECTION_MIN_WINDOW = 10  # samples each side of a detected changepoint used to classify it
                            # warming/cooling. FIXED for every method/min_size (was max(this, min_size*2))
                            # so all configs are treated identically.

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
    # min_size) combos coincide with memory jumps.
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


class OptunaSearch:
    """Bundles the pooled train/validation data and evaluation logic. Three
    disjoint roles, never blurred:

      - train:      what `objective` scores every single trial against. This is
                     the only thing Optuna's sampler ever sees, and it's fine to
                     look at train scores/diagnostics as often as you want.
      - validation: what `evaluate_validation` scores a candidate winning config
                     against, once per config you're considering. Use this to
                     compare experimental choices (a narrower vs. wider pen
                     range, a different min_size grid, a different sampler, a
                     different set of kept methods/costs) and decide which
                     search *procedure* to finalize. Safe to check repeatedly
                     while iterating -- that's its whole job.
      - test:       deliberately NOT loaded by this class at all. It's loaded and
                     scored exactly once, by the separate evaluate_test.py script,
                     once every method/cost/range/sampler decision is already
                     locked in via validation. If you find yourself wanting to
                     look at a second test score to decide anything, that decision
                     belongs in validation instead.
    """

    def __init__(self, train_path, val_path, tolerance, n_jobs, objective_metric='warming',
                 max_call_seconds=DEFAULT_MAX_CALL_SECONDS):
        (self.time_arr, self.db_train, self.gt_times_train, self.gt_dirs_train,
         self.n_train) = load_split(train_path)
        val_time_arr, self.db_val, self.gt_times_val, self.gt_dirs_val, self.n_val = load_split(val_path)
        # evaluate_validation (via _evaluate) reuses self.time_arr (TRAIN's) for every
        # series it scores, including validation ones -- val_time_arr itself is discarded
        # above. This assertion makes the assumption that makes it safe -- train and
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
        self.max_call_seconds = max_call_seconds   # hard per-series _detect timeout -- see CallTimeoutError
        self.objective_metric = objective_metric   # 'all' or 'warming' -- which f1_* column
                                                     # `objective` hands the sampler; see module docstring

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
    config against it. Call this once, after every design decision -- kept
    methods/costs, pen ranges, min_size grid, sampler settings -- is already
    finalized via validation. Do not change anything and call it again."""
    _, db_test, gt_times_test, gt_dirs_test, n_test = load_split(test_path)
    seed_idx = np.arange(n_test)
    scored = Parallel(n_jobs=n_jobs)(
        delayed(_score_one)(time_arr, db_test[s], gt_times_test[s], gt_dirs_test[s],
                             params['method'], params['cost'], params['pen'], params['n_bkps'],
                             params['min_size'], params['jump'], tolerance, max_call_seconds)
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
        (n_startup_trials=30) or before every (method, cost) branch of this
        conditional search space has been sampled at least a few times.
      - min_delta: an "improvement" only resets the patience counter if it clears
        this margin, so the study doesn't keep running indefinitely chasing
        float-noise-sized F1 gains.

    Recomputes best-so-far from the full study.trials history (filtered to
    TrialState.COMPLETE) on every call rather than tracking running state, so it
    also does the right thing on a resumed study that already plateaued in an
    earlier run (a study resumed past its own plateau will re-trigger quickly).
    O(n) per call / O(n^2) over a whole study -- irrelevant next to per-trial
    evaluation cost here.

    `stop_info` is a dict the caller owns (mutated in place, not returned) --
    records whether *this* callback is what ended the study and at which trial,
    since study.optimize() otherwise gives no way to tell "stopped early" apart
    from "ran out of --timeout/--n-trials" after the fact. main() writes it into
    best_params.json so that distinction survives on disk, not just in the run's
    stdout log."""
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
    """Saves the thing you actually need to decide whether a patience rule is
    justified: cumulative-best objective value vs. trial number, as both a CSV
    (optuna_convergence.csv -- best_value_so_far column added to a value/state/
    number slice of trials_df, re-plottable however you like) and a quick PNG
    (optuna_convergence.png -- raw per-trial value as points, cumulative best as
    a line) so the shape is visible without opening a notebook. Only COMPLETE
    trials are considered (matches what `_make_early_stopping_callback` itself
    watches); state/number/value columns come straight from
    study.trials_dataframe(), which optuna_trials.csv is also saved from."""
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
    parser.add_argument("--n-trials", type=int, default=400,
                         help="hard cap on the TOTAL number of trials this study will ever run, "
                              "counted across every resumed invocation via --storage -- not trials "
                              "added by this invocation alone. If the study already has >= this "
                              "many trials when the run starts, this invocation runs zero more and "
                              "just re-reports the existing best/validation. Raise this (and rerun "
                              "the same --storage/--study-name) to let a resumed study keep exploring "
                              "past a prior cap.")
    parser.add_argument("--timeout", type=int, default=None,
                         help="optional wall-clock safety net in seconds (default: disabled -- "
                              "--n-trials is the budget mechanism). Set this only if you also want a "
                              "hard stop under e.g. a SLURM time limit regardless of trial count.")
    parser.add_argument("--n-jobs", type=int, default=-1,
                         help="joblib n_jobs; -1 = all allocated cores.")
    parser.add_argument("--objective", choices=["all", "warming"], default="warming",
                         help="which F1 the Optuna sampler chases: 'warming' (default -- the "
                              "eventual application, NGRIP DO events, is warming-only) or 'all' "
                              "(direction-agnostic, the old single-F1 behavior). f1_all/"
                              "f1_warming/f1_cooling are always computed and saved regardless -- "
                              "run once per --objective (with distinct --outdir/--study-name) "
                              "and use compare_objectives.py to see how much it actually matters.")
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--storage", default=None,
                         help="Optuna storage URL for a persistent, resumable study (default: a "
                              "sqlite file under --outdir). Rerunning with the same --storage and "
                              "--study-name reconnects to the existing study and keeps sampling on "
                              "top of its trial history -- safe to interrupt (SLURM time limit, OOM, "
                              "manual kill) and resume later with no lost trials.")
    parser.add_argument("--study-name", default=None,
                         help="must match across runs you want to resume into the same study. "
                              "Default: 'optuna_ruptures_search_<objective>' -- namespaced by "
                              "--objective so an --objective all run and an --objective warming "
                              "run never collide into the same study by accident.")
    parser.add_argument("--patience", type=int, default=80,
                         help="stop the whole study early once this many COMPLETE trials have passed "
                              "with no improvement greater than --min-delta over the running best "
                              "(default 80), instead of always running until the --n-trials cap is "
                              "reached -- see module docstring's 'Early stopping' section. Counts are "
                              "over the WHOLE persisted study history (all resumed runs combined), not "
                              "just this invocation. Set to 0 to disable and always run the full budget.")
    parser.add_argument("--min-trials", type=int, default=100,
                         help="early stopping cannot trigger until at least this many trials have "
                              "completed (default 100) -- a warm-up floor so the study can't stop "
                              "during TPE's random-startup phase or before every (method, cost) branch "
                              "has been sampled a few times. Only matters when --patience > 0.")
    parser.add_argument("--min-delta", type=float, default=0.01,
                         help="an improvement only resets the --patience counter if it exceeds this "
                              "margin over the running best f1 (default 0.01), so the study doesn't "
                              "run indefinitely chasing noise-sized gains. Only matters when "
                              "--patience > 0.")
    parser.add_argument("--max-call-seconds", type=int, default=DEFAULT_MAX_CALL_SECONDS,
                         help="hard wall-time cap (seconds) on a SINGLE series' _detect call "
                             f"(default {DEFAULT_MAX_CALL_SECONDS}s = 2h -- see DEFAULT_MAX_CALL_SECONDS; "
                             "a general safety net, not tuned to any specific method). A call that "
                             "exceeds this raises CallTimeoutError, which fails just that one series "
                             "(same path as any other detection crash), escalating to that trial "
                             "being marked FAIL with the reason recorded in its user_attrs.")
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
    study_name = args.study_name or f"optuna_ruptures_search_{args.objective}"

    search = OptunaSearch(args.train_path, args.val_path, TOLERANCE, args.n_jobs,
                           objective_metric=args.objective, max_call_seconds=args.max_call_seconds)
    print(f"train seeds: {search.n_train}   validation seeds: {search.n_val}", flush=True)
    print(f"max_call_seconds={args.max_call_seconds}s (hard per-series _detect timeout)", flush=True)
    budget_desc = (f"n_trials cap={args.n_trials} (total across all resumed invocations)"
                   if args.n_trials is not None else f"timeout={args.timeout}s")
    print(f"budget: {budget_desc}   n_jobs={args.n_jobs}   jump={JUMP}", flush=True)

    sampler = TPESampler(seed=RANDOM_SEED, multivariate=True, group=True, n_startup_trials=30)
    study = optuna.create_study(direction="maximize", sampler=sampler, storage=storage,
                                 study_name=study_name, load_if_exists=True)
    n_trials_before = len(study.trials)
    print(f"study {study_name!r} @ {storage!r}: {n_trials_before} trial(s) already recorded "
          f"before this run", flush=True)

    # Guards against resuming this study under conditions that no longer match what its
    # already-completed trials were scored against. n_train alone wouldn't catch
    # train.npz being swapped for a different file with the same series COUNT, so this
    # hashes the actual loaded array; objective/tolerance/search-space/SCORING_VERSION
    # cover the rest of what defines "what a trial's score means" -- any change here
    # would let two eras of the same study be silently comparing scores that aren't
    # actually comparable, corrupting best-trial selection.
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
        print("early stopping: disabled (--patience 0) -- will run the full --n-trials budget. "
              "Check optuna_convergence.png/.csv afterward to decide whether a patience rule is "
              "justified for future runs.", flush=True)

    t0 = time.perf_counter()
    if n_trials_to_run == 0:
        print(f"study already has {n_trials_before} trials >= --n-trials cap of {args.n_trials} -- "
              f"nothing to run this invocation; re-reporting the existing best.", flush=True)
    else:
        # catch=(SeriesEvaluationError,) -- a series/batch failure (see
        # OptunaSearch.objective) marks just that one trial FAIL and moves on to the
        # next trial, instead of the default (any uncaught exception aborts the
        # whole study.optimize() call and kills the job).
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
              f"reasons in each trial's user_attrs (an 'error' entry) -- SeriesEvaluationError "
              f"means every trial hit a genuine execution error, likely a real bug, not a tuning "
              f"issue. Stopping here without writing best_params.json/best_train_raw.csv/"
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

    print(f"\ntest.npz was never opened this run. Once validation results are locked in, "
          f"score the winning config on held-out TEST with: "
          f"python evaluate_test.py --outdir {outdir}")
    print(f"To see whether --objective all vs --objective warming actually matters, run the "
          f"other objective too (separate --outdir) and compare both best_params.json with: "
          f"python compare_objectives.py --all-params <...>/best_params.json "
          f"--warming-params <...>/best_params.json")


if __name__ == "__main__":
    main()
