"""Cached per-(params, alpha, seed) scoring for lerm_grid_red_snr3.ipynb.

Kept in a real module (not defined inline in the notebook) so joblib.Memory's cache
path is stable across kernels/SLURM jobs -- same rationale as dynp_scoring.py.

LERM's pipeline is staged (embed -> find_epsilon -> recurrence matrix -> Laplacian
eigenmaps -> smooth -> threshold detection). The first stages -- embedding, the
find_epsilon density search, and the O(n^3) eigendecomposition inside
laplacian_eigenmaps -- are the expensive part and depend only on
(m, target_density, w_size, w_incre, seed); block_size (smoothing) and alpha
(detection threshold) are cheap post-processing on top of that FI series.
`compute_fi_series` below is the disk-cached boundary between the two: everything
upstream of it runs once per (m, target_density, w_size, w_incre, seed) and is reused
for every block_size/alpha combination scored on top of it.

Known limitation: the cache is keyed at the FI-series level (after laplacian_eigenmaps
has already run), not at the eigenvector level, so changing only w_size/w_incre for an
otherwise-identical (m, target_density, seed) still reruns the full eigendecomposition
rather than reusing it. Splitting the eigendecomposition out as its own cached stage
would require duplicating laplacian_eigenmaps' internals instead of calling ammonyte's
public API -- only worth doing if Section 8's benchmarking shows the w_size/w_incre
grid is large enough for that recomputation to dominate total runtime.
"""
import copy
import functools

import numpy as np
import ammonyte as amt
from ammonyte.utils.metrics import evaluate_detection
from joblib import Memory

_memory = Memory("lerm_score_cache", verbose=0)


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


def eps_start_for_m(m, eps_start_base, m_reference):
    """Starting guess for find_epsilon's search, scaled by sqrt(m / m_reference).

    Median embedded pairwise distance grows roughly like sqrt(m) -- confirmed on this
    dataset (seed 0: median pairwise distance 0.69 -> 1.17 -> 1.51 for m=3,7,11, close
    to the sqrt(m) ratio). find_epsilon's search converges to the same target_density
    regardless of the starting guess, so this doesn't change *what* it converges to --
    only how many iterations it takes. Left unscaled, a guess tuned for one end of the
    m grid can be off by 2x+ at the other end, and that cost multiplies across every
    (target_density, seed) combination run at that m.
    """
    return eps_start_base * np.sqrt(m / m_reference)


@_memory.cache
def compute_fi_series(data_path, seed, m, target_density, w_size, w_incre,
                       eps_start_base, m_reference, density_tolerance, amp):
    """Expensive stage: embed -> find_epsilon -> recurrence matrix -> Laplacian
    eigenmaps -> raw (unsmoothed) Fisher Information series (ammonyte.RQARes).

    tau is left at its per-series auto default (TimeEmbeddedSeries' tau=None ->
    tau_search) rather than gridded or passed in -- see the notebook's intro markdown
    for why.
    """
    time_arr, db_noisy, _ = _load_data(data_path)
    series = amt.Series(time=time_arr, value=db_noisy[seed], label='grid', verbose=False)

    embedded = amt.TimeEmbeddedSeries(series, m=m, tau=None)
    eps_start = eps_start_for_m(m, eps_start_base, m_reference)
    eps_result = embedded.find_epsilon(eps=eps_start, target_density=target_density,
                                        tolerance=density_tolerance, amp=amp,
                                        parallelize=False, verbose=False)
    rm = eps_result['Output']
    return rm.laplacian_eigenmaps(w_size=w_size, w_incre=w_incre)


def _bootstrap_w(fi_len, w_fraction, w_min):
    """Bootstrap draw size for confidence_interval, scaled to *this* grid point's own
    (smoothed) FI-series length rather than one flat constant -- a fixed w would be a
    small, reasonable fraction of a long FI series at one w_size/w_incre grid point and
    most of the entire series at another. See the notebook's intro markdown."""
    return max(w_min, int(fi_len * w_fraction))


def score_one(data_path, params, alpha, seed, detection_tolerance,
              eps_start_base, m_reference, density_tolerance, amp,
              n_samples_bootstrap, w_fraction, w_min):
    """Full pipeline for one (params, alpha, seed) combo: reuse the cached FI series
    for (m, target_density, w_size, w_incre, seed), apply this combo's block_size
    smoothing and alpha-derived detection threshold (both cheap, uncached), score
    against ground truth."""
    _, _, gt_times = _load_data(data_path)

    try:
        fi_series = compute_fi_series(
            data_path, seed, params['m'], params['target_density'],
            params['w_size'], params['w_incre'],
            eps_start_base, m_reference, density_tolerance, amp)

        # smooth_series modifies its input in place (per its docstring) -- deep-copy
        # first so repeated block_size values scored against the same cached
        # fi_series (a joblib.Memory hit can hand back the same in-process object)
        # don't corrupt each other.
        smoothed = amt.utils.fisher.smooth_series(copy.deepcopy(fi_series),
                                                   block_size=params['block_size'])

        w = _bootstrap_w(len(smoothed.value), w_fraction, w_min)
        lower_pct = 100 * (alpha / 2)
        upper_pct = 100 * (1 - alpha / 2)

        transitions = smoothed.lerm_transitions(
            upper=upper_pct, lower=lower_pct, w=w, n_samples=n_samples_bootstrap)
        detected = _clean_times(transitions.jump_times)
    except Exception:
        return dict(precision=0.0, recall=0.0, f1=0.0, false_positives=np.nan)

    metrics = evaluate_detection(detected=detected, ground_truth=gt_times[seed],
                                  tolerance=detection_tolerance)
    return dict(precision=metrics.precision, recall=metrics.recall, f1=metrics.f1_score,
                false_positives=metrics.false_positives)
