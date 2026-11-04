"""Helpers for ensemble.ipynb (Optuna detectors -> candidate table -> logistic regression).

Warming only, tolerance 0.5 kyr, one-to-one matching as in ammonyte's evaluate_detection. Detection code lives in
detectors.py (it imports the Optuna scripts, so detections are identical to what each study scored).
"""
import os
import pickle
from collections import Counter

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.neighbors import KernelDensity

import detectors
from ammonyte.utils.metrics import evaluate_detection

TOLERANCE = 0.5


def load_split(npz_path):
    """(time, standardized series, true times, true directions, n_series); identical for ruptures and KS."""
    return detectors.module("ruptures").load_split(npz_path)


def run_split(npz_path, configs, cache_path=None, n_jobs=-1):
    """Returns (det, gt): det[series][cfg_id] = warming detection times, gt[series] = true warming onset times.
    configs: {cfg_id: detector params dict, see detectors.load_trials}. Detections are cached per config in cache_path."""
    time_arr, db, gt_times, gt_dirs, n = load_split(npz_path)
    cache = pickle.load(open(cache_path, "rb")) if cache_path and os.path.exists(cache_path) else {}
    cfg_used = cache.pop("_cfg", {})   # config values each cached entry was computed with
    # a cached entry is only valid if its config (not just its id, e.g. the trial number) is unchanged
    cache = {k: v for k, v in cache.items() if k in configs and cfg_used.get(k) == dict(configs[k])}
    todo = {k: v for k, v in configs.items() if k not in cache}
    if todo:
        out = Parallel(n_jobs=n_jobs)(
            delayed(lambda v: {k: detectors.detect_warming(time_arr, v, c) for k, c in todo.items()})(db[i]) for i in range(n))
        cache.update({k: [o[k] for o in out] for k in todo})
        if cache_path:
            pickle.dump({**cache, "_cfg": {k: dict(configs[k]) for k in cache}}, open(cache_path, "wb"))
    det = [{k: cache[k][i] for k in configs} for i in range(n)]
    gt = [np.asarray(t, float)[np.asarray(d) == "warming"] for t, d in zip(gt_times, gt_dirs)]
    return det, gt


def match_flags(detected, ground_truth, tol=TOLERANCE):
    """True per detection (input order) if it is a TP: same greedy one-to-one rule as ammonyte's evaluate_detection."""
    detected, gt = np.asarray(detected, float), np.sort(np.asarray(ground_truth, float))
    used, flags = np.zeros(len(gt), bool), np.zeros(len(detected), bool)
    for i in np.argsort(detected):
        d = np.where(used, np.inf, np.abs(gt - detected[i]))
        if d.size and d.min() <= tol:
            used[d.argmin()] = flags[i] = True
    return flags


def containment(strict, loose, tol=TOLERANCE):
    """Fraction of `strict`'s detections matched one-to-one to a `loose` detection within tol (1.0 = fully nested).
    One-to-one (same rule as match_flags) so several close `strict` detections cannot share one `loose` detection."""
    if len(strict) == 0:
        return np.nan
    return float(match_flags(strict, loose, tol).mean())


def nesting(a, b, tol=TOLERANCE):
    """Order-free nesting of two configs on one series: containment of the config with FEWER detections in the other."""
    small, large = (a, b) if len(a) <= len(b) else (b, a)
    return containment(small, large, tol)


def build_table(det, gt, groups, tol=TOLERANCE):
    """One row per candidate event. Detections of all configs in a series are pooled and a new candidate starts when
    a detection is more than `tol` after the first detection of the current one (position = median).
    groups: {cfg_id: group}; support_<group> = fraction of the group's configs detecting the candidate.
    label = 1 if the candidate matches a true event (one-to-one)."""
    size = Counter(groups.values())
    rows = []
    for s, (d, g) in enumerate(zip(det, gt)):
        cands = []
        for t, cid in sorted((t, cid) for cid, ts in d.items() for t in ts):
            if cands and t - cands[-1][0][0] <= tol:
                cands[-1].append((t, cid))
            else:
                cands.append([(t, cid)])
        pos = np.array([np.median([t for t, _ in c]) for c in cands])
        for c, p, y in zip(cands, pos, match_flags(pos, g, tol)):
            n_sup = Counter(groups[cid] for cid in {cid for _, cid in c})
            rows.append({"series": s, "pos": p, "label": int(y), **{f"support_{k}": n_sup[k] / size[k] for k in sorted(size)}})
    return pd.DataFrame(rows)


def pooled_counts(detected, gt, tol=TOLERANCE):
    """TP/FP/FN pooled over series (lists of detections / truths), via ammonyte evaluate_detection."""
    m = [evaluate_detection(d, g, tol) for d, g in zip(detected, gt)]
    return tuple(int(sum(getattr(x, k) for x in m)) for k in ("true_positives", "false_positives", "false_negatives"))


def event_counts(table, gt, accept, tol=TOLERANCE):
    """Event-level TP/FP/FN of the accepted candidates, re-matched one-to-one to the truth (FN includes true events
    no candidate covered)."""
    return pooled_counts([table["pos"][(table["series"] == s).values & accept] for s in range(len(gt))], gt, tol)


def event_curve(table, score, gt, tol=TOLERANCE, thresholds=np.round(np.arange(0.02, 0.99, 0.01), 2)):
    """Event-level precision/recall/F1 at each score threshold (recall is vs ALL true events)."""
    rows = []
    for t in thresholds:
        tp, fp, fn = event_counts(table, gt, score >= t, tol)
        p, r = tp / max(tp + fp, 1), tp / (tp + fn)
        rows.append(dict(threshold=t, TP=tp, FP=fp, FN=fn, precision=p, recall=r, f1=2 * p * r / (p + r) if tp else 0.0))
    return pd.DataFrame(rows)


def detection_density(times, grid, bandwidth=TOLERANCE / 2, weights=None):
    """Gaussian KDE of detection times on `grid`, scaled so each detection contributes `weight` (default 1): the curve
    height is the (weighted) number of detections per unit time, so a peak = many configs/candidates agree."""
    times = np.asarray(times, float)
    if len(times) == 0:
        return np.zeros(len(grid))
    w = np.ones(len(times)) if weights is None else np.asarray(weights, float)
    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(times[:, None], sample_weight=w)
    return np.exp(kde.score_samples(np.asarray(grid, float)[:, None])) * w.sum()
