"""Detector registry for the ensemble: for each Optuna study, how to read its trials and how to reproduce its detections.

A detector is one unit of the ensemble. All ruptures algorithms (Pelt, KernelCPD, Binseg, ...) together are ONE detector
("ruptures"; the algorithm stays visible in `family`), the KS test is one detector, LERM (+ KernelCPD on the Fisher information) is one detector. Nothing is pooled across detectors before the logistic regression.
Detection and direction code is imported from the Optuna scripts, so detections are identical to what each study scored.
"""
import copy
import importlib.util
import os
import sys
import warnings

import ammonyte as amt
import numpy as np
import optuna

HERE = os.path.dirname(os.path.abspath(__file__))
METHODS_DIR = os.path.dirname(HERE)

# kind -> (folder, script, study.db). All studies must be scored on the same (noisier) train split.
SOURCES = {
    "ruptures": ("optuna_search", "optuna_search.py", os.path.join("optuna_search", "results", "study.db")),
    "kstest": ("optuna_search_kstest", "optuna_search_kstest.py", os.path.join("optuna_search_kstest", "noisier_results", "study.db")),
    "lerm": ("optuna_search_lerm", "optuna_search_lerm_kcpd.py", os.path.join("optuna_search_lerm", "results_lerm_kcpd", "study.db")),
}
_modules = {}


def module(kind):
    """The Optuna script of `kind`, loaded under its own name (ruptures and KS scripts both define _detect/load_split)."""
    if kind not in _modules:
        folder, script, _ = SOURCES[kind]
        if METHODS_DIR not in sys.path:
            sys.path.insert(0, METHODS_DIR)             # method_configs.py, imported by optuna_search.py
        spec = importlib.util.spec_from_file_location(folder, os.path.join(METHODS_DIR, folder, script))
        _modules[kind] = mod = importlib.util.module_from_spec(spec)
        sys.modules[folder] = mod
        spec.loader.exec_module(mod)
    return _modules[kind]


def _complete_trials(kind):
    storage = f"sqlite:///{os.path.join(METHODS_DIR, SOURCES[kind][2])}"
    study = optuna.load_study(study_name=optuna.study.get_all_study_names(storage)[0], storage=storage)
    df = study.trials_dataframe().rename(columns={
        "number": "trial", "user_attrs_mean_precision_warming": "P", "user_attrs_mean_recall_warming": "R",
        "user_attrs_mean_f1_warming": "F1"})
    return df[df["state"] == "COMPLETE"].dropna(subset=["P", "R"]).copy()


def load_trials(kind):
    """One row per completed trial: cfg (unique id), detector, family (configs that can be nested), P, R, F1 (train, warming)
    and params (dict passed to detect_warming)."""
    df = _complete_trials(kind)
    if kind == "ruptures":
        df["cost"] = df.filter(like="params_cost_").bfill(axis=1).iloc[:, 0]
        df["pen"] = df.apply(lambda r: r[f"params_pen_{r['params_method']}_{r['cost']}"], axis=1)
        df["detector"] = "ruptures"             # all ruptures algorithms are ONE detector (one pooled Pareto front, one support feature)
        df["family"] = df["params_method"] + "_" + df["cost"]
        df["cfg"] = df["params_method"] + "|" + df["cost"] + "|t" + df["trial"].astype(str)
        df["params"] = [dict(kind=kind, method=r.params_method, cost=r.cost, pen=float(r.pen), min_size=int(r.params_min_size))
                        for r in df.itertuples()]
    elif kind == "kstest":
        keys = ["n_c", "w_min", "w_max", "n_w", "d_c", "s_c"]
        df["detector"] = df["family"] = "KS"
        df["cfg"] = "KS|t" + df["trial"].astype(str)
        df["params"] = [dict(kind=kind, **{k: getattr(r, f"params_{k}") for k in keys}) for r in df.itertuples()]
    elif kind == "lerm":
        # the study stores only the best penalty multiplier k of each trial, so a config = trial parameters + that k
        df["detector"] = df["family"] = "LERM"
        df["cfg"] = "LERM|t" + df["trial"].astype(str)
        df["params"] = [dict(kind=kind, m=r.params_m, tau=r.params_tau, target_density=r.params_target_density, w_size=r.params_w_size,
                             w_incre=r.user_attrs_w_incre, k=r.user_attrs_best_k) for r in df.itertuples()]
    else:
        raise NotImplementedError(kind)
    return df[["cfg", "trial", "detector", "family", "P", "R", "F1", "params"]]


def detect_warming(time_arr, value, cfg, rng=None):
    """Warming-classified detections of one config on one series; rng=(lo, hi) keeps only detections inside it."""
    kind = cfg["kind"]
    mod = module(kind)
    if kind == "ruptures":
        min_size = int(cfg["min_size"])
        det, _ = mod._detect(time_arr, value, cfg["method"], cfg["cost"], float(cfg["pen"]), None, min_size, mod.JUMP,
                             mod.DEFAULT_MAX_CALL_SECONDS)
    elif kind == "kstest":
        min_size = int(cfg["n_c"])
        det, _ = mod._detect(time_arr, value, float(cfg["w_min"]), float(cfg["w_max"]), int(cfg["n_w"]), float(cfg["d_c"]),
                             min_size, float(cfg["s_c"]), mod.DEFAULT_MAX_CALL_SECONDS)
    elif kind == "lerm":
        # same steps as optuna_search_lerm_kcpd._score_one: embed -> Fisher information -> smooth -> standardize -> KernelCPD
        try:
            fi = mod._fisher_series(time_arr, value, int(cfg["m"]), int(cfg["tau"]), float(cfg["target_density"]),
                                    int(cfg["w_size"]), int(cfg["w_incre"]))
        except RuntimeError as e:     # find_epsilon can fail (the study lost ~16% of large-m trials to it): no detections
            warnings.warn(f"LERM {dict(cfg)} failed on one series: {e}")
            return np.array([])
        fi_std = mod._standardized_fi(amt.utils.fisher.smooth_series(copy.deepcopy(fi), block_size=mod.BLOCK_SIZE))
        det, _ = mod._detect_kcpd(fi_std, float(np.median(np.diff(time_arr))), float(cfg["k"]))
        directions = lambda d: mod._classify_detected_directions(value, time_arr, d)
    else:
        raise NotImplementedError(kind)
    det = np.asarray(det, float)
    if rng is not None:
        det = det[(det >= rng[0]) & (det <= rng[1])]
    if kind != "lerm":
        directions = lambda d: mod._classify_detected_directions(value, time_arr, d, min_size)
    return np.sort(det[directions(det) == "warming"])
