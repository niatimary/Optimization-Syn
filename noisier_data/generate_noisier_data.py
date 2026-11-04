#!/usr/bin/env python
"""
Generate the NOISIER synthetic Melcher25 datasets (3 event-count regimes x 3 noise colors x 3 noise levels x N seeds).

Same dynamics, regimes and file layout as Generate_Data_{15,25,40}events.ipynb, with two differences:
  1. NOISIER levels:  sigma_noise = gap / k  with  k in {1, 0.5, 0.25}  (SNR_dB = 20*log10(k) = 0, -6, -12 dB).
     The previous levels were k in {4, 3, 2}  (+12, +9.5, +6 dB).
  2. NEW seeds: base seeds (true trajectories) and noise seeds are offset so nothing is shared with the
     previous datasets.

Output (one folder per regime, same naming scheme as before):
    noisier_data/regime_<15|25|40>event/noise_<color>_<level>.npz  (+ noise_grid_config.json, noise_grid_manifest.csv)

Run with an environment that has climatecritters + pyleoclim (e.g. pb_env):
    python generate_noisier_data.py
"""
import json
import os

import numpy as np
import pandas as pd

from climatecritters.model_critters.melcher25 import Melcher25, classify_bistable_states
from climatecritters.utils.noise import from_param

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- fixed dynamics, identical to the previous datasets (only sigma differs per regime) ----
GAMMA, ALPHA = 1.5, -0.5
T_SPAN, DT, Y0 = (0, 120), 0.02, [1.0, 0.0]
REGIME_SIGMA = {'15event': 0.37, '25event': 0.5, '40event': 0.68}   # process noise per regime (as before)

# ---- noise: color x level ----
NOISE_COLORS = {'white': 0.0, 'pink': 1.0, 'red': 2.0}               # CN spectral exponent beta
K_LEVELS = {'k1': 1.0, 'k0p5': 0.5, 'k0p25': 0.25}                   # sigma_noise = gap / k
N_SEEDS = 30                                                          # seeds 0-19 are identical to the earlier 20-seed version

# ---- seeds: disjoint from the previous data (which used base seeds 0-19, noise seeds 1000*i+s) ----
BASE_SEED_OFFSET = 5000
NOISE_SEED_OFFSET = 500000


def generate_colored_noise(beta, amplitude, length, seed):
    """Zero-mean noise with target spectral color (beta) and amplitude (std)."""
    surr = from_param(method='CN', noise_param=[beta], length=length, number=1, seed=seed)
    return surr.series_list[0].standardize(scale=amplitude).value


def simulate_condition(sigma, beta, amplitude, base_seed, noise_seed):
    model = Melcher25(sigma=sigma, gamma=GAMMA, alpha=ALPHA)
    output = model.integrate(t_span=T_SPAN, y0=Y0, method='heun_maruyama', dt=DT,
                             kwargs={'random_seed': base_seed, 'si': DT})
    time = output.time
    db_true = np.asarray(output.state_variables['db']).copy()
    true_states = np.asarray(output.diagnostic_variables['states']).copy()
    noise = generate_colored_noise(beta, amplitude, length=len(db_true), seed=noise_seed)
    output.add_noise('db', noise)
    db_noisy = np.asarray(output.state_variables['db'])
    return time, db_true, db_noisy, true_states, classify_bistable_states(db_noisy, alpha=ALPHA)


def main():
    gap_model = Melcher25(sigma=0.5, gamma=GAMMA, alpha=ALPHA)
    gap_model.integrate(t_span=T_SPAN, y0=Y0, method='heun_maruyama', dt=DT, kwargs={'random_seed': 0, 'si': DT})  # thresholds are set during integrate
    gap = float(gap_model.interstadial_threshold - gap_model.stadial_threshold)
    levels = {name: gap / k for name, k in K_LEVELS.items()}
    print(f"gap = {gap:.5f};  noise std per level: " + ", ".join(f"{n}={a:.4f}" for n, a in levels.items()))

    for regime, sigma in REGIME_SIGMA.items():
        out_dir = os.path.join(HERE, f"regime_{regime}")
        os.makedirs(out_dir, exist_ok=True)
        rows = []
        for ci, (color, beta) in enumerate(NOISE_COLORS.items()):
            for li, (level, amp) in enumerate(levels.items()):
                stacks = {k: [] for k in ('db_true', 'db_noisy', 'state_true', 'state_observed')}
                time_ref = None
                for s in range(N_SEEDS):
                    noise_seed = NOISE_SEED_OFFSET + 1000 * (ci * len(levels) + li) + s
                    time, db_true, db_noisy, st, so = simulate_condition(
                        sigma, beta, amp, base_seed=BASE_SEED_OFFSET + s, noise_seed=noise_seed)
                    time_ref = time if time_ref is None else time_ref
                    for k, v in zip(stacks, (db_true, db_noisy, st, so)):
                        stacks[k].append(v)
                    rows.append(dict(color=color, level=level, amplitude=amp, seed=s,
                                     n_true_transitions=int((np.diff(st) != 0).sum()),
                                     n_observed_transitions=int((np.diff(so) != 0).sum()),
                                     n_warming=int((np.diff(st) == -1).sum())))
                np.savez(os.path.join(out_dir, f"noise_{color}_{level}.npz"), time=time_ref,
                         seeds=np.arange(N_SEEDS), **{k: np.vstack(v) for k, v in stacks.items()})
                print(f"[{regime}] done {color}/{level} (std={amp:.4f})")
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, "noise_grid_manifest.csv"), index=False)
        with open(os.path.join(out_dir, "noise_grid_config.json"), 'w') as f:
            json.dump(dict(regime=regime, sigma=sigma, gamma=GAMMA, alpha=ALPHA, t_span=T_SPAN, dt=DT, y0=Y0,
                           noise_colors=NOISE_COLORS, k_levels=K_LEVELS, noise_levels=levels, gap=gap,
                           n_seeds=N_SEEDS, base_seed_offset=BASE_SEED_OFFSET,
                           noise_seed_offset=NOISE_SEED_OFFSET), f, indent=2)


if __name__ == "__main__":
    main()
