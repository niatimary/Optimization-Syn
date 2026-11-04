#!/usr/bin/env python
"""
Build train / validation / test splits from the NOISIER datasets (generate_noisier_data.py).

Same rules as build_train_test_split.ipynb (v2):
  * 3 event-count regimes (15 / 25 / 40 events). Train + validation pool all three; TEST uses ONLY 25event.
  * Leakage safety is enforced at the (regime, seed) level -- a seed fixes the TRUE trajectory, shared across all
    9 (color, level) files of a regime -- with disjoint seed pools per split, asserted before AND after drawing.
  * Test is filtered to a 20-30 warming-event band (NGRIP has ~25).
  * Sizes: test 30 (25event, 3-4 per (color, level)), train 60 (20 per regime), validation 30 (all 27 regime x color x level cells + 3 extra).
  * Validation uses its own RNG so it cannot perturb the train/test draws.

Output: noisier_data/train_test_split_data/{train,validation,test}.npz + *_manifest.csv  (same format as before;
the level names are k1 / k0p5 / k0p25 instead of snr2 / snr3 / snr4).

    python build_noisier_split.py
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "train_test_split_data")
os.makedirs(OUT_DIR, exist_ok=True)

REGIMES = ['15event', '25event', '40event']
COLORS = ['white', 'pink', 'red']
LEVELS = ['k1', 'k0p5', 'k0p25']
COMBOS = [(c, l) for c in COLORS for l in LEVELS]
N_SEEDS_PER_REGIME = 30
RANDOM_SEED = 42
EVENT_BAND_LOW, EVENT_BAND_HIGH = 20, 30
TEST_TARGET, TRAIN_TARGET, VAL_TARGET = 30, 60, 30               # series; validation = all 27 (regime, color, level) cells + 3 extra

# Disjoint seed pools per split (a seed fixes the TRUE trajectory). Test = 25event only, in-band seeds only.
TEST_SEED_POOL = list(range(14, 20)) + [25, 26, 28, 29]                                  # 25event only
TRAIN_SEED_POOLS = {'15event': list(range(0, 20)), '25event': list(range(0, 10)) + [24, 27], '40event': list(range(0, 20))}
VAL_SEED_POOLS = {'15event': list(range(20, 30)), '25event': list(range(10, 14)) + list(range(20, 24)), '40event': list(range(20, 30))}

# ---- pool definitions must be disjoint and fully partition each regime's seeds ----
ALL_SEEDS = set(range(N_SEEDS_PER_REGIME))
for r in REGIMES:
    tr, va = set(TRAIN_SEED_POOLS[r]), set(VAL_SEED_POOLS[r])
    te = set(TEST_SEED_POOL) if r == '25event' else set()
    assert not tr & va and not tr & te and not va & te, f"overlapping seed pools in {r}"
    assert tr | va | te == ALL_SEEDS, f"seed pools do not partition {r}"

# ---- load everything ----
data = {(r, c, l): np.load(os.path.join(HERE, f"regime_{r}", f"noise_{c}_{l}.npz"))
        for r in REGIMES for c, l in COMBOS}


def n_warming_per_seed(regime):
    return (np.diff(data[(regime, 'red', 'k1')]['state_true'], axis=1) == -1).sum(1)   # identical across combos


def make_row(regime, color, level, seed):
    d = data[(regime, color, level)]
    st = d['state_true'][seed]
    return dict(source=regime, color=color, level=level, seed=int(seed), time=d['time'],
                db_true=d['db_true'][seed], db_noisy=d['db_noisy'][seed], state_true=st,
                state_observed=d['state_observed'][seed],
                n_events=int((np.diff(st) != 0).sum()), n_warming=int((np.diff(st) == -1).sum()),
                n_cooling=int((np.diff(st) == 1).sum()))


def draw_seeds(candidates, n, rng):
    candidates = list(candidates)
    return rng.choice(candidates, size=n, replace=n > len(candidates))


def spread_counts(total, n_bins, rng):
    base = total // n_bins
    counts = [base] * n_bins
    for i in rng.choice(n_bins, size=total - base * n_bins, replace=False):
        counts[i] += 1
    return counts


# ---- test: 25event only, in-band seeds ----
w25 = n_warming_per_seed('25event')
TEST_SEED_CANDIDATES = sorted(s for s in TEST_SEED_POOL if EVENT_BAND_LOW <= w25[s] <= EVENT_BAND_HIGH)
assert len(TEST_SEED_CANDIDATES) >= 3, f"only {len(TEST_SEED_CANDIDATES)} test seeds in the {EVENT_BAND_LOW}-{EVENT_BAND_HIGH} band"
print("test seed pool warming counts:", {s: int(w25[s]) for s in TEST_SEED_POOL}, "-> in-band:", TEST_SEED_CANDIDATES)

rng = np.random.default_rng(RANDOM_SEED)
test_rows = []
for (color, level), n in zip(COMBOS, spread_counts(TEST_TARGET, len(COMBOS), rng)):
    for s in draw_seeds(TEST_SEED_CANDIDATES, n, rng):
        test_rows.append(make_row('25event', color, level, s))
assert len(test_rows) == TEST_TARGET

# ---- train: pooled across the 3 regimes ----
train_rows = []
per_regime = spread_counts(TRAIN_TARGET, len(REGIMES), rng)
for regime, n_train in zip(REGIMES, per_regime):
    for (color, level), nt in zip(COMBOS, spread_counts(n_train, len(COMBOS), rng)):
        for s in draw_seeds(TRAIN_SEED_POOLS[regime], nt, rng):
            train_rows.append(make_row(regime, color, level, s))
assert len(train_rows) == TRAIN_TARGET

# ---- validation: every (regime, color, level) cell exactly once (27), then the remaining series on randomly chosen extra cells; own RNG ----
val_rng = np.random.default_rng(RANDOM_SEED)
cells = [(r, c, l) for r in REGIMES for c, l in COMBOS]
extra = [cells[i] for i in val_rng.choice(len(cells), size=VAL_TARGET - len(cells), replace=False)]
val_rows, used = [], set()
for regime, color, level in cells + extra:
    pool = [s for s in VAL_SEED_POOLS[regime] if (regime, color, level, s) not in used]
    s = int(val_rng.choice(pool)); used.add((regime, color, level, s))
    val_rows.append(make_row(regime, color, level, s))
assert len(val_rows) == VAL_TARGET
assert {(r['source'], r['color'], r['level']) for r in val_rows} == set(cells), "validation misses a (regime, color, level) cell"

# ---- zero leakage on the rows actually drawn ----
keys = {n: {(r['source'], r['seed']) for r in rows} for n, rows in (('train', train_rows), ('validation', val_rows), ('test', test_rows))}
assert not keys['train'] & keys['validation'], "LEAKAGE train/validation"
assert not keys['train'] & keys['test'], "LEAKAGE train/test"
assert not keys['validation'] & keys['test'], "LEAKAGE validation/test"
print("VERIFIED: zero (regime, seed) overlap between train, validation and test.")


def save_split(rows, name):
    arrs = {k: np.vstack([r[k] for r in rows]) for k in ('db_true', 'db_noisy', 'state_true', 'state_observed')}
    np.savez(os.path.join(OUT_DIR, f"{name}.npz"), time=rows[0]['time'], seeds=np.arange(len(rows)), **arrs)
    pd.DataFrame([{'row': i, 'source': r['source'], 'color': r['color'], 'level': r['level'], 'orig_seed': r['seed'],
                   'n_events': r['n_events'], 'n_warming': r['n_warming'], 'n_cooling': r['n_cooling']}
                  for i, r in enumerate(rows)]).to_csv(os.path.join(OUT_DIR, f"{name}_manifest.csv"), index=False)


for name, rows in (('train', train_rows), ('validation', val_rows), ('test', test_rows)):
    save_split(rows, name)
    print(f"saved {name}: {len(rows)} series, mean warming events {np.mean([r['n_warming'] for r in rows]):.1f}")
