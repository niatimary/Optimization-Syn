#!/usr/bin/env python
"""
Sanity-checks train/validation/test_manifest.csv after (re)running
build_noisier_split.py -- run this any time the splits are regenerated.

Checks, per split:
  1. (color, level) coverage -- every one of the 9 combos must have >=1 series, in
     EVERY split (train, validation, test). This is the check that would have caught
     validation's red/snr2 = 0 bug before it went unnoticed.
  2. Event-count regime (category) coverage -- train and validation must each include
     all three regimes (15event/25event/40event); test must include ONLY 25event
     (by design -- it's the regime that matches NGRIP's real ~25 DO events).
  3. Zero leakage -- no (source, orig_seed) pair appears in more than one split (a
     pair fixes the TRUE trajectory, shared across all 9 color/level noise draws
     within a regime -- see build_noisier_split.py Section 5's docstring).
  4. Within-split duplicate (source, orig_seed, color, level) rows -- informational,
     not a failure: train draws with replacement when a regime's reserved seed pool
     is smaller than what a (regime, combo) cell needs, so some repeats there are
     expected. Flagged here just so it's visible, not silent.

Usage:
    python check_split_coverage.py
    python check_split_coverage.py --split-dir /path/to/train_test_split_data
Exit code 0 if every REQUIRED check (1-3) passes for every split, 1 otherwise.
"""
import argparse
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

COLORS = ['white', 'pink', 'red']
LEVELS = ['k1', 'k0p5', 'k0p25']
ALL_COMBOS = {(c, l) for c in COLORS for l in LEVELS}
ALL_REGIMES = {'15event', '25event', '40event'}

# split name -> regimes it's REQUIRED to contain, exactly (no more, no less)
REQUIRED_REGIMES = {
    'train': ALL_REGIMES,
    'validation': ALL_REGIMES,
    'test': {'25event'},
}


def load_manifests(split_dir):
    manifests = {}
    for name in ('train', 'validation', 'test'):
        path = os.path.join(split_dir, f"{name}_manifest.csv")
        if not os.path.exists(path):
            sys.exit(f"ERROR: {path} not found -- run build_noisier_split.py first.")
        manifests[name] = pd.read_csv(path)
    return manifests


def check_combo_coverage(name, df, failures):
    present = set(zip(df['color'], df['level']))
    missing = ALL_COMBOS - present
    counts = df.groupby(['color', 'level']).size()
    thin = counts[counts < 2]
    status = "FAIL" if missing else "OK"
    print(f"  ({name}) color x level coverage: {status} -- {len(present)}/9 combos present"
          + (f"; MISSING: {sorted(missing)}" if missing else "")
          + (f"; thin (<2 series): {list(thin.index)}" if len(thin) else ""))
    if missing:
        failures.append(f"[{name}] missing (color, level) combo(s): {sorted(missing)}")
        return False
    return True


def check_regime_coverage(name, df, failures):
    present = set(df['source'].unique())
    required = REQUIRED_REGIMES[name]
    missing = required - present
    unexpected = present - required
    ok = not missing and not unexpected
    if missing:
        failures.append(f"[{name}] missing required regime(s): {sorted(missing)}")
    if unexpected:
        failures.append(f"[{name}] contains regime(s) it shouldn't: {sorted(unexpected)}")
    print(f"  ({name}) regime coverage: {'OK' if ok else 'FAIL'} -- "
          f"present={sorted(present)}  required={sorted(required)}")
    return ok


def check_validation_cells(df, failures):
    present = set(zip(df['source'], df['color'], df['level']))
    needed = {(r, c, l) for r in ALL_REGIMES for c, l in ALL_COMBOS}
    missing = needed - present
    print(f"  (validation) regime x color x level coverage: {'OK' if not missing else 'FAIL'} -- {len(present & needed)}/27 cells present")
    if missing:
        failures.append(f"[validation] missing (regime, color, level) cell(s): {sorted(missing)}")


def check_leakage(manifests, failures):
    keysets = {}
    for name, df in manifests.items():
        keysets[name] = set(zip(df['source'], df['orig_seed']))

    ok = True
    names = list(keysets)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            overlap = keysets[a] & keysets[b]
            if overlap:
                ok = False
                failures.append(f"LEAKAGE between {a} and {b}: {sorted(overlap)}")
    print(f"  cross-split (source, orig_seed) leakage check: {'OK' if ok else 'FAIL'}")
    return ok


def report_within_split_duplicates(name, df):
    dupe_counts = df.groupby(['source', 'orig_seed', 'color', 'level']).size()
    dupes = dupe_counts[dupe_counts > 1]
    if len(dupes):
        print(f"  ({name}) within-split duplicate rows (informational, not a failure -- "
              f"expected from draw-with-replacement when a regime's seed pool is small): "
              f"{len(dupes)} distinct (source, seed, color, level) tuple(s) repeated")
    else:
        print(f"  ({name}) within-split duplicates: none")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split-dir", default=HERE)
    args = parser.parse_args()

    manifests = load_manifests(args.split_dir)
    failures = []

    print("=== (color, level) coverage -- required in every split ===")
    for name, df in manifests.items():
        check_combo_coverage(name, df, failures)

    print("\n=== event-count regime (category) coverage ===")
    for name, df in manifests.items():
        check_regime_coverage(name, df, failures)

    print("\n=== validation covers all 27 regime x color x level cells ===")
    check_validation_cells(manifests['validation'], failures)

    print("\n=== cross-split leakage ===")
    check_leakage(manifests, failures)

    print("\n=== within-split duplicates (informational) ===")
    for name, df in manifests.items():
        report_within_split_duplicates(name, df)

    print()
    if failures:
        print(f"FAILED -- {len(failures)} problem(s):")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("ALL REQUIRED CHECKS PASSED.")
        sys.exit(0)


if __name__ == "__main__":
    main()
