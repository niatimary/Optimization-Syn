"""Per-(method, cost) penalty search bounds, (lo, hi), consumed directly by
optuna_search.py as the range for a continuous `suggest_float(..., log=True)`
search -- one shared Optuna study now optimizes method/cost/pen/min_size
jointly, replacing the old per-(method,cost) brute-force grid search
(run_grid_experiment.py + grid_common.py, retired to
_archive_old_grid_search/). min_size is likewise no longer sourced from here:
optuna_search.py searches it over one shared discrete set (MIN_SIZE_CHOICES)
for every method.

These bounds used to be the endpoints of a `logrange(lo, hi, n)` grid built
for that old brute-force search (binseg_grid_red_snr3.ipynb,
pelt_grid_red_snr3.ipynb, bottomup_grid_red_snr3.ipynb,
kernelcpd_grid_red_snr3.ipynb). Since a continuous search only ever needs the
two endpoints, not the discrete points between them, they're written directly
as (lo, hi) here.

Ranges below are set from that old grid search's actual winning `pen` per
(method, cost) -- see {Pelt,Binseg,BottomUp,KernelCPD}/results*/*_conclusion.csv
-- rather than the old grids' original (mostly arbitrary, and inconsistent
across methods for the same cost, e.g. mahalanobis's hi ranged 52-200) caps:

  cost          winning pen (Binseg / Pelt / BottomUp)   -> new (lo, hi)
  l1            4.61 / 4.61 / 4.61                        (0.05, 100)
  l2            2.169 / 2.169 / 2.169                      (0.05, 100)
  mahalanobis   16.8 / 19.3 / 29.1                         (1, 80)
  rank          13.9 / 26.8 / 26.8                         (1, 60)
  rbf           13.9 / 16.8 / 13.9  (+ KernelCPD 13.9)     (1, 40)
  normal        76.3 / 123.6 / 123.6                       (1, 200)  unchanged --
                already uses over half its range, no evidence to shrink it

Each hi leaves several x of headroom above the highest observed winner (the
old grid was discrete, so the true continuous optimum could sit a bit above
its nearest grid point) while cutting the dead space a wide-open cap like 200
wasted for costs (l1, l2 especially) whose real optimum sits under 5. If the
new, larger pooled dataset shifts an optimum toward either edge, widen that
cost's bound rather than assume the edge is real -- Optuna will report it via
an edge-hugging best_trial the same way the old grid's `on_edge` flag did.

Dynp (dynamic programming -- needs a separate n_bkps-based calibration scheme)
and the KS test are intentionally not included here.

`cosine` is dropped everywhere (as it is in the notebooks): for a univariate
series it structurally collapses to a sign comparison, blind to magnitude, so
it never adds information the other cost functions don't already capture.

Decisions baked into which costs appear below (see {Pelt,Binseg,BottomUp,
KernelCPD,Window}/results*/*_conclusion.csv and
Dynp/results/dynp_grid_red_snr3_conclusion.csv for the numbers):

  - Window:  dropped entirely. Underperforms every other method across every
             cost tested -- a single fixed window width can't match
             irregularly-spaced transitions, and its failure mode shifts
             unpredictably across costs (over-detection under l2/rbf/rank,
             severe under-detection under mahalanobis [precision 1.0, recall
             0.23], near-total non-detection under clinear [recall 0.006]).
             That points to a limitation of the approach, not something more
             tuning fixes.
  - clinear: dropped everywhere. Piecewise-linear, continuity-constrained
             cost -- can't represent an abrupt jump, blurs the transition
             instead. Its best score across every method it was tried with
             (0.533, BottomUp) is still below the typical score of the other
             costs.
  - linear:  dropped for Binseg/Pelt/BottomUp. With no covariate supplied,
             this cost has nothing to fit a coefficient against and
             mathematically reduces to a plain sum of squares, which is
             exactly additive across any split -- so for these penalty-based
             methods (which pick the split count by minimizing
             cost + pen * n_bkps), more segments never lower the total cost
             at any penalty, and test_f1 is exactly 0.0000 across the entire
             grid. KEPT for KernelCPD, where 'linear' instead selects a
             kernel (similarity-based, not a regression fit) -- unaffected by
             the missing-covariate issue, and tied for second-best result in
             the whole study (0.857 test_f1, essentially identical to Pelt's
             l2 result since ruptures' linear-kernel cost and l2 cost are
             equivalent up to scaling, which is also why their (lo, hi) match
             below).

NOTE: 'ar' shows the identical exact-0.0000-everywhere degeneracy as 'linear'
for Binseg/Pelt/BottomUp (best_pen always lands on the grid's lower edge,
1.0) -- flagged but not yet dropped from the dicts below since it wasn't part
of the linear/clinear/Window decision this file was last revised for.
"""

BINSEG_COSTS = {
    'ar':          (1, 100),
    'l1':          (0.05, 100),
    'l2':          (0.05, 100),
    'normal':      (1, 200),
    'rank':        (1, 60),
    'rbf':         (1, 40),
    'mahalanobis': (1, 80),
}

PELT_COSTS = {
    'ar':          (1, 100),
    'l1':          (0.05, 100),
    'l2':          (0.05, 100),
    'normal':      (1, 200),
    'rank':        (1, 60),
    'rbf':         (1, 40),
    'mahalanobis': (1, 80),
}

BOTTOMUP_COSTS = {
    'ar':          (1, 100),
    'l1':          (0.05, 100),
    'l2':          (0.05, 100),
    'normal':      (1, 200),
    'rank':        (1, 60),
    'rbf':         (1, 40),
    'mahalanobis': (1, 80),
}

# KernelCPD only supports 2 of ruptures' 3 kernels usefully here (`cosine`
# dropped, same reasoning as above) -- mirrors kernelcpd_grid_red_snr3.ipynb.
KERNELCPD_COSTS = {
    'linear': (0.05, 100),
    'rbf':    (1, 40),
}
