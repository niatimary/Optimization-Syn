"""Shared grid-search engine for ruptures method x cost-function experiments.

Ported from the per-method notebooks (binseg_grid_red_snr3.ipynb,
pelt_grid_red_snr3.ipynb, bottomup_grid_red_snr3.ipynb,
kernelcpd_grid_red_snr3.ipynb, window_grid_red_snr3.ipynb) so a non-interactive
SLURM run and the notebooks share identical scoring / train-test-split /
selection logic. Kept in a real module (not inline in a notebook) so joblib
worker processes can pickle these functions by reference, and so a script run
can import it directly.

Train/test split methodology (same as every notebook this was ported from):
grid search picks the winning (cost, pen, min_size[, width]) combo by mean f1
over a train seed pool only; that single combo is then applied once to a
held-out test seed pool for the honest, reported numbers. A pen sweep at the
winning min_size (and width, for Window) on the test pool gives the
precision/recall tradeoff curve.
"""
import time as _time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers the 3d projection)
import ammonyte as amt
from ammonyte.utils.metrics import evaluate_detection
from joblib import Parallel, delayed


def logrange(lo, hi, n):
    return list(np.unique(np.round(np.logspace(np.log10(lo), np.log10(hi), n), 3)))


def transition_times_from_states(time, states):
    """True transition times (both directions) from a state-label array.

    Mirrors the same helper in cv_noise_sensitivity.py / the per-method
    notebooks: any stadial/interstadial change counts as ground truth,
    regardless of direction.
    """
    change_idx = np.where(np.diff(states) != 0)[0] + 1
    return time[change_idx]


def _clean_times(times):
    times = np.asarray(times, dtype=float)
    return times[~np.isnan(times)]


def load_condition(data_path):
    d = np.load(data_path)
    time_arr = d['time']
    db_noisy = d['db_noisy']
    state_true = d['state_true']
    n_seeds = db_noisy.shape[0]
    gt_times = [transition_times_from_states(time_arr, state_true[s]) for s in range(n_seeds)]
    return time_arr, db_noisy, gt_times, n_seeds


def seed_pool(n_seeds_total, pool_size, random_state):
    """Fixed subsample of `pool_size` seed indices out of n_seeds_total."""
    rng = np.random.RandomState(random_state)
    return np.sort(rng.choice(n_seeds_total, size=pool_size, replace=False))


def outer_split(pool, n_train, n_test, random_state):
    """Split `pool` into disjoint train/test seed-index arrays, once."""
    rng = np.random.RandomState(random_state)
    shuffled = rng.permutation(pool)
    train_idx = np.sort(shuffled[:n_train])
    test_idx = np.sort(shuffled[n_train:n_train + n_test])
    return train_idx, test_idx


def split_for_cost(cost, n_seeds_total, heavy_costs, light_budget, heavy_budget, random_state):
    budget = heavy_budget if cost in heavy_costs else light_budget
    pool = seed_pool(n_seeds_total, budget['pool_size'], random_state=random_state)
    return outer_split(pool, budget['n_train'], budget['n_test'], random_state=random_state)


class GridRunner:
    """Bundles one (algo, dataset) run's shared state -- data, tolerance, seed
    budgets -- so `run_one_cost` only needs (cost, pens, min_sizes, ...) per
    call, the same call shape as each notebook's Section 7 cells.
    """

    def __init__(self, algo, data_path, tolerance, random_seed, n_jobs,
                 heavy_costs, light_budget, heavy_budget):
        self.algo = algo
        self.data_path = data_path
        self.tolerance = tolerance
        self.random_seed = random_seed
        self.n_jobs = n_jobs
        self.heavy_costs = heavy_costs
        self.light_budget = light_budget
        self.heavy_budget = heavy_budget
        self.time_arr, self.db_noisy, self.gt_times, self.n_seeds = load_condition(data_path)
        self.results = {}   # every run_one_cost result lands here, keyed by cost name

    def _detect(self, params, seed):
        series = amt.Series(time=self.time_arr, value=self.db_noisy[seed], label='grid', verbose=False)
        t0 = _time.perf_counter()
        transitions = series.ruptures(algo=self.algo, **params)
        elapsed_s = _time.perf_counter() - t0
        return _clean_times(transitions.jump_times), elapsed_s

    def _score_one(self, params, seed):
        try:
            detected, elapsed_s = self._detect(params, seed)
        except Exception as e:
            # Same fallback the notebooks use (some costs, e.g. ar/linear, are
            # documented to legitimately score 0 everywhere). `error` is purely
            # diagnostic -- it does not affect any statistic -- so a systemic
            # wiring bug (every call raising the same exception) is
            # distinguishable after the fact from a genuine all-zero result
            # (which would have error=None throughout).
            return dict(precision=0.0, recall=0.0, f1=0.0, false_positives=np.nan,
                        error=f"{type(e).__name__}: {e}"[:300], elapsed_s=np.nan)
        m = evaluate_detection(detected=detected, ground_truth=self.gt_times[seed], tolerance=self.tolerance)
        return dict(precision=m.precision, recall=m.recall, f1=m.f1_score, false_positives=m.false_positives,
                    error=None, elapsed_s=elapsed_s)

    def _split_for_cost(self, cost):
        return split_for_cost(cost, self.n_seeds, self.heavy_costs, self.light_budget,
                               self.heavy_budget, self.random_seed)

    def grid_search_train(self, param_grid, train_idx):
        """Score every (param, seed in train_idx) combo, aggregate per param, pick the
        winner by mean f1 over the train pool only."""
        jobs = [(pi, params, s) for pi, params in enumerate(param_grid) for s in train_idx]
        scored = Parallel(n_jobs=self.n_jobs)(
            delayed(self._score_one)(params, s) for pi, params, s in jobs
        )
        rows = []
        for (pi, params, s), res in zip(jobs, scored):
            row = dict(params)
            row.update(pi=pi, seed=int(s), split='train', **res)
            rows.append(row)
        train_raw = pd.DataFrame(rows)

        param_cols = list(param_grid[0].keys())
        train_summary = (train_raw.groupby(param_cols)
                          .agg(mean_precision=('precision', 'mean'), std_precision=('precision', 'std'),
                               mean_recall=('recall', 'mean'), std_recall=('recall', 'std'),
                               mean_f1=('f1', 'mean'), std_f1=('f1', 'std'),
                               mean_fp=('false_positives', 'mean'),
                               mean_elapsed_s=('elapsed_s', 'mean'))
                          .reset_index())

        best_row = train_summary.loc[train_summary['mean_f1'].idxmax()]
        best_params = {c: best_row[c] for c in param_cols}
        return best_params, best_row, train_summary, train_raw

    def evaluate_test(self, best_params, test_idx):
        """Apply best_params to the untouched-until-now test pool, once."""
        scored = Parallel(n_jobs=self.n_jobs)(
            delayed(self._score_one)(best_params, s) for s in test_idx
        )
        rows = []
        for s, res in zip(test_idx, scored):
            row = dict(best_params)
            row.update(seed=int(s), split='test', **res)
            rows.append(row)
        test_raw = pd.DataFrame(rows)
        return (test_raw['precision'].mean(), test_raw['recall'].mean(),
                test_raw['f1'].mean(), test_raw['false_positives'].mean(), test_raw)

    def pen_sweep_test(self, cost, pen_grid, min_size, test_idx, extra_params=None):
        """Sweep pen at the winning min_size (and any other winning params, e.g.
        width/jump, passed via extra_params), test seeds only -- precision/recall
        tradeoff curve for this cost function (the pen-analog of a PR curve)."""
        extra_params = extra_params or {}
        jobs = [(pen, s) for pen in pen_grid for s in test_idx]
        scored = Parallel(n_jobs=self.n_jobs)(
            delayed(self._score_one)(dict(cost=cost, pen=pen, min_size=min_size, **extra_params), s)
            for pen, s in jobs
        )
        rows = []
        for (pen, s), res in zip(jobs, scored):
            rows.append(dict(cost=cost, pen=pen, min_size=min_size, seed=int(s), split='test_sweep', **res))
        sweep_raw = pd.DataFrame(rows)
        sweep_summary = (sweep_raw.groupby('pen')
                          .agg(mean_precision=('precision', 'mean'), mean_recall=('recall', 'mean'),
                               mean_f1=('f1', 'mean'), mean_fp=('false_positives', 'mean'))
                          .reset_index())
        return sweep_summary, sweep_raw

    def run_one_cost(self, cost, pens, min_sizes, widths=None, jumps=None):
        t0 = _time.perf_counter()
        train_idx, test_idx = self._split_for_cost(cost)
        budget = 'HEAVY' if cost in self.heavy_costs else 'light'

        axes = [('pen', pens), ('min_size', min_sizes)]
        if widths is not None:
            axes.append(('width', widths))
        if jumps is not None:
            axes.append(('jump', jumps))

        param_grid = [{'cost': cost}]
        for name, values in axes:
            param_grid = [dict(g, **{name: v}) for g in param_grid for v in values]

        best_params, train_row, train_summary, train_raw = self.grid_search_train(param_grid, train_idx)
        test_p, test_r, test_f1, test_fp, test_raw = self.evaluate_test(best_params, test_idx)

        sweep_extra = {}
        if widths is not None:
            sweep_extra['width'] = best_params['width']
        if jumps is not None:
            sweep_extra['jump'] = best_params['jump']
        sweep_summary, sweep_raw = self.pen_sweep_test(cost, pens, best_params['min_size'], test_idx,
                                                        extra_params=sweep_extra or None)

        on_edge = (best_params['pen'] in (min(pens), max(pens))
                   or best_params['min_size'] in (min(min_sizes), max(min_sizes)))
        if widths is not None:
            on_edge = on_edge or best_params['width'] in (min(widths), max(widths))

        conclusion_row = dict(
            cost=cost, best_pen=best_params['pen'], best_min_size=best_params['min_size'],
            best_width=best_params.get('width', np.nan), best_jump=best_params.get('jump', 1),
            train_f1=train_row['mean_f1'], train_precision=train_row['mean_precision'],
            train_recall=train_row['mean_recall'], mean_train_call_s=train_row['mean_elapsed_s'],
            test_f1=test_f1, test_precision=test_p, test_recall=test_r, test_fp=test_fp,
            mean_test_call_s=test_raw['elapsed_s'].mean(),
            pen_range=f"{min(pens):.3g}-{max(pens):.3g}",
            min_size_range=f"{min(min_sizes):.0f}-{max(min_sizes):.0f}",
            width_range=(f"{min(widths):.0f}-{max(widths):.0f}" if widths is not None else ""),
            on_edge=on_edge, seed_budget=budget,
            n_train_seeds=len(train_idx), n_test_seeds=len(test_idx),
            train_seeds=",".join(map(str, train_idx)), test_seeds=",".join(map(str, test_idx)),
        )

        result = dict(train_idx=train_idx, test_idx=test_idx, best_params=best_params,
                      train_summary=train_summary, train_raw=train_raw, test_raw=test_raw,
                      sweep_summary=sweep_summary, sweep_raw=sweep_raw, conclusion_row=conclusion_row)

        elapsed = _time.perf_counter() - t0
        edge_note = "  <-- ON GRID EDGE" if on_edge else ""
        width_note = f" width={best_params['width']:>4.0f}" if widths is not None else ""
        print(f"[{budget:5s}] {cost:12s}: pen={best_params['pen']:>8.3g} "
              f"min_size={best_params['min_size']:>3.0f}{width_note} jump={best_params.get('jump', 1)}  "
              f"train_f1={train_row['mean_f1']:.3f}  test_f1={test_f1:.3f}  ({elapsed:.0f}s){edge_note}",
              flush=True)

        self.results[cost] = result
        return result


def plot_train_surface(cost, result, out_prefix):
    """3D response surface of mean_f1 -- (pen, min_size) normally, or (pen,
    width) sliced at the winning min_size when a width axis is present
    (Window), matching window_grid_red_snr3.ipynb's surface slice."""
    summary = result['train_summary']
    best_params = result['best_params']
    has_width = 'width' in summary.columns

    if 'jump' in summary.columns:
        summary = summary[summary['jump'] == best_params.get('jump', 1)]

    if has_width:
        best_ms = best_params['min_size']
        summary = summary[summary['min_size'] == best_ms]
        x_col, y_col = 'pen', 'width'
        best_x, best_y = best_params['pen'], best_params['width']
        y_label = 'width'
        title_extra = f" (min_size={best_ms:.0f} fixed)"
    else:
        x_col, y_col = 'pen', 'min_size'
        best_x, best_y = best_params['pen'], best_params['min_size']
        y_label = 'min_size'
        title_extra = ""

    pivot = summary.pivot(index=y_col, columns=x_col, values='mean_f1')
    x_vals, y_vals = pivot.columns.values, pivot.index.values
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = pivot.values

    fig = plt.figure(figsize=(6, 5.5))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', vmin=0, vmax=np.nanmax(Z),
                     edgecolor='k', linewidth=0.2, antialiased=True)

    yi, xi = list(y_vals).index(best_y), list(x_vals).index(best_x)
    ax.scatter([best_x], [best_y], [Z[yi, xi]], color='red', s=70, marker='*', zorder=10)

    on_edge = result['conclusion_row']['on_edge']
    flag = 'edge -- NOT converged' if on_edge else 'interior -- converged'
    ax.set_title(f"{cost}: train mean_f1{title_extra}\nbest={Z[yi, xi]:.3f} @ pen={best_x:.3g}, "
                 f"{y_label}={best_y:.0f}\n({flag})",
                 fontsize=9, color='#b3261e' if on_edge else '#1a1a1a')
    ax.set_xlabel('pen', fontsize=8)
    ax.set_ylabel(y_label, fontsize=8)
    ax.set_zlabel('mean_f1 (train)', fontsize=8)
    ax.set_zlim(0, max(np.nanmax(Z) * 1.05, 0.05))
    ax.view_init(elev=25, azim=-60)
    fig.suptitle(f"cost='{cost}': TRAIN response surface -- this is what the winner was picked from", fontsize=10)
    plt.tight_layout()
    fig.savefig(f"{out_prefix}_{cost}_surface.png", dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_test_pr(cost, result, out_prefix):
    """Test-set precision/recall tradeoff (pen swept at the winning min_size/width)."""
    sweep, row = result['sweep_summary'], result['conclusion_row']
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(sweep['mean_recall'], sweep['mean_precision'], color='#2a78d6',
             linewidth=1.5, marker='o', markersize=4, zorder=2)
    ax.scatter([row['test_recall']], [row['test_precision']], color='#e34948',
               s=90, marker='*', zorder=5, label='selected pen')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('recall')
    ax.set_ylabel('precision')
    ax.set_title(f"{cost}: TEST precision/recall tradeoff")
    ax.legend(frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(color='#e1e0d9', linewidth=0.6)
    ax.set_axisbelow(True)
    plt.tight_layout()
    fig.savefig(f"{out_prefix}_{cost}_pr.png", dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_test_pr_grid(results, out_prefix, ncols=4):
    """All costs' test PR curves together, one overview panel per method."""
    costs = list(results.keys())
    nrows = int(np.ceil(len(costs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.2 * nrows), squeeze=False)

    for idx, cost in enumerate(costs):
        ax = axes[idx // ncols][idx % ncols]
        res = results[cost]
        sweep = res['sweep_summary']
        row = res['conclusion_row']

        ax.plot(sweep['mean_recall'], sweep['mean_precision'], color='#2a78d6',
                 linewidth=1.5, marker='o', markersize=3, zorder=2)
        ax.scatter([row['test_recall']], [row['test_precision']], color='#e34948',
                   s=80, marker='*', zorder=5, label='selected pen')

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(cost, fontsize=10)
        ax.set_xlabel('recall', fontsize=7)
        ax.set_ylabel('precision', fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(color='#e1e0d9', linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)

    for j in range(len(costs), nrows * ncols):
        axes[j // ncols][j % ncols].axis('off')

    fig.subplots_adjust(hspace=0.7, wspace=0.5, top=0.92)
    fig.suptitle("TEST set: precision/recall tradeoff (pen swept at winning min_size, test seeds only)", y=0.98)
    fig.savefig(f"{out_prefix}_test_pr_grid.png", dpi=200, bbox_inches='tight')
    plt.close(fig)


def save_csvs(results, out_prefix):
    """raw: every scored (param, seed) row across every cost function, tagged by
    split -- 'train', 'test', 'test_sweep'. summary: the TRAIN grid summary per
    cost function (what selection was based on). conclusion: one row per cost
    function -- train_f1 (selection score, optimistic) next to test_f1 (the
    honest, held-out score)."""
    raw_frames = []
    for result in results.values():
        raw_frames.append(result['train_raw'])
        raw_frames.append(result['test_raw'])
        raw_frames.append(result['sweep_raw'])
    raw_df = pd.concat(raw_frames, ignore_index=True, sort=False)
    raw_path = f"{out_prefix}_raw.csv"
    raw_df.to_csv(raw_path, index=False)

    summary_df = pd.concat([r['train_summary'] for r in results.values()], ignore_index=True, sort=False)
    summary_path = f"{out_prefix}_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    conclusion_df = (pd.DataFrame([r['conclusion_row'] for r in results.values()])
                      .sort_values('test_f1', ascending=False).reset_index(drop=True))
    conclusion_path = f"{out_prefix}_conclusion.csv"
    conclusion_df.to_csv(conclusion_path, index=False)

    return raw_path, summary_path, conclusion_path, conclusion_df
