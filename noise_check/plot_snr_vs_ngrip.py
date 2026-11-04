"""Compare synthetic series at each (color, SNR) against real NGRIP, on a z-scored scale,
and compute a roughness metric (std of 1st differences / std of series) for each."""
import numpy as np, pandas as pd, matplotlib.pyplot as plt, os
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
ng = pd.read_csv(os.path.join(ROOT, 'data/NGRIP.csv'), comment='#', skiprows=8, header=None).iloc[:, :2].astype(float).values
t_ng, y_ng = ng[:, 0], ng[:, 1]
z = lambda x: (x - x.mean()) / x.std()
rough = lambda x: np.diff(x).std() / x.std()
def ac1(x): d = np.diff(x); return np.corrcoef(d[:-1], d[1:])[0, 1]   # white noise -> diff ac1 ~ -0.5

COLORS, LEVELS, SEED = ['white', 'pink', 'red'], ['snr4', 'snr3', 'snr2'], 0
fig, axes = plt.subplots(len(LEVELS) + 1, len(COLORS), figsize=(16, 9), sharey=True)
rows = []
for j in range(3):
    axes[0, j].plot(t_ng, z(y_ng), lw=.5, c='k'); axes[0, j].set_title('NGRIP (z-scored)')
for i, lv in enumerate(LEVELS, 1):
    for j, c in enumerate(COLORS):
        d = np.load(os.path.join(ROOT, f'noise_sensitivity_data/noise_{c}_{lv}.npz'))
        y, yt = d['db_noisy'][SEED], d['db_true'][SEED]
        ax = axes[i, j]; ax.plot(d['time'], z(y), lw=.5, c='#888'); ax.plot(d['time'], (yt - y.mean()) / y.std(), lw=1, c='C1')
        ax.set_title(f'{c} {lv}  rough={rough(y):.3f}')
        rows.append(dict(color=c, level=lv, roughness=np.mean([rough(r) for r in d['db_noisy']]), diff_ac1=np.mean([ac1(r) for r in d['db_noisy']])))
fig.tight_layout(); fig.savefig(os.path.join(HERE, 'snr_vs_ngrip.png'), dpi=130)
rows.append(dict(color='NGRIP', level='real', roughness=rough(y_ng), diff_ac1=ac1(y_ng)))
print(pd.DataFrame(rows).round(3).to_string(index=False))
