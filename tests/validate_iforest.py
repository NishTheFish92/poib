"""
Validate the Isolation Forest with synthetic faults.

We don't have true anomaly labels, so we plant faults with known locations
into real sensor data and measure how well the FOG pipeline (5-reading
windows -> mean/std/slope features -> Isolation Forest) ranks them above
normal windows.

To avoid testing on training data, the model is retrained here with the exact
same pipeline as train.ipynb on the first 70% of the data (by time). Faults are
planted only in the remaining 30%. The whole experiment is repeated over many
random seeds and compared against a simple z-score baseline.

Fault types evaluated: spike, dropout, drift.
"""
import os
import sys

# Run from the project root so data/model paths and FOGInference resolve
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, roc_curve


N_SEEDS = 30
EVENTS_PER_TYPE = 4   # per seed, so N_SEEDS * EVENTS_PER_TYPE events per fault type in total
TRAIN_FRACTION = 0.7
WINDOW_SIZE = 5       # same as FOGInference.py
GAP = 6               # clean rows kept between events so no window spans two faults

fields = {
    'field3': 'Temperature',
    'field4': 'Humidity',
    'field5': 'eCO2',
    'field7': 'Dust',
}

# Fault type -> (length in rows, fields it can hit)
FAULTS = {
    'spike':   (1,  list(fields)),
    'dropout': (1,  ['field3', 'field4', 'field5']),  # dust is legitimately 0 sometimes
    'drift':   (15, list(fields)),
}


# ---------------------------------------------------------------
# Feature pipeline (mirrors train.ipynb / FOGInference.py)
# ---------------------------------------------------------------
def window_features(values):
    """values: (n_rows, n_fields) array -> (n_windows, 3 * n_fields) of mean/std/slope."""
    windows = np.lib.stride_tricks.sliding_window_view(values, WINDOW_SIZE, axis=0)  # (n_win, n_fields, W)
    i = np.arange(1, WINDOW_SIZE + 1)
    i_centered = i - i.mean()

    mean = windows.mean(axis=2)
    std = windows.std(axis=2)
    slope = (windows - mean[..., None]) @ i_centered / np.sum(i_centered ** 2)

    # interleave per field: [f_mean, f_std, f_slope, ...]
    return np.stack([mean, std, slope], axis=2).reshape(len(windows), -1)


def window_labels(row_is_fault):
    """A window is a fault window if any of its readings is faulty."""
    return np.lib.stride_tricks.sliding_window_view(row_is_fault, WINDOW_SIZE).any(axis=1)


# ---------------------------------------------------------------
# 1. Load data and split by time
# ---------------------------------------------------------------
df = pd.read_csv("data/indoor_data.csv")
df['created_at'] = pd.to_datetime(df['created_at'])
df = df.sort_values('created_at').reset_index(drop=True)
df[list(fields)] = df[list(fields)].ffill().bfill()

split = int(len(df) * TRAIN_FRACTION)
train, test = df.iloc[:split], df.iloc[split:].reset_index(drop=True)
test_hours = (test['created_at'].iloc[-1] - test['created_at'].iloc[0]).total_seconds() / 3600


# ---------------------------------------------------------------
# 2. Train on the first 70% with the train.ipynb pipeline
# ---------------------------------------------------------------
train_feats = window_features(train[list(fields)].to_numpy())
mu = train_feats.mean(axis=0)
sigma = train_feats.std(axis=0, ddof=1)

iso = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
iso.fit(np.nan_to_num((train_feats - mu) / sigma))

# Baseline: per-reading z-score against training data, worst reading in the window
raw_mu = train[list(fields)].mean().to_numpy()
raw_sigma = train[list(fields)].std().to_numpy()


def score_iforest(values):
    z = np.nan_to_num((window_features(values) - mu) / sigma)
    return -iso.decision_function(z), iso.predict(z) == -1   # higher = more anomalous


def score_zscore(values):
    z = np.abs((values - raw_mu) / raw_sigma).max(axis=1)
    return np.lib.stride_tricks.sliding_window_view(z, WINDOW_SIZE).max(axis=1)


# ---------------------------------------------------------------
# 3. Plant faults at random, non-overlapping positions
# ---------------------------------------------------------------
test_std = test[list(fields)].std()


def plant_faults(rng):
    faulty = test[list(fields)].copy()
    fault = np.array(['normal'] * len(faulty), dtype=object)
    taken = np.zeros(len(faulty), dtype=bool)
    events = []

    for name, (length, allowed_fields) in FAULTS.items():
        placed = 0
        while placed < EVENTS_PER_TYPE:
            start = int(rng.integers(GAP, len(faulty) - length - GAP))
            if taken[start - GAP:start + length + GAP].any():
                continue

            field = rng.choice(allowed_fields)
            rows = slice(start, start + length)
            col = faulty.columns.get_loc(field)

            if name == 'spike':
                faulty.iloc[start, col] += 6 * test_std[field]
            elif name == 'dropout':
                faulty.iloc[start, col] = 0
            elif name == 'drift':
                faulty.iloc[rows, col] += np.linspace(0, 5 * test_std[field], length)

            fault[rows] = name
            taken[start:start + length] = True
            events.append({'fault': name, 'field': fields[field], 'start': start, 'end': start + length})
            placed += 1

    return faulty, fault, pd.DataFrame(events)


# ---------------------------------------------------------------
# 4. Run all seeds
# ---------------------------------------------------------------
auc_rows = []
pooled = {k: {'y': [], 'iforest': [], 'zscore': []} for k in [*FAULTS, 'all faults']}
detections = []
false_alarms = []

for seed in range(N_SEEDS):
    rng = np.random.default_rng(seed)
    faulty, row_fault, events = plant_faults(rng)
    values = faulty.to_numpy()

    if_score, if_flag = score_iforest(values)
    z_score = score_zscore(values)

    # window label = fault type present in the window (one type at most, thanks to GAP)
    win_fault = np.array([
        next((f for f in w if f != 'normal'), 'normal')
        for w in np.lib.stride_tricks.sliding_window_view(row_fault, WINDOW_SIZE)
    ])
    normal = win_fault == 'normal'

    for name in [*FAULTS, 'all faults']:
        keep = normal | (win_fault == name) if name in FAULTS else np.ones_like(normal)
        y = (~normal[keep]).astype(int)
        auc_rows.append({
            'seed': seed, 'fault': name,
            'iforest': roc_auc_score(y, if_score[keep]),
            'zscore': roc_auc_score(y, z_score[keep]),
        })
        pooled[name]['y'].append(y)
        pooled[name]['iforest'].append(if_score[keep])
        pooled[name]['zscore'].append(z_score[keep])

    # Event detected if any window containing it is flagged at the deployed threshold
    for ev in events.itertuples():
        first_win = max(ev.start - WINDOW_SIZE + 1, 0)
        detections.append({'fault': ev.fault, 'detected': if_flag[first_win:ev.end].any()})

    false_alarms.append(if_flag[normal].sum() / test_hours)

    if seed == 0:
        example = (faulty, events, if_score, if_flag)

auc = pd.DataFrame(auc_rows)
summary = auc.groupby('fault', sort=False)[['iforest', 'zscore']].agg(['mean', 'std'])
detections = pd.DataFrame(detections)
det_summary = detections.groupby('fault', sort=False)['detected'].agg(['sum', 'count'])


# ---------------------------------------------------------------
# 5. Report
# ---------------------------------------------------------------
print(f"\nTrain: first {split} rows | Test: last {len(test)} rows ({test_hours:.1f} h)")
print(f"{N_SEEDS} seeds x {EVENTS_PER_TYPE} events per fault type "
      f"= {N_SEEDS * EVENTS_PER_TYPE} planted events per type\n")

print("ROC-AUC (mean ± std over seeds)")
for name, r in summary.iterrows():
    print(f"  {name:<11} Isolation Forest {r[('iforest', 'mean')]:.3f} ± {r[('iforest', 'std')]:.3f}"
          f"   |   z-score baseline {r[('zscore', 'mean')]:.3f} ± {r[('zscore', 'std')]:.3f}")

print("\nDetection at the deployed threshold (predict == -1)")
for name, r in det_summary.iterrows():
    print(f"  {name:<11} {r['sum']:>3}/{r['count']} events caught ({r['sum'] / r['count']:.0%})")


# ---------------------------------------------------------------
# 6. Plots
# ---------------------------------------------------------------
fig, (ax_roc, ax_bar) = plt.subplots(1, 2, figsize=(13, 5.5))

for name, p in pooled.items():
    y = np.concatenate(p['y'])
    fpr, tpr, _ = roc_curve(y, np.concatenate(p['iforest']))
    style = {'lw': 2.5, 'color': 'black'} if name == 'all faults' else {'lw': 1.5}
    ax_roc.plot(fpr, tpr, label=f"{name} (AUC = {summary.loc[name, ('iforest', 'mean')]:.2f})", **style)

y = np.concatenate(pooled['all faults']['y'])
fpr, tpr, _ = roc_curve(y, np.concatenate(pooled['all faults']['zscore']))
ax_roc.plot(fpr, tpr, lw=1.5, ls=':', color='gray',
            label=f"z-score baseline, all faults (AUC = {summary.loc['all faults', ('zscore', 'mean')]:.2f})")
ax_roc.plot([0, 1], [0, 1], ls='--', color='lightgray', label='random guess (AUC = 0.50)')
ax_roc.set_xlabel('False positive rate (normal windows flagged)')
ax_roc.set_ylabel('True positive rate (fault windows flagged)')
ax_roc.set_title(f'ROC on held-out data, {N_SEEDS} seeds pooled')
ax_roc.legend(loc='lower right', fontsize=8)

names = list(summary.index)
x = np.arange(len(names))
ax_bar.bar(x - 0.2, summary[('iforest', 'mean')], 0.4, yerr=summary[('iforest', 'std')],
           capsize=4, label='Isolation Forest')
ax_bar.bar(x + 0.2, summary[('zscore', 'mean')], 0.4, yerr=summary[('zscore', 'std')],
           capsize=4, color='gray', label='z-score baseline')
ax_bar.axhline(0.5, ls='--', color='lightgray')
ax_bar.set_xticks(x, names)
ax_bar.set_ylim(0, 1.05)
ax_bar.set_ylabel('ROC-AUC (mean ± std)')
ax_bar.set_title('Isolation Forest vs baseline')
ax_bar.legend(loc='lower right')

fig.tight_layout()
fig.savefig("results/validate_iforest.png", dpi=120)

# Example trace: one seed, planted faults shaded, anomaly score underneath
faulty, events, if_score, if_flag = example
fig, axes = plt.subplots(len(fields) + 1, 1, figsize=(13, 9), sharex=True)
t = np.arange(len(faulty))
colors = {'spike': 'tab:red', 'dropout': 'tab:orange', 'drift': 'tab:purple'}

for ax, (field, label) in zip(axes, fields.items()):
    ax.plot(t, faulty[field], lw=0.8, color='tab:blue')
    ax.set_ylabel(label)
for ev in events.itertuples():
    for ax in axes:
        ax.axvspan(ev.start - 0.5, ev.end - 0.5, color=colors[ev.fault], alpha=0.25)

win_t = t[WINDOW_SIZE - 1:]
axes[-1].plot(win_t, if_score, lw=0.8, color='black')
axes[-1].scatter(win_t[if_flag], if_score[if_flag], s=8, color='tab:red', zorder=3, label='flagged')
axes[-1].set_ylabel('Anomaly score')
axes[-1].set_xlabel('Reading index (held-out test set)')
axes[-1].legend(loc='upper right')
axes[0].legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.4) for c in colors.values()],
               labels=list(colors), loc='upper right', ncol=3)
axes[0].set_title('Example: planted faults (shaded) and Isolation Forest anomaly score')

fig.tight_layout()
fig.savefig("results/validate_iforest_trace.png", dpi=120)
plt.show()
