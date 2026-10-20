"""
Validate the Isolation Forest with synthetic faults.

We don't have true anomaly labels, so we plant faults with known locations
into the real sensor data and measure how well the deployed FOG model
(isolation_forest_model.pkl, 5-reading windows) ranks them above normal rows.
"""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import FOGInference
from FOGInference import process_new_reading


SEED = 42
EVENTS_PER_TYPE = 8
GAP = 6  # rows kept clean between events so windows don't overlap two faults

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
    'stuck':   (12, list(fields)),
}

rng = np.random.default_rng(SEED)


# ---------------------------------------------------------------
# 1. Load clean data
# ---------------------------------------------------------------
df = pd.read_csv("indoor_data.csv")
df['created_at'] = pd.to_datetime(df['created_at'])
df = df.sort_values('created_at').reset_index(drop=True)
df[list(fields)] = df[list(fields)].ffill().bfill()

std = df[list(fields)].std()


# ---------------------------------------------------------------
# 2. Plant faults at random, non-overlapping positions
# ---------------------------------------------------------------
faulty = df.copy()
faulty['fault'] = 'normal'
events = []
taken = np.zeros(len(faulty), dtype=bool)

for fault, (length, allowed_fields) in FAULTS.items():
    placed = 0
    while placed < EVENTS_PER_TYPE:
        start = int(rng.integers(GAP, len(faulty) - length - GAP))
        if taken[start - GAP:start + length + GAP].any():
            continue

        field = rng.choice(allowed_fields)
        rows = range(start, start + length)
        col = faulty.columns.get_loc(field)

        if fault == 'spike':
            faulty.iloc[start, col] += 6 * std[field]
        elif fault == 'dropout':
            faulty.iloc[start, col] = 0
        elif fault == 'drift':
            faulty.iloc[rows, col] += np.linspace(0, 5 * std[field], length)
        elif fault == 'stuck':
            faulty.iloc[rows, col] = faulty.iloc[start, col]

        faulty.iloc[rows, faulty.columns.get_loc('fault')] = fault
        taken[start:start + length] = True
        events.append({'fault': fault, 'field': fields[field], 'start': start, 'end': start + length})
        placed += 1

events = pd.DataFrame(events)

# ---------------------------------------------------------------
# 3. Score with the FOG model (higher = more anomalous)
# ---------------------------------------------------------------
FOGInference.buffer.clear()
scores = []
for _, row in faulty.iterrows():
    out = process_new_reading(row['field3'], row['field4'], row['field5'], row['field7'])
    scores.append(-out['score'] if isinstance(out, dict) else np.nan)

faulty['score'] = scores
faulty = faulty.dropna(subset=['score'])


# ---------------------------------------------------------------
# 4. ROC-AUC per fault type (each fault type vs all normal rows)
# ---------------------------------------------------------------
normal = faulty[faulty['fault'] == 'normal']
curves = {}
results = []

for fault in FAULTS:
    subset = pd.concat([normal, faulty[faulty['fault'] == fault]])
    y = (subset['fault'] != 'normal').astype(int)
    curves[fault] = roc_curve(y, subset['score'])
    results.append({'fault': fault, 'roc_auc': roc_auc_score(y, subset['score'])})

# All fault types together
y_all = (faulty['fault'] != 'normal').astype(int)
curves['all faults'] = roc_curve(y_all, faulty['score'])
results.append({'fault': 'all faults', 'roc_auc': roc_auc_score(y_all, faulty['score'])})

results = pd.DataFrame(results).set_index('fault')

print(f"\nPlanted {len(events)} faults ({EVENTS_PER_TYPE} per type) into {len(faulty)} rows.\n")
with pd.option_context('display.float_format', '{:.3f}'.format):
    print(results)


# ---------------------------------------------------------------
# 5. Plot ROC curves
# ---------------------------------------------------------------
plt.figure(figsize=(7, 6))

for fault, (fpr, tpr, _) in curves.items():
    style = {'lw': 2.5, 'color': 'black'} if fault == 'all faults' else {'lw': 1.5}
    plt.plot(fpr, tpr, label=f"{fault} (AUC = {results.loc[fault, 'roc_auc']:.2f})", **style)

plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='random guess (AUC = 0.50)')
plt.xlabel('False positive rate (normal rows flagged)')
plt.ylabel('True positive rate (fault rows flagged)')
plt.title('FOG Isolation Forest: ROC on planted faults')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig("validate_iforest.png", dpi=120)
plt.show()
