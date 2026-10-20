import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, precision_recall_curve,
)
import FOGInference
from FOGInference import process_new_reading


HORIZONS = [10, 15, 30]
TUNE_FRACTION = 0.5   # first half (by time) picks thresholds, second half is the test set


def create_future_label(df, minutes):
    labels = []

    for i in range(len(df)):
        current_time = df.iloc[i]['created_at']

        future = df[
            (df['created_at'] > current_time) &
            (df['created_at'] <= current_time + pd.Timedelta(minutes=minutes))
        ]

        if len(future) == 0:
            labels.append(0)
        else:
            labels.append(1 if future['anomaly'].max() == 1 else 0)

    return labels


def best_f1_threshold(y_true, scores):
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    # last precision/recall pair has no threshold
    return thresholds[np.argmax(f1[:-1])]


# Load dataset
df = pd.read_csv("indoor_data.csv")
df['created_at'] = pd.to_datetime(df['created_at'])
df = df.sort_values('created_at').reset_index(drop=True)

fields = [
    'field3',  # Temperature
    'field4',  # Humidity
    'field5',  # eCO2
    'field7'   # Dust
]

# Handle missing values
df[fields] = df[fields].ffill().bfill()


# ---------------------------------------------------------------
# 1. Ground-truth labels: point-wise Isolation Forest anomalies
# ---------------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[fields])

iso_model = IsolationForest(
    n_estimators=200,
    contamination=0.05,
    random_state=42
)
iso_model.fit(X_scaled)

df['anomaly'] = (iso_model.predict(X_scaled) == -1).astype(int)

# Naive baseline: current point's anomaly score as risk (0 to 1)
anomaly_score = iso_model.decision_function(X_scaled)
df['naive_risk'] = (anomaly_score.max() - anomaly_score)
df['naive_risk'] = df['naive_risk'] / df['naive_risk'].max()

for h in HORIZONS:
    df[f'actual_{h}'] = create_future_label(df, h)


# ---------------------------------------------------------------
# 2. Predictions: FOG pipeline (windowed IF + Markov chain forecast)
# ---------------------------------------------------------------
FOGInference.buffer.clear()

fog_rows = []
for _, row in df.iterrows():
    out = process_new_reading(row['field3'], row['field4'], row['field5'], row['field7'])
    if isinstance(out, dict):
        fog_rows.append({f'fog_{h}': out[f'Risk_{h}min'] for h in HORIZONS})
    else:
        fog_rows.append({f'fog_{h}': np.nan for h in HORIZONS})

df = pd.concat([df, pd.DataFrame(fog_rows)], axis=1)

# Drop rows before the first full window
df = df.dropna(subset=[f'fog_{h}' for h in HORIZONS]).reset_index(drop=True)


# ---------------------------------------------------------------
# 3. Time-based split: tune thresholds on the past, test on the future
# ---------------------------------------------------------------
split = int(len(df) * TUNE_FRACTION)
tune, test = df.iloc[:split], df.iloc[split:]


def evaluate(h, score_col, name):
    y_tune, y_test = tune[f'actual_{h}'], test[f'actual_{h}']
    threshold = best_f1_threshold(y_tune, tune[score_col])
    y_pred = (test[score_col] >= threshold).astype(int)
    corr, _ = pearsonr(test[score_col], y_test)

    return {
        'horizon': f'{h} min',
        'model': name,
        'threshold': threshold,
        'accuracy': accuracy_score(y_test, y_pred),
        'baseline_acc': 1 - y_test.mean(),   # always predicting "no anomaly"
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, test[score_col]),
        'pr_auc': average_precision_score(y_test, test[score_col]),
        'pos_rate': y_test.mean(),           # PR-AUC of a random guess
        'correlation': corr,
    }


results = []
for h in HORIZONS:
    results.append(evaluate(h, 'naive_risk', 'naive'))
    results.append(evaluate(h, f'fog_{h}', 'fog'))

results = pd.DataFrame(results).set_index(['horizon', 'model'])

print(f"\nTune rows: {len(tune)}   Test rows: {len(test)}")
print("Metrics below are on the test set only.\n")
with pd.option_context('display.float_format', '{:.3f}'.format, 'display.width', 200, 'display.max_columns', None):
    print(results)


# ---------------------------------------------------------------
# 4. Plot
# ---------------------------------------------------------------
fog_threshold = results.loc[('10 min', 'fog'), 'threshold']

plt.figure(figsize=(12, 5))
plt.plot(df['created_at'], df['fog_10'], label='FOG Risk (10min)')
plt.scatter(
    df['created_at'],
    df['actual_10'],
    color='red',
    s=10,
    label='Actual Anomaly (10min)'
)
plt.axhline(y=fog_threshold, linestyle='--', color='green',
            label=f'Threshold ({fog_threshold:.2f})')
plt.axvline(x=test['created_at'].iloc[0], linestyle=':', color='gray',
            label='Tune / Test split')
plt.legend()
plt.title("FOG Risk vs Future Anomaly")
plt.show()
