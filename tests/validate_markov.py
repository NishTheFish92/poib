"""
Validate the Markov chain risk forecast.

The Markov chain answers one question: "what is the probability the system is
in a fault state k minutes from now?". So we test exactly that:

  - Each 5-reading window gets a state (OK / Thermal / CO2 / Dust / Sensor)
    using the same rule as train.ipynb, from the deployed FOG pipeline.
  - The transition matrix is learned on the first half of the data (by time)
    and tested on the second half, which it has never seen.
  - The forecast for minute k is compared with the state actually observed at
    minute k (1 step = 1 reading, about 1 minute).

The forecast is a probability, so it is scored with the Brier score (mean
squared error of a probability, lower is better) against two simpler guesses:
  - base rate: always predict the usual fault rate from the training half
  - P_A now:   use the current anomaly probability as the future probability
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
from sklearn.metrics import brier_score_loss
import FOGInference as F


HORIZONS = [10, 15, 30]   # minutes ahead = steps ahead (1 reading per minute)
TRAIN_FRACTION = 0.5
STATES = ['OK', 'Thermal', 'CO2', 'Dust', 'Sensor']


# ---------------------------------------------------------------
# 1. Run the deployed FOG pipeline over every 5-reading window
# ---------------------------------------------------------------
df = pd.read_csv("data/indoor_data.csv")
df['created_at'] = pd.to_datetime(df['created_at'])
df = df.sort_values('created_at').reset_index(drop=True)

fields = ['field3', 'field4', 'field5', 'field7']
df[fields] = df[fields].ffill().bfill()

P_A, fault_probs = [], []
for end in range(F.WINDOW_SIZE, len(df) + 1):
    window = df.iloc[end - F.WINDOW_SIZE:end]
    z = ((F.extract_features(window) - F.baseline_mu) / F.baseline_sigma).values.reshape(1, -1)
    P_A.append(F.anomaly_probability(F.model.decision_function(z)[0]))
    fault_probs.append(F.fault_probabilities(z))

P_A = np.array(P_A)
fault_probs = np.array(fault_probs)

# Same state rule as train.ipynb: OK unless anomalous, then the most likely fault type
state = np.where(P_A < 0.5, 0, 1 + fault_probs.argmax(axis=1))
is_fault = (state != 0).astype(int)


# ---------------------------------------------------------------
# 2. Learn the transition matrix on the first half only
# ---------------------------------------------------------------
split = int(len(state) * TRAIN_FRACTION)

# Add-one smoothing: states never seen in the training half would otherwise get
# an all-zero row, which is not a valid probability distribution
counts = np.ones((len(STATES), len(STATES)))
for a, b in zip(state[:split - 1], state[1:split]):
    counts[a, b] += 1
T = counts / counts.sum(axis=1, keepdims=True)


def markov_risk(p_a, probs, steps):
    """Probability of being in any fault state after `steps` (mirrors FOGInference.future_risk)."""
    dist = np.column_stack([1 - p_a, p_a[:, None] * probs])
    return (dist @ np.linalg.matrix_power(T, steps))[:, 1:].sum(axis=1)


base_rate = is_fault[:split].mean()


# ---------------------------------------------------------------
# 3. Forecast on the second half and compare with what happened
# ---------------------------------------------------------------
results = []

for h in HORIZONS:
    now = np.arange(split, len(state) - h)
    actual = is_fault[now + h]

    forecasts = {
        'Markov chain': markov_risk(P_A[now], fault_probs[now], h),
        'P_A now': P_A[now],
        'base rate': np.full(len(now), base_rate),
    }
    brier = {name: brier_score_loss(actual, p) for name, p in forecasts.items()}

    results.append({
        'horizon': f'{h} min',
        **{f'brier ({name})': b for name, b in brier.items()},
        'improvement vs base rate': 1 - brier['Markov chain'] / brier['base rate'],
        'improvement vs P_A now': 1 - brier['Markov chain'] / brier['P_A now'],
    })

results = pd.DataFrame(results).set_index('horizon')


# ---------------------------------------------------------------
# 4. Report
# ---------------------------------------------------------------
print(f"\nTransition matrix learned on the first {split} windows, tested on the last {len(state) - split}.")
print(f"Fault-state rate in the training half: {base_rate:.1%}\n")

print("Learned transition matrix (row = now, column = next minute):")
print(pd.DataFrame(T, index=STATES, columns=STATES).round(3).to_string(), "\n")

print("Brier score on the test half (lower is better)")
for h, r in results.iterrows():
    print(f"  {h:<7} Markov {r['brier (Markov chain)']:.4f}   "
          f"P_A now {r['brier (P_A now)']:.4f}   "
          f"base rate {r['brier (base rate)']:.4f}   "
          f"-> {r['improvement vs base rate']:.0%} better than base rate, "
          f"{r['improvement vs P_A now']:.0%} better than P_A now")


# ---------------------------------------------------------------
# 5. Plot
# ---------------------------------------------------------------
fig, ax_bar = plt.subplots(figsize=(7, 5.5))

names = ['Markov chain', 'P_A now', 'base rate']
colors = ['tab:blue', 'tab:orange', 'gray']
x = np.arange(len(HORIZONS))
for i, (name, color) in enumerate(zip(names, colors)):
    ax_bar.bar(x + (i - 1) * 0.27, results[f'brier ({name})'], 0.27, color=color, label=name)
ax_bar.set_xticks(x, results.index)
ax_bar.set_xlabel('Forecast horizon')
ax_bar.set_ylabel('Brier score (lower is better)')
ax_bar.set_title('Forecast error on held-out data')
ax_bar.legend()

fig.tight_layout()
fig.savefig("results/validate_markov.png", dpi=120)
plt.show()
