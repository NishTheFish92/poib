import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from FOGInference import process_new_reading




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

# Load dataset
df = pd.read_csv("indoor_data.csv")
df['created_at'] = pd.to_datetime(df['created_at'])
df = df.sort_values('created_at')

fields = [
    'field3',  # Temperature
    'field4',  # Humidity
    'field5',  # eCO2
    'field7'   # Dust
]

# Select only sensor values
X = df[fields]

# Handle missing values
X = X.ffill().bfill()

# 🔥 Scale data (IMPORTANT)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Isolation Forest
iso_model = IsolationForest(
    n_estimators=200,          # slightly stronger
    contamination=0.05,
    random_state=42
)

iso_model.fit(X_scaled)

# Predict anomalies
preds = iso_model.predict(X_scaled)

# Convert labels
df['anomaly'] = (preds == -1).astype(int)

# Anomaly score
df['anomaly_score'] = iso_model.decision_function(X_scaled)

# Normalize score → risk (0 to 1)
df['risk_score'] = (df['anomaly_score'].max() - df['anomaly_score'])
df['risk_score'] = df['risk_score'] / df['risk_score'].max()




df['actual_10'] = create_future_label(df, 10)
df['actual_15'] = create_future_label(df, 15)
df['actual_30'] = create_future_label(df, 30)


THRESHOLD = 0.6  # you can tune this

df['pred_10'] = (df['risk_score'] > THRESHOLD).astype(int)
df['pred_15'] = (df['risk_score'] > THRESHOLD).astype(int)
df['pred_30'] = (df['risk_score'] > THRESHOLD).astype(int)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate(h):
    y_true = df[f'actual_{h}']
    y_pred = df[f'pred_{h}']

    print(f"\n===== {h} MINUTES =====")
    print("Accuracy :", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall   :", recall_score(y_true, y_pred))
    print("F1 Score :", f1_score(y_true, y_pred))

evaluate(10)
evaluate(15)
evaluate(30)


from scipy.stats import pearsonr

for h in [10, 15, 30]:
    corr, _ = pearsonr(df['risk_score'], df[f'actual_{h}'])
    print(f"Correlation ({h} min):", corr)



import matplotlib.pyplot as plt

plt.figure(figsize=(12,5))

plt.plot(df['created_at'], df['risk_score'], label='Risk Score')

plt.scatter(
    df['created_at'],
    df['actual_10'],
    color='red',
    s=10,
    label='Actual Anomaly (10min)'
)

plt.axhline(y=THRESHOLD, linestyle='--', label='Threshold (0.6)', color= 'green')
plt.legend()
plt.title("Risk vs Future Anomaly")
plt.show()