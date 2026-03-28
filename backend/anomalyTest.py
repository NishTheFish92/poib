import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

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

print(df.head(20))

df.to_csv("indoor_data_with_anomalies.csv", index=False)