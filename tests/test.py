import os
import sys

# Run from the project root so data/model paths and FOGInference resolve
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import joblib
import numpy as np

model = joblib.load("model/isolation_forest_model.pkl")
print(model.get_params())

# Example: 1 sample with correct number of features
X_test = np.random.rand(1, model.n_features_in_)

score = model.decision_function(X_test)
label = model.predict(X_test)

baseline_mu = joblib.load("model/baseline_mu.pkl")

print(baseline_mu)

