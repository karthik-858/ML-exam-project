import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib

# Load dataset
df = pd.read_csv("behavior_dataset.csv")

X = df

model = IsolationForest(
    n_estimators=100,
    contamination=0.1,
    random_state=42
)

model.fit(X)

joblib.dump(model, "isolation_model.pkl")

print("Model trained and saved successfully.")

# STEP 5 TEST
# Check full dataset
predictions = model.predict(X)

print("Total Normal:", sum(predictions == 1))
print("Total Suspicious:", sum(predictions == -1))