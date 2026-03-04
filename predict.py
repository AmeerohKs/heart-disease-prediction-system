import joblib
import pandas as pd

# Load pipeline
model = joblib.load("models/final_svm_pipeline.pkl")

# Load original dataset structure
data = pd.read_csv("data/heart.csv")  # your original dataset

# Take first row as template
sample = data.drop("target", axis=1).iloc[[0]].copy()

print("=== Heart Disease Prediction System ===")

# Modify some fields
sample["age"] = float(input("Age: "))
sample["trestbps"] = float(input("Resting Blood Pressure: "))
sample["chol"] = float(input("Cholesterol: "))
sample["thalach"] = float(input("Max Heart Rate: "))
sample["oldpeak"] = float(input("Oldpeak: "))

prediction = model.predict(sample)
probability = model.predict_proba(sample)[0][1]

if prediction[0] == 1:
    print(f"\n⚠️ High Risk (Probability: {probability:.2f})")
else:
    print(f"\n✅ Low Risk (Probability: {probability:.2f})")