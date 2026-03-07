import joblib
import pandas as pd

print("Loading model...")

model = joblib.load("models/final_svm_pipeline.pkl")

print("\n=== Heart Disease Prediction System ===")

age = float(input("Age: "))
trestbps = float(input("Resting Blood Pressure: "))
chol = float(input("Cholesterol: "))
thalach = float(input("Max Heart Rate: "))
oldpeak = float(input("Oldpeak: "))

sample = pd.DataFrame([{
    "age": age,
    "trestbps": trestbps,
    "chol": chol,
    "thalach": thalach,
    "oldpeak": oldpeak
}])

prediction = model.predict(sample)
probability = model.predict_proba(sample)[0][1]

if prediction[0] == 1:
    print(f"\n⚠️ High Risk (Probability: {probability:.2f})")
else:
    print(f"\n✅ Low Risk (Probability: {probability:.2f})")