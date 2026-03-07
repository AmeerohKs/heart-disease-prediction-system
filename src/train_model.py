import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve

import matplotlib.pyplot as plt

# -------------------------
# Load Dataset
# -------------------------
print("Loading dataset...")
df = pd.read_csv("heart.csv")

# -------------------------
# Select ONLY 5 features
# -------------------------
features = ["age", "trestbps", "chol", "thalach", "oldpeak"]

X = df[features]
y = df["target"]

# -------------------------
# Train/Test Split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# -------------------------
# Decision Tree
# -------------------------
print("\nTraining Decision Tree...")

dt_model = DecisionTreeClassifier(
    max_depth=6,
    min_samples_split=5,
    min_samples_leaf=3,
    random_state=42
)

dt_model.fit(X_train, y_train)

# -------------------------
# Logistic Regression
# -------------------------
print("\nTraining Logistic Regression...")

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

# -------------------------
# Support Vector Machine
# -------------------------
print("\nTraining SVM...")

svm_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel='rbf', C=1, gamma='scale', probability=True))
])

svm_pipeline.fit(X_train, y_train)

# -------------------------
# Evaluation Function
# -------------------------
def evaluate(model, name):

    print(f"\n===== {name} Results =====")

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print("Accuracy:", accuracy)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    if hasattr(model, "predict_proba"):

        y_proba = model.predict_proba(X_test)[:,1]

        auc = roc_auc_score(y_test, y_proba)

        print("ROC-AUC Score:", auc)

        fpr, tpr, _ = roc_curve(y_test, y_proba)

        plt.figure()
        plt.plot(fpr, tpr)
        plt.plot([0,1], [0,1])
        plt.title(f"{name} ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.show()

    return accuracy


# -------------------------
# Evaluate Models
# -------------------------
dt_acc = evaluate(dt_model, "Decision Tree")

lr_acc = evaluate(lr_model, "Logistic Regression")

svm_acc = evaluate(svm_pipeline, "SVM")

# -------------------------
# Save Best Model (SVM)
# -------------------------
print("\nSaving best model...")

joblib.dump(svm_pipeline, "models/final_svm_pipeline.pkl")

print("Model saved to models/final_svm_pipeline.pkl")

print("\nTraining completed successfully.")