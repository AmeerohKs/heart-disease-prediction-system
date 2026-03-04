from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -------------------------
# Decision Tree
# -------------------------
def train_decision_tree(X_train, y_train):
    model = DecisionTreeClassifier(
        max_depth=6,
        min_samples_split=5,
        min_samples_leaf=3,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

# -------------------------
# Logistic Regression
# -------------------------
def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

# -------------------------
# Support Vector Machine (Pipeline)
# -------------------------
def train_svm(X_train, y_train):
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel='rbf', C=1, gamma='scale', probability=True))
    ])

    pipeline.fit(X_train, y_train)
    return pipeline

# -------------------------
# Evaluation Function
# -------------------------
def evaluate_model(model, X_test, y_test, model_name="Model"):
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)

    print(f"\n===== {model_name} Results =====")
    print("Accuracy:", accuracy)
    print("\nClassification Report:\n", report)
    print("Confusion Matrix:\n", matrix)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
        print("ROC-AUC Score:", auc)

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.figure()
        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1])
        plt.title(f"{model_name} ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.show()

    return accuracy