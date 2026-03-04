from src.data_preprocessing import load_and_clean_data, split_data, scale_data
from sklearn.model_selection import cross_val_score
import joblib
from src.train_model import (
    train_decision_tree,
    train_logistic_regression,
    train_svm,
    evaluate_model,
)


def main():
    df = load_and_clean_data("data/heart_disease_uci.csv")
    X_train, X_test, y_train, y_test = split_data(df)
    X_train, X_test = scale_data(X_train, X_test)

    # Decision Tree
    dt_model = train_decision_tree(X_train, y_train)
    dt_acc = evaluate_model(dt_model, X_test, y_test, "Decision Tree")

    # Logistic Regression
    lr_model = train_logistic_regression(X_train, y_train)
    lr_acc = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")

    # SVM
    svm_model = train_svm(X_train, y_train)
    svm_acc = evaluate_model(svm_model, X_test, y_test, "SVM")
    
    joblib.dump(svm_model, "models/final_svm_model.pkl")
    print("\nSVM model saved successfully!")
    print("\n=== Model Comparison ===")
    print(f"Decision Tree Accuracy: {dt_acc}")
    print(f"Logistic Regression Accuracy: {lr_acc}")
    print(f"SVM Accuracy: {svm_acc}")
    cv_scores = cross_val_score(svm_model, X_train, y_train, cv=5)
    print("\nSVM Cross-Validation Accuracy Scores:", cv_scores)
    print("Mean CV Accuracy:", cv_scores.mean())

if __name__ == "__main__":
    main()