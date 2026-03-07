🫀 Heart Disease Prediction System (HDPS)
📌 Overview

This project develops a machine learning-based Heart Disease Prediction System (HDPS) to classify patients into high-risk or low-risk categories using clinical features.

The system compares multiple models and selects the best-performing algorithm based on statistical evaluation.

----------

🎯 Objectives

Develop predictive models for heart disease detection

Compare Decision Tree, Logistic Regression, and SVM

Evaluate using Accuracy, ROC-AUC, and Cross-Validation

Deploy the best model for prediction

----------

🧠 Models Used

Decision Tree

Logistic Regression

Support Vector Machine (Final Model)

----------

📊 Final Model Performance
| Metric           | Value  |
| ---------------- | ------ |
| Accuracy         | 86.96% |
| ROC-AUC          | 0.912  |
| Mean CV Accuracy | 82.88% |
| False Negatives  | 13     |


The SVM model achieved the highest performance and was selected as the final predictive engine.

----------

🏗 Project Structure

heart-disease-prediction-system/
│
├── data/
├── models/
├── src/
├── main.py
├── predict.py
└── README.md

----------

▶️ How to Run

1. Install dependencies:

pip install -r requirements.txt

2. Train models:

python main.py

3. Run prediction:

python predict.py

----------

📈 Evaluation Methods

- Accuracy
- Precision / Recall / F1-score
- Confusion Matrix
- ROC-AUC
- 5-Fold Cross-Validation

---------- 

🚀 Future Improvements

- Deep Learning integration
- Web-based deployment (Streamlit)
- Integration with hospital databases
- Real-time patient monitoring
