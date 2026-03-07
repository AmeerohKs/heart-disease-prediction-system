import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('D:\Document\DSA\DSA 3\DSA3_2-2568\Final Project Proposal\data\heart_disease_uci.csv')
df = df.rename(columns={
    'thalch': 'thalach',
    'num': 'target'
})

#Age distribution of patients
plt.hist(df['age'], bins=20)
plt.title("Age Distribution of Patients")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

#Cholesterol Level Distribution
plt.hist(df['chol'], bins=20)
plt.title("Cholesterol Level Distribution")
plt.xlabel("Cholesterol")
plt.ylabel("Frequency")
plt.show()

#Maximum Heart Rate Distribution
plt.hist(df['thalach'], bins=20)
plt.title("Maximum Heart Rate Distribution")
plt.xlabel("Heart Rate")
plt.ylabel("Frequency")
plt.show()

#Heart Disease Class Distribution
plt.title("Heart Disease Class Distribution")
plt.xlabel("Class")
plt.ylabel("Count")
plt.xticks([0,1],["Normal","Heart Disease"])
plt.show()

#Feature Correlation Heatmap