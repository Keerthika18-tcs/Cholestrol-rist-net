🧠 Cholesterol Risk Net
A Machine Learning Framework for Predicting High Cholesterol Levels Using SMOTE and XGBoost
📌 Project Overview

Cholesterol Risk Net is a machine learning project designed to predict whether a person has high cholesterol levels using health-related data.
The model addresses class imbalance using SMOTE and applies XGBoost, a powerful gradient boosting algorithm, to achieve high prediction accuracy.

This project helps in early identification of cholesterol risk, which can support preventive healthcare decisions.

🎯 Objective

Predict high cholesterol (binary classification)

Handle imbalanced medical data

Improve prediction performance using SMOTE + XGBoost

Visualize results for better interpretation

🧪 Dataset

Source: Cardio dataset (cardio_train.csv)

Target Variable:
chol_high

0 → Normal cholesterol

1 → High cholesterol (cholesterol ≥ 2)

⚙️ Technologies & Libraries Used

Python

Pandas & NumPy – Data handling

Scikit-learn – Preprocessing & evaluation

Imbalanced-learn (SMOTE) – Handling class imbalance

XGBoost – Classification model

Matplotlib & Seaborn – Data visualization

🛠 Methodology
1️⃣ Data Loading & Preprocessing

Load dataset using Pandas

Create a binary target variable chol_high

Remove unnecessary columns

Create an additional feature gluc_high

2️⃣ Feature Scaling

Standardize features using StandardScaler

3️⃣ Handling Class Imbalance

Apply SMOTE to balance minority and majority classes

4️⃣ Model Training

Train XGBoost Classifier with tuned hyperparameters

5️⃣ Model Evaluation

Accuracy Score

ROC-AUC Score

Confusion Matrix

Classification Report

📊 Model Evaluation Metrics

Accuracy

AUC Score

Precision, Recall, F1-score

Confusion Matrix

📈 Visualizations

The project includes the following visualizations:

🔹 Confusion Matrix Heatmap

🔹 ROC Curve with AUC Score

🔹 Feature Importance Plot from XGBoost

These visualizations help in understanding model performance and feature impact.
