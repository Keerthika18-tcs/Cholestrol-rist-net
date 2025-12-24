import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report,
    confusion_matrix, roc_curve, auc
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb

# 1️⃣ Load
df = pd.read_csv("cardio_train.csv", sep=";")
df['chol_high'] = (df['cholesterol'] >= 2).astype(int)

# 2️⃣ Features / Target
X = df.drop(columns=['id','cardio','cholesterol','gluc','chol_high'])
X['gluc_high'] = (df['gluc'] >= 2).astype(int)
y = df['chol_high']

# 3️⃣ Scale + Balance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_bal, y_bal = SMOTE(random_state=42).fit_resample(X_scaled, y)

X_train, X_test, y_train, y_test = train_test_split(
    X_bal, y_bal, test_size=0.2, random_state=42, stratify=y_bal
)

# 4️⃣ XGBoost
model = xgb.XGBClassifier(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    use_label_encoder=False,
    random_state=42
)
model.fit(X_train, y_train)

# 5️⃣ Evaluate
preds = model.predict(X_test)
probs = model.predict_proba(X_test)[:,1]

print("Accuracy :", accuracy_score(y_test, preds))
print("AUC      :", roc_auc_score(y_test, probs))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, preds))
print("\nClassification Report:\n", classification_report(y_test, preds))

# =======================
# 📈 VISUALIZATIONS
# =======================

# --- Confusion Matrix Heatmap
cm = confusion_matrix(y_test, preds)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Normal Chol","High Chol"],
            yticklabels=["Normal Chol","High Chol"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# --- ROC Curve
fpr, tpr, _ = roc_curve(y_test, probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, color="darkorange", lw=2,
         label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0,1], [0,1], color="navy", lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic")
plt.legend(loc="lower right")
plt.show()
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names = [f"F{i}" for i in range(X.shape[1])]  # generic names if you prefer
# if you kept original columns:
# feature_names = X.columns

plt.figure(figsize=(8,6))
sns.barplot(x=importances[indices], y=np.array(feature_names)[indices], color="teal")
plt.title("XGBoost Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()