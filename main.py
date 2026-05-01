import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve

from xgboost import XGBClassifier

# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv("data/raw/churn.csv")

# =========================
# 2. CLEAN DATA
# =========================
df.drop("customerID", axis=1, inplace=True)

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(inplace=True)

# =========================
# CUSTOM ENCODING (ONLY REQUIRED FEATURES)
# =========================

df["gender"] = df["gender"].map({"Male": 1, "Female": 0})

contract_map = {
    "Month-to-month": 0,
    "One year": 1,
    "Two year": 2
}
df["Contract"] = df["Contract"].map(contract_map)
# Convert target variable to numeric
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
# =========================
# 4. SPLIT DATA
# =========================
selected_features = [
    "gender",
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "Contract"
]

X = df[selected_features]
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =========================
# 5. SCALE DATA
# =========================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# 6. TRAIN MODEL (XGBOOST)
# =========================
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# =========================
# 7. PREDICT
# =========================
preds = model.predict(X_test)

# =========================
# 8. EVALUATION
# =========================
print("Accuracy:", accuracy_score(y_test, preds))
print("\nClassification Report:\n", classification_report(y_test, preds))
print("ROC-AUC:", roc_auc_score(y_test, preds))

# =========================
# 9. ROC CURVE
# =========================
fpr, tpr, _ = roc_curve(y_test, preds)

plt.figure()
plt.plot(fpr, tpr)
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.savefig("outputs/roc_curve.png")

# =========================
# 10. FEATURE IMPORTANCE
# =========================
feature_importance = model.feature_importances_
features = X.columns

plt.figure(figsize=(10,5))
plt.barh(features, feature_importance)
plt.title("Feature Importance")
plt.savefig("outputs/feature_importance.png")

# =========================
# 11. SAVE MODEL
# =========================
pickle.dump(model, open("models/churn_model.pkl", "wb"))
pickle.dump(scaler, open("models/scaler.pkl", "wb"))

print("✅ Model saved successfully!")