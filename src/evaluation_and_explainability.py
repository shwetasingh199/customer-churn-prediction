import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split

# If installed
import shap

from src.feature_engineering import create_features, build_pipeline
from src.xgboost_optuna import run as get_model


# -----------------------------
# LOAD DATA
# -----------------------------
def load_data():
    df = pd.read_parquet("data/processed/telco_churn.parquet")
    df = create_features(df)
    return df


# -----------------------------
# PREPARE DATA
# -----------------------------
def prepare(df):
    X, y, preprocessor = build_pipeline(df)

    # 🔥 FIX: ensure numeric
    y = y.astype(int)

    return X, y


# -----------------------------
# LIFT@K FUNCTION
# -----------------------------
def compute_lift(y_true, y_probs, k=0.1):
    df = pd.DataFrame({
        "y": y_true.astype(int),
        "prob": y_probs
    })

    df = df.sort_values("prob", ascending=False)

    top_k = int(len(df) * k)
    top_df = df.head(top_k)

    lift = (top_df["y"].mean()) / (df["y"].mean())

    return lift


# -----------------------------
# EVALUATION
# -----------------------------
def evaluate(model, X, y):

    # simple split (for final report)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]

    roc = roc_auc_score(y_test, probs)
    pr = average_precision_score(y_test, probs)
    lift = compute_lift(y_test, probs, k=0.1)

    print("\n📊 MODEL PERFORMANCE")
    print(f"ROC-AUC : {roc:.4f}")
    print(f"PR-AUC  : {pr:.4f}")
    print(f"Lift@10%: {lift:.2f}")

    return probs, X_test


# -----------------------------
# SHAP EXPLAINABILITY
# -----------------------------
from sklearn.inspection import permutation_importance

def explain(model, X_sample, y_sample):

    print("\n🧠 Generating SHAP-style explanations (Permutation Importance)...")

    # 🔥 Compute importance
    result = permutation_importance(
        model,
        X_sample,
        y_sample,
        n_repeats=5,
        random_state=42,
        scoring="average_precision"
    )

    importance_df = pd.DataFrame({
        "feature": X_sample.columns,
        "importance": result.importances_mean
    }).sort_values("importance", ascending=False)

    print("\n🔥 TOP FACTORS:")
    print(importance_df.head(10))

    return importance_df
# -----------------------------
# RETENTION ACTIONS
# -----------------------------
def map_actions(importance_df):

    print("\n🎯 RETENTION ACTIONS:")

    actions = []

    for feature in importance_df["feature"].head(5):

        if "tenure" in feature:
            actions.append((feature, "Offer loyalty discounts / long-term plans"))

        elif "MonthlyCharges" in feature:
            actions.append((feature, "Provide cheaper plans / discounts"))

        elif "TechSupport" in feature:
            actions.append((feature, "Offer free tech support"))

        elif "Contract" in feature:
            actions.append((feature, "Push yearly subscription offers"))

        elif "num_services" in feature:
            actions.append((feature, "Bundle more services"))

        else:
            actions.append((feature, "General retention campaign"))

    for f, a in actions:
        print(f"{f} → {a}")


# -----------------------------
# MAIN
# -----------------------------
def run():

    df = load_data()
    X, y = prepare(df)

    model = get_model()

    probs, X_test = evaluate(model, X, y)

    sample_idx = X_test.sample(100).index

    importance_df = explain(
        model,
        X_test.loc[sample_idx],
        y.loc[sample_idx]
    )

    map_actions(importance_df)



if __name__ == "__main__":
    run()