import pandas as pd
import numpy as np

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# Import your previous functions
from src.feature_engineering import create_features, build_pipeline


# -----------------------------
# LOAD DATA
# -----------------------------
def load_data():
    df = pd.read_parquet("data/processed/telco_churn.parquet")

    # Add time simulation again
    df["snapshot_date"] = pd.date_range(
        start="2020-01-01",
        periods=len(df),
        freq="D"
    )

    df = df.sort_values("snapshot_date")
    return df


# -----------------------------
# PREPARE DATA
# -----------------------------
def prepare_data(df):
    df = create_features(df)
    X, y, preprocessor = build_pipeline(df)

    return X, y, preprocessor


# -----------------------------
# TRAIN + EVALUATE
# -----------------------------
def train_and_evaluate(X, y, preprocessor):

    tscv = TimeSeriesSplit(n_splits=5)

    models = {
        "LogisticRegression": LogisticRegression(class_weight="balanced", max_iter=1000),
        "RandomForest": RandomForestClassifier(class_weight="balanced", n_estimators=100)
    }

    results = {}

    for name, model in models.items():
        print(f"\n🚀 Training {name}...")

        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        pr_aucs = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            pipeline.fit(X_train, y_train)

            y_probs = pipeline.predict_proba(X_test)[:, 1]

            pr_auc = average_precision_score(y_test, y_probs)
            pr_aucs.append(pr_auc)

            print(f"Fold {fold+1} PR-AUC: {pr_auc:.4f}")

        avg_score = np.mean(pr_aucs)
        results[name] = avg_score

        print(f"✅ {name} Average PR-AUC: {avg_score:.4f}")

    return results


# -----------------------------
# MAIN
# -----------------------------
def run_training():
    df = load_data()
    X, y, preprocessor = prepare_data(df)

    results = train_and_evaluate(X, y, preprocessor)

    print("\n📊 FINAL RESULTS:")
    for model, score in results.items():
        print(f"{model}: {score:.4f}")


if __name__ == "__main__":
    run_training()