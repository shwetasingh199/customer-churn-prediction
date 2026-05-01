import joblib
from src.xgboost_optuna import run

def save():

    print("🚀 Training and saving model...")

    model = run()

    joblib.dump(model, "models/churn_model.pkl")

    print("✅ Model saved at models/churn_model.pkl")

if __name__ == "__main__":
    save()