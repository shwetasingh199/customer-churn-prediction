# src/data_ingestion.py

import pandas as pd
import os

# -----------------------------
# 1. FILE PATHS
# -----------------------------
RAW_PATH = "data/raw/telco_churn.csv"
PROCESSED_PATH = "data/processed/telco_churn.parquet"

# -----------------------------
# 2. DEFINE SCHEMA (IMPORTANT)
# -----------------------------
DTYPES = {
    "customerID": "string",
    "gender": "category",
    "SeniorCitizen": "int64",
    "Partner": "category",
    "Dependents": "category",
    "tenure": "int64",
    "PhoneService": "category",
    "MultipleLines": "category",
    "InternetService": "category",
    "OnlineSecurity": "category",
    "OnlineBackup": "category",
    "DeviceProtection": "category",
    "TechSupport": "category",
    "StreamingTV": "category",
    "StreamingMovies": "category",
    "Contract": "category",
    "PaperlessBilling": "category",
    "PaymentMethod": "category",
    "MonthlyCharges": "float64",
    "TotalCharges": "float64",  # will fix below
    "Churn": "category"
}

# -----------------------------
# 3. LOAD DATA
# -----------------------------
def load_data():
    print("📥 Loading raw dataset...")
    
    df = pd.read_csv(RAW_PATH)
    
    print("✅ Raw data loaded")
    return df

# -----------------------------
# 4. CLEAN DATA
# -----------------------------
def clean_data(df):
    print("🧹 Cleaning data...")

    # Fix TotalCharges (important bug in dataset)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Drop missing values
    df = df.dropna()

    print("✅ Data cleaned")
    return df

# -----------------------------
# 5. ENFORCE DATA TYPES
# -----------------------------
def enforce_schema(df):
    print("🧠 Enforcing schema...")

    for col, dtype in DTYPES.items():
        if col in df.columns:
            try:
                df[col] = df[col].astype(dtype)
            except Exception as e:
                print(f"⚠️ Issue converting {col}: {e}")

    print("✅ Schema enforced")
    return df

# -----------------------------
# 6. SAVE AS PARQUET
# -----------------------------
def save_parquet(df):
    print("💾 Saving as Parquet...")

    os.makedirs("data/processed", exist_ok=True)

    df.to_parquet(PROCESSED_PATH, index=False)

    print(f"✅ Saved to {PROCESSED_PATH}")

# -----------------------------
# 7. MAIN PIPELINE
# -----------------------------
def run_pipeline():
    df = load_data()
    df = clean_data(df)
    df = enforce_schema(df)
    save_parquet(df)

    print("\n🎉 Data ingestion pipeline completed!")

# -----------------------------
# ENTRY POINT
# -----------------------------
if __name__ == "__main__":
    run_pipeline()