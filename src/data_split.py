import pandas as pd

# -----------------------------
# LOAD PROCESSED DATA
# -----------------------------
def load_data():
    df = pd.read_parquet("data/processed/telco_churn.parquet")
    return df


# -----------------------------
# ADD FAKE TIME COLUMN (SIMULATION)
# -----------------------------
def add_time_index(df):
    """
    Since Telco dataset has no timestamp,
    we simulate time for learning purposes.
    """
    df = df.copy()
    df["snapshot_date"] = pd.date_range(
        start="2020-01-01",
        periods=len(df),
        freq="D"
    )
    return df


# -----------------------------
# TIME-BASED SPLIT
# -----------------------------
def time_based_split(df, split_date="2020-06-01"):
    """
    Train: data before split_date
    Test: data after split_date
    """
    
    train_df = df[df["snapshot_date"] < split_date]
    test_df  = df[df["snapshot_date"] >= split_date]

    print(f"Train size: {len(train_df)}")
    print(f"Test size: {len(test_df)}")

    return train_df, test_df


# -----------------------------
# REMOVE LEAKAGE FEATURES
# -----------------------------
def remove_leakage(df):
    """
    Drop columns that can leak future info
    """
    
    leakage_cols = [
        "customerID"  # ID column (not useful)
    ]

    df = df.drop(columns=leakage_cols, errors="ignore")

    return df


# -----------------------------
# MAIN FUNCTION
# -----------------------------
def run_split_pipeline():
    df = load_data()

    print("📦 Adding time simulation...")
    df = add_time_index(df)

    print("🚫 Removing leakage columns...")
    df = remove_leakage(df)

    print("⏳ Applying time-based split...")
    train_df, test_df = time_based_split(df)

    return train_df, test_df


if __name__ == "__main__":
    train_df, test_df = run_split_pipeline()