"""
Step 1 - Data cleaning for the IBM Telco Customer Churn dataset.

Input : data/raw/Telco-Customer-Churn.csv        (7,043 real customers, IBM/Kaggle source)
Output: data/processed/telco_churn_cleaned.csv    (7,032 rows after removing invalid records)

The raw file has 11 blank TotalCharges values (customers with tenure == 0, i.e. brand-new
accounts that have not been billed yet). These rows are dropped rather than imputed, since
they represent an edge case outside the modelling scope (a customer who cannot yet churn).
"""
import pandas as pd

RAW_PATH = "data/raw/Telco-Customer-Churn.csv"
OUT_PATH = "data/processed/telco_churn_cleaned.csv"


def main():
    df = pd.read_csv(RAW_PATH)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["TotalCharges"]).reset_index(drop=True)
    dropped = before - len(df)

    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    df = df.drop(columns=["customerID"])

    df.to_csv(OUT_PATH, index=False)

    churn_rate = (df["Churn"] == "Yes").mean()
    print(f"Loaded {before} rows, dropped {dropped} rows with missing TotalCharges.")
    print(f"Cleaned dataset: {len(df)} rows, {df.shape[1]} columns -> {OUT_PATH}")
    print(f"Churn rate: {churn_rate:.2%}")


if __name__ == "__main__":
    main()
