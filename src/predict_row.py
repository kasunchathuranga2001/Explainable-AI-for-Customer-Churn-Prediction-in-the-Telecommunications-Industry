"""
Quick sanity-check tool: run the trained pipeline against one or more REAL rows from the
cleaned dataset (not synthetic) and compare the model's prediction to the true label. Also
prints the row's values in the exact order the Streamlit sidebar needs, so you can re-enter
the same customer manually in the app and confirm the UI gives the same prediction.

Usage:
  python src/predict_row.py                 # 3 random real rows
  python src/predict_row.py --index 42       # a specific row by index
  python src/predict_row.py --n 10           # 10 random rows
  python src/predict_row.py --churned-only    # only sample real customers who actually churned
"""
import argparse

import joblib
import pandas as pd

from features import ALL_FEATURE_COLS

DATA_PATH = "data/processed/telco_churn_cleaned.csv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=None, help="specific row index to test")
    parser.add_argument("--n", type=int, default=3, help="number of random rows to test")
    parser.add_argument("--churned-only", action="store_true", help="only sample rows where Churn == Yes")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    model = joblib.load("models/customer_churn_model.pkl")
    preprocessor = joblib.load("models/preprocessor.pkl")
    df = pd.read_csv(DATA_PATH)

    if args.index is not None:
        rows = df.loc[[args.index]]
    else:
        pool = df[df["Churn"] == "Yes"] if args.churned_only else df
        rows = pool.sample(n=args.n, random_state=args.seed)

    X = rows[ALL_FEATURE_COLS]
    proba = model.predict_proba(preprocessor.transform(X))[:, 1]
    pred = (proba >= 0.5).astype(int)

    for i, (idx, row) in enumerate(rows.iterrows()):
        actual = row["Churn"]
        predicted = "Yes" if pred[i] == 1 else "No"
        match = "correct" if predicted == actual else "MISMATCH"
        print(f"\n=== Row index {idx} ({match}) ===")
        print(f"Predicted churn probability: {proba[i]:.2%}  ->  {predicted}   |   Actual: {actual}")
        print("Sidebar values for this customer:")
        for col in ALL_FEATURE_COLS:
            print(f"  {col:18s} = {row[col]}")


if __name__ == "__main__":
    main()
