"""
Step 2 - Split the REAL cleaned data into train/test BEFORE any synthetic augmentation.

WHY THIS STEP EXISTS: augmentation (step 3, augment.py) creates many jittered near-duplicates
of each real row. If augmentation ran on the full dataset and *then* we split 80/20, several
near-duplicate copies of the same real customer would land in both the train and test sets -
the model would effectively be evaluated on data it had almost already seen (data leakage),
inflating every reported metric. Splitting the real data first and augmenting only the train
portion guarantees the test set is 100% real, untouched, and never influenced the model.

Input : data/processed/telco_churn_cleaned.csv         (7,032 real rows)
Output: data/processed/real_train.csv                    (5,625 real rows, 80%)
        data/processed/real_test.csv                     (1,407 real rows, 20% - held out, real-only)
"""
import pandas as pd
from sklearn.model_selection import train_test_split

IN_PATH = "data/processed/telco_churn_cleaned.csv"
TRAIN_OUT = "data/processed/real_train.csv"
TEST_OUT = "data/processed/real_test.csv"
SEED = 42


def main():
    df = pd.read_csv(IN_PATH)
    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df["Churn"], random_state=SEED
    )
    train_df.to_csv(TRAIN_OUT, index=False)
    test_df.to_csv(TEST_OUT, index=False)

    print(f"Real train: {len(train_df)} rows (churn rate {(train_df['Churn'] == 'Yes').mean():.2%}) -> {TRAIN_OUT}")
    print(f"Real test:  {len(test_df)} rows (churn rate {(test_df['Churn'] == 'Yes').mean():.2%}) -> {TEST_OUT}")
    print("The test set is real, untouched data - it is NEVER augmented and NEVER seen during training.")


if __name__ == "__main__":
    main()
