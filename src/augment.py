"""
Step 3 - Synthetic augmentation of the REAL TRAINING split to 100,000 rows.

WHY: The authentic IBM Telco dataset only has 7,032 usable rows, which is too small for
the large-sample research goal of this project (100k rows). No public real-world telecom
churn dataset of that size with a matching feature schema exists, so this step generates a
larger *synthetic* dataset that preserves the statistical structure of the real data. This
is an established data-augmentation technique for small tabular datasets, and it is reported
here explicitly so the synthetic origin of ~94% of the training data is never confused with
real customer records.

IMPORTANT: this script only ever reads data/processed/real_train.csv (the 80% train split
produced by split_real.py). It must never touch real_test.csv - augmenting before splitting
would let jittered near-duplicates of the same real customer land in both train and test,
leaking information into the evaluation set and inflating every reported metric.

METHOD: stratified bootstrap resampling with continuous-feature jitter.
  1. Rows are sampled with replacement, stratified by Churn, so the class balance of the
     real data is preserved exactly.
  2. Each sampled row is duplicated as a whole record (preserving every real feature
     correlation, e.g. InternetService <-> OnlineSecurity <-> MonthlyCharges), then Gaussian
     noise is added to the 3 continuous columns (tenure, MonthlyCharges, TotalCharges) so
     rows are not exact duplicates. Noise scale is a fraction of the *within-segment*
     standard deviation (segment = Contract x InternetService), so jitter never invents
     combinations that violate the real distribution's shape.
  3. TotalCharges is re-derived from the jittered tenure/MonthlyCharges so the well-known
     real-world identity TotalCharges ~= tenure * MonthlyCharges holds after jittering.
  4. All values are clipped to the observed real-world ranges.

Input : data/processed/real_train.csv               (5,625 real rows - training split only)
Output: data/processed/telco_churn_augmented_100000.csv
"""
import numpy as np
import pandas as pd

IN_PATH = "data/processed/real_train.csv"
OUT_PATH = "data/processed/telco_churn_augmented_100000.csv"
TARGET_ROWS = 100_000
JITTER_FRACTION = 0.10  # noise std = 10% of within-segment std
SEED = 42


def jitter_group(group: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    tenure_std = max(group["tenure"].std(ddof=0), 1.0)
    charges_std = max(group["MonthlyCharges"].std(ddof=0), 1.0)

    n = len(group)
    tenure = group["tenure"].to_numpy() + rng.normal(0, tenure_std * JITTER_FRACTION, n)
    monthly = group["MonthlyCharges"].to_numpy() + rng.normal(0, charges_std * JITTER_FRACTION, n)

    tenure = np.clip(np.round(tenure), 0, 72)
    monthly = np.clip(np.round(monthly, 2), 18.0, 120.0)
    # keep the real-world identity TotalCharges ~= tenure * MonthlyCharges, with a small
    # amount of extra noise to reflect prorated/partial billing months seen in the real data
    total = tenure * monthly + rng.normal(0, 5.0, n)
    total = np.clip(np.round(total, 2), 0.0, None)

    group = group.copy()
    group["tenure"] = tenure.astype(int)
    group["MonthlyCharges"] = monthly
    group["TotalCharges"] = total
    return group


def main():
    real = pd.read_csv(IN_PATH)
    rng = np.random.default_rng(SEED)

    # stratified bootstrap by Churn to preserve the real class balance exactly
    frames = []
    for churn_value, sub in real.groupby("Churn"):
        n_target = round(TARGET_ROWS * len(sub) / len(real))
        idx = rng.choice(sub.index, size=n_target, replace=True)
        frames.append(real.loc[idx].reset_index(drop=True))
    synthetic = pd.concat(frames, ignore_index=True)
    synthetic = synthetic.sample(frac=1, random_state=SEED).reset_index(drop=True)

    # jitter continuous columns within Contract x InternetService segments
    synthetic = (
        synthetic.groupby(["Contract", "InternetService"], group_keys=False)
        .apply(lambda g: jitter_group(g, rng), include_groups=True)
        .reset_index(drop=True)
    )

    synthetic.insert(0, "customerID", [f"SYN-{i:06d}" for i in range(len(synthetic))])
    synthetic.to_csv(OUT_PATH, index=False)

    real_rate = (real["Churn"] == "Yes").mean()
    synth_rate = (synthetic["Churn"] == "Yes").mean()
    print(f"Real rows: {len(real)}  ->  Synthetic rows: {len(synthetic)}")
    print(f"Churn rate  real: {real_rate:.2%}   synthetic: {synth_rate:.2%}")
    print(f"Saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
