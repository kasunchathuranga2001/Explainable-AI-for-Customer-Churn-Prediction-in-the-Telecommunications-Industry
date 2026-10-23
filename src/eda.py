"""
Step 3.5 - Exploratory Data Analysis. Produces the standard figures a research write-up
needs before modelling: class balance, feature correlation structure, and a check that the
synthetic augmentation (step 3, augment.py) didn't distort the real data's distributions.

Run after split_real.py and augment.py.

Outputs (reports/figures/):
  eda_churn_distribution.png     - class balance on the real data
  eda_correlation_heatmap.png    - correlation among numeric features + churn
  eda_real_vs_synthetic.png      - real vs augmented distributions for tenure/MonthlyCharges/TotalCharges
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from features import NUMERIC_COLS

REAL_TRAIN_PATH = "data/processed/real_train.csv"
AUGMENTED_PATH = "data/processed/telco_churn_augmented_100000.csv"


def plot_churn_distribution(real_train: pd.DataFrame):
    counts = real_train["Churn"].value_counts()
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.bar(counts.index, counts.values, color=["#4C72B0", "#C44E52"])
    for i, v in enumerate(counts.values):
        ax.text(i, v + 20, f"{v}\n({v / counts.sum():.1%})", ha="center")
    ax.set_title("Churn Class Balance (real training data)")
    ax.set_ylabel("Number of customers")
    fig.tight_layout()
    fig.savefig("reports/figures/eda_churn_distribution.png", dpi=150)
    plt.close(fig)


def plot_correlation_heatmap(real_train: pd.DataFrame):
    df = real_train.copy()
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    df["SeniorCitizen"] = (df["SeniorCitizen"] == "Yes").astype(int)
    corr_cols = NUMERIC_COLS + ["SeniorCitizen", "Churn"]
    corr = df[corr_cols].corr()

    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr_cols)), corr_cols, rotation=45, ha="right")
    ax.set_yticks(range(len(corr_cols)), corr_cols)
    for i in range(len(corr_cols)):
        for j in range(len(corr_cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Feature Correlation (real training data)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig("reports/figures/eda_correlation_heatmap.png", dpi=150)
    plt.close(fig)


def plot_real_vs_synthetic(real_train: pd.DataFrame, augmented: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, col in zip(axes, NUMERIC_COLS):
        ax.hist(real_train[col], bins=30, density=True, alpha=0.6, label="Real (train split)", color="#4C72B0")
        ax.hist(augmented[col], bins=30, density=True, alpha=0.4, label="Synthetic-augmented", color="#DD8452")
        ax.set_title(col)
        ax.set_ylabel("Density")
    axes[0].legend()
    fig.suptitle("Real vs Synthetic-Augmented Distributions - does augmentation preserve shape?")
    fig.tight_layout()
    fig.savefig("reports/figures/eda_real_vs_synthetic.png", dpi=150)
    plt.close(fig)


def main():
    real_train = pd.read_csv(REAL_TRAIN_PATH)
    augmented = pd.read_csv(AUGMENTED_PATH)

    plot_churn_distribution(real_train)
    plot_correlation_heatmap(real_train)
    plot_real_vs_synthetic(real_train, augmented)

    print("Saved EDA figures -> reports/figures/eda_churn_distribution.png")
    print("Saved EDA figures -> reports/figures/eda_correlation_heatmap.png")
    print("Saved EDA figures -> reports/figures/eda_real_vs_synthetic.png")


if __name__ == "__main__":
    main()
