"""
Step 4 - Tune and compare churn classifiers on the 100k augmented TRAINING pool via 5-fold
cross-validated hyperparameter search, then report final performance on the untouched,
100% real held-out test set (data/processed/real_test.csv) - never augmented, never used
for tuning, so the reported metrics reflect genuine generalization to real customers.

Preprocessing is a single fitted sklearn ColumnTransformer (StandardScaler on the 3 numeric
columns + OneHotEncoder on the 16 categorical columns, categories fixed from
features.CATEGORICAL_OPTIONS so a category absent from a given batch - e.g. a single-row
prediction - still encodes correctly). It is fit ONLY on the training pool.

Models compared: Logistic Regression, Random Forest, XGBoost, CatBoost, LightGBM - each
tuned with RandomizedSearchCV (5-fold StratifiedKFold, scoring=roc_auc).

Class imbalance (only ~27% of customers churn) is handled TWO ways, run as separate
experiments, so the better technique can be reported rather than assumed:
  A. "class_weight"  - each estimator's built-in balanced class-weighting.
  B. "smote"          - SMOTE oversampling applied to the encoded training pool only
                         (never to the real test set), estimators trained with no
                         class-weighting since the resampled data is already balanced.
All 10 (model x strategy) combinations are evaluated on the same untouched real test set,
and the single best-by-real-test-ROC-AUC combination is deployed.

Outputs:
  models/customer_churn_model.pkl   - best classifier overall (refit on the full training pool)
  models/preprocessor.pkl           - fitted ColumnTransformer
  models/model_columns.pkl          - encoded feature names, in order (for SHAP labelling)
  reports/model_comparison.csv      - CV score (train pool) + final metrics (real test set),
                                       for all 10 (model x imbalance-strategy) combinations
  reports/figures/confusion_matrix.png
  reports/figures/roc_curves.png
"""
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from scipy.stats import loguniform, randint
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    RocCurveDisplay, accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from features import (
    ALL_FEATURE_COLS, CATEGORICAL_COLS, CATEGORICAL_OPTIONS, NUMERIC_COLS, TARGET_COL,
)

TRAIN_PATH = "data/processed/telco_churn_augmented_100000.csv"  # synthetic-augmented, training only
TEST_PATH = "data/processed/real_test.csv"                       # 100% real, untouched holdout
SEED = 42
CV_FOLDS = 5
N_ITER = 6  # RandomizedSearchCV iterations per (model, strategy) combination


def build_preprocessor():
    category_lists = [CATEGORICAL_OPTIONS[col] for col in CATEGORICAL_COLS]
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_COLS),
        ("cat", OneHotEncoder(categories=category_lists, drop="first", handle_unknown="ignore"), CATEGORICAL_COLS),
    ])


TREE_PARAM_DIST = {
    "n_estimators": randint(100, 300),
    "max_depth": [4, 6, 8, 12],
    "learning_rate": loguniform(0.02, 0.2),
}


def build_search_specs(strategy: str, pos_weight: float):
    """(model, hyperparameter distribution) pairs for RandomizedSearchCV.
    strategy: 'class_weight' (built-in balanced weighting) or 'smote' (no weighting -
    the training data passed in has already been SMOTE-balanced)."""
    cw = "balanced" if strategy == "class_weight" else None
    scale_pos = pos_weight if strategy == "class_weight" else 1.0
    catboost_auto_weights = "Balanced" if strategy == "class_weight" else None

    return {
        "Logistic Regression": (
            LogisticRegression(max_iter=1000, class_weight=cw),
            {"C": loguniform(1e-3, 1e2)},
        ),
        "Random Forest": (
            RandomForestClassifier(class_weight=cw, random_state=SEED, n_jobs=-1),
            {
                "n_estimators": randint(100, 300),
                "max_depth": [6, 8, 12, 16, None],
                "min_samples_leaf": randint(1, 6),
            },
        ),
        "XGBoost": (
            XGBClassifier(eval_metric="logloss", scale_pos_weight=scale_pos, random_state=SEED),
            {**TREE_PARAM_DIST, "subsample": [0.8, 0.9, 1.0]},
        ),
        "CatBoost": (
            CatBoostClassifier(
                auto_class_weights=catboost_auto_weights, random_state=SEED, verbose=0,
                allow_writing_files=False,
            ),
            {"iterations": randint(100, 300), "depth": [4, 6, 8], "learning_rate": loguniform(0.02, 0.2)},
        ),
        "LightGBM": (
            LGBMClassifier(class_weight=cw, random_state=SEED, verbose=-1),
            {**TREE_PARAM_DIST, "num_leaves": randint(15, 63)},
        ),
    }


def run_strategy(strategy, X_train_enc, y_train, X_test_enc, y_test, cv):
    """Fit + tune all 5 models under one imbalance-handling strategy. Returns
    (results_rows, fitted_models_dict)."""
    if strategy == "smote":
        X_fit, y_fit = SMOTE(random_state=SEED).fit_resample(X_train_enc, y_train)
    else:
        X_fit, y_fit = X_train_enc, y_train

    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    results = []
    fitted = {}
    for name, (estimator, param_dist) in build_search_specs(strategy, pos_weight).items():
        print(f"Tuning {name} [{strategy}] ({N_ITER} candidates x {CV_FOLDS}-fold CV)...")
        search = RandomizedSearchCV(
            estimator, param_distributions=param_dist, n_iter=N_ITER, cv=cv,
            scoring="roc_auc", random_state=SEED, n_jobs=-1, refit=True,
        )
        search.fit(X_fit, y_fit)
        best_model = search.best_estimator_
        key = f"{name} ({strategy})"
        fitted[key] = best_model

        pred = best_model.predict(X_test_enc)
        proba = best_model.predict_proba(X_test_enc)[:, 1]
        results.append({
            "model": name,
            "imbalance_strategy": strategy,
            "best_params": search.best_params_,
            "cv_roc_auc_mean": search.best_score_,
            "cv_roc_auc_std": search.cv_results_["std_test_score"][search.best_index_],
            "test_accuracy": accuracy_score(y_test, pred),
            "test_precision": precision_score(y_test, pred),
            "test_recall": recall_score(y_test, pred),
            "test_f1": f1_score(y_test, pred),
            "test_roc_auc": roc_auc_score(y_test, proba),
        })
        print(f"  CV ROC-AUC: {search.best_score_:.4f}   |   Real-test ROC-AUC: {results[-1]['test_roc_auc']:.4f}")
    return results, fitted


def main():
    train_df = pd.read_csv(TRAIN_PATH).drop(columns=["customerID"])
    test_df = pd.read_csv(TEST_PATH)

    X_train = train_df[ALL_FEATURE_COLS]
    y_train = (train_df[TARGET_COL] == "Yes").astype(int)
    X_test = test_df[ALL_FEATURE_COLS]
    y_test = (test_df[TARGET_COL] == "Yes").astype(int)

    preprocessor = build_preprocessor()
    X_train_enc = preprocessor.fit_transform(X_train)
    X_test_enc = preprocessor.transform(X_test)
    if hasattr(X_train_enc, "toarray"):
        X_train_enc = X_train_enc.toarray()
    if hasattr(X_test_enc, "toarray"):
        X_test_enc = X_test_enc.toarray()
    model_columns = list(preprocessor.get_feature_names_out())

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)

    all_results = []
    all_fitted = {}
    for strategy in ["class_weight", "smote"]:
        results, fitted = run_strategy(strategy, X_train_enc, y_train, X_test_enc, y_test, cv)
        all_results.extend(results)
        all_fitted.update(fitted)

    results_df = pd.DataFrame(all_results).sort_values("test_roc_auc", ascending=False)
    results_df.to_csv("reports/model_comparison.csv", index=False)
    print("\n" + results_df.drop(columns=["best_params"]).to_string(index=False))

    best_row = results_df.iloc[0]
    best_key = f"{best_row['model']} ({best_row['imbalance_strategy']})"
    best_model = all_fitted[best_key]
    print(f"\nBest overall (model x imbalance-strategy): {best_key}  ->  test ROC-AUC {best_row['test_roc_auc']:.4f}")

    # confusion matrix figure for the best combination, on the real test set
    pred = best_model.predict(X_test_enc)
    cm = confusion_matrix(y_test, pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    ax.set_xticks([0, 1], ["Stay", "Churn"])
    ax.set_yticks([0, 1], ["Stay", "Churn"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix - {best_key} (real test set)")
    fig.tight_layout()
    fig.savefig("reports/figures/confusion_matrix.png", dpi=150)
    plt.close(fig)

    # ROC curves - one line per model, best strategy for each model, on the real test set
    best_per_model = results_df.loc[results_df.groupby("model")["test_roc_auc"].idxmax()]
    fig, ax = plt.subplots(figsize=(6, 6))
    for _, row in best_per_model.iterrows():
        key = f"{row['model']} ({row['imbalance_strategy']})"
        model = all_fitted[key]
        RocCurveDisplay.from_predictions(y_test, model.predict_proba(X_test_enc)[:, 1], ax=ax, name=key)
    ax.set_title("ROC Curves - Best Strategy per Model (real test set)")
    fig.tight_layout()
    fig.savefig("reports/figures/roc_curves.png", dpi=150)
    plt.close(fig)

    joblib.dump(best_model, "models/customer_churn_model.pkl")
    joblib.dump(preprocessor, "models/preprocessor.pkl")
    joblib.dump(model_columns, "models/model_columns.pkl")

    # small unscaled/unencoded background sample (from the training pool only) for LIME/DiCE
    X_train.sample(n=min(2000, len(X_train)), random_state=SEED).to_csv(
        "data/processed/background_sample.csv", index=False
    )
    print("\nSaved best model, preprocessor, model_columns, and background sample.")


if __name__ == "__main__":
    main()
