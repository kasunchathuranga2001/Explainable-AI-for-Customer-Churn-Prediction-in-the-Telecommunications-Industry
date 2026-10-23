"""
Step 4 - Compute global SHAP feature importance for the trained model and save a summary
plot. Run offline (not on every Streamlit page load) since SHAP over hundreds of rows takes
a few seconds - the app just displays the resulting PNG in the "Global (SHAP)" tab.

Output: reports/figures/shap_summary.png
"""
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from explain import clean_feature_names

BACKGROUND_PATH = "data/processed/background_sample.csv"
SAMPLE_SIZE = 500
SEED = 42


def main():
    model = joblib.load("models/customer_churn_model.pkl")
    preprocessor = joblib.load("models/preprocessor.pkl")
    background = pd.read_csv(BACKGROUND_PATH).sample(
        n=min(SAMPLE_SIZE, 2000), random_state=SEED
    )

    X_enc = preprocessor.transform(background)
    if hasattr(X_enc, "toarray"):
        X_enc = X_enc.toarray()
    feature_names = clean_feature_names(preprocessor.get_feature_names_out())

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_enc)
        # binary classifiers: take the "churn" (positive) class contributions
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        elif shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]
    except Exception:
        explainer = shap.Explainer(model.predict_proba, X_enc)
        sv = explainer(X_enc)
        shap_values = sv.values[:, :, 1] if sv.values.ndim == 3 else sv.values

    fig = plt.figure(figsize=(8, 8))
    shap.summary_plot(shap_values, X_enc, feature_names=feature_names, show=False)
    plt.title("Global Feature Importance (SHAP) - What Drives Churn Company-Wide")
    plt.tight_layout()
    fig.savefig("reports/figures/shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved -> reports/figures/shap_summary.png")


if __name__ == "__main__":
    main()
