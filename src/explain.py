"""
Shared explainability helpers (SHAP / LIME / DiCE), used by both the offline report
generator (generate_shap_summary.py) and the live Streamlit app (app.py).
"""
import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
from sklearn.pipeline import Pipeline

from features import ALL_FEATURE_COLS, CATEGORICAL_COLS, NUMERIC_COLS


def clean_feature_names(raw_names):
    """'cat__Contract_Two year' -> 'Contract_Two year', 'num__tenure' -> 'tenure'."""
    return [name.split("__", 1)[-1] for name in raw_names]


def build_full_pipeline(preprocessor, model) -> Pipeline:
    """Combine the already-fitted preprocessor + classifier into one Pipeline so tools like
    DiCE can call .predict_proba directly on a raw (unencoded) customer record. Wrapping
    pre-fitted steps is safe here since we only ever call .predict/.predict_proba on it,
    never .fit()."""
    return Pipeline([("preprocess", preprocessor), ("classifier", model)])


def make_lime_explainer(background_raw: pd.DataFrame, preprocessor) -> LimeTabularExplainer:
    background_enc = preprocessor.transform(background_raw)
    if hasattr(background_enc, "toarray"):
        background_enc = background_enc.toarray()
    feature_names = clean_feature_names(preprocessor.get_feature_names_out())
    return LimeTabularExplainer(
        training_data=np.asarray(background_enc),
        feature_names=feature_names,
        class_names=["Stay", "Churn"],
        mode="classification",
        discretize_continuous=True,
    )


def explain_instance_lime(explainer: LimeTabularExplainer, preprocessor, model, raw_row: pd.DataFrame, num_features=10):
    row_enc = preprocessor.transform(raw_row)
    if hasattr(row_enc, "toarray"):
        row_enc = row_enc.toarray()
    row_enc = np.asarray(row_enc)[0]

    def predict_fn(x):
        return model.predict_proba(x)

    return explainer.explain_instance(row_enc, predict_fn, num_features=num_features)


def generate_counterfactuals(dice_exp, raw_row: pd.DataFrame, total_cfs=3, vary_features=None):
    """desired_class='opposite' flips the customer's current predicted outcome, e.g. for a
    customer currently predicted to churn, it finds the smallest realistic feature changes
    that would flip the prediction to 'stay'."""
    kwargs = {}
    if vary_features:
        kwargs["features_to_vary"] = vary_features
    return dice_exp.generate_counterfactuals(
        raw_row, total_CFs=total_cfs, desired_class="opposite", **kwargs
    )
