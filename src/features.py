"""
Shared feature schema for the churn model.

This module is the single source of truth for which raw columns feed the model and what
values each categorical column can take. Training (train.py), the explainability helpers
(explain.py) and the Streamlit app (app.py) all import from here so the sidebar widgets can
never drift out of sync with what the model was actually trained on.
"""

TARGET_COL = "Churn"

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

CATEGORICAL_COLS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]

ALL_FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS

# allowed values per categorical column - used both to build the Streamlit sidebar widgets
# and as the fixed category list for the OneHotEncoder (so a category unseen in a given
# batch, e.g. a single-row prediction, still produces a valid encoding).
CATEGORICAL_OPTIONS = {
    "gender": ["Female", "Male"],
    "SeniorCitizen": ["No", "Yes"],
    "Partner": ["No", "Yes"],
    "Dependents": ["No", "Yes"],
    "PhoneService": ["No", "Yes"],
    "MultipleLines": ["No", "Yes", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["No", "Yes", "No internet service"],
    "OnlineBackup": ["No", "Yes", "No internet service"],
    "DeviceProtection": ["No", "Yes", "No internet service"],
    "TechSupport": ["No", "Yes", "No internet service"],
    "StreamingTV": ["No", "Yes", "No internet service"],
    "StreamingMovies": ["No", "Yes", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["No", "Yes"],
    "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
}
