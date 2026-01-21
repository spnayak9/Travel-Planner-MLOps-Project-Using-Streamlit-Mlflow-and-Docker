import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sklearn

# ==========================================================
# CONFIGURATION
# ==========================================================
MODEL_PATH = "Models/Regression_Model/model.pkl"
DATASET_PATH = "dataset"

REQUIRED_SKLEARN_VERSION = "1.1.3"  # must match training environment

DROP_COLUMNS = [
    "travelCode",
    "userCode",
    "price",
    "date",
    "date_flight",
    "date_hotel"
]

# Categorical columns encoded during training
ENCODED_CATEGORICAL_COLS = ["flightType", "agency", "gender"]

# User-facing categorical inputs
UI_CATEGORICAL_COLS = ["from", "to", "place", "company"]

# ==========================================================
# ENVIRONMENT VALIDATION
# ==========================================================
def validate_environment():
    if sklearn.__version__ != REQUIRED_SKLEARN_VERSION:
        raise RuntimeError(
            f"[FP-E100] scikit-learn version mismatch.\n\n"
            f"Expected: {REQUIRED_SKLEARN_VERSION}\n"
            f"Found: {sklearn.__version__}\n\n"
            f"Fix: pip install scikit-learn=={REQUIRED_SKLEARN_VERSION}"
        )

# ==========================================================
# MODEL LOADING
# ==========================================================
@st.cache_resource
def load_model():
    validate_environment()

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("[FP-E001] model.pkl not found")

    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        raise RuntimeError(
            "[FP-E101] Model deserialization failed.\n"
            "This indicates a binary incompatibility with scikit-learn.\n\n"
            f"Original error:\n{e}"
        )

# ==========================================================
# FEATURE ENGINEERING (TRAINING PARITY)
# ==========================================================
@st.cache_data
def build_feature_frame():
    # Load datasets
    flights = pd.read_csv(os.path.join(DATASET_PATH, "flights.csv"))
    users = pd.read_csv(os.path.join(DATASET_PATH, "users.csv"))
    hotels = pd.read_csv(os.path.join(DATASET_PATH, "hotels.csv"))

    # Merge datasets
    merged = flights.merge(
        users, left_on="userCode", right_on="code", how="left"
    ).merge(
        hotels,
        on=["travelCode", "userCode"],
        how="left",
        suffixes=("_flight", "_hotel")
    )

    # ---------------- Date features ----------------
    if "date_flight" not in merged.columns:
        raise KeyError("[FP-E010] date_flight missing after merge")

    merged["date_flight"] = pd.to_datetime(
        merged["date_flight"], errors="coerce"
    )

    merged["month"] = merged["date_flight"].dt.month.fillna(0).astype(int)
    merged["day"] = merged["date_flight"].dt.day.fillna(0).astype(int)
    merged["is_weekend"] = (
        merged["date_flight"].dt.weekday.isin([5, 6]).astype(int)
    )

    # ---------------- Hotel features ----------------
    if "price_hotel" not in merged.columns:
        raise KeyError("[FP-E011] price_hotel missing after merge")

    merged["avg_price_per_day"] = (
        merged["price_hotel"] / merged["days"]
    ).replace([np.inf, -np.inf], 0).fillna(0)

    merged["total_stay_cost"] = (
        merged["price_hotel"] * merged["days"]
    ).fillna(0)

    # ---------------- Ratio feature ----------------
    merged["stay_length_ratio"] = (
        merged["days"] / merged["time"]
    ).replace([np.inf, -np.inf], 0).fillna(0)

    # ---------------- Feature selection ----------------
    features = merged.drop(columns=DROP_COLUMNS, errors="ignore")

    # Remove any datetime remnants
    features = features.drop(
        columns=features.select_dtypes(
            include=["datetime64[ns]"]
        ).columns,
        errors="ignore"
    )

    # Encode known categoricals
    cat_df = features[ENCODED_CATEGORICAL_COLS].astype(str)
    num_df = features.drop(columns=ENCODED_CATEGORICAL_COLS)

    cat_encoded = pd.get_dummies(cat_df, drop_first=True)

    final_features = pd.concat([num_df, cat_encoded], axis=1)

    # Ensure ALL features numeric
    for col in final_features.columns:
        if final_features[col].dtype == "object":
            final_features[col] = (
                final_features[col]
                .astype("category")
                .cat.codes
                .astype(float)
            )

    return merged, final_features

# ==========================================================
# STREAMLIT PAGE
# ==========================================================
def flight_price_prediction_page():
    st.title("Flight Price Prediction")

    # -------- Load model & features --------
    try:
        model = load_model()
        merged_df, feature_df = build_feature_frame()
    except Exception as e:
        st.error(str(e))
        return

    # -------- Schema validation --------
    if feature_df.shape[1] != model.n_features_in_:
        st.error(
            f"[FP-E002] Feature schema mismatch.\n"
            f"Expected: {model.n_features_in_}, "
            f"Found: {feature_df.shape[1]}"
        )
        return

    st.success("Feature schema validated (26 features)")

    # -------- Reference row --------
    st.subheader("Select Reference Trip")
    idx = st.selectbox("Choose a base record", merged_df.index, index=0)
    base_row = feature_df.loc[idx]

    # -------- UI mappings --------
    category_maps = {}
    for col in UI_CATEGORICAL_COLS:
        if col in merged_df.columns:
            values = sorted(merged_df[col].dropna().unique())
            category_maps[col] = {v: i for i, v in enumerate(values)}

    # -------- Inputs --------
    st.subheader("Adjust Feature Values")
    user_inputs = {}

    for col in feature_df.columns:

        # User-friendly categorical dropdowns
        if col in UI_CATEGORICAL_COLS:
            options = list(category_maps[col].keys())
            selected = st.selectbox(col, options)
            user_inputs[col] = category_maps[col][selected]

        # Hide dummy variables
        elif col.startswith(("flightType_", "agency_", "gender_")):
            user_inputs[col] = base_row[col]

        # Numeric fields
        else:
            user_inputs[col] = st.number_input(
                col, value=float(base_row[col])
            )

    # -------- Prediction --------
    if st.button("Predict Flight Price"):
        try:
            X = pd.DataFrame([user_inputs], columns=feature_df.columns)
            prediction = model.predict(X)[0]
            st.success(f"Predicted Flight Price: $ {prediction:,.2f}")
        except Exception as e:
            st.error(f"[FP-E003] Prediction failed: {e}")
