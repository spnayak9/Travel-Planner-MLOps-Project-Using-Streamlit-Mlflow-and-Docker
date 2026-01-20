import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
MODEL_PATH = "Models/Regression_Model/model.pkl"
DATASET_PATH = "dataset"

# ------------------------------------------------------------------
# Constants (training contract)
# ------------------------------------------------------------------
DROP_COLUMNS = [
    "travelCode",
    "userCode",
    "price",
    "date",
    "date_flight",
    "date_hotel"
]

CATEGORICAL_COLS = ["flightType", "agency", "gender"]

# ------------------------------------------------------------------
# Load model
# ------------------------------------------------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("[FP-E001] model.pkl not found")
    return joblib.load(MODEL_PATH)

# ------------------------------------------------------------------
# Build feature matrix (FULL training parity)
# ------------------------------------------------------------------
@st.cache_data
def build_feature_matrix():
    flights = pd.read_csv(os.path.join(DATASET_PATH, "flights.csv"))
    users = pd.read_csv(os.path.join(DATASET_PATH, "users.csv"))
    hotels = pd.read_csv(os.path.join(DATASET_PATH, "hotels.csv"))

    # Merge flights + users
    merged = flights.merge(
        users,
        left_on="userCode",
        right_on="code",
        how="left"
    )

    # Merge hotels
    merged = merged.merge(
        hotels,
        on=["travelCode", "userCode"],
        how="left",
        suffixes=("_flight", "_hotel")
    )

    # --------------------------------------------------------------
    # Date features (FLIGHT DATE ONLY)
    # --------------------------------------------------------------
    if "date_flight" not in merged.columns:
        raise KeyError("[FP-E010] 'date_flight' column missing after merge")

    merged["date_flight"] = pd.to_datetime(
        merged["date_flight"], errors="coerce"
    )

    merged["month"] = merged["date_flight"].dt.month.fillna(0).astype(int)
    merged["day"] = merged["date_flight"].dt.day.fillna(0).astype(int)

    merged["is_weekend"] = (
        merged["date_flight"].dt.weekday.isin([5, 6]).astype(int)
    )

    # --------------------------------------------------------------
    # Hotel aggregations
    # --------------------------------------------------------------
    if "price_hotel" not in merged.columns:
        raise KeyError("[FP-E011] 'price_hotel' column missing after merge")

    merged["avg_price_per_day"] = (
        merged["price_hotel"] / merged["days"]
    ).replace([np.inf, -np.inf], 0).fillna(0)

    merged["total_stay_cost"] = (
        merged["price_hotel"] * merged["days"]
    ).fillna(0)

    # --------------------------------------------------------------
    # Ratio feature
    # --------------------------------------------------------------
    merged["stay_length_ratio"] = (
        merged["days"] / merged["time"]
    ).replace([np.inf, -np.inf], 0).fillna(0)

    # --------------------------------------------------------------
    # Feature selection
    # --------------------------------------------------------------
    features = merged.drop(columns=DROP_COLUMNS, errors="ignore")

    # Defensive: remove any datetime columns
    datetime_cols = features.select_dtypes(
        include=["datetime64[ns]"]
    ).columns
    features = features.drop(columns=datetime_cols, errors="ignore")

    # Split categorical / numeric
    cat_df = features[CATEGORICAL_COLS].astype(str)
    num_df = features.drop(columns=CATEGORICAL_COLS)

    # Selective one-hot encoding
    cat_encoded = pd.get_dummies(cat_df, drop_first=True)

    final_features = pd.concat([num_df, cat_encoded], axis=1)
    # Ensure all features are numeric (critical for prediction)
    
    for col in final_features.columns:
        if final_features[col].dtype == "object":
            final_features[col] = (
                final_features[col]
                .astype("category")
                .cat.codes
                .astype(float)
            )


    return merged, final_features

# ------------------------------------------------------------------
# Streamlit page
# ------------------------------------------------------------------
def flight_price_prediction_page():
    st.title("Flight Price Prediction")

    try:
        model = load_model()
        merged_df, feature_df = build_feature_matrix()
    except Exception as e:
        st.error(str(e))
        return

    # Schema validation
    if feature_df.shape[1] != model.n_features_in_:
        st.error(
            f"[FP-E002] Feature schema mismatch: "
            f"model expects {model.n_features_in_}, "
            f"but got {feature_df.shape[1]}"
        )
        return

    st.success("Feature schema validated (26 features)")

    # --------------------------------------------------------------
    # Reference record selection
    # --------------------------------------------------------------
    st.subheader("Select Reference Trip")

    idx = st.selectbox(
        "Choose a base record",
        merged_df.index,
        index=0
    )

    base_row = feature_df.loc[idx]

    # --------------------------------------------------------------
    # Dynamic UI (TYPE SAFE)
    # --------------------------------------------------------------
    st.subheader("Adjust Feature Values")

    user_inputs = {}

    for col in feature_df.columns:
        col_data = feature_df[col]
        value = base_row[col]

        # Numeric features
        if pd.api.types.is_numeric_dtype(col_data):
            user_inputs[col] = st.number_input(
                label=col,
                value=float(value)
            )

        # Categorical / string features
        else:
            options = sorted(col_data.dropna().unique().tolist())

            # Fallback if unseen value
            default_index = options.index(value) if value in options else 0

            user_inputs[col] = st.selectbox(
                label=col,
                options=options,
                index=default_index
            )

    # --------------------------------------------------------------
    # Prediction
    # --------------------------------------------------------------
    if st.button("Predict Flight Price"):
        try:
            X = pd.DataFrame([user_inputs], columns=feature_df.columns)
            prediction = model.predict(X)[0]

            st.success(f"Predicted Flight Price: ₹ {prediction:,.2f}")

        except Exception as e:
            st.error(f"[FP-E003] Prediction failed: {e}")

