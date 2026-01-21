"""
Hotel Recommendation Module (Production Ready)

- User selection by NAME (resolved from dataset/users.csv)
- Collaborative Filtering predictions
- Deterministic hotel name resolution
- Strict artifact validation
- Diagnostic error codes for ops support

Error Codes:
HR-E001  Model artifact loading failure
HR-E002  User dataset loading / validation failure
HR-E003  User resolution failure
HR-E004  Recommendation inference failure
HR-W001  Cold-start or empty recommendation set
"""

import os
import joblib
import streamlit as st
import pandas as pd
from typing import Dict, Tuple


# ==================================================
# CONFIGURATION
# ==================================================

MODEL_DIR = "Models/Hotel_Recommendation_Model"
DATASET_DIR = "dataset"
USERS_FILE = "users.csv"
TOP_K = 5


# ==================================================
# CUSTOM EXCEPTION
# ==================================================

class HotelRecError(Exception):
    """Base exception for hotel recommendation failures."""
    pass


# ==================================================
# LOAD USERS (SOURCE OF TRUTH)
# ==================================================

@st.cache_resource(show_spinner=False)
def load_users() -> Tuple[Dict[int, str], Dict[str, int]]:
    """
    Loads users.csv and builds:
    - user_id -> user_name
    - user_name -> user_id
    """

    users_path = os.path.join(DATASET_DIR, USERS_FILE)

    try:
        users_df = pd.read_csv(users_path)
    except Exception as e:
        raise HotelRecError(f"[HR-E002] Failed to load users.csv: {e}")

    required_cols = {"code", "name"}
    if not required_cols.issubset(users_df.columns):
        raise HotelRecError(
            f"[HR-E002] users.csv must contain columns: {required_cols}"
        )

    users_df = users_df.dropna(subset=["code", "name"])

    user_lookup = (
        users_df[["code", "name"]]
        .drop_duplicates()
        .set_index("code")["name"]
        .to_dict()
    )

    if not user_lookup:
        raise HotelRecError("[HR-E002] users.csv contains no valid users")

    name_to_user = {v: k for k, v in user_lookup.items()}

    return user_lookup, name_to_user


# ==================================================
# LOAD MODEL ARTIFACTS
# ==================================================

@st.cache_resource(show_spinner=False)
def load_model_artifacts():
    """
    Loads and validates all model artifacts.
    """

    try:
        cf_preds = joblib.load(
            os.path.join(MODEL_DIR, "cf_predictions_df.pkl")
        )
        user_to_idx = joblib.load(
            os.path.join(MODEL_DIR, "user_to_idx.pkl")
        )
        hotel_lookup = joblib.load(
            os.path.join(MODEL_DIR, "hotel_lookup.pkl")
        )
    except Exception as e:
        raise HotelRecError(f"[HR-E001] Artifact loading failed: {e}")

    if not isinstance(cf_preds, pd.DataFrame):
        raise HotelRecError("[HR-E001] cf_predictions_df is invalid")

    if not isinstance(user_to_idx, dict):
        raise HotelRecError("[HR-E001] user_to_idx is invalid")

    if not isinstance(hotel_lookup, dict):
        raise HotelRecError("[HR-E001] hotel_lookup is invalid")

    return cf_preds, user_to_idx, hotel_lookup


# ==================================================
# RECOMMENDATION ENGINE
# ==================================================

def recommend_hotels(
    user_name: str,
    cf_preds: pd.DataFrame,
    user_to_idx: Dict[int, int],
    name_to_user: Dict[str, int],
    hotel_lookup: Dict[int, str],
    top_k: int = TOP_K
) -> pd.DataFrame:
    """
    Returns top-K hotel recommendations for a given user name.
    """

    if user_name not in name_to_user:
        raise HotelRecError(
            f"[HR-E003] User '{user_name}' not found in users.csv"
        )

    user_id = name_to_user[user_name]

    if user_id not in user_to_idx:
        raise HotelRecError(
            f"[HR-E003] User '{user_name}' has no trained interactions"
        )

    user_idx = user_to_idx[user_id]

    try:
        scores = (
            cf_preds.iloc[:, user_idx]
            .sort_values(ascending=False)
            .head(top_k)
        )
    except Exception as e:
        raise HotelRecError(f"[HR-E004] Prediction failed: {e}")

    if scores.empty:
        return pd.DataFrame(columns=["Hotel", "Score"])

    results = []
    for hotel_encoded, score in scores.items():
        hotel_name = hotel_lookup.get(hotel_encoded)
        if hotel_name:
            results.append({
                "Hotel": hotel_name,
                "Score": round(float(score), 3)
            })

    return pd.DataFrame(results)


# ==================================================
# STREAMLIT PAGE
# ==================================================

def hotel_recommendation_page():
    st.title("Hotel Recommendation")

    try:
        _, name_to_user = load_users()
        cf_preds, user_to_idx, hotel_lookup = load_model_artifacts()
    except HotelRecError as e:
        st.error(str(e))
        st.stop()

    user_name = st.selectbox(
        "Select User",
        sorted(name_to_user.keys()),
        help="Users loaded from dataset/users.csv"
    )

    if st.button("Get Recommendations"):
        try:
            recommendations = recommend_hotels(
                user_name=user_name,
                cf_preds=cf_preds,
                user_to_idx=user_to_idx,
                name_to_user=name_to_user,
                hotel_lookup=hotel_lookup
            )

            if recommendations.empty:
                st.warning(
                    "[HR-W001] No recommendations available for this user"
                )
            else:
                st.table(recommendations)

        except HotelRecError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"[HR-E999] Unexpected system error: {e}")


# ==================================================
# ENTRY POINT
# ==================================================

if __name__ == "__main__":
    hotel_recommendation_page()
