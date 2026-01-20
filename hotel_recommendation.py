import streamlit as st
import pandas as pd
import joblib
import os


MODEL_DIR = "models/hotel_recommendation"


@st.cache_resource
def load_cf_artifacts():
    try:
        cf_preds = joblib.load(os.path.join(MODEL_DIR, "cf_predictions_df.pkl"))
        hotel_lookup = joblib.load(os.path.join(MODEL_DIR, "hotel_lookup.pkl"))
        user_to_idx = joblib.load(os.path.join(MODEL_DIR, "user_to_idx.pkl"))
        idx_to_hotel = joblib.load(os.path.join(MODEL_DIR, "hotel_to_idx.pkl"))
        return cf_preds, hotel_lookup, user_to_idx, idx_to_hotel
    except Exception as e:
        raise RuntimeError(f"[HR-E001] Failed to load CF artifacts: {e}")


def hotel_recommendation_page():
    st.title("Hotel Recommendation System")

    try:
        cf_preds, hotel_lookup, user_to_idx, idx_to_hotel = load_cf_artifacts()
    except Exception as e:
        st.error(str(e))
        return

    user_ids = sorted(user_to_idx.keys())
    user_id = st.selectbox("Select User ID", user_ids)

    if st.button("Get Recommendations"):
        try:
            user_idx = user_to_idx[user_id]
            scores = cf_preds.iloc[:, user_idx].sort_values(ascending=False)

            top_hotels = []
            for hotel_idx in scores.index[:5]:
                hotel_name = hotel_lookup.get(hotel_idx, "Unknown Hotel")
                top_hotels.append({
                    "Hotel": hotel_name,
                    "Score": round(scores.loc[hotel_idx], 3)
                })

            st.table(pd.DataFrame(top_hotels))

        except KeyError:
            st.error("[HR-E002] User not found in recommendation matrix.")
        except Exception as e:
            st.error(f"[HR-E003] Recommendation failed: {e}")
