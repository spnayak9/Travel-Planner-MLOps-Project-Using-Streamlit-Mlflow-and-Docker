import streamlit as st
import pandas as pd
import joblib
import os


BASE_DIR = "Models/Hotel Recommendation Model"


@st.cache_resource
def load_cf_artifacts():
    try:
        cf_preds = joblib.load(os.path.join(BASE_DIR, "cf_predictions_df.pkl"))
        hotel_lookup = joblib.load(os.path.join(BASE_DIR, "hotel_lookup.pkl"))
        user_to_idx = joblib.load(os.path.join(BASE_DIR, "user_to_idx.pkl"))
        return cf_preds, hotel_lookup, user_to_idx
    except Exception as e:
        raise RuntimeError(f"[HR-E001] Artifact loading failed: {e}")


def hotel_recommendation_page():
    st.title("Hotel Recommendation")

    try:
        cf_preds, hotel_lookup, user_to_idx = load_cf_artifacts()
    except Exception as e:
        st.error(str(e))
        return

    user_id = st.selectbox("Select User ID", sorted(user_to_idx.keys()))

    if st.button("Get Recommendations"):
        try:
            user_idx = user_to_idx[user_id]
            scores = cf_preds.iloc[:, user_idx].sort_values(ascending=False)

            results = []
            for hotel_idx in scores.index[:5]:
                results.append({
                    "Hotel": hotel_lookup.get(hotel_idx, "Unknown"),
                    "Score": round(scores.loc[hotel_idx], 3)
                })

            st.table(pd.DataFrame(results))

        except Exception as e:
            st.error(f"[HR-E002] Recommendation error: {e}")
