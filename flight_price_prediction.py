import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os


MODEL_DIR = r"Models\Regression_Model"


@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
        le_from = joblib.load(os.path.join(MODEL_DIR, "le_from.pkl"))
        le_to = joblib.load(os.path.join(MODEL_DIR, "le_to.pkl"))
        le_type = joblib.load(os.path.join(MODEL_DIR, "le_type.pkl"))
        le_agency = joblib.load(os.path.join(MODEL_DIR, "le_agency.pkl"))
        route_df = pd.read_csv(os.path.join(MODEL_DIR, "new_df.csv"))
        return model, le_from, le_to, le_type, le_agency, route_df
    except Exception as e:
        raise RuntimeError(f"[FP-E001] Failed to load regression artifacts: {e}")


def flight_price_prediction_page():
    st.title("Flight Price Prediction")

    try:
        model, le_from, le_to, le_type, le_agency, route_df = load_artifacts()
    except Exception as e:
        st.error(str(e))
        return

    from_city = st.selectbox("From", sorted(route_df["from"].unique()))
    to_city = st.selectbox("To", sorted(route_df["to"].unique()))

    if from_city == to_city:
        st.warning("[FP-E002] Source and destination cannot be the same.")
        return

    route = route_df[
        (route_df["from"] == from_city) &
        (route_df["to"] == to_city)
    ]

    if route.empty:
        st.error("[FP-E003] Route data not available.")
        return

    time = float(route.iloc[0]["time"])
    distance = float(route.iloc[0]["distance"])

    st.info(f"Estimated Time: {time} hrs | Distance: {distance} km")

    flight_type = st.selectbox("Flight Type", le_type.classes_)
    agency = st.selectbox("Agency", le_agency.classes_)
    passengers = st.number_input("Passengers", min_value=1, value=1)

    if st.button("Predict Price"):
        try:
            X = np.array([[
                le_from.transform([from_city])[0],
                le_to.transform([to_city])[0],
                le_agency.transform([agency])[0],
                le_type.transform([flight_type])[0],
                time,
                distance
            ]])

            price = model.predict(X)[0] * passengers
            st.success(f"Predicted Price: ₹ {price:,.2f}")

        except Exception as e:
            st.error(f"[FP-E004] Prediction failed: {e}")
