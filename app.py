import streamlit as st

from flight_price_prediction import flight_price_prediction_page
from gender_classification import gender_classification_page
from hotel_recommendation import hotel_recommendation_page


def main():
    st.set_page_config(
        page_title="Voyage Analytics",
        page_icon="🌍",
        layout="wide"
    )

    st.sidebar.title("Voyage Analytics")
    selection = st.sidebar.radio(
        "Navigation",
        [
            "Home",
            "Flight Price Prediction",
            "Gender Classification",
            "Hotel Recommendation"
        ]
    )

    if selection == "Home":
        st.title("Voyage Analytics: Integrating MLOps in Travel")
        st.markdown("""
        **Production-ready ML systems** for:
        - Flight price prediction
        - Name-based gender classification
        - Personalized hotel recommendations

        All models are **MLflow-tracked** and **artifact-driven** (no retraining at inference).
        """)

    elif selection == "Flight Price Prediction":
        flight_price_prediction_page()

    elif selection == "Gender Classification":
        gender_classification_page()

    elif selection == "Hotel Recommendation":
        hotel_recommendation_page()


if __name__ == "__main__":
    main()
