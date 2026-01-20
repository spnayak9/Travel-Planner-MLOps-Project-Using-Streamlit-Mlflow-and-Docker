import streamlit as st

from flight_price_prediction import flight_price_prediction_page
from gender_classification import gender_classification_page
from hotel_recommendation import hotel_recommendation_page


def main():
    st.set_page_config(
        page_title="Voyage Analytics",
        layout="wide"
    )

    st.sidebar.title("Voyage Analytics")
    choice = st.sidebar.radio(
        "Select Module",
        [
            "Home",
            "Flight Price Prediction",
            "Gender Classification",
            "Hotel Recommendation"
        ]
    )

    if choice == "Home":
        st.title("Voyage Analytics – Travel Planner MLOps")
        st.markdown("""
        **Production ML Applications**
        - Flight Price Prediction (Regression)
        - Gender Classification (Deep Learning)
        - Hotel Recommendation (Collaborative Filtering)

        All models are served **from prebuilt MLflow artifacts**.
        """)

    elif choice == "Flight Price Prediction":
        flight_price_prediction_page()

    elif choice == "Gender Classification":
        gender_classification_page()

    elif choice == "Hotel Recommendation":
        hotel_recommendation_page()


if __name__ == "__main__":
    main()
