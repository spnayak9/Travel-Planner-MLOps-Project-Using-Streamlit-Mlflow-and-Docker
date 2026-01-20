import streamlit as st
import tensorflow as tf
import joblib
import numpy as np
import os


BASE_DIR = "Models/Gender Classification Model"


@st.cache_resource
def load_gender_artifacts():
    try:
        model = tf.keras.models.load_model(
            os.path.join(BASE_DIR, "model.keras")
        )
        tokenizer = joblib.load(os.path.join(BASE_DIR, "tokenizer.pkl"))
        label_encoder = joblib.load(os.path.join(BASE_DIR, "label_encoder.pkl"))
        return model, tokenizer, label_encoder
    except Exception as e:
        raise RuntimeError(f"[GC-E001] Failed loading artifacts: {e}")


def gender_classification_page():
    st.title("Gender Classification")

    try:
        model, tokenizer, label_encoder = load_gender_artifacts()
    except Exception as e:
        st.error(str(e))
        return

    name = st.text_input("Enter Name")

    if st.button("Predict Gender") and name.strip():
        try:
            seq = tokenizer.texts_to_sequences([name])
            padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=20)
            preds = model.predict(padded)
            gender = label_encoder.inverse_transform([np.argmax(preds)])[0]
            confidence = np.max(preds) * 100

            st.success(f"Predicted Gender: {gender}")
            st.info(f"Confidence: {confidence:.2f}%")

        except Exception as e:
            st.error(f"[GC-E002] Prediction error: {e}")
