import streamlit as st
import numpy as np
import joblib
import tensorflow as tf
import os


MODEL_DIR = "models/gender"


@st.cache_resource
def load_gender_model():
    try:
        model = tf.keras.models.load_model(
            os.path.join(MODEL_DIR, "model.keras")
        )
        tokenizer = joblib.load(os.path.join(MODEL_DIR, "tokenizer.pkl"))
        label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
        return model, tokenizer, label_encoder
    except Exception as e:
        raise RuntimeError(f"[GC-E001] Failed to load gender model artifacts: {e}")


def gender_classification_page():
    st.title("Gender Classification from Name")

    try:
        model, tokenizer, label_encoder = load_gender_model()
    except Exception as e:
        st.error(str(e))
        return

    name = st.text_input("Enter a name")

    if st.button("Predict Gender") and name.strip():
        try:
            seq = tokenizer.texts_to_sequences([name])
            padded = tf.keras.preprocessing.sequence.pad_sequences(
                seq, maxlen=20
            )

            preds = model.predict(padded)
            gender = label_encoder.inverse_transform([np.argmax(preds)])

            confidence = float(np.max(preds)) * 100
            st.success(f"Predicted Gender: {gender[0]} ({confidence:.2f}%)")

        except Exception as e:
            st.error(f"[GC-E002] Prediction error: {e}")
