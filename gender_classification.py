import os
import joblib
import numpy as np
import streamlit as st
import tensorflow as tf
import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences

# =====================================================
# ERROR CLASS
# =====================================================

class GenderClassificationError(Exception):
    def __init__(self, code, message):
        self.code = code
        self.message = message
        super().__init__(f"[{code}] {message}")

# =====================================================
# PATHS
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "Models", "Gender_Classification_Model")

TOKENIZER_PATH = os.path.join(MODEL_DIR, "artifacts", "tokenizer.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "artifacts", "label_encoder.pkl")

# =====================================================
# LOAD MODEL (KERAS 3 WAY)
# =====================================================

@st.cache_resource(show_spinner=False)
def load_gender_model():
    try:
        tfsm_layer = keras.layers.TFSMLayer(
            MODEL_DIR,
            call_endpoint="serving_default"
        )

        tokenizer = joblib.load(TOKENIZER_PATH)
        label_encoder = joblib.load(ENCODER_PATH)

        return tfsm_layer, tokenizer, label_encoder

    except Exception as e:
        raise GenderClassificationError(
            "GC-002",
            f"Failed to load model: {str(e)}"
        )

# =====================================================
# PREDICTION
# =====================================================

def predict_gender(name: str):
    if not name or not name.strip():
        raise GenderClassificationError("GC-003", "Invalid name input")

    try:
        model_layer, tokenizer, label_encoder = load_gender_model()

        seq = tokenizer.texts_to_sequences([name.lower().strip()])
        if not seq or not seq[0]:
            raise GenderClassificationError("GC-004", "Tokenization failed")

        padded = pad_sequences(seq, maxlen=17, padding="post")

        # Cast to float32 (required by SavedModel signature)
        padded = tf.convert_to_tensor(padded, dtype=tf.float32)

        # 🔴 KERAS 3 FIX: TFSMLayer returns dict
        outputs = model_layer(padded, training=False)
        preds = list(outputs.values())[0].numpy()

        idx = int(np.argmax(preds, axis=1)[0])
        confidence = float(np.max(preds))

        gender = label_encoder.inverse_transform([idx])[0]

        return {
            "gender": gender,
            "confidence": round(confidence, 4)
        }

    except GenderClassificationError:
        raise
    except Exception as e:
        raise GenderClassificationError(
            "GC-005",
            f"Inference failed: {str(e)}"
        )

# =====================================================
# STREAMLIT PAGE
# =====================================================

def gender_classification_page():
    st.subheader("Gender Classification")

    name = st.text_input("Enter full name")

    if st.button("Predict Gender"):
        try:
            result = predict_gender(name)
            st.success(f"Predicted Gender: **{result['gender']}**")
            st.caption(f"Confidence: {result['confidence']}")
        except GenderClassificationError as e:
            st.error(f"Error [{e.code}]")
            st.caption(e.message)
