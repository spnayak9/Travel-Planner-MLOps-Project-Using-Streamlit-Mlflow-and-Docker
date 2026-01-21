import joblib
import pandas as pd
import os

MODEL_DIR = "Models/Hotel_Recommendation_Model"

# Load existing artifact
hotel_lookup_df = joblib.load(
    os.path.join(MODEL_DIR, "hotel_lookup.pkl")
)

# Validate
if not isinstance(hotel_lookup_df, pd.DataFrame):
    raise ValueError("hotel_lookup.pkl is not a DataFrame")

required_cols = {"hotel_encoded", "name"}
if not required_cols.issubset(hotel_lookup_df.columns):
    raise ValueError("hotel_lookup DataFrame missing required columns")

# Convert to dict
hotel_lookup_dict = (
    hotel_lookup_df
    .set_index("hotel_encoded")["name"]
    .to_dict()
)

# Overwrite artifact
joblib.dump(
    hotel_lookup_dict,
    os.path.join(MODEL_DIR, "hotel_lookup.pkl")
)

print("hotel_lookup.pkl converted to dictionary successfully")
