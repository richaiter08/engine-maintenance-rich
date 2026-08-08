import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

FEATURE_COLUMNS = [
    "Engine_RPM",
    "Lub_Oil_Pressure",
    "Fuel_Pressure",
    "Coolant_Pressure",
    "Lub_Oil_Temperature",
    "Coolant_Temperature",
]
TARGET_LABELS = {
    0: "Normal / Healthy",
    1: "Maintenance Required",
}

BASE_DIR = Path(__file__).resolve().parent
MODEL_CANDIDATES = [
    BASE_DIR / "artifacts" / "best_model.pkl",
    BASE_DIR / "src" / "best_random_forest.pkl",
]


def load_model():
    for model_path in MODEL_CANDIDATES:
        if model_path.exists():
            try:
                return joblib.load(model_path)
            except (ValueError, AttributeError):
                continue

    # Fallback: train if no saved model exists or all candidates failed to load
    from train_pipeline import train_and_save_model

    model, _ = train_and_save_model()
    return model


@st.cache_resource
def get_model():
    return load_model()


st.set_page_config(page_title="Predictive Maintenance", layout="wide")
st.title("Engine Predictive Maintenance Dashboard")
st.caption("ML deployment demo for engine maintenance prediction")

with st.sidebar:
    st.header("Input Parameters")
    user_inputs = {
        "Engine_RPM": st.number_input(
            "Engine RPM", min_value=0.0, value=750.0, step=10.0
        ),
        "Lub_Oil_Pressure": st.number_input(
            "Lub Oil Pressure", min_value=0.0, value=4.0, step=0.1
        ),
        "Fuel_Pressure": st.number_input(
            "Fuel Pressure", min_value=0.0, value=6.0, step=0.1
        ),
        "Coolant_Pressure": st.number_input(
            "Coolant Pressure", min_value=0.0, value=2.5, step=0.1
        ),
        "Lub_Oil_Temperature": st.number_input(
            "Lub Oil Temperature (°C)", min_value=0.0, value=80.0, step=1.0
        ),
        "Coolant_Temperature": st.number_input(
            "Coolant Temperature (°C)", min_value=0.0, value=78.0, step=1.0
        ),
    }

    predict_button = st.button("Predict Engine Condition")

if predict_button:
    model = get_model()
    df = pd.DataFrame([user_inputs], columns=FEATURE_COLUMNS)
    prediction = int(model.predict(df)[0])
    prob = model.predict_proba(df)[0]
    confidence = round(float(prob.max() * 100), 2)

    st.subheader("Prediction Result")
    label = TARGET_LABELS.get(prediction, str(prediction))
    st.markdown(f"## {label}")
    st.metric("Confidence", f"{confidence}%")

    st.write("Prediction probabilities:")
    st.dataframe(
        pd.DataFrame(
            {
                "Class": [TARGET_LABELS.get(i, str(i)) for i in range(len(prob))],
                "Probability": prob.round(4),
            }
        )
    )

    with open(BASE_DIR / "artifacts" / "metrics.json", "r", encoding="utf-8") as f:
        metrics = json.load(f)
    st.subheader("Model Metrics")
    st.json(metrics)

else:
    st.info(
        "Use the sidebar to enter engine sensor values and click on Predict Engine Condition."
    )
