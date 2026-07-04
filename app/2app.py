import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# =====================================================
# CONFIG
# =====================================================
st.set_page_config(
    page_title="City Power Consumption Prediction",
    page_icon="⚡",
    layout="wide"
)

# =====================================================
# LOAD MODEL
# =====================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")
model = joblib.load(MODEL_PATH)

FEATURES = [
    "Temperature",
    "Humidity",
    "Wind Speed",
    "general diffuse flows",
    "diffuse flows",
    "Air Quality Index (PM)",
    "Cloudiness"
]

# =====================================================
# HEADER
# =====================================================
st.markdown("""
<h1 style='text-align:center; color:#00BFFF;'>
⚡ City Power Consumption Prediction System
</h1>
<hr>
""", unsafe_allow_html=True)

# =====================================================
# STEP 1: UPLOAD FILE (MANDATORY FIRST STEP)
# =====================================================
uploaded_file = st.file_uploader("📂 Upload Excel/CSV file to continue", type=["csv", "xlsx"])

# Stop here until file is uploaded
if uploaded_file is None:
    st.info("⬆️ Please upload a file to continue")
    st.stop()

# =====================================================
# STEP 2: READ FILE
# =====================================================
if uploaded_file.name.endswith(".csv"):
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_excel(uploaded_file)

df = df.drop(
    columns=["Power Consumption in A Zone", "Serial Number"],
    errors="ignore"
)

df = df[FEATURES]

st.success("✅ File uploaded successfully")

st.subheader("📊 Uploaded Data Preview")
st.dataframe(df)

# =====================================================
# STEP 3: TAKE FIRST ROW AS DEFAULT VALUES
# =====================================================
base = df.iloc[0].to_dict()

# =====================================================
# STEP 4: SLIDERS (ONLY AFTER UPLOAD)
# =====================================================
st.subheader("⚙️ Adjust Parameters Before Prediction")

col1, col2, col3 = st.columns(3)

with col1:
    temperature = st.slider("🌡 Temperature", 0.0, 50.0, float(base["Temperature"]))

with col2:
    humidity = st.slider("💧 Humidity", 0.0, 100.0, float(base["Humidity"]))

with col3:
    wind_speed = st.slider("🌬 Wind Speed", 0.0, 10.0, float(base["Wind Speed"]))

col4, col5, col6, col7 = st.columns(4)

with col4:
    general_diffuse = st.slider("☀️ General Diffuse", 0.0, 1200.0, float(base["general diffuse flows"]))

with col5:
    diffuse = st.slider("🌤 Diffuse Flows", 0.0, 1000.0, float(base["diffuse flows"]))

with col6:
    aqi = st.slider("🌫 AQI (PM)", 150.0, 160.0, float(base["Air Quality Index (PM)"]))

with col7:
    cloudiness = st.select_slider("☁️ Cloudiness", [0, 1], value=int(base["Cloudiness"]))

# =====================================================
# STEP 5: PREDICT BUTTON
# =====================================================
if st.button("🚀 Predict Power Consumption"):

    input_data = pd.DataFrame([{
        "Temperature": temperature,
        "Humidity": humidity,
        "Wind Speed": wind_speed,
        "general diffuse flows": general_diffuse,
        "diffuse flows": diffuse,
        "Air Quality Index (PM)": aqi,
        "Cloudiness": cloudiness
    }])

    prediction = model.predict(input_data)[0]

    st.success("✅ Prediction Completed")

    st.metric("⚡ Power Consumption", f"{prediction:.2f}")

    st.subheader("📊 Final Input Used")
    st.dataframe(input_data)

    # download
    result = input_data.copy()
    result["Prediction"] = prediction

    csv = result.to_csv(index=False).encode("utf-8")

    st.download_button(
        "📥 Download Result",
        csv,
        "prediction.csv",
        "text/csv"
    )