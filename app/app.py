import streamlit as st
import joblib
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "models", "best_model.pkl")
model = joblib.load(model_path)

st.set_page_config(
    page_title="CITY POWER CONSUMPTION PROJECT",
    page_icon="⚡",
    layout="wide"
)

st.markdown("""
<style>
.main {
    background-color: #0f172a;
    color: white;
}
h1 {
    text-align: center;
    color: #00E5FF;
    font-weight: bold;
}
.kpi-card {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.4);
    text-align: center;
}
.kpi-title {
    font-size: 14px;
    color: #94a3b8;
}
.kpi-value {
    font-size: 22px;
    font-weight: bold;
    color: #38bdf8;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>⚡ CITY POWER CONSUMPTION PROJECT</h1>", unsafe_allow_html=True)
st.markdown("### Predict  Electricity  Demand")

st.markdown("---")


st.sidebar.header("🎛 Control Panel")

temperature = st.sidebar.slider("🌡 Temperature", 0.0, 50.0, 25.0)
humidity = st.sidebar.slider("💧 Humidity", 0.0, 100.0, 60.0)
wind_speed = st.sidebar.slider("🌬 Wind Speed", 0.0, 10.0, 1.0)
general_diffuse = st.sidebar.slider("☀ General Diffuse", 0.0, 1200.0, 100.0)
diffuse = st.sidebar.slider("🌤 Diffuse Flows", 0.0, 1000.0, 50.0)
aqi = st.sidebar.slider("🏭 AQI (PM)", 150.0, 160.0, 155.0)
cloudiness = st.sidebar.selectbox("☁ Cloudiness", [0, 1])

predict_btn = st.sidebar.button("🚀 Predict Now")


if predict_btn:

    input_data = pd.DataFrame({
        "Temperature": [temperature],
        "Humidity": [humidity],
        "Wind Speed": [wind_speed],
        "general diffuse flows": [general_diffuse],
        "diffuse flows": [diffuse],
        "Air Quality Index (PM)": [aqi],
        "Cloudiness": [cloudiness]
    })

    prediction = model.predict(input_data)[0]

  
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("🌡 Temperature", f"{temperature} °C")

    with col2:
        st.metric("💧 Humidity", f"{humidity} %")

    with col3:
        st.metric("⚡ Predicted Power", f"{prediction:.2f} KW")


    st.markdown("---")

 
    st.subheader("📊 Power Consumption Trend Simulation")

    environmental_factors = [
        "Temperature",
        "Humidity",
        "Wind Speed",
        "General Diffuse",
        "Diffuse Flows",
        "AQI",
        "Cloudiness"
    ]

    values = [
        temperature,
        humidity,
        wind_speed,
        general_diffuse,
        diffuse,
        aqi,
        cloudiness
    ]

    fig, ax = plt.subplots(figsize=(10, 5))

    bars = ax.bar(
        environmental_factors,
        values,
        color="#38bdf8",
        edgecolor="black"
    )

    # Display values on top of each bar
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold"
        )

    ax.set_title("Current Environmental Conditions")
    ax.set_xlabel("Environmental Factors")
    ax.set_ylabel("Measured Values")
    ax.tick_params(axis="x", rotation=20)
    st.pyplot(fig)
  
    csv = pd.DataFrame({
        "Temperature": [temperature],
        "Humidity": [humidity],
        "Wind Speed": [wind_speed],
        "Prediction": [prediction]
    }).to_csv(index=False).encode("utf-8")

    st.download_button("📥 Download Result", csv, "prediction.csv", "text/csv")

    st.success("Prediction Generated Successfully 🚀")


st.markdown("---")
st.markdown("<center style='color:gray'>AI Powered Energy Analytics | ML Dashboard Project</center>", unsafe_allow_html=True)