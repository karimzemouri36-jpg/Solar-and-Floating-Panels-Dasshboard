import streamlit as st
import joblib
import numpy as np

# Load models
model_land = joblib.load("model_land.pkl")
model_water = joblib.load("model_water.pkl")
model_total = joblib.load("model_total.pkl")

st.title("🌞 Solar & Floating Panels Energy Dashboard")

# User inputs
month = st.slider("Month", 1, 12, 6)
hour = st.slider("Hour", 6, 18, 12)
irr_land = st.number_input("Land Irradiance (W/m²)", value=800.0)
temp_land = st.number_input("Land Temperature (°C)", value=25.0)
irr_water = st.number_input("Water Irradiance (W/m²)", value=850.0)
temp_water = st.number_input("Water Temperature (°C)", value=22.0)

# Prediction button
if st.button("Predict Energy"):
    land_pred = model_land.predict([[month, hour, irr_land, temp_land]])[0]
    water_pred = model_water.predict([[month, hour, irr_water, temp_water]])[0]
    total_pred = model_total.predict([[month, hour, irr_land, temp_land, irr_water, temp_water]])[0]

    st.success("### Prediction Results")
    st.write(f"🌱 Land Energy: **{land_pred:.3f} kWh**")
    st.write(f"🌊 Water Energy: **{water_pred:.3f} kWh**")
    st.write(f"⚡ Total Energy: **{total_pred:.3f} kWh**")
