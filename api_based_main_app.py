
import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime

API_URL = "http://localhost:5000"

st.set_page_config(page_title="Hospital Resource Forecasting", page_icon="🏥", layout="centered")
st.title("🏥 Unified Hospital Resource Forecasting App (via API)")
st.markdown("Forecast medicine stock levels and doctor availability by calling your Flask API.")

tab1, tab2 = st.tabs(["📦 Medicine Stock Forecast", "🩺 Doctor Availability Forecast"])

with tab1:
    st.header("📦 Predict Medicine Stock")
    med_id = st.number_input("Medicine ID", min_value=1, step=1)
    day = st.selectbox("Day of the Week", range(7), format_func=lambda x: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][x])
    prev_demand = st.number_input("Previous Day Demand", min_value=0)
    rolling_3 = st.number_input("3-Day Rolling Avg", min_value=0.0)
    rolling_7 = st.number_input("7-Day Rolling Avg", min_value=0.0)
    cumulative = st.number_input("Cumulative Demand", min_value=0)

    if st.button("🔮 Predict Stock Level"):
        payload = {
            "med_id": med_id,
            "day": day,
            "prev_day_demand": prev_demand,
            "rolling_3day_avg": rolling_3,
            "rolling_7day_avg": rolling_7,
            "cumulative_demand": cumulative
        }
        try:
            response = requests.post(f"{API_URL}/predict_stock", json=payload)
            result = response.json()
            st.success(f"📦 Predicted Stock Level: {int(result['predicted_stock_level'])} units")
        except Exception as e:
            st.error(f"❌ Failed: {e}")

with tab2:
    st.header("🩺 Predict Doctor Availability")
    doctors = [
        {"doctor_id": 401, "experience": 12, "shift_type": 0},
        {"doctor_id": 402, "experience": 8, "shift_type": 1},
        {"doctor_id": 403, "experience": 15, "shift_type": 0},
        {"doctor_id": 404, "experience": 22, "shift_type": 1},
        {"doctor_id": 405, "experience": 10, "shift_type": 0}
    ]

    selected_date = st.date_input("📅 Select a Date", value=datetime.today())
    weekday = selected_date.weekday()
    is_weekend = 1 if weekday in [5, 6] else 0
    week_number = pd.Timestamp(selected_date).isocalendar().week

    prev_day_seen = 6
    rolling_3 = 6
    rolling_7 = 6
    rolling_14 = 6
    cumulative = 300

    results = []
    for doc in doctors:
        payload = {
            "doctor_id": doc["doctor_id"],
            "day": weekday,
            "experience": doc["experience"],
            "shift_type": doc["shift_type"],
            "is_weekend": is_weekend,
            "week_number": int(week_number),
            "prev_day_seen": prev_day_seen,
            "rolling_3day_seen": rolling_3,
            "rolling_7day_seen": rolling_7,
            "rolling_14day_seen": rolling_14,
            "cumulative_seen": cumulative,
            "patients_exp_interaction": rolling_3 * doc["experience"]
        }
        try:
            response = requests.post(f"{API_URL}/predict_staff", json=payload)
            result = response.json()
            results.append({
                "Doctor ID": doc["doctor_id"],
                "Experience (yrs)": doc["experience"],
                "Shift": "Morning" if doc["shift_type"] == 0 else "Evening",
                "Predicted Available Staff": round(result["predicted_available_staff"], 2)
            })
        except Exception as e:
            st.error(f"Prediction failed for Doctor {doc['doctor_id']}: {e}")

    if results:
        result_df = pd.DataFrame(results).sort_values(by="Predicted Available Staff", ascending=False)
        st.subheader(f"📋 Predicted Doctor Availability on {selected_date.strftime('%A, %B %d, %Y')}")
        st.dataframe(result_df.reset_index(drop=True))
