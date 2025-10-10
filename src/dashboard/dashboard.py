import streamlit as st
import requests
from PIL import Image
import io
import time
import pandas as pd
import os

# st.set_page_config(title="AegisAI Dashboard", page_icon="🛡️")
st.set_page_config(page_title="AegisAI Dashboard", page_icon="🛡️")
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.title("🛡️ AegisAI: The AI Immune System")
page = st.sidebar.selectbox("Navigate", ["AI Model Security", "Cloud & Identity Monitoring"])

if page == "AI Model Security":
    st.subheader("Adversarial Attack Simulation")
    uploaded_file = st.file_uploader("Upload an image to test model robustness (png, jpg, jpeg)", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        with st.spinner("Running attack simulation..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            try:
                response = requests.post(f"{API_URL}/test_model_robustness", files=files)
                if response.status_code == 200:
                    data = response.json()
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Clean Prediction", data["clean_prediction"])
                    with col2:
                        if data["clean_prediction"] != data["attacked_prediction"]:
                            st.error(f"Attacked Prediction: {data['attacked_prediction']}")
                        else:
                            st.success(f"Attacked Prediction: {data['attacked_prediction']}")
                else:
                    st.error(f"Error: {response.content}")
            except Exception as e:
                st.error(f"API call failed: {e}")

elif page == "Cloud & Identity Monitoring":
    st.subheader("Real-Time Anomaly Detection Feed")
    SAMPLE_LOGS = [
        {"user_id": "user_123", "event": "Login from new device", "features": {"impossible_travel_speed": 0, "login_frequency_1hr": 2, "ip_change_count_24hr": 3}},
        {"user_id": "user_456", "event": "API key usage spike", "features": {"impossible_travel_speed": 0, "login_frequency_1hr": 35, "ip_change_count_24hr": 1}},
        {"user_id": "user_789", "event": "Successful login", "features": {"impossible_travel_speed": 0, "login_frequency_1hr": 0, "ip_change_count_24hr": 5}}, # Quick fix normal
        {"user_id": "user_123", "event": "Impossible travel detected", "features": {"impossible_travel_speed": 8500, "login_frequency_1hr": 5, "ip_change_count_24hr": 8}},
        {"user_id": "user_456", "event": "Regular API call", "features": {"impossible_travel_speed": 0, "login_frequency_1hr": 8, "ip_change_count_24hr": 2}},
    ]
    if st.button("Start Simulation"):
        placeholder = st.empty()
        log_history = []
        for log in SAMPLE_LOGS:
            features = log["features"]
            description = f"**{log['user_id']}** · {log['event']}"
            try:
                response = requests.post(f"{API_URL}/predict_anomaly", json=features)
                if response.status_code == 200:
                    result = response.json()
                    if result["is_anomaly"]:
                        msg = f":warning: 🔴 **Anomaly:** {description} ([score: {result['model_score']}])"
                    else:
                        msg = f":shield: 🟢 **Normal:** {description} ([score: {result['model_score']}])"
                else:
                    msg = f":grey_question: Could not get API result."
            except Exception as e:
                msg = f":grey_question: API call failed: {e}"
            log_history.insert(0, msg)
            placeholder.markdown("\n".join(log_history))
            time.sleep(2)
