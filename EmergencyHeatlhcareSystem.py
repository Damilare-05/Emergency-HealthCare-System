
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the saved model and scaler
best_gb = joblib.load("best_gradient_boosting_model.pkl")
scaler = joblib.load("scaler.pkl")  # Save your StandardScaler separately

# Define urgency labels
urgency_labels = ["Low Urgency", "Medium Urgency", "High Urgency"]

# Streamlit app title
st.title("Emergency Urgency Level Prediction")

st.write("This app predicts the urgency level of an emergency call based on given features.")

# Sidebar for user input
st.sidebar.header("User Input Features")

# Input fields
def user_input_features():
    DISPATCH_RESPONSE_SECONDS_QY = st.sidebar.number_input("Dispatch Response Time (seconds)", min_value=0)
    INCIDENT_RESPONSE_SECONDS_QY = st.sidebar.number_input("Incident Response Time (seconds)", min_value=0)
    INCIDENT_TRAVEL_TM_SECONDS_QY = st.sidebar.number_input("Incident Travel Time (seconds)", min_value=0)
    YEAR = st.sidebar.number_input("Year", min_value=2000, max_value=2050, value=2024)
    CALL_CATEGORY_ENCODED = st.sidebar.number_input("Call Category (Encoded)", min_value=0, max_value=10)
    INCIDENT_DURATION = st.sidebar.number_input("Incident Duration (seconds)", min_value=0)

    data = {
        "DISPATCH_RESPONSE_SECONDS_QY": DISPATCH_RESPONSE_SECONDS_QY,
        "INCIDENT_RESPONSE_SECONDS_QY": INCIDENT_RESPONSE_SECONDS_QY,
        "INCIDENT_TRAVEL_TM_SECONDS_QY": INCIDENT_TRAVEL_TM_SECONDS_QY,
        "YEAR": YEAR,
        "CALL_CATEGORY_ENCODED": CALL_CATEGORY_ENCODED,
        "INCIDENT_DURATION": INCIDENT_DURATION
    }

    return pd.DataFrame([data])

# File uploader for batch predictions
uploaded_file = st.file_uploader("Upload CSV for batch prediction", type=["csv"])

# Predict function
def predict(data):
    # Scale input data
    data_scaled = scaler.transform(data)

    # Get predictions
    prediction = best_gb.predict(data_scaled)
    prediction_labels = [urgency_labels[i] for i in prediction]

    return prediction_labels

# User input (manual)
input_df = user_input_features()

# Display user input
st.subheader("User Input Features")
st.write(input_df)

# Make predictions if the user submits data
if st.button("Predict Urgency"):
    result = predict(input_df)
    st.success(f"Predicted Urgency Level: {result[0]}")

# Batch processing (CSV file)
if uploaded_file is not None:
    st.subheader("Batch Prediction")
    batch_data = pd.read_csv(uploaded_file)

    # Ensure the file contains the correct columns
    expected_columns = [
        "DISPATCH_RESPONSE_SECONDS_QY",
        "INCIDENT_RESPONSE_SECONDS_QY",
        "INCIDENT_TRAVEL_TM_SECONDS_QY",
        "YEAR",
        "CALL_CATEGORY_ENCODED",
        "INCIDENT_DURATION"
    ]

    if all(col in batch_data.columns for col in expected_columns):
        batch_predictions = predict(batch_data)
        batch_data["Predicted Urgency"] = batch_predictions
        st.write(batch_data)

        # Option to download results
        csv = batch_data.to_csv(index=False).encode("utf-8")
        st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")

    else:
        st.error(f"Uploaded file is missing required columns. Expected: {expected_columns}")

# Run the app with: streamlit run app.py
