import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# Load the saved model and scaler
best_gb = joblib.load("best_gradient_boosting_model (1).pkl")
scaler = joblib.load("scaler.pkl")
expected_features = joblib.load("feature_names.pkl")  # Load saved feature names

# Define call category mapping
call_category_mapping = {
    'Medical': {'SICK', 'INJURY', 'CARD', 'DRUG', 'SEIZR', 'RESPIR', 'ASTHMB', 'CVA', 'OBLAB', 'BURNMA', 'GYNHEM', 'ANAPH', 'TRAUMA'},
    'Fire': {'BURNMA', 'BURNMI', 'INHALE', 'ELECT'},
    'Traffic': {'MVA', 'MVAINJ', 'PEDSTR', 'CVAC', 'STRANS', 'ACC'},
    'Crime': {'ARREST', 'SHOT', 'STAB', 'JUMPDN', 'JUMPUP', 'DDOA'},
    'Mental_Health': {'EDP', 'EDPC', 'ALTMEN', 'ALTMFC', 'UNC', 'UNCFC', 'STNDBM'},
    'Other': {'OTHER', 'SAFE', 'TEST', 'DRILL', 'ACTIVE', 'T-TRMA', 'T-INJ'}
}

# Function to categorize call types
def categorize_call(call_type):
    for category, types in call_category_mapping.items():
        if call_type in types:
            return category
    return 'Other'



# Streamlit UI
st.title("<h1 class='title'>🚑 Emergency Healthcare Prediction System</h1>", unsafe_allow_html=True)

# Dropdown for INITIAL_CALL_TYPE
call_types = [call for category in call_category_mapping.values() for call in category]
selected_call_type = st.selectbox("Select Call Type:", call_types)

# Other inputs
incident_datetime = st.text_input("Incident Datetime (YYYY-MM-DD HH:MM:SS)", "2025-03-31 14:30:00")
incident_close_datetime = st.text_input("Incident Close Datetime (YYYY-MM-DD HH:MM:SS)", "2025-03-31 15:00:00")
first_assignment_datetime = st.text_input("First Assignment Datetime (YYYY-MM-DD HH:MM:SS)", "2025-03-31 14:35:00")

# Convert to datetime
incident_dt = datetime.strptime(incident_datetime, "%Y-%m-%d %H:%M:%S")
incident_close_dt = datetime.strptime(incident_close_datetime, "%Y-%m-%d %H:%M:%S")
first_assignment_dt = datetime.strptime(first_assignment_datetime, "%Y-%m-%d %H:%M:%S")

# Feature engineering
incident_duration = (incident_close_dt - incident_dt).total_seconds()
dispatch_response_time = (first_assignment_dt - incident_dt).total_seconds()
hour_of_incident = incident_dt.hour
call_category = categorize_call(selected_call_type)

# Encode CALL_CATEGORY
call_category_encoded = pd.Series(call_category).astype("category").cat.codes[0]

# Create input DataFrame
input_df = pd.DataFrame([{
    "INITIAL_CALL_TYPE": selected_call_type,
    "CALL_CATEGORY_ENCODED": call_category_encoded,
    "HOUR": hour_of_incident,
    "INCIDENT_DURATION": incident_duration,
    "DISPATCH_RESPONSE_TIME": dispatch_response_time
}])

# Ensure input features match training features
missing_cols = set(expected_features) - set(input_df.columns)
extra_cols = set(input_df.columns) - set(expected_features)

# Add missing columns
for col in missing_cols:
    input_df[col] = 0  # Default value for missing features

# Reorder columns
input_df = input_df[expected_features]

# Scale data
try:
    data_scaled = scaler.transform(input_df)
except ValueError as e:
    st.error(f"Feature mismatch error: {e}")
    st.write("Expected feature names:", expected_features)
    st.write("Input feature names:", input_df.columns.tolist())
    st.stop()

# Make prediction
prediction = best_gb.predict(data_scaled)[0]

# Display result
st.subheader("🚑 Prediction Result:")
st.write(f"The predicted outcome for this emergency call is: **{prediction}**")
