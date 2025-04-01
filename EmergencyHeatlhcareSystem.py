import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# Load the saved model and scaler
best_gb = joblib.load("best_gradient_boosting_model (1).pkl")
scaler = joblib.load("scaler.pkl")  

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

# Encode call categories into numbers
call_category_encoding = {category: idx for idx, category in enumerate(call_category_mapping.keys())}

# Define urgency level mapping
urgency_levels = {0: "Low", 1: "Medium", 2: "High"}

# Streamlit UI
st.set_page_config(page_title="Emergency Urgency Predictor", page_icon="🚨", layout="centered")

# Set darker background color
st.markdown(
    """
    <style>
        body {
            background-color: #1E1E1E;
            color: white;
        }
        .stApp {
            background-color: #1E1E1E;
        }
        h1 {
            color: #FF4B4B;
            text-align: center;
        }
        .stTextInput, .stNumberInput, .stSelectbox, .stButton {
            color: black;
        }
        .stButton>button {
            background-color: #FF4B4B;
            color: white;
            font-size: 16px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🚨 Emergency Urgency Level Predictor")

st.markdown("### Enter Incident Details Below:")

# Dropdown for INITIAL_CALL_TYPE (all call types)
all_call_types = sorted(set().union(*call_category_mapping.values()))
initial_call_type = st.selectbox("Select Initial Call Type", all_call_types)

# Automatically determine CALL_CATEGORY based on the selected call type
call_category = categorize_call(initial_call_type)
call_category_encoded = call_category_encoding[call_category]

st.write(f"📌 Mapped to Category: **{call_category}** ")

# User inputs for numeric features
dispatch_response_seconds = st.number_input("Dispatch Response Seconds", min_value=0)
incident_response_seconds = st.number_input("Incident Response Seconds", min_value=0)
incident_travel_time = st.number_input("Incident Travel Time (Seconds)", min_value=0)
year = st.number_input("Year", min_value=2000, max_value=2100)
hour = st.number_input("Hour of Incident (0-23)", min_value=0, max_value=23)
incident_duration = st.number_input("Incident Duration (Seconds)", min_value=0)

# Use the same value for DISPATCH_RESPONSE_TIME as DISPATCH_RESPONSE_SECONDS_QY
dispatch_response_time = dispatch_response_seconds

# Create input DataFrame
input_data = pd.DataFrame([[
    dispatch_response_seconds, incident_response_seconds, incident_travel_time, 
    year, hour, call_category_encoded, incident_duration, dispatch_response_time
]], columns=[
    'DISPATCH_RESPONSE_SECONDS_QY', 'INCIDENT_RESPONSE_SECONDS_QY',
    'INCIDENT_TRAVEL_TM_SECONDS_QY', 'YEAR', 'HOUR',
    'CALL_CATEGORY_ENCODED', 'INCIDENT_DURATION', 'DISPATCH_RESPONSE_TIME'
])

# Scale input data
input_data_scaled = scaler.transform(input_data)

# Prediction button
if st.button("Predict Urgency Level"):
    prediction = best_gb.predict(input_data_scaled)
    urgency_label = urgency_levels.get(prediction[0], "Unknown")  # Convert to label
    st.success(f"🚨 Predicted Urgency Level: **{urgency_label}**")
