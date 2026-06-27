# Import all the necessary libraries
import pandas as pd
import numpy as np
import joblib
import streamlit as st

# Load the model and model structure
model = joblib.load("pollution_model.pkl")
model_cols = joblib.load("model_columns.pkl")

# Pollutant safe limits (source: Parameters_WQM_RMS.pdf)
SAFE_LIMITS = {
    'O2': 5,           # > 5 mg/L is good
    'NO3': 10,         # < 10 mg/L nitrate-nitrogen
    'NO2': 0.1,        # < 0.1 mg/L
    'SO4': 250,        # < 250 mg/L
    'PO4': 0.1,        # < 0.1 mg/L for surface water
    'CL': 250          # < 250 mg/L
}

# UI
st.title("💧 Water Pollutants Predictor")
st.write("Predict key water pollutant levels for a given Year and Station ID")

# User inputs
year_input = st.number_input("Enter Year", min_value=2000, max_value=2100, value=2022)
station_id = st.text_input("Enter Station ID", value='1')

# Predict button
if st.button('Predict'):
    if not station_id:
        st.warning('Please enter the station ID')
    else:
        # Prepare input
        input_df = pd.DataFrame({'year': [year_input], 'id': [station_id]})
        input_encoded = pd.get_dummies(input_df, columns=['id'])

        # Align with training model columns
        for col in model_cols:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        input_encoded = input_encoded[model_cols]

        # Predict pollutants
        predicted_pollutants = model.predict(input_encoded)[0]
        pollutants = ['O2', 'NO3', 'NO2', 'SO4', 'PO4', 'CL']
        predicted_values = dict(zip(pollutants, predicted_pollutants))

        # Display results
        st.subheader(f"🔍 Predicted Pollutant Levels for Station '{station_id}' in {year_input}:")
        for pollutant, value in predicted_values.items():
            st.write(f"**{pollutant}: {value:.2f} mg/L**")

            # Safe limit alert
            if pollutant in SAFE_LIMITS:
                limit = SAFE_LIMITS[pollutant]
                if (pollutant == 'O2' and value < limit):
                    st.error(f"⚠️ Low {pollutant} — below healthy oxygen level (< {limit} mg/L)")
                elif (pollutant != 'O2' and value > limit):
                    st.error(f"⚠️ High {pollutant} — exceeds safe limit (> {limit} mg/L)")
                else:
                    st.success(f"✅ {pollutant} is within the safe range.")
