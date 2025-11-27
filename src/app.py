import streamlit as st
import pandas as pd
import joblib
import os

# --- PATH SETUP ---
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    model_file = os.path.join(base_dir, "..", "models", "house_price_model.joblib")
    scaler_file = os.path.join(base_dir, "..", "models", "scaler_transform.joblib")

    # Resolve absolute paths
    model_file = os.path.abspath(model_file)
    scaler_file = os.path.abspath(scaler_file)

    # Debug print for Streamlit Cloud
    st.write("Model path:", model_file)
    st.write("Scaler path:", scaler_file)

    # Check existence BEFORE loading
    if not os.path.exists(model_file):
        st.error("Model file missing.")
        st.stop()

    if not os.path.exists(scaler_file):
        st.error("Scaler file missing.")
        st.stop()

    # Load both
    try:
        model = joblib.load(model_file)
        scaler = joblib.load(scaler_file)
        return model, scaler
    except Exception as e:
        st.error(f"Loading error: {e}")
        st.stop()


model ,scaler= load_model()

# --- APP INTERFACE ---
st.title("House Price Predictor 🏠")

# 1. Numerical Input
area = st.number_input("Area (in sq ft)", value=500.0, min_value=100.0, max_value=10000.0, step=50.0)

# 2. Count Inputs (Sliders)
st.write("### House Details")
col1, col2 = st.columns(2)
with col1:
    bedrooms = st.slider("Bedrooms", 1, 6, 3)
    stories = st.slider("Stories", 1, 4, 2)
with col2:
    bathrooms = st.slider("Bathrooms", 1, 4, 1)
    parking = st.slider("Parking Spaces", 0, 3, 0)

# 3. Categorical Inputs
st.write("### Features")
col3, col4 = st.columns(2)

# Create the translator (Dictionary)
input_map = {'Yes': 1, 'No': 0}

with col3:
    mainroad = st.selectbox("Main Road", ["Yes", "No"])
    mainroad_val = input_map[mainroad]

    guestroom = st.selectbox("Guestroom", ["Yes", "No"])
    guestroom_val = input_map[guestroom]

    basement = st.selectbox("Basement", ["Yes", "No"])
    basement_val = input_map[basement]

with col4:
    hotwaterheating = st.selectbox("Hot Water Heating", ["Yes", "No"])
    hotwaterheating_val = input_map[hotwaterheating]

    airconditioning = st.selectbox("Air Conditioning", ["Yes", "No"])
    airconditioning_val = input_map[airconditioning]

    prefarea = st.selectbox("Preferred Area", ["Yes", "No"])
    prefarea_val = input_map[prefarea]

# 4. Furnishing Status
st.write("### Furnishing")
furnishing = st.selectbox("Furnishing Status", ["Furnished", "Semi-Furnished", "Unfurnished"])

# The Logic (Matches One-Hot Encoding)
furnishingstatus_semi = 1 if furnishing == 'Semi-Furnished' else 0
furnishingstatus_unfurnished = 1 if furnishing == 'Unfurnished' else 0

# --- PREDICTION LOGIC ---
if st.button("Predict Price", type="primary"):
    # A. Package the inputs into a DataFrame
    # Note: These column names must match your training data exactly!
    input_data = pd.DataFrame({
        'area': [area],
        'bedrooms': [bedrooms],
        'bathrooms': [bathrooms],
        'stories': [stories],
        'mainroad': [mainroad_val],
        'guestroom': [guestroom_val],
        'basement': [basement_val],
        'hotwaterheating': [hotwaterheating_val],
        'airconditioning': [airconditioning_val],
        'parking': [parking],
        'prefarea': [prefarea_val],
        'furnishingstatus_semi-furnished': [furnishingstatus_semi],
        'furnishingstatus_unfurnished': [furnishingstatus_unfurnished]
    })

    # B. Scale the data using the saved scaler
    try:
        input_data_scaled = scaler.transform(input_data)
        
        # C. Make the prediction
        prediction = model.predict(input_data_scaled)

        # D. Display the result
        st.success(f"Estimated Price: ${prediction[0]:,.2f}")
        
    except ValueError as e:
        st.error(f"Error during prediction: {e}")
        st.write("Debug - Columns expected by scaler:", scaler.feature_names_in_)
        st.write("Debug - Input data:", input_data) 