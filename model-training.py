import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- 1. Load Model Artifacts ---
try:
    model = joblib.load('house_price_model.pkl')
    scaler = joblib.load('scaler_transform.pkl')
except FileNotFoundError:
    st.error("Error: Model or Scaler files not found.")
    st.write("Please ensure 'house_price_model.pkl' and 'scaler_transform.pkl' are in the same directory.")
    st.stop()

# --- 2. Define Feature Order (Must match training) ---
FEATURE_ORDER = [
    'area', 'bedrooms', 'bathrooms', 'stories', 'parking',
    'mainroad', 'guestroom', 'basement', 'hotwaterheating',
    'airconditioning', 'prefarea',
    'furnishingstatus_semi-furnished', 'furnishingstatus_unfurnished'
]

# --- 3. Input Preparation Function ---
def prepare_input_features(input_dict):
    """
    Prepares raw user input dictionary into the scaled array required by the model.
    """
    df = pd.DataFrame([input_dict])

    # 3a. Handle Binary Columns ('yes': 1, 'no': 0)
    binary_cols = ['mainroad', 'guestroom', 'basement', 
                   'hotwaterheating', 'airconditioning', 'prefarea']
    
    for col in binary_cols:
        df[col] = df[col].apply(lambda x: 1 if x == 'Yes' else 0)

    # 3b. Handle One-Hot Encoding for Furnishing Status
    # 'furnished' is the baseline (0, 0)
    status = df['furnishingstatus'].iloc[0]
    
    df['furnishingstatus_semi-furnished'] = 1 if status == 'Semi-Furnished' else 0
    df['furnishingstatus_unfurnished'] = 1 if status == 'Unfurnished' else 0
    
    # 3c. Reorder and Select Columns
    df_final = df[FEATURE_ORDER]

    # 3d. Scale Features
    input_scaled = scaler.transform(df_final)
    
    return input_scaled

# --- 4. Streamlit UI and Logic ---
st.set_page_config(page_title="House Price Predictor", layout="centered")
st.title("🏘️ Delhi Housing Price Predictor")
st.markdown("Use the form below to enter the features of the house and get a predicted price.")

# Create the form layout using columns
with st.form("prediction_form"):
    st.header("🏠 House Details")

    # Column 1: Numerical/Quantitative Features
    col1, col2 = st.columns(2)
    
    with col1:
        area = st.number_input("Area (sq ft)", min_value=1000, max_value=20000, value=5000)
        bedrooms = st.slider("Bedrooms", min_value=1, max_value=6, value=3)
        bathrooms = st.slider("Bathrooms", min_value=1, max_value=4, value=2)
        stories = st.slider("Stories", min_value=1, max_value=4, value=2)
        parking = st.slider("Parking Spaces", min_value=0, max_value=3, value=1)
        
    # Column 2: Categorical/Binary Features
    with col2:
        mainroad = st.selectbox("Main Road Access", ['Yes', 'No'])
        guestroom = st.selectbox("Guest Room", ['No', 'Yes'])
        basement = st.selectbox("Basement", ['No', 'Yes'])
        hotwaterheating = st.selectbox("Hot Water Heating", ['No', 'Yes'])
        airconditioning = st.selectbox("Air Conditioning", ['No', 'Yes'])
        prefarea = st.selectbox("Preferred Area", ['No', 'Yes'])
        
        furnishingstatus = st.selectbox("Furnishing Status", 
                                        ['Furnished', 'Semi-Furnished', 'Unfurnished'])
        
    submitted = st.form_submit_button("Get Prediction")

# --- 5. Prediction Execution ---
if submitted:
    # 5a. Collect all inputs into a dictionary
    raw_input_data = {
        'area': area,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'stories': stories,
        'parking': parking,
        'mainroad': mainroad,
        'guestroom': guestroom,
        'basement': basement,
        'hotwaterheating': hotwaterheating,
        'airconditioning': airconditioning,
        'prefarea': prefarea,
        'furnishingstatus': furnishingstatus
    }
    
    # 5b. Process Input
    processed_features = prepare_input_features(raw_input_data)
    
    # 5c. Predict
    log_price_prediction = model.predict(processed_features)
    
    # 5d. Inverse Transform (np.exp is crucial because the target was log-transformed)
    actual_price_prediction = np.exp(log_price_prediction)[0]
    
    st.success("## 💰 Predicted House Price")
    st.success(f"**₹ {actual_price_prediction:,.0f}**")
    st.info("Note: The model was trained on the natural logarithm of price, so the prediction was inverse-transformed using $e^x$ to get this final rupee value.")