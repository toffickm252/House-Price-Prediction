# --- Imports ---
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Page Config ---
st.set_page_config(page_title="Kumasi Housing Price Prediction", page_icon=":house:")

# --- Utility Functions ---
def _max_width_(prcnt_width:int = 70):
    max_width_str = f"max-width: {prcnt_width}%;"
    st.markdown(f""" 
                <style> 
                .appview-container .main .block-container{{{max_width_str}}}
                </style>    
                """, 
                unsafe_allow_html=True,
    )

# --- Cached Resources ---
@st.cache_resource
def load_model(filepath: str):
    with st.spinner('Loading model...'):
        return joblib.load(filepath)

@st.cache_resource
def load_scaler(filepath: str):
    with st.spinner('Loading scaler...'):
        return joblib.load(filepath)

# --- Load Artifacts ---
try:
    model_log = load_model('house_price_model.pkl')
    scaler = load_scaler('scaler_transform.pkl')
except Exception as e:
    st.error(f"Error loading model or scaler: {e}")
    st.stop()

# --- Session State Initialization ---
def initialize_session_states():
    if 'prediction' not in st.session_state:
        st.session_state['prediction'] = None
    if 'inputs' not in st.session_state:
        st.session_state['inputs'] = {}

initialize_session_states()

# --- Input Function ---
def user_input_features():
    st.header("Enter House Attributes")

    col1, col2 = st.columns(2)
    with col1:
        area = st.number_input("Area (sq ft)", min_value=1500, max_value=16200, value=7500, step=100)
        bedrooms = st.slider("Bedrooms 🛏️", 1, 6, 3)
        bathrooms = st.slider("Bathrooms 🛁", 1, 4, 2)
        stories = st.slider("Stories", 1, 4, 2)
        parking = st.slider("Parking Spots 🅿️", 0, 3, 1)

    with col2:
        mainroad = st.selectbox("Main Road Access", ("Yes", "No"))
        guestroom = st.selectbox("Has Guest Room", ("Yes", "No"))
        basement = st.selectbox("Has Basement", ("Yes", "No"))
        hotwaterheating = st.selectbox("Hot Water Heating", ("Yes", "No"))
        airconditioning = st.selectbox("Air Conditioning", ("Yes", "No"))
        prefarea = st.selectbox("Preferred Area", ("Yes", "No"))
        furnishingstatus = st.selectbox("Furnishing Status", ("furnished", "semi-furnished", "unfurnished"))

    # Encode inputs
    data = {
        'area': area,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'stories': stories,
        'parking': parking,
        'mainroad': 1 if mainroad == 'Yes' else 0,
        'guestroom': 1 if guestroom == 'Yes' else 0,
        'basement': 1 if basement == 'Yes' else 0,
        'hotwaterheating': 1 if hotwaterheating == 'Yes' else 0,
        'airconditioning': 1 if airconditioning == 'Yes' else 0,
        'prefarea': 1 if prefarea == 'Yes' else 0,
        'furnishingstatus_semi-furnished': 1 if furnishingstatus == 'semi-furnished' else 0,
        'furnishingstatus_unfurnished': 1 if furnishingstatus == 'unfurnished' else 0
    }

    column_order = [
        'area', 'bedrooms', 'bathrooms', 'stories', 'parking',
        'mainroad', 'guestroom', 'basement', 'hotwaterheating',
        'airconditioning', 'prefarea',
        'furnishingstatus_semi-furnished', 'furnishingstatus_unfurnished'
    ]

    features = pd.DataFrame([data], columns=column_order)
    return features

# --- Main Layout ---
_max_width_(70)
st.title("Kumasi Housing Prices Prediction 🏠")
st.markdown("""
##### A web application for predicting housing prices in Kumasi, Ghana.

This app uses a log-transformed linear regression model trained on housing data. 
Provide house attributes, and the model will predict the expected price in Ghana Cedis (GHS).
""")

# --- Input Section ---
df_input = user_input_features()
st.subheader("User-Selected Features")
st.write(df_input)

# --- Prediction Button ---
if st.button("Predict", use_container_width=True):
    try:
        scaled_features = scaler.transform(df_input)
        log_prediction = model_log.predict(scaled_features)
        final_prediction = float(np.exp(log_prediction).squeeze())
        st.session_state['prediction'] = final_prediction
        st.success("Prediction complete!")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# --- Display Prediction ---
if st.session_state['prediction']:
    pred = st.session_state['prediction']
    st.markdown(
        """
        <style>
        [data-testid="stMetricValue"] {
            font-size: 34px;
            color: green;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.metric(label="Predicted House Price", value=f"GHS {pred:,.2f}")
    st.info("Note: The predicted price is an estimate based on the provided features and may vary in real market conditions.")