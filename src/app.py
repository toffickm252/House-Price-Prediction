import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('C:\\Users\\Surface\\OneDrive\\Documentos\\GitHub\\House-Price-Prediction\\models\\house_price_model.joblib')
scaler = joblib.load('C:\\Users\\Surface\\OneDrive\\Documentos\\GitHub\\House-Price-Prediction\\models\\scaler_transform.joblib')

print("Model and scaler loaded successfully.")

st.title("House Price Predictor")

# 1. Numerical Input
area = st.number_input("Area (in sq ft)", value=0.0)

# Create the translator (Dictionary)
input_map = {'Yes': 1, 'No': 0}

# Mainroad Input
mainroad = st.selectbox("Mainroad", ["Yes", "No"])
# if mainroad == "Yes":
#     mainroad = 1
# else:
#     mainroad = 0
mainroad_val = input_map[mainroad]

# Guestroom Input
guestroom = st.selectbox("Guestroom", ["Yes", "No"])
# if guestroom == "Yes":
#     guestroom = 1
# else:
#     guestroom = 0

guestroom_val = input_map[guestroom]

# Airconditioning Input
airconditioning = st.selectbox("Airconditioning", ["Yes", "No"])
# if airconditioning == "Yes":
#     airconditioning = 1
# else:
#     airconditioning = 0
airconditioning_val = input_map[airconditioning]

# Remaining Yes/No Inputs
# basement 
basement = st.selectbox("Basement", ["Yes", "No"])
basement_val = input_map[basement]
# hotwaterheating
hotwaterheating = st.selectbox("Hot Water Heating", ["Yes", "No"])
hotwaterheating_val = input_map[hotwaterheating]
#Preferred Area
prefarea = st.selectbox("Preferred Area", ["Yes", "No"])
prefarea_val = input_map[prefarea]

# parking


# Furnishing Status
furnishing = st.selectbox("Furnishing Status", ["Furnished", "Semi-Furnished", "Unfurnished"])
if furnishing == "Furnished":
    furnishing_val = 1
elif furnishing == "Semi-Furnished":
    furnishing_val = 2
else:
    furnishing_val = 3

furnishingstatus_semi = 1 if furnishing_val == 2 else 0
furnishingstatus_unfurnished = 1 if furnishing_val == 3 else 0

st.write("### House Details")
bedrooms = st.slider("Bedrooms", 1, 6, 3)
bathrooms = st.slider("Bathrooms", 1, 4, 1)
stories = st.slider("Stories", 1, 4, 2)
parking = st.slider("Parking Spaces", 0, 3, 0)

# --- 3. The Prediction Logic ---
if st.button("Predict Price"):
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
    # The model expects values between 0 and 1
    input_data_scaled = scaler.transform(input_data)

    # C. Make the prediction
    prediction = model.predict(input_data_scaled)

    # D. Display the result
    st.success(f"Estimated Price: ${prediction[0]:,.2f}")