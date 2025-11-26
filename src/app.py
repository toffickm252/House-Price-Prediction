import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('C:\\Users\\Surface\\OneDrive\\Documentos\\GitHub\\House-Price-Prediction\\models\\house_price_model.joblib')
scaler = joblib.load('C:\\Users\\Surface\\OneDrive\\Documentos\\GitHub\\House-Price-Prediction\\models\\scaler_transform.joblib')

print("Model and scaler loaded successfully.")