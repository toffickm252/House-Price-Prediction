# --- train_model.py ---
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import joblib

# --- Load Dataset ---
data = pd.read_csv('data/Housing.csv')  # Update path if needed

# --- Encode Binary Columns ---
binary_cols = ['mainroad', 'guestroom', 'basement',
               'hotwaterheating', 'airconditioning', 'prefarea']
data[binary_cols] = data[binary_cols].replace({'yes': 1, 'no': 0})

# --- One-Hot Encode Furnishing Status ---
df = pd.get_dummies(data, columns=['furnishingstatus'], drop_first=False, dtype=int)

# Ensure all furnishingstatus columns exist
for col in ['furnishingstatus_furnished', 'furnishingstatus_semi-furnished', 'furnishingstatus_unfurnished']:
    if col not in df.columns:
        df[col] = 0

data = df.copy()

# --- Prepare Features and Target ---
y = data['price']
X = data.drop('price', axis=1)

# --- Define Exact Feature Order ---
feature_order = [
    'area', 'bedrooms', 'bathrooms', 'stories', 'parking',
    'mainroad', 'guestroom', 'basement', 'hotwaterheating',
    'airconditioning', 'prefarea',
    'furnishingstatus_semi-furnished', 'furnishingstatus_unfurnished'
]
X = X[feature_order]

# --- Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# --- Remove Outliers from Training Set ---
Q1 = y_train.quantile(0.25)
Q3 = y_train.quantile(0.75)
IQR = Q3 - Q1
upper_limit = Q3 + 1.5 * IQR
y_train_filtered = y_train[y_train < upper_limit]
X_train_filtered = X_train.loc[y_train_filtered.index]
X_train = X_train_filtered
y_train = y_train_filtered

# --- Feature Scaling ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Log Transform Target ---
y_train = np.log(y_train)
y_test = np.log(y_test)

# --- Train Model ---
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# --- Save Model and Scaler ---
joblib.dump(model, 'house_price_model.pkl')
joblib.dump(scaler, 'scaler_transform.pkl')

print("Model and scaler saved successfully!")
# --- Function to Prepare Input Features ---