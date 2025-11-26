import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os

# 1. Load the Cleaned Data
# This assumes you ran data_cleaning.py first!
if not os.path.exists('C:\\Users\\Surface\\OneDrive\\Documentos\\GitHub\\House-Price-Prediction\\data\\Housing_Cleaned.csv'):
    print("Error: 'Housing_Cleaned.csv' not found. Please run data_cleaning.py first.")
    exit()

df = pd.read_csv('C:\\Users\\Surface\\OneDrive\\Documentos\\GitHub\\House-Price-Prediction\\data\\Housing_Cleaned.csv')
# 2. Define Features (X) and Target (y)
X = df.drop('price', axis=1)
y = df['price']

# 3. Scale the Features
# We use MinMaxScaler to map all features to the range [0, 1]
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 4. Train/Test Split
# We train on 80% of the data, test on 20%
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 5. Train the Model
model = LinearRegression()
model.fit(X_train, y_train)

# 6. Evaluate
score = model.score(X_test, y_test)
print(f"Model Training Complete.")
print(f"R-squared Score on Test Data: {score:.4f}")

# 7. Save Model and Scaler
# Create 'models' directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

joblib.dump(model, 'C:\\Users\\Surface\\OneDrive\\Documentos\\GitHub\\House-Price-Prediction\\models\\house_price_model.joblib')
joblib.dump(scaler, 'C:\\Users\\Surface\\OneDrive\\Documentos\\GitHub\\House-Price-Prediction\\models\\scaler_transform.joblib')

print("Model and Scaler saved to the 'models/' folder.")