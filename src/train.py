# Import necessary libraries
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# Load the data
HousingData = pd.read_csv('C:\\Users\\Surface\\OneDrive\\Documentos\\GitHub\\House-Price-Prediction\\data\\Housing_Cleaned.csv')

# 1. Separate Price (Target) from Features
y = HousingData['price']                  # The Answer Key
X = HousingData.drop('price', axis=1)     # The Questions

# 2. Separate Training Set from Testing Set
# We use the train_test_split function from the sklearn library
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 3. Create the Model
model = LinearRegression()

# 4. Train the Model
model_run=model.fit(X_train, y_train)

# 5. Make Predictions
y_pred = model_run.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Print the results
print("--- Model Performance ---")
print(f"Mean Squared Error (MSE): {mse:,.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:,.2f}")
print(f"R-squared Score: {r2:.4f}")

comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print("\nFirst 5 Predictions:")
print(comparison.head())

# Save the model
import joblib   
joblib.dump(model_run, 'C:\\Users\\Surface\\OneDrive\\Documentos\\GitHub\\House-Price-Prediction\\models\\house_price_model.joblib')
# joblib.dump(scaler, '../models/near_regression_model.joblib')

