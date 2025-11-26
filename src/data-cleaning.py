# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

HousingData = pd.read_csv('C:\\Users\\Surface\\OneDrive\\Documentos\\GitHub\\House-Price-Prediction\\data\\Housing.csv')
print(HousingData.head())

# data cleaning 
# Duplicates 
print(HousingData.duplicated().sum())

# Missing values
print(HousingData.isnull().sum())

# List of columns that contain 'yes' and 'no' values
binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']

# Map 'yes' to 1 and 'no' to 0 for each of these columns
for col in binary_cols:
    HousingData[col] = HousingData[col].map({'yes': 1, 'no': 0})

# One-hot encoding for 'furnishingstatus'
furnishing_dummies = pd.get_dummies(HousingData['furnishingstatus'], prefix='furnishingstatus', drop_first=True)
HousingData = pd.concat([HousingData, furnishing_dummies], axis=1)

# Drop the original 'furnishingstatus' column
HousingData.drop('furnishingstatus', axis=1, inplace=True)

# outliers
def remove_outliers_iqr(df, column):
    """
    Removes outliers from a dataframe column using the IQR method.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Filter the data to keep only values within the bounds
    df_filtered = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df_filtered

# --- Visualizing Before Removal ---
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.boxplot(y=HousingData['price'])
plt.title('Price Before Outlier Removal')

plt.subplot(1, 2, 2)
sns.boxplot(y=HousingData['area'])
plt.title('Area Before Outlier Removal')
plt.show()

# --- Removing Outliers ---
# Apply the function to 'price' and then to 'area'
df_clean = remove_outliers_iqr(HousingData, 'price')
df_clean = remove_outliers_iqr(df_clean, 'area')

print(f"Original shape: {HousingData.shape}")
print(f"New shape after removing outliers: {df_clean.shape}")

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.boxplot(y=df_clean['price'])
plt.title('Price After Outlier Removal')

plt.subplot(1, 2, 2)
sns.boxplot(y=df_clean['area'])
plt.title('Area After Outlier Removal')
plt.show()

# Feature Scaling
# Define columns to scale
cols_to_scale = ['price', 'area', 'bedrooms', 'bathrooms', 'stories', 'parking']

# Initialize the scaler
scaler = MinMaxScaler()

# Apply scaling
# This replaces the original values with the scaled values (0 to 1)
df_clean[cols_to_scale] = scaler.fit_transform(df_clean[cols_to_scale])

# Save the scaler
import joblib
joblib.dump(scaler, 'C:\\Users\\Surface\\OneDrive\\Documentos\\GitHub\\House-Price-Prediction\\models\\scaler_transform.joblib')

# Display the result
print(df_clean.head())

# dtypes
print(df_clean.dtypes)

# Save cleaned data
df_clean.to_csv('C:\\Users\\Surface\\OneDrive\\Documentos\\GitHub\\House-Price-Prediction\\data\\Housing_Cleaned.csv', index=False)

