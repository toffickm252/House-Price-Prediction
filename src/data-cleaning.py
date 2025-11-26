import pandas as pd

# 1. Load the Raw Data
# Ensure 'Housing.csv' is in the same folder or update the path
df = pd.read_csv('C:\\Users\\Surface\\OneDrive\\Documentos\\GitHub\\House-Price-Prediction\\data\\Housing.csv')
print("Original Shape:", df.shape)

# 2. Encoding Binary Columns (Yes/No -> 1/0)
binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
for col in binary_cols:
    df[col] = df[col].map({'yes': 1, 'no': 0})

# 3. One-Hot Encoding for Furnishing Status
# drop_first=True creates 'semi-furnished' and 'unfurnished' columns
df = pd.get_dummies(df, columns=['furnishingstatus'], drop_first=True)

# 4. Outlier Removal Function
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    # Filter the data
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Apply outlier removal
df = remove_outliers_iqr(df, 'price')
df = remove_outliers_iqr(df, 'area')

print("Cleaned Shape:", df.shape)

# 5. Save the Cleaned Data
# We index=False so we don't save the row numbers as a column
df.to_csv('C:\\Users\\Surface\\OneDrive\\Documentos\\GitHub\\House-Price-Prediction\\data\\Housing_Cleaned.csv', index=False)
print("Data cleaning complete! Saved to 'Housing_Cleaned.csv'")