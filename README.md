# House-Price-Prediction
- Project-2-of-my-learn-by-doing-journey

A machine learning model that estimates housing prices in Kronum, Kumasi. It analyzes features including house area, bedroom/bathroom count, stories, air conditioning, parking availability, and furnishing status to predict property values.

## Technologies Used 🛠️
1. Streamlit: For the interactive web interface.
2. Scikit-Learn: For building the predictive machine learning model.
3. Pandas: For data manipulation and analysis.
4. NumPy: For numerical computations.
5. Seaborn: For data visualization.
6. Joblib: For model persistence.

## Installation ⚙️
1. Clone the repository:
   ```
   git clone https://github.com/toffickm252/housing-price-prediction.git
   ```

2. Navigate to the project directory:
   ```
   cd housing-price-prediction
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage 🚀
To launch the application, run the following command in your terminal:
```
streamlit run src/app.py
```

## Data 📊
- The dataset used for this model was sourced from Kaggle. It is a static dataset containing records of housing properties with the following key features:

1. Numerical Features: Area, Number of Bedrooms, Bathrooms, Stories, Parking.

2. Categorical Features: Main Road access, Guestroom, Basement, Hot water heating, Air conditioning, Preferred area, Furnishing status.

3. Target Variable: Price.

## Methodology 🧪
1. Data Preprocessing:
   - Encoding: Converted categorical variables (like furnishingstatus and mainroad) into numeric formats using encoding techniques to make them machine-readable.
   - Scaling: Applied feature scaling to numerical variables (like area) to ensure all features contribute equally to the model.
   - Splitting: Divided the dataset into training and testing sets to evaluate model performance on unseen data.

2. Model:
   Utilized Linear Regression (or the specific regressor you used) to establish the relationship between the features and the housing price.

## Model Performance 📉
The model was evaluated using the following metrics:

- R² Score: [0.657] - Indicates the proportion of variance in the housing prices predictable from the features.
- Mean Absolute Error (MAE): [0.190] - The average absolute difference between predicted and actual prices.
- Mean Squared Error (MSE): [0.24] - Measures the average of the squares of the errors.

## Save the Model 💾
- After training, persist the model and any preprocessing objects with `joblib`.
- Example:
  ```python
  from joblib import dump, load
  dump((trained_model, scaler, encoder), "artifacts/house_price_model.joblib")
  ```
- To reuse:
  ```python
  from joblib import load
  model, scaler, encoder = load("artifacts/house_price_model.joblib")
  ```
- Note the file path, version, and date so future runs know which artifact was deployed.

## Problems Faced
- The Python environment was having dependency issues so I had to switch to conda environment.
- By installing Anaconda and changing the Environment Interpreter to conda.
- (base)

## Contributing
Feel free to fork the repository and submit pull requests for improvements.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
