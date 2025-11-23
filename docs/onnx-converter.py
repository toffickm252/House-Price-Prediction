from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import joblib
import numpy as np
from skl2onnx import to_onnx
import onnx
from sklearn.ensemble import RandomForestClassifier

# Load model 
model = joblib.load('house_price_model.pkl')
n_features = model.n_features_in_

# Convert model to ONNX format
initial_type = [('input', FloatTensorType([None, n_features]))]

onnx_model = convert_sklearn(model, initial_types=initial_type)

with open("model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
print("Model successfully converted to ONNX format and saved as 'model.onnx'.")