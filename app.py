from flask import Flask, request, jsonify, send_from_directory
import joblib
import pandas as pd
import numpy as np
import os
from flask_cors import CORS
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
# Load the trained model, preprocessor, and selected feature indices
try:
    model = joblib.load('heart_failure_model.joblib')
    preprocessor = joblib.load('preprocessor.joblib')
    selected_feature_indices = joblib.load('selected_features.joblib')
    print("Model, preprocessor, and selected features loaded successfully.")
except FileNotFoundError:
    print("Error: Model files not found. Please run model_training.py first.")
    model = None
    preprocessor = None
    selected_feature_indices = None

# Define the expected features (order matters for prediction)
# These should match the original DataFrame columns used for training, excluding 'DEATH_EVENT'
EXPECTED_FEATURES = ['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 
                     'ejection_fraction', 'high_blood_pressure', 'platelets', 
                     'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time']

@app.route('/')
def serve_index():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return send_from_directory(current_dir, 'index.html')
@app.route('/predict', methods=['POST'])
def predict():
    if model is None or preprocessor is None or selected_feature_indices is None:
        return jsonify({"error": "Model not loaded. Please ensure model_training.py was run."}), 500

    try:
        data = request.get_json(force=True)
        
        # Convert input data to a DataFrame
        input_df = pd.DataFrame([data])
        
        # Ensure all expected features are present, fill missing with NaN if necessary
        # This is crucial for consistent preprocessing
        for feature in EXPECTED_FEATURES:
            if feature not in input_df.columns:
                input_df[feature] = np.nan # Or a sensible default/imputation strategy

        # Reorder columns to match training data
        input_df = input_df[EXPECTED_FEATURES]

        # Preprocess the input data
        processed_data = preprocessor.transform(input_df)
        
        # Select the same features as during training
        selected_data = processed_data[:, selected_feature_indices]

        # Make prediction
        prediction = model.predict(selected_data)[0]
        prediction_proba = model.predict_proba(selected_data)[0][1] # Probability of death event

        result = {
            "prediction": "Death Event" if prediction == 1 else "Survival",
            "probability_of_death": f"{prediction_proba:.4f}"
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# This part is for local development only.
# In production, a WSGI server (like Gunicorn) would manage the application.
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)