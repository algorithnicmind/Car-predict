"""
Car Price Prediction — Flask Web Application

Serves a professional frontend and exposes a /predict API endpoint
that uses the trained Random Forest model to predict car prices.
"""

import os
import json
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# ------------------------------------------------------------------
#  Initialise Flask
# ------------------------------------------------------------------

app = Flask(
    __name__,
    static_folder='static',
    template_folder='templates',
)
CORS(app)

# ------------------------------------------------------------------
#  Load model + metadata
# ------------------------------------------------------------------

MODEL_PATH    = os.path.join(os.path.dirname(__file__), 'model.pkl')
METADATA_PATH = os.path.join(os.path.dirname(__file__), 'model_metadata.json')

if not os.path.exists(MODEL_PATH) or not os.path.exists(METADATA_PATH):
    print("\n[ERROR] Model files not found!")
    print("   Run 'python train_model.py' first to train and save the model.")
    print(f"   Expected: {MODEL_PATH}")
    print(f"   Expected: {METADATA_PATH}")
    exit(1)

model    = joblib.load(MODEL_PATH)
with open(METADATA_PATH, 'r') as f:
    metadata = json.load(f)

MODEL_COLUMNS           = metadata['model_columns']
NUMERICAL_FEATURES      = metadata['numerical_features']
CATEGORICAL_FEATURES    = metadata['categorical_features']   # dict: col -> [values]
FEATURE_STATS           = metadata['feature_stats']
MODEL_METRICS           = metadata['model_metrics']

print(f"[OK] Model loaded (R2: {MODEL_METRICS['r2_percentage']}%)")


# ------------------------------------------------------------------
#  Routes
# ------------------------------------------------------------------

@app.route('/')
def index():
    """Serve the main frontend page."""
    return render_template('index.html')


@app.route('/robots.txt')
def static_from_root():
    """Serve robots.txt at the root URL."""
    return app.send_static_file('robots.txt')


@app.errorhandler(404)
def page_not_found(e):
    """Render custom 404 page."""
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_server_error(e):
    """Return JSON error for internal server errors."""
    return jsonify({'error': 'Internal server error occurred.'}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint for monitoring and deployment."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_accuracy': MODEL_METRICS.get('r2_percentage', 'N/A'),
    })


@app.route('/api/metadata', methods=['GET'])
def get_metadata():
    """Return model metadata (feature info, stats, etc.) for the frontend."""
    return jsonify({
        'numerical_features': NUMERICAL_FEATURES,
        'categorical_features': CATEGORICAL_FEATURES,
        'feature_stats': FEATURE_STATS,
        'model_metrics': MODEL_METRICS,
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """Accept car attributes as JSON and return predicted price."""
    try:
        data = request.get_json(force=True)

        # Build a single-row DataFrame with the raw input
        row = {}
        for feat in NUMERICAL_FEATURES:
            val = data.get(feat)
            if val is None:
                return jsonify({'error': f'Missing field: {feat}'}), 400
            row[feat] = float(val)

        for col, valid_values in CATEGORICAL_FEATURES.items():
            val = data.get(col)
            if val is None:
                return jsonify({'error': f'Missing field: {col}'}), 400
            if val not in valid_values:
                return jsonify({'error': f'Invalid value for {col}: {val}. Must be one of {valid_values}'}), 400
            row[col] = val

        input_df = pd.DataFrame([row])

        # One-hot encode (same way as training)
        cat_cols = list(CATEGORICAL_FEATURES.keys())
        input_encoded = pd.get_dummies(input_df, columns=cat_cols, drop_first=True)

        # Align columns with the model (add missing dummies as 0)
        for col in MODEL_COLUMNS:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        input_encoded = input_encoded[MODEL_COLUMNS]

        # Predict
        prediction = model.predict(input_encoded)[0]

        return jsonify({
            'predicted_price': round(float(prediction), 2),
            'formatted_price': f"Rs. {int(prediction):,}",
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/contribute', methods=['POST'])
def contribute_data():
    """Accept new car data from the user and save it to user_data.csv."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # Validation
        required_numerical = NUMERICAL_FEATURES
        required_categorical = list(CATEGORICAL_FEATURES.keys())
        required_fields = required_numerical + required_categorical + ['selling_price']

        row_data = {}
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400
            row_data[field] = data[field]

        # Prepare full CSV row
        full_row = {
            '': '', 
            'car_name': data.get('car_name', 'User Contributed'),
            'brand': data.get('brand', 'Unknown'),
            'model': data.get('model', 'Unknown'),
            'vehicle_age': row_data['vehicle_age'],
            'km_driven': row_data['km_driven'],
            'seller_type': row_data['seller_type'],
            'fuel_type': row_data['fuel_type'],
            'transmission_type': row_data['transmission_type'],
            'mileage': row_data['mileage'],
            'engine': row_data['engine'],
            'max_power': row_data['max_power'],
            'seats': row_data['seats'],
            'selling_price': row_data['selling_price']
        }

        # Append to user_data.csv
        contributed_df = pd.DataFrame([full_row])
        contributed_df.to_csv('user_data.csv', mode='a', header=False, index=False)

        return jsonify({'message': 'Data contributed successfully! Thank you for helping the AI get smarter.'}), 201

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ------------------------------------------------------------------
#  Run
# ------------------------------------------------------------------

if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("  Car Price Prediction Web App")
    print("  http://localhost:5000")
    print("=" * 50 + "\n")
    app.run(debug=True, port=5000)
