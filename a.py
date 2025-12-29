from flask import Flask, request, jsonify, render_template, send_from_directory
import pickle
import joblib
import numpy as np
from flask_cors import CORS
import os

app = Flask(__name__, static_folder='.')
CORS(app, resources={r"/*": {"origins": "*"}})

# Configure Flask to ignore virtual environment and cache files
import os
class Config:
    DEBUG = True
    # Ignore virtual environment and cache directories
    RELOAD_IGNORE_DIRS = [
        '.venv', 
        '__pycache__', 
        'site-packages',
        '.git',
        'tmp'
    ]

# Initialize models as None
fault_model = None
line_model = None
label_encoder = None
xgb_model = None

# Define confidence calculation function
def calculate_confidence(currents, fault_type):
    avg = sum(currents) / 3
    median = sorted(currents)[1]
    std = (sum((x - avg) ** 2 for x in currents) / 3) ** 0.5
    imbalance = std / (avg if avg != 0 else 1)
    
    if fault_type == 'Open Phase Fault':
        return min(98, 85 + 15 * (1 - min(currents) / (avg if avg > 0 else 1)))
    elif fault_type == 'Short Circuit':
        return min(99, 90 + 10 * (max(currents) / (median if median > 0 else 1) - 1))
    elif fault_type == 'Unbalanced Load':
        return min(92, 70 + 30 * imbalance)
    else:  # Normal
        return max(60, 100 - 40 * imbalance)

def preprocess_features(currents, voltages):
    """Calculate additional features needed by the model."""
    # Basic statistics
    avg_current = sum(currents) / 3
    max_current = max(currents)
    min_current = min(currents)
    
    # Calculate moving averages (simplified as regular averages for real-time)
    Ia_wmean_1 = avg_current
    Ia_wmean_2 = avg_current
    Ib_wmean_1 = avg_current
    Ib_wmean_2 = avg_current
    Ic_wmean_1 = avg_current
    Ic_wmean_2 = avg_current
    
    # Combine all features
    features = currents + voltages + [
        Ia_wmean_1, Ia_wmean_2,
        Ib_wmean_1, Ib_wmean_2,
        Ic_wmean_1, Ic_wmean_2
    ]
    return np.array([features])

def simple_ml(currents, voltages):
    # Fallback ML logic when models are not available
    avg = sum(currents) / 3
    median = sorted(currents)[1]
    std = (sum((x - avg) ** 2 for x in currents) / 3) ** 0.5
    imbalance = std / (avg if avg != 0 else 1)
    
    # Open phase detection
    for i, current in enumerate(currents):
        if current < 0.25 * avg and current < 400:
            return "Open Phase Fault", f"L{i+1}", 0.95
            
    # Short circuit detection
    for i, current in enumerate(currents):
        if current > 1.6 * median and current > 800:
            return "Short Circuit", f"L{i+1}", 0.98
            
    # Unbalanced load
    if imbalance > 0.25:
        max_idx = currents.index(max(currents))
        return "Unbalanced Load", f"L{max_idx+1}", 0.85
        
    return "Normal", "Normal", 0.90

# Load models
try:
    with open("cat_model_fault (2).pkl", "rb") as f:
        fault_model = pickle.load(f)
    print("✔ Loaded fault model")
    # Print expected feature names if available
    if hasattr(fault_model, 'feature_names_'):
        print("Expected features:", fault_model.feature_names_)
except Exception as e:
    print("❌ Error loading cat_model_fault (2).pkl:", e)
    fault_model = None

try:
    with open("cat_model_line_cat.pkl", "rb") as f:
        line_model = pickle.load(f)
    print("✔ Loaded line model")
except Exception as e:
    print("❌ Error loading cat_model_line_cat.pkl:", e)

try:
    label_encoder = joblib.load("label_encoder_line.pkl")
    print("✔ Loaded label encoder")
except Exception as e:
    print("❌ Error loading label_encoder_line.pkl with joblib:", e)

# Skip loading XGB pipeline as we're using our own confidence calculation
xgb_model = None
print("ℹ️ Using custom confidence calculation instead of XGB pipeline")

@app.route('/')
def home():
    return send_from_directory('.', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        currents = data.get('currents')   # [L1, L2, L3]
        voltages = data.get('voltages')   # [V1, V2, V3]

        print(f"Received prediction request - currents: {currents}, voltages: {voltages}")

        # Validate input data
        if not currents or not voltages:
            return jsonify({"error": "Missing currents or voltages data"}), 400
            
        if len(currents) != 3 or len(voltages) != 3:
            return jsonify({"error": "Must provide exactly 3 values for both currents and voltages"}), 400
            
        try:
            currents = [float(c) for c in currents]
            voltages = [float(v) for v in voltages]
        except (ValueError, TypeError):
            return jsonify({"error": "All values must be numeric"}), 400

        # Preprocess features for ML models
        try:
            features = preprocess_features(currents, voltages)
            print("Preprocessed features shape:", features.shape)
        except Exception as e:
            print("Error preprocessing features:", e)
            return jsonify({
                "error": "Feature preprocessing failed",
                "details": str(e)
            }), 500

        # First try ML models, fall back to simple_ml if needed
        try:
            if all(model is not None for model in [fault_model, line_model, label_encoder]):
                try:
                    print("Using ML models for prediction...")
                    print("Input feature names:", [f"Feature_{i}" for i in range(features.shape[1])])
                    if hasattr(fault_model, 'feature_names_'):
                        print("Model expects features:", fault_model.feature_names_)
                    
                    # Make predictions
                    fault_pred = fault_model.predict(features)[0]
                    print(f"Fault prediction: {fault_pred}")
                    
                    line_pred_raw = line_model.predict(features)[0]
                    line_label = label_encoder.inverse_transform([line_pred_raw])[0]
                    print(f"Line prediction: {line_label}")
                    
                    # Calculate confidence using our function instead of XGB
                    confidence = calculate_confidence(currents, str(fault_pred))
                    print(f"Calculated confidence: {confidence}")
                    
                    return jsonify({
                        "fault": str(fault_pred),
                        "line": str(line_label),
                        "xgb_composite": str(confidence / 100),
                        "using_ml_model": True
                    })
                except Exception as e:
                    print(f"ML model prediction failed: {str(e)}")
                    print("Falling back to simple_ml...")
                    raise  # Re-raise to fall back to simple_ml
            else:
                print("One or more ML models not loaded, using simple_ml")
                raise Exception("ML models not available")
        except Exception as e:
            # Fall back to simple_ml
            print(f"Using simple_ml due to: {str(e)}")
            fault_pred, line_label, confidence = simple_ml(currents, voltages)
            
            return jsonify({
                "fault": str(fault_pred),
                "line": str(line_label),
                "xgb_composite": str(confidence / 100),
                "using_ml_model": False,
                "fallback_reason": str(e)
            })

    except Exception as e:
        print("Unexpected error in prediction route:", str(e))
        return jsonify({
            "error": "Internal server error",
            "details": str(e)
        }), 500

if __name__ == '__main__':
    app.config.from_object(Config)
    # Run without debug mode to avoid frequent reloads
    app.run(host='127.0.0.1', port=5000)