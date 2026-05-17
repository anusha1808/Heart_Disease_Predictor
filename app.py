"""
Heart Disease Predictor - Complete Flask App with Firebase
"""

from flask import Flask, request, render_template, jsonify, session, redirect
import joblib
import pandas as pd
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
import secrets
from admin import admin_bp
import os
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# Initialize Firebase Admin (Backend)
if os.path.exists('serviceAccountKey.json'):
    cred = credentials.Certificate('serviceAccountKey.json')
    print("Firebase initialized from local file (serviceAccountKey.json)")
else:
    # Then check for environment variable (production)
    firebase_creds_str = os.environ.get('FIREBASE_ADMIN_SDK_JSON')
    if firebase_creds_str:
        cred_dict = json.loads(firebase_creds_str)
        cred = credentials.Certificate(cred_dict)
        print("Firebase initialized from environment variable")
    else:
        print("ERROR: No Firebase credentials found")
        raise Exception("Missing Firebase credentials")

firebase_admin.initialize_app(cred)
db = firestore.client()

# Register admin blueprint
app.register_blueprint(admin_bp)

# Load model
model = joblib.load('model.pkl')

# Helper function to get Firestore db
def get_db():
    return db

# Helper function to update statistics
def update_stats(prediction, confidence):
    try:
        stats_ref = db.collection('stats').document('summary')
        stats = stats_ref.get().to_dict() or {}
        
        total = stats.get('total_predictions', 0) + 1
        high_risk = stats.get('high_risk_count', 0) + (1 if prediction == 'HIGH RISK' else 0)
        low_risk = stats.get('low_risk_count', 0) + (1 if prediction == 'LOW RISK' else 0)
        
        # Calculate running average of confidence
        old_avg = stats.get('avg_confidence', 0)
        old_total = stats.get('total_predictions', 0)
        new_avg = (old_avg * old_total + confidence) / total if total > 0 else confidence
        
        stats_ref.set({
            'total_predictions': total,
            'high_risk_count': high_risk,
            'low_risk_count': low_risk,
            'avg_confidence': new_avg,
            'last_updated': firestore.SERVER_TIMESTAMP
        })
    except Exception as e:
        print(f"Stats update error: {e}")

def generate_patient_id(form_data, timestamp):
    """Generate meaningful patient ID"""
    # Option 1: Timestamp-based (simple)
    patient_id = f"PAT-{timestamp.strftime('%Y%m%d%H%M%S')}-{hash(str(form_data)) % 10000:04d}"
    
    return patient_id

# # Helper function to save prediction to Firebase
def save_to_firebase(form_data, prediction, prediction_code, confidence, ip_address):
    try:
        timestamp = datetime.now()
        patient_id = generate_patient_id(form_data, timestamp)
        created_by_email = session.get('admin_email', 'unknown')
        created_by_name = session.get('user_name', 'Unknown')
        
        doc_ref = db.collection('predictions').document(patient_id).set({
            'patient_id': patient_id,
            'timestamp': timestamp,
            'age': form_data.get('age'),
            'sex': form_data.get('sex'),
            'cp': form_data.get('cp'),
            'trestbps': form_data.get('trestbps'),
            'chol': form_data.get('chol'),
            'fbs': form_data.get('fbs'),
            'restecg': form_data.get('restecg'),
            'thalch': form_data.get('thalch'),
            'exang': form_data.get('exang'),
            'oldpeak': form_data.get('oldpeak'),
            'slope': form_data.get('slope'),
            'ca': form_data.get('ca'),
            'thal': form_data.get('thal'),
            'prediction': prediction,
            'prediction_code': prediction_code,
            'confidence': confidence,
            'ip_address': ip_address,
            'created_by_email': created_by_email,
            'created_by_name': created_by_name
        })
        print(f"Saved prediction for patient: {patient_id}")
        update_stats(prediction, confidence)
        return True
    except Exception as e:
        print(f"Firebase save error: {e}")
        return False

# Helper function for input validation
def validate_input(data):
    required_fields = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
                      'restecg', 'thalch', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    
    missing = [field for field in required_fields if field not in data and data.get(field) is not None]
    if missing and len(missing) > 0:
        return False, f"Missing fields: {missing}"
    
    try:
        age = float(data['age'])
        if age < 0 or age > 120:
            return False, "Age must be between 0 and 120"
    except (ValueError, TypeError):
        return False, "Age must be a number"
    
    try:
        trestbps = float(data['trestbps'])
        if trestbps < 50 or trestbps > 250:
            return False, "Blood pressure must be between 50 and 250"
    except (ValueError, TypeError):
        return False, "Blood pressure must be a number"
    
    try:
        chol = float(data['chol'])
        if chol < 100 or chol > 600:
            return False, "Cholesterol must be between 100 and 600"
    except (ValueError, TypeError):
        return False, "Cholesterol must be a number"
    
    try:
        thalch = float(data['thalch'])
        if thalch < 60 or thalch > 220:
            return False, "Max heart rate must be between 60 and 220"
    except (ValueError, TypeError):
        return False, "Max heart rate must be a number"
    
    valid_sex = ['Male', 'Female']
    if data['sex'] not in valid_sex:
        return False, f"Sex must be one of {valid_sex}"
    
    valid_cp = ['typical angina', 'atypical angina', 'non-anginal pain', 'asymptomatic']
    if data['cp'] not in valid_cp:
        return False, f"Chest pain type must be one of {valid_cp}"
    
    return True, "Valid"

# Web route
@app.route('/', methods=['GET', 'POST'])
def home():

    if not session.get('admin_logged_in'):
        return redirect('/login')
     
    form_data = {}
    prediction = None
    confidence = None
    
    if request.method == 'POST':
        try:
            form_data = {
                'age': request.form.get('age', ''),
                'sex': request.form.get('sex', 'Male'),
                'cp': request.form.get('cp', 'typical angina'),
                'trestbps': request.form.get('trestbps', ''),
                'chol': request.form.get('chol', ''),
                'fbs': request.form.get('fbs', 'false'),
                'restecg': request.form.get('restecg', 'normal'),
                'thalch': request.form.get('thalch', ''),
                'exang': request.form.get('exang', 'false'),
                'oldpeak': request.form.get('oldpeak', ''),
                'slope': request.form.get('slope', 'upsloping'),
                'ca': request.form.get('ca', ''),
                'thal': request.form.get('thal', 'normal')
            }
            
            features = {
                'age': float(form_data['age']),
                'sex': form_data['sex'],
                'cp': form_data['cp'],
                'trestbps': float(form_data['trestbps']),
                'chol': float(form_data['chol']),
                'fbs': form_data['fbs'],
                'restecg': form_data['restecg'],
                'thalch': float(form_data['thalch']),
                'exang': form_data['exang'],
                'oldpeak': float(form_data['oldpeak']),
                'slope': form_data['slope'],
                'ca': float(form_data['ca']),
                'thal': form_data['thal']
            }
            
            input_df = pd.DataFrame([features])
            prediction_result = int(model.predict(input_df)[0])
            probability = float(model.predict_proba(input_df)[0][1])
            
            prediction = "HIGH RISK" if prediction_result == 1 else "LOW RISK"
            if prediction_result == 1:
                confidence_percent = probability
            else:
                confidence_percent = 1 - probability
            confidence = f"{confidence_percent:.1%}"
                        
            # Save to Firebase
            ip = request.remote_addr
            save_to_firebase(form_data, prediction, prediction_result, confidence_percent, ip)
            
        except Exception as e:
            prediction = "ERROR"
            confidence = str(e)
            print(f"Error: {e}")
    
    return render_template('index.html', 
                         form_data=form_data,
                         prediction=prediction, 
                         confidence=confidence)

# REST API endpoint
@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No JSON data provided'
            }), 400
        
        is_valid, message = validate_input(data)
        if not is_valid:
            return jsonify({
                'status': 'error',
                'message': message
            }), 400
        
        features = {
            'age': float(data['age']),
            'sex': data['sex'],
            'cp': data['cp'],
            'trestbps': float(data['trestbps']),
            'chol': float(data['chol']),
            'fbs': data['fbs'],
            'restecg': data.get('restecg', 'normal'),
            'thalch': float(data['thalch']),
            'exang': data['exang'],
            'oldpeak': float(data['oldpeak']),
            'slope': data['slope'],
            'ca': float(data['ca']),
            'thal': data['thal']
        }
        
        input_df = pd.DataFrame([features])
        prediction_result = int(model.predict(input_df)[0])
        probability = float(model.predict_proba(input_df)[0][1])
        
        # Calculate confidence for the predicted class (ONCE)
        if prediction_result == 1:
            prediction_text = "HIGH RISK"
            display_text = "High Risk"
            confidence_value = probability
        else:
            prediction_text = "LOW RISK"
            display_text = "Low Risk"
            confidence_value = 1 - probability
        
        # Save to Firebase with correct confidence
        form_data = {k: str(v) for k, v in data.items()}
        ip = request.remote_addr
        save_to_firebase(form_data, prediction_text, prediction_result, confidence_value, ip)
        
        return jsonify({
            'status': 'success',
            'prediction': display_text,
            'prediction_code': prediction_result,
            'confidence': round(confidence_value, 3),
            'confidence_percentage': f"{confidence_value:.1%}",
            'input_received': {
                'age': data['age'],
                'sex': data['sex'],
                'chest_pain_type': data['cp']
            }
        }), 200
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500
    
    
# Health check
@app.route('/api/health', methods=['GET'])
def health_check():
    try:
        stats_ref = db.collection('stats').document('summary')
        stats = stats_ref.get().to_dict() or {}
        return jsonify({
            'status': 'healthy',
            'model_loaded': True,
            'database': 'Firebase Firestore',
            'predictions_stored': stats.get('total_predictions', 0),
            'endpoints': ['/ (web)', '/api/predict (POST)', '/api/health (GET)', '/admin/dashboard']
        })
    except Exception as e:
        return jsonify({
            'status': 'degraded',
            'error': str(e)
        }), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("HEART DISEASE PREDICTOR - FIREBASE DEPLOYMENT")
    print("="*60)
    print("Web Interface: http://127.0.0.1:5000/")
    print("="*60 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)