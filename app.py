from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import traceback

app = Flask(__name__)
model = joblib.load('model.pkl')

# Helper function for input validation
def validate_input(data):
    """Validate all required fields and data types"""
    required_fields = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
                      'restecg', 'thalch', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    
    # Check missing fields
    missing = [field for field in required_fields if field not in data]
    if missing:
        return False, f"Missing fields: {missing}"
    
    # Validate age
    try:
        age = float(data['age'])
        if age < 0 or age > 120:
            return False, "Age must be between 0 and 120"
    except (ValueError, TypeError):
        return False, "Age must be a number"
    
    # Validate trestbps (blood pressure)
    try:
        trestbps = float(data['trestbps'])
        if trestbps < 50 or trestbps > 250:
            return False, "Blood pressure must be between 50 and 250"
    except (ValueError, TypeError):
        return False, "Blood pressure must be a number"
    
    # Validate cholesterol
    try:
        chol = float(data['chol'])
        if chol < 100 or chol > 600:
            return False, "Cholesterol must be between 100 and 600"
    except (ValueError, TypeError):
        return False, "Cholesterol must be a number"
    
    # Validate thalch (max heart rate)
    try:
        thalch = float(data['thalch'])
        if thalch < 60 or thalch > 220:
            return False, "Max heart rate must be between 60 and 220"
    except (ValueError, TypeError):
        return False, "Max heart rate must be a number"
    
    # Validate ca (major vessels)
    try:
        ca = float(data['ca'])
        if ca < 0 or ca > 3:
            return False, "Major vessels must be between 0 and 3"
    except (ValueError, TypeError):
        return False, "Major vessels must be a number"
    
    # Validate categorical values
    valid_sex = ['Male', 'Female']
    if data['sex'] not in valid_sex:
        return False, f"Sex must be one of {valid_sex}"
    
    valid_cp = ['typical angina', 'atypical angina', 'non-anginal pain', 'asymptomatic']
    if data['cp'] not in valid_cp:
        return False, f"Chest pain type must be one of {valid_cp}"
    
    valid_fbs = ['false', 'true']
    if data['fbs'] not in valid_fbs:
        return False, f"FBS must be one of {valid_fbs}"
    
    valid_exang = ['false', 'true']
    if data['exang'] not in valid_exang:
        return False, f"Exercise angina must be one of {valid_exang}"
    
    valid_slope = ['upsloping', 'flat', 'downsloping']
    if data['slope'] not in valid_slope:
        return False, f"Slope must be one of {valid_slope}"
    
    valid_thal = ['normal', 'fixed defect', 'reversible defect']
    if data['thal'] not in valid_thal:
        return False, f"Thalassemia must be one of {valid_thal}"
    
    return True, "Valid"

# Web route (existing)
@app.route('/', methods=['GET', 'POST'])
def home():
    form_data = {}
    prediction = None
    confidence = None
    
    if request.method == 'POST':
        try:
            # Get form data
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
            
            # Prepare features for prediction
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
            
            # Convert to DataFrame
            input_df = pd.DataFrame([features])
            
            # Predict
            prediction_result = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0][1]
            
            prediction = "HIGH RISK" if prediction_result == 1 else "LOW RISK"
            confidence = f"{probability:.1%}"
            
        except Exception as e:
            prediction = "ERROR"
            confidence = str(e)
            print(f"Error: {e}")
    
    return render_template('index.html', 
                         form_data=form_data,
                         prediction=prediction, 
                         confidence=confidence)

# NEW: REST API endpoint (Phase 3)
@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    REST API endpoint for prediction
    Accepts JSON, returns JSON
    """
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No JSON data provided. Please send JSON with patient data.'
            }), 400
        
        # Validate input
        is_valid, message = validate_input(data)
        if not is_valid:
            return jsonify({
                'status': 'error',
                'message': message
            }), 400
        
        # Prepare features
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
        
        # Convert to DataFrame
        input_df = pd.DataFrame([features])
        
        # Predict
        prediction_result = int(model.predict(input_df)[0])
        probability = float(model.predict_proba(input_df)[0][1])
        
        # Return JSON response
        return jsonify({
            'status': 'success',
            'prediction': 'High Risk' if prediction_result == 1 else 'Low Risk',
            'prediction_code': prediction_result,
            'confidence': round(probability, 3),
            'confidence_percentage': f"{probability:.1%}",
            'input_received': {
                'age': data['age'],
                'sex': data['sex'],
                'chest_pain_type': data['cp']
            }
        }), 200
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e),
            'traceback': traceback.format_exc()
        }), 500

# Optional: GET endpoint to check API health
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': True,
        'endpoints': ['/ (web)', '/api/predict (POST)', '/api/health (GET)']
    })

if __name__ == '__main__':
    print("\n" + "="*50)
    print("SERVER STARTING")
    print("="*50)
    print("Web interface: http://127.0.0.1:5000/")
    print("API endpoint: http://127.0.0.1:5000/api/predict")
    print("API health check: http://127.0.0.1:5000/api/health")
    print("="*50 + "\n")
    app.run(debug=True)