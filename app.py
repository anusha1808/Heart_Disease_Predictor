from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load('model.pkl')

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
            
            prediction = "❤️ HIGH RISK" if prediction_result == 1 else "💚 LOW RISK"
            confidence = f"{probability:.1%}"
            
        except Exception as e:
            prediction = "⚠️ ERROR"
            confidence = str(e)
    
    return render_template('index.html', 
                         form_data=form_data,
                         prediction=prediction, 
                         confidence=confidence)

if __name__ == '__main__':
    print("\n🚀 Server starting...")
    print("📍 Open http://127.0.0.1:5000 in your browser")
    print("⚠️ Press CTRL+C to stop\n")
    app.run(debug=True)