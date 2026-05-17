"""
Admin Section with Google Authentication and Roles
"""

from flask import Blueprint, request, render_template, redirect, session, jsonify
from firebase_admin import auth as admin_auth, firestore
import pandas as pd
from functools import wraps
from datetime import datetime, timedelta
import json
import os

admin_bp = Blueprint('admin', __name__)

# User roles and permissions
USERS = {
    'doctor.smith@hospital.com': {'role': 'doctor', 'name': 'Dr. Smith'},
    'dr.jones@clinic.org': {'role': 'doctor', 'name': 'Dr. Jones'},
    'isha18082004@gmail.com': {'role': 'admin', 'name': 'System Admin'},
    'anushagupta1808@gmail.com': {'role': 'doctor', 'name': 'Dr.Anusha'},
    'anusha.23bce10866@vitbhopal.ac.in': {'role': 'doctor', 'name': 'Dr. John'},
    # Add more doctors here
}

def get_user_role(email):
    """Get role for a user"""
    if email in USERS:
        return USERS[email]['role']
    # Check domain-based authorization for doctors
    domain = email.split('@')[-1]
    if domain in ['hospital.com', 'clinic.org']:
        return 'doctor'
    return None

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('admin_logged_in'):
            return redirect('/login')
        if session.get('user_role') != 'admin':
            return "Access denied. Admin privileges required.", 403
        return f(*args, **kwargs)
    return decorated_function

def doctor_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('admin_logged_in'):
            return redirect('/login')
        if session.get('user_role') not in ['admin', 'doctor']:
            return "Access denied. Doctor privileges required.", 403
        return f(*args, **kwargs)
    return decorated_function


@admin_bp.route('/login')
def admin_login():
    """Google login page"""
    firebase_config = {
        'apiKey': os.environ.get('FIREBASE_API_KEY'),
        'authDomain': os.environ.get('FIREBASE_AUTH_DOMAIN'),
        'projectId': os.environ.get('FIREBASE_PROJECT_ID'),
        'storageBucket': os.environ.get('FIREBASE_STORAGE_BUCKET'),
        'messagingSenderId': os.environ.get('FIREBASE_MESSAGING_SENDER_ID'),
        'appId': os.environ.get('FIREBASE_APP_ID')
    }
    return render_template('admin_login_google.html', firebase_config=firebase_config)

@admin_bp.route('/admin/google-auth', methods=['POST'])
def google_auth():
    """Verify Google ID token and create session"""
    try:
        data = request.get_json()
        id_token = data.get('id_token')
        
        # Verify the ID token
        decoded_token = admin_auth.verify_id_token(id_token)
        email = decoded_token.get('email')
        
        if not email:
            return jsonify({'status': 'error', 'message': 'No email provided'}), 400
        
        # Get user role
        role = get_user_role(email)
        
        if role:
            user_info = USERS.get(email, {})
            custom_name = user_info.get('name', decoded_token.get('name', email))
            session['admin_logged_in'] = True
            session['admin_email'] = email
            session['user_role'] = role
            session['user_name'] = custom_name
            
            # Redirect based on role
            return jsonify({'status': 'success', 'redirect': '/'}), 200
        else:
            return jsonify({'status': 'error', 'message': f'Email {email} is not authorized'}), 403
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

# Admin routes (full access)
@admin_bp.route('/admin/dashboard')
@admin_required
def admin_dashboard():
    try:
        db = firestore.client()
        
        # Get all predictions (no filters)
        predictions_ref = db.collection('predictions').order_by('timestamp', direction=firestore.Query.DESCENDING).limit(100)
        predictions = list(predictions_ref.get())
        
        # Get stats
        stats_ref = db.collection('stats').document('summary')
        stats = stats_ref.get().to_dict() or {}
        
        total_predictions = stats.get('total_predictions', 0)
        high_risk_count = stats.get('high_risk_count', 0)
        low_risk_count = stats.get('low_risk_count', 0)
        avg_confidence = stats.get('avg_confidence', 0)
        avg_confidence_percent = round(avg_confidence * 100, 1)
        
        predictions_list = []
        confidence_bins = [0, 0, 0, 0, 0]
        
        for doc in predictions:
            data = doc.to_dict()
            timestamp = data.get('timestamp', '')
            if hasattr(timestamp, 'strftime'):
                time_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
            else:
                time_str = str(timestamp)
            
            confidence = data.get('confidence', 0)
            
            if confidence < 0.2:
                confidence_bins[0] += 1
            elif confidence < 0.4:
                confidence_bins[1] += 1
            elif confidence < 0.6:
                confidence_bins[2] += 1
            elif confidence < 0.8:
                confidence_bins[3] += 1
            else:
                confidence_bins[4] += 1
            
            predictions_list.append({
                'patient_id': doc.id,
                'time': time_str,
                'age': data.get('age', ''),
                'sex': data.get('sex', ''),
                'cp': data.get('cp', ''),
                'trestbps': data.get('trestbps', ''),
                'chol': data.get('chol', ''),
                'prediction': data.get('prediction', ''),
                'confidence': confidence,
                'doctor_name': data.get('created_by_name', 'Unknown'),
                'doctor_email': data.get('created_by_email', 'Unknown')
            })
        
        # Trend calculation
        trend_dates = []
        trend_values = []
        from datetime import datetime, timedelta
        
        for i in range(6, -1, -1):
            date = datetime.now() - timedelta(days=i)
            date_str = date.strftime('%m/%d')
            trend_dates.append(date_str)
            
            day_predictions = [p for p in predictions if p.to_dict().get('timestamp') and hasattr(p.to_dict().get('timestamp'), 'strftime') and p.to_dict().get('timestamp').strftime('%Y-%m-%d') == date.strftime('%Y-%m-%d')]
            day_high = sum(1 for p in day_predictions if p.to_dict().get('prediction') == 'HIGH RISK')
            day_total = len(day_predictions)
            percentage = (day_high / day_total * 100) if day_total > 0 else 0
            trend_values.append(round(percentage, 1))
        
        confidence_bins_json = json.dumps(confidence_bins)
        trend_dates_json = json.dumps(trend_dates)
        trend_values_json = json.dumps(trend_values)
        
        return render_template('admin_dashboard.html',
                              email=session.get('admin_email'),
                              name=session.get('user_name'),
                              role=session.get('user_role'),
                              total_predictions=total_predictions,
                              high_risk_count=high_risk_count,
                              low_risk_count=low_risk_count,
                              avg_confidence_percent=avg_confidence_percent,
                              predictions=predictions_list,
                              confidence_bins=confidence_bins_json,
                              trend_dates=trend_dates_json,
                              trend_values=trend_values_json)
    
    except Exception as e:
        return f"Dashboard Error: {str(e)}"

@admin_bp.route('/admin/export')
@admin_required
def export_csv():
    try:
        db = firestore.client()
        predictions_ref = db.collection('predictions').order_by('timestamp', direction=firestore.Query.DESCENDING)
        predictions = list(predictions_ref.get())
        
        data = []
        for doc in predictions:
            item = doc.to_dict()
            item['patient_id'] = doc.id
            if hasattr(item.get('timestamp'), 'strftime'):
                item['timestamp'] = item['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            data.append(item)
        
        df = pd.DataFrame(data)
        csv_data = df.to_csv(index=False)
        
        from flask import Response
        return Response(
            csv_data,
            mimetype="text/csv",
            headers={"Content-disposition": "attachment; filename=predictions_export.csv"}
        )
    except Exception as e:
        return f"Export error: {e}"

@admin_bp.route('/logout')
def logout():
    session.clear()
    return redirect('/login')

@admin_bp.route('/admin/patient/<patient_id>')
@doctor_required
def patient_detail(patient_id):
    try:
        db = firestore.client()
        doc = db.collection('predictions').document(patient_id).get()
        
        if not doc.exists:
            return "Patient not found", 404
        
        data = doc.to_dict()
        
        timestamp = data.get('timestamp', '')
        if hasattr(timestamp, 'strftime'):
            timestamp = timestamp.strftime('%Y-%m-%d %H:%M:%S')
        
        return render_template('patient_detail.html',
                              patient_id=patient_id,
                              timestamp=timestamp,
                              data=data,
                              session = session)
    
    except Exception as e:
        return f"Error loading patient: {str(e)}"

@admin_bp.route('/admin/history')
@admin_required
def admin_history():
    try:
        db = firestore.client()
        
        # Get all predictions (no filters)
        predictions_ref = db.collection('predictions').order_by('timestamp', direction=firestore.Query.DESCENDING)
        predictions = list(predictions_ref.get())
        
        predictions_list = []
        for doc in predictions:
            data = doc.to_dict()
            timestamp = data.get('timestamp', '')
            if hasattr(timestamp, 'strftime'):
                time_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
            else:
                time_str = str(timestamp)
            
            predictions_list.append({
                'patient_id': doc.id,
                'time': time_str,
                'age': data.get('age', ''),
                'sex': data.get('sex', ''),
                'prediction': data.get('prediction', ''),
                'confidence': data.get('confidence', 0),
                'doctor_name': data.get('created_by_name', 'Unknown'),
                'doctor_email': data.get('created_by_email', 'Unknown')
            })
        
        return render_template('admin_history.html',
                              email=session.get('admin_email'),
                              role=session.get('user_role'),
                              predictions=predictions_list)
    except Exception as e:
        return f"Error: {e}"
        
# Doctor dashboard (limited access)
@admin_bp.route('/doctor/dashboard')
@doctor_required
def doctor_dashboard():
    try:
        db = firestore.client()
        doctor_email = session.get('admin_email')
        
        # Get all predictions by this doctor (for stats)
        all_ref = db.collection('predictions').where('created_by_email', '==', doctor_email)
        all_predictions = list(all_ref.get())
        
        # Calculate stats
        total_count = len(all_predictions)
        high_risk_count = sum(1 for p in all_predictions if p.to_dict().get('prediction') == 'HIGH RISK')
        low_risk_count = total_count - high_risk_count
        
        # Get only last 5 predictions for dashboard
        predictions_ref = db.collection('predictions').where('created_by_email', '==', doctor_email).order_by('timestamp', direction=firestore.Query.DESCENDING).limit(5)
        predictions = list(predictions_ref.get())
        
        predictions_list = []
        for doc in predictions:
            data = doc.to_dict()
            timestamp = data.get('timestamp', '')
            if hasattr(timestamp, 'strftime'):
                time_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
            else:
                time_str = str(timestamp)
            
            predictions_list.append({
                'patient_id': doc.id,
                'time': time_str,
                'age': data.get('age', ''),
                'sex': data.get('sex', ''),
                'prediction': data.get('prediction', ''),
                'confidence': data.get('confidence', 0)
            })
        
        return render_template('doctor_dashboard.html',
                              name=session.get('user_name'),
                              email=doctor_email,
                              predictions=predictions_list,
                              total_count=total_count,
                              high_risk_count=high_risk_count,
                              low_risk_count=low_risk_count)
    except Exception as e:
        return f"Doctor Dashboard Error: {str(e)}"

@admin_bp.route('/doctor/history')
@doctor_required
def doctor_history():
    try:
        db = firestore.client()
        doctor_email = session.get('admin_email')
        
        # Get ALL predictions by this doctor
        predictions_ref = db.collection('predictions').where('created_by_email', '==', doctor_email).order_by('timestamp', direction=firestore.Query.DESCENDING)
        predictions = list(predictions_ref.get())
        
        predictions_list = []
        for doc in predictions:
            data = doc.to_dict()
            timestamp = data.get('timestamp', '')
            if hasattr(timestamp, 'strftime'):
                time_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
            else:
                time_str = str(timestamp)
            
            predictions_list.append({
                'patient_id': doc.id,
                'time': time_str,
                'age': data.get('age', ''),
                'sex': data.get('sex', ''),
                'prediction': data.get('prediction', ''),
                'confidence': data.get('confidence', 0)
            })
        
        return render_template('doctor_history.html',
                              name=session.get('user_name'),
                              email=doctor_email,
                              predictions=predictions_list,
                              total=len(predictions_list))
    except Exception as e:
        return f"Doctor History Error: {str(e)}"