"""
Admin Section with Firebase Authentication
"""

from flask import Blueprint, request, render_template, redirect, session
from firebase_admin import firestore
import pandas as pd
from functools import wraps
from datetime import datetime, timedelta
import json

admin_bp = Blueprint('admin', __name__)

ADMIN_EMAIL = "admin@heartpredictor.com"

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('admin_logged_in'):
            return redirect('/admin/login')
        return f(*args, **kwargs)
    return decorated_function

@admin_bp.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        if email == ADMIN_EMAIL and password == 'admin123':
            session['admin_logged_in'] = True
            session['admin_email'] = email
            return redirect('/admin/dashboard')
        else:
            return render_template('admin_login.html'), 401
    
    return render_template('admin_login.html')

@admin_bp.route('/admin/dashboard')
@admin_required
def admin_dashboard():
    try:
        db = firestore.client()
        
        predictions_ref = db.collection('predictions').order_by('timestamp', direction=firestore.Query.DESCENDING).limit(100)
        predictions = list(predictions_ref.get())
        
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
                'confidence': confidence
            })
        
        trend_dates = []
        trend_values = []
        
        for i in range(6, -1, -1):
            date = datetime.now() - timedelta(days=i)
            date_str = date.strftime('%m/%d')
            trend_dates.append(date_str)
            
            day_high = 0
            day_total = 0
            
            for doc in predictions:
                data = doc.to_dict()
                ts = data.get('timestamp')
                if hasattr(ts, 'strftime') and ts.strftime('%Y-%m-%d') == date.strftime('%Y-%m-%d'):
                    day_total += 1
                    if data.get('prediction') == 'HIGH RISK':
                        day_high += 1
            
            percentage = (day_high / day_total * 100) if day_total > 0 else 0
            trend_values.append(round(percentage, 1))
        
        # Convert to JSON strings for JavaScript
        confidence_bins_json = json.dumps(confidence_bins)
        trend_dates_json = json.dumps(trend_dates)
        trend_values_json = json.dumps(trend_values)
        
        return render_template('admin_dashboard.html',
                              email=session.get('admin_email'),
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

@admin_bp.route('/admin/logout')
def admin_logout():
    session.pop('admin_logged_in', None)
    session.pop('admin_email', None)
    return redirect('/')