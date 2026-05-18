# ❤️ Heart Disease Prediction System

A full-stack machine learning web application that predicts heart disease risk using 13 clinical parameters, with role-based authentication (Admin/Doctor), real-time predictions, and interactive dashboards. The project demonstrates how ML can assist clinical decision-making in an educational context.

---

## Table of Contents
- Project Overview
- Key Features
- Model Performance
- Project Structure
- Machine Learning Pipeline
- Frontend & Dashboard
- Getting Started
- Technologies Used
- API Documentation
- Deployment
- Uptime Monitoring
- Live Deployment Link

---

## Project Overview

Heart disease is a leading cause of death worldwide. Early and accurate risk assessment can significantly improve patient outcomes. This project builds an end-to-end ML web application to:

- Predict heart disease risk from clinical parameters
- Provide confidence scores for each prediction
- Support role-based access (Admin for oversight, Doctors for patient management)
- Visualize prediction analytics and trends

**Important:** This is an educational demonstration project, not a real medical diagnostic tool.

---

##  Key Features

### Machine Learning
- **Multi-model comparison:** Logistic Regression, Random Forest, SVM
- **Hyperparameter tuning:** GridSearchCV with 5-fold cross-validation
- **Random Forest:** 85.9% accuracy, 92.2% recall


### Role-Based Authentication
- **Admin:** Full dashboard, analytics charts, export CSV, view all predictions
- **Doctor:** View only their own predictions, patient history


### Dashboard Features
- Statistics cards (total predictions, high/low risk cases)
- Risk distribution pie chart
- Confidence distribution bar chart
- 7-day risk trend line chart
- Recent predictions table
- Full prediction history


### Patient Management
- Clinical reports with 13 medical parameters
- Print-ready patient reports
- Google OAuth 2.0 authentication

---

##  Model Performance

| Metric | Value |
|--------|-------|
| Accuracy | 85.9% |
| Recall | 92.2% |
| Precision | 83.9% |
| F1-Score | 87.9% |
| ROC-AUC | 0.916 |


**Models Compared:**

| Model | Accuracy |
|-------|----------|
| Logistic Regression | 84.2% |
| Random Forest | 85.9% |
| SVM | 82.6% |

---

##  Project Structure

```
Heart_Disease_Predictor/
│
├── app.py 
├── admin.py 
├── train.py 
├── requirements.txt 
├── model.pkl 
│
├── templates/ 
│ ├── index.html 
│ ├── admin_login_google.html 
│ ├── admin_dashboard.html 
│ ├── admin_history.html 
│ ├── doctor_dashboard.html 
│ ├── doctor_history.html 
│ └── patient_detail.html 
│
├── static/css/ 
│ ├── admin_dashboard.css
│ ├── admin_history.css
│ ├── doctor_dashboard.css
│ ├── doctor_history.css
│ ├── index.css
│ ├── login.css
│ └── patient_detail.css
│
└── .env 

```
---

##  Machine Learning Pipeline

The ML pipeline follows a structured sequence:

### 1. Data Preprocessing
- Load 920 patient records with 13 clinical parameters
- Median imputation for missing values
- StandardScaler for numerical features
- OneHotEncoder for categorical features

### 2. Feature Engineering
- Categorical columns: sex, cp, fbs, restecg, exang, slope, thal
- Numerical columns: age, trestbps, chol, thalch, oldpeak, ca

### 3. Model Training & Tuning
- GridSearchCV with 5-fold cross-validation
- Hyperparameters tuned for each model
- Random Forest selected as best performer

### 4. Model Evaluation
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix, ROC Curve, AUC

### 5. Model Export
- Best model saved as `model.pkl` for API serving

---

##  Frontend & Dashboard

The frontend is built using Flask templates with Chart.js visualizations and consumes predictions from the trained model.

### Visual Elements
- **Risk Distribution Pie Chart:** High vs Low risk proportions
- **Confidence Bar Chart:** Distribution across 5 confidence bins
- **Risk Trend Line Chart:** 7-day high-risk percentage trend
- **Predictions Table:** Recent predictions with doctor names
- **Patient Reports:** Full clinical data with print support

---

##  Getting Started

### Prerequisites

- Python 3.11 or higher
- pip package manager
- Firebase project (for authentication & database)

### Installation

```bash
# Clone the repository
git clone https://github.com/anusha1808/Heart_Disease_Predictor.git
cd Heart_Disease_Predictor
```

Create virtual environment
```bash
python -m venv .venv
source .venv/bin/activate
```
Install dependencies
```bash
pip install -r requirements.txt
```
Train the model
```bash
python train.py
```
Run the application
```bash
python app.py
```

---

### Firebase Setup
- Create a project on Firebase Console
- Enable Authentication → Google Sign-in method
- Create Firestore Database in test mode
- Generate Service Account Key
- Create .env file with your Firebase config

---

## API Documentation

Health Check
```bash
GET /api/health
```
Prediction Endpoint
```bash
POST /api/predict
Content-Type: application/json
```

Request Example:
```bash
{
  "age": 68,
  "sex": "Male",
  "cp": "asymptomatic",
  "trestbps": 155,
  "chol": 280,
  "fbs": "true",
  "restecg": "lv hypertrophy",
  "thalch": 115,
  "exang": "true",
  "oldpeak": 2.5,
  "slope": "flat",
  "ca": 2,
  "thal": "reversible defect"
}
```
Response:
```bash
{
  "status": "success",
  "prediction": "High Risk",
  "confidence": 0.922,
  "confidence_percentage": "92.2%"
}
```

---

## Technologies Used

### Backend Stack

| Technology | Version | Purpose |
|------------|---------|---------|
| Python | 3.11 | Programming language |
| Flask | 2.3.3 | Web framework |
| scikit-learn | 1.3.0 | ML algorithms |
| Gunicorn | 21.2.0 | Production WSGI server |


### Database & Authentication

| Technology | Purpose |
|------------|---------|
| Firebase Firestore | Cloud database |
| Firebase Auth | Google OAuth 2.0 |


### Frontend Stack

| Technology | Purpose |
|------------|---------|
| HTML5 | UI structure |
| CSS3 | Styling |
| Chart.js | Interactive visualizations |


### DevOps & Monitoring

| Technology | Purpose |
|------------|---------|
| Render | Cloud deployment |
| Git / GitHub | Version control |
| CI/CD | Automatic deployment |
| UptimeRobot | 5-min interval monitoring |

---

### Deployment
Deploy on Render

1. Push code to GitHub
2. Create Web Service on [Render](https://render.com)
3. Configure:

| Setting | Value |
|---------|-------|
| Build Command | `pip install -r requirements.txt && python train.py` |
| Start Command | `gunicorn app:app --bind 0.0.0.0:$PORT` |
4. Add environment variables
5. Click **Create Web Service**

---

### Uptime Monitoring
UptimeRobot configured to ping `/api/health` every 5 minutes, eliminating cold starts.

---

### Live Deployment Link
Web Application	- [Heart Disease Prediction System](https://heart-disease-predictor-1-cs1n.onrender.com)
