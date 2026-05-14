"""
Initialize Firebase Firestore Collections
Run this once after creating Firebase project
"""

import firebase_admin
from firebase_admin import credentials, firestore, auth
import time

print("="*60)
print("FIREBASE INITIALIZATION")
print("="*60)

# Initialize Firebase Admin
try:
    cred = credentials.Certificate('serviceAccountKey.json')
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("Connected to Firebase successfully")
except Exception as e:
    print(f"Error: {e}")
    print("Make sure serviceAccountKey.json is in the project folder")
    exit()

# Create collections and add sample data
print("\nCreating collections...")

# 1. Users collection (for admin users)
users_ref = db.collection('users')

# Check if admin exists
admin_exists = False
for doc in users_ref.where('email', '==', 'admin@heartpredictor.com').get():
    admin_exists = True

if not admin_exists:
    users_ref.add({
        'email': 'admin@heartpredictor.com',
        'role': 'admin',
        'created_at': firestore.SERVER_TIMESTAMP
    })
    print("Created admin user: admin@heartpredictor.com")
else:
    print("Admin user already exists")

# 2. Predictions collection (will be populated by app)
print("Predictions collection ready (will be populated by app)")

# 3. Stats collection
stats_ref = db.collection('stats').document('summary')
if not stats_ref.get().exists:
    stats_ref.set({
        'total_predictions': 0,
        'high_risk_count': 0,
        'low_risk_count': 0,
        'last_updated': firestore.SERVER_TIMESTAMP
    })
    print("Created stats document")

print("\n" + "="*60)
print("FIREBASE INITIALIZATION COMPLETE")
print("="*60)