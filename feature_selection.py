"""
FEATURE SELECTION - AFTER Baseline Model
Uses model-based importance to decide which features matter
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import joblib

# Import preprocessing pipeline from train.py
# First, load the saved model or recreate pipeline

print("="*80)
print("FEATURE SELECTION - Model-Based Importance")
print("="*80)

# Load data
df = pd.read_csv('heart_disease_uci.csv')
df = df.dropna(subset=['num'])
X = df.drop(['num', 'id', 'dataset'], axis=1)
y = (df['num'] > 0).astype(int)

# Define columns
categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
numerical_cols = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca']

# Split data first
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Load the trained pipeline from train.py
# Option 1: If you have saved the full pipeline in train.py
try:
    pipeline = joblib.load('model.pkl')
    print("Loaded existing pipeline from model.pkl")
except:
    # Option 2: Recreate pipeline (only if model.pkl doesn't exist)
    print("model.pkl not found. Recreating pipeline...")
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer([
        ('num', numerical_pipeline, numerical_cols),
        ('cat', categorical_pipeline, categorical_cols)
    ])
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42, n_estimators=100))
    ])
    
    pipeline.fit(X_train, y_train)

# Get feature names after preprocessing
preprocessor = pipeline.named_steps['preprocessor']
feature_names = []

for name, transformer, columns in preprocessor.transformers_:
    if name == 'num':
        feature_names.extend(columns)
    else:
        encoder = transformer.named_steps['onehot']
        feature_names.extend(encoder.get_feature_names_out(columns))

# Get importance scores from the trained model
feature_importance = pipeline.named_steps['classifier'].feature_importances_

# Create importance dataframe
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print("\nFeature Importance Ranking (Random Forest):")
print("-" * 55)
for i, row in importance_df.head(10).iterrows():
    bar = "=" * int(row['importance'] * 50)
    print(f"{row['feature']:30s}: {row['importance']:.3f} {bar}")

# Plot importance
plt.figure(figsize=(10, 8))
top_features = importance_df.head(10)
plt.barh(top_features['feature'], top_features['importance'], color='steelblue')
plt.xlabel('Importance')
plt.title('Top 10 Features - Random Forest Importance')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=100)
print("\nSaved: feature_importance.png")

# Cumulative importance analysis
print("\n" + "="*80)
print("FEATURE SELECTION DECISION")
print("="*80)

cumulative_importance = importance_df['importance'].cumsum()
print(f"\nCumulative importance:")
print(f"   Top 5 features: {importance_df.head(5)['importance'].sum():.1%} of importance")
print(f"   Top 10 features: {importance_df.head(10)['importance'].sum():.1%} of importance")

# Check how many features needed for 95% importance
cumsum = 0
features_needed = 0
for imp in importance_df['importance']:
    cumsum += imp
    features_needed += 1
    if cumsum >= 0.95:
        break

print(f"   Features needed for 95% importance: {features_needed} out of {len(importance_df)}")

# Show bottom features
print("\nLowest importance features (candidates for removal):")
print("-" * 55)
for i, row in importance_df.tail(5).iterrows():
    print(f"   {row['feature']:30s}: {row['importance']:.3f}")

print("\nFeature selection complete.")