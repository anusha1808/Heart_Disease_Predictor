import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
import joblib
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

print("="*70)
print("PHASE 1: ADVANCED MODEL TRAINING WITH HYPERPARAMETER TUNING")
print("="*70)

# Load dataset
print("\nStep 1: Loading Dataset...")
df = pd.read_csv('heart_disease_uci.csv')
df = df.dropna(subset=['num'])
print(f"Loaded {len(df)} patient records")

# Prepare features and target
X = df.drop(['num', 'id', 'dataset'], axis=1)
y = (df['num'] > 0).astype(int)
print(f"Features: {X.shape[1]} clinical parameters")
print(f"Disease prevalence: {y.mean()*100:.1f}%")

# Define columns
categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
numerical_cols = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca']

# Preprocessing pipeline
print("\nStep 2: Building Preprocessing Pipeline...")
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

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# ============================================================
# HYPERPARAMETER TUNING WITH GRIDSEARCHCV
# ============================================================
print("\n" + "="*70)
print("Step 3: Hyperparameter Tuning with GridSearchCV")
print("="*70)

# 1. Logistic Regression Tuning
print("\nTuning Logistic Regression...")
log_reg_params = {
    'classifier__C': [0.01, 0.1, 1, 10],
    'classifier__penalty': ['l1', 'l2'],
    'classifier__solver': ['liblinear', 'saga']
}

log_reg_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'))
])

log_reg_grid = GridSearchCV(
    log_reg_pipeline, 
    log_reg_params, 
    cv=5, 
    scoring='accuracy',
    n_jobs=-1
)
log_reg_grid.fit(X_train, y_train)
print(f"Best params: {log_reg_grid.best_params_}")
print(f"Best CV accuracy: {log_reg_grid.best_score_:.3f}")

# 2. Random Forest Tuning
print("\nTuning Random Forest...")
rf_params = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [10, 20, None],
    'classifier__min_samples_split': [2, 5, 10]
}

rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))
])

rf_grid = GridSearchCV(
    rf_pipeline, 
    rf_params, 
    cv=5, 
    scoring='accuracy',
    n_jobs=-1
)
rf_grid.fit(X_train, y_train)
print(f"Best params: {rf_grid.best_params_}")
print(f"Best CV accuracy: {rf_grid.best_score_:.3f}")

# 3. SVM Tuning
print("\nTuning SVM...")
svm_params = {
    'classifier__C': [0.1, 1, 10],
    'classifier__kernel': ['rbf', 'linear'],
    'classifier__gamma': ['scale', 'auto']
}

svm_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', SVC(random_state=42, probability=True, class_weight='balanced'))
])

svm_grid = GridSearchCV(
    svm_pipeline, 
    svm_params, 
    cv=5, 
    scoring='accuracy',
    n_jobs=-1
)
svm_grid.fit(X_train, y_train)
print(f"Best params: {svm_grid.best_params_}")
print(f"Best CV accuracy: {svm_grid.best_score_:.3f}")

# Store tuned models
tuned_models = {
    'Logistic Regression': log_reg_grid.best_estimator_,
    'Random Forest': rf_grid.best_estimator_,
    'SVM': svm_grid.best_estimator_
}

print("\n" + "="*70)
print("Step 4: Evaluating Tuned Models on Test Set")
print("="*70)

results = {}
best_model = None
best_accuracy = 0

for name, model in tuned_models.items():
    print(f"\nEvaluating {name}...")
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    accuracy = accuracy_score(y_test, y_pred)
    
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    
    results[name] = {
        'pipeline': model,
        'accuracy': accuracy,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    
    print(f"Test Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    print(f"CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_model_name = name

print("\n" + "="*70)
print(f"BEST MODEL AFTER TUNING: {best_model_name} (Accuracy: {best_accuracy:.3f})")
print("="*70)

# Save best model
joblib.dump(best_model, 'model.pkl')
print("\nBest model saved as 'model.pkl'")

# Detailed evaluation
print("\n" + "="*70)
print("Step 5: Detailed Evaluation Metrics")
print("="*70)

y_pred_best = results[best_model_name]['y_pred']

print("\nClassification Report:")
print(classification_report(y_test, y_pred_best, target_names=['No Disease', 'Disease']))

cm = confusion_matrix(y_test, y_pred_best)
tn, fp, fn, tp = cm.ravel()

print("\nConfusion Matrix:")
print(f"True Negatives: {tn}, False Positives: {fp}")
print(f"False Negatives: {fn}, True Positives: {tp}")

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"\nPrecision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1-Score: {f1:.3f}")

if results[best_model_name]['y_pred_proba'] is not None:
    roc_auc = roc_auc_score(y_test, results[best_model_name]['y_pred_proba'])
    print(f"ROC-AUC: {roc_auc:.3f}")

# Visualizations
print("\n" + "="*70)
print("Step 6: Generating Visualizations")
print("="*70)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Disease', 'Disease'],
            yticklabels=['No Disease', 'Disease'])
plt.title(f'Confusion Matrix - {best_model_name}')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=100)
print("Saved: confusion_matrix.png")

plt.figure(figsize=(10, 8))
for name, data in results.items():
    if data['y_pred_proba'] is not None:
        fpr, tpr, _ = roc_curve(y_test, data['y_pred_proba'])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves - Tuned Models')
plt.legend()
plt.tight_layout()
plt.savefig('roc_curves.png', dpi=100)
print("Saved: roc_curves.png")

print("\n" + "="*70)
print("TRAINING COMPLETE")
print("="*70)