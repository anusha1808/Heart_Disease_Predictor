"""
COMPLETE EDA WITH ACTIONABLE INSIGHTS
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("COMPLETE EDA FOR FEATURE UNDERSTANDING")
print("="*80)

# Load data
df = pd.read_csv('heart_disease_uci.csv')
df = df.dropna(subset=['num'])
df['target'] = (df['num'] > 0).astype(int)

# Convert boolean columns to string for display
for col in df.select_dtypes(include=['bool']).columns:
    df[col] = df[col].astype(str)

print(f"\nDataset: {df.shape[0]} rows, {df.shape[1]} columns")

# ============================================================
# 1. MISSING VALUES ANALYSIS
# ============================================================
print("\n" + "="*80)
print("1. MISSING VALUES ANALYSIS")
print("="*80)

missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({'Missing': missing, 'Percentage': missing_pct})
missing_df = missing_df[missing_df['Missing'] > 0].sort_values('Percentage', ascending=False)

if len(missing_df) > 0:
    print(missing_df)
    print("\nDecision: Use imputation (median for numerical, mode for categorical)")
else:
    print("No missing values found")

# ============================================================
# 2. FEATURE VS TARGET ANALYSIS
# ============================================================
print("\n" + "="*80)
print("2. FEATURE VS TARGET ANALYSIS")
print("="*80)

# Numerical features
numerical_cols = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca']

print("\nNumerical Features - Statistical Comparison:")
print("-" * 70)
print(f"{'Feature':12s} | {'No Disease':>10s} | {'Disease':>10s} | {'Difference':>10s} | {'P-value':>10s} | {'Signif':>6s}")
print("-" * 70)

for col in numerical_cols:
    no_disease_mean = df[df['target']==0][col].mean()
    disease_mean = df[df['target']==1][col].mean()
    diff = disease_mean - no_disease_mean
    
    t_stat, p_value = stats.ttest_ind(
        df[df['target']==0][col].dropna(),
        df[df['target']==1][col].dropna()
    )
    
    if p_value < 0.001:
        significance = "***"
    elif p_value < 0.01:
        significance = "**"
    elif p_value < 0.05:
        significance = "*"
    else:
        significance = "ns"
    
    print(f"{col:12s} | {no_disease_mean:10.1f} | {disease_mean:10.1f} | {diff:10.2f} | {p_value:10.4f} | {significance:>6s}")

# Categorical features
categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']

print("\nCategorical Features - Disease Rate by Category:")
print("-" * 60)

for col in categorical_cols:
    if col in df.columns:
        print(f"\n{col.upper()}:")
        df[col] = df[col].astype(str)
        cross_tab = pd.crosstab(df[col], df['target'], normalize='index') * 100
        for category in cross_tab.index:
            disease_rate = cross_tab.loc[category, 1]
            bar = "=" * int(disease_rate / 5)
            print(f"   {str(category):20s}: {disease_rate:5.1f}% disease {bar}")

# ============================================================
# 3. CORRELATION ANALYSIS
# ============================================================
print("\n" + "="*80)
print("3. CORRELATION ANALYSIS")
print("="*80)

for col in numerical_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

corr_matrix = df[numerical_cols + ['target']].corr()
target_corr = corr_matrix['target'].drop('target').sort_values(ascending=False)

print("\nCorrelation with Target (Heart Disease):")
for feature, corr in target_corr.items():
    if corr > 0.4:
        strength = "Strong positive"
    elif corr > 0.2:
        strength = "Moderate positive"
    elif corr > 0:
        strength = "Weak positive"
    else:
        strength = "Negative"
    print(f"   {feature:12s}: {corr:+.3f} ({strength})")

# ============================================================
# 4. OUTLIER DETECTION
# ============================================================
print("\n" + "="*80)
print("4. OUTLIER DETECTION")
print("="*80)

for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
    outlier_pct = len(outliers) / len(df) * 100
    print(f"   {col:12s}: {len(outliers):3d} outliers ({outlier_pct:.1f}%)")

print("\nDecision: Keep outliers (medical data can have extreme cases)")

# ============================================================
# 5. SKEWNESS ANALYSIS
# ============================================================
print("\n" + "="*80)
print("5. SKEWNESS ANALYSIS")
print("="*80)

for col in numerical_cols:
    skewness = df[col].skew()
    if abs(skewness) > 1:
        shape = "Highly skewed"
    elif abs(skewness) > 0.5:
        shape = "Moderately skewed"
    else:
        shape = "Approximately normal"
    print(f"   {col:12s}: skewness = {skewness:6.3f} ({shape})")

# ============================================================
# 6. FEATURE IMPORTANCE RANKING (EDA-Based)
# ============================================================
print("\n" + "="*80)
print("6. FEATURE IMPORTANCE RANKING")
print("="*80)

importance_scores = {}

for col in numerical_cols:
    corr_score = abs(target_corr.get(col, 0))
    _, p_value = stats.ttest_ind(
        df[df['target']==0][col].dropna(),
        df[df['target']==1][col].dropna()
    )
    p_score = 1 - min(p_value, 0.1) / 0.1
    importance_scores[col] = (corr_score * 0.6 + p_score * 0.4)

for col in categorical_cols:
    if col in df.columns:
        try:
            contingency = pd.crosstab(df[col].astype(str), df['target'])
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
            importance_scores[col] = 1 - min(p_value, 0.1) / 0.1
        except:
            importance_scores[col] = 0.5

sorted_features = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)

print("\nFeature Importance Ranking (Higher = More Predictive):")
print("-" * 50)
for i, (feature, score) in enumerate(sorted_features, 1):
    bar = "=" * int(score * 50)
    print(f"{i:2d}. {feature:15s}: {score:.3f} {bar}")

print("\nRecommendation:")
print("   Keep ALL features initially (no premature dropping)")
print("   Feature selection will happen AFTER model training")

# ============================================================
# 7. GENERATE VISUALIZATIONS
# ============================================================
print("\n" + "="*80)
print("7. GENERATING EDA VISUALIZATIONS")
print("="*80)

fig = plt.figure(figsize=(20, 15))

# 1. Target distribution
ax1 = plt.subplot(3, 3, 1)
df['target'].value_counts().plot(kind='bar', color=['#51cf66', '#ff6b6b'], ax=ax1)
ax1.set_title('Target Distribution', fontweight='bold')
ax1.set_xlabel('Heart Disease')
ax1.set_ylabel('Count')
ax1.set_xticklabels(['No Disease', 'Disease'])

# 2. Correlation heatmap
ax2 = plt.subplot(3, 3, 2)
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax2)
ax2.set_title('Correlation Matrix', fontweight='bold')

# 3. Age distribution by disease
ax3 = plt.subplot(3, 3, 3)
for target, color, label in [(0, '#51cf66', 'No Disease'), (1, '#ff6b6b', 'Disease')]:
    df[df['target']==target]['age'].hist(bins=20, alpha=0.5, color=color, label=label, ax=ax3)
ax3.set_title('Age Distribution by Disease', fontweight='bold')
ax3.set_xlabel('Age')
ax3.set_ylabel('Frequency')
ax3.legend()

# 4. Chest pain analysis
ax4 = plt.subplot(3, 3, 4)
cp_disease = df.groupby('cp')['target'].mean() * 100
cp_disease.plot(kind='bar', color='coral', ax=ax4)
ax4.set_title('Disease Rate by Chest Pain Type', fontweight='bold')
ax4.set_xlabel('Chest Pain Type')
ax4.set_ylabel('Disease Rate (%)')
ax4.tick_params(axis='x', rotation=45)

# 5. Oldpeak boxplot
ax5 = plt.subplot(3, 3, 5)
df.boxplot(column='oldpeak', by='target', ax=ax5)
ax5.set_title('ST Depression (oldpeak) by Disease', fontweight='bold')
ax5.set_xlabel('Disease (0=No, 1=Yes)')

# 6. Max heart rate boxplot
ax6 = plt.subplot(3, 3, 6)
df.boxplot(column='thalch', by='target', ax=ax6)
ax6.set_title('Max Heart Rate by Disease', fontweight='bold')
ax6.set_xlabel('Disease (0=No, 1=Yes)')

# 7. Feature importance bar chart
ax7 = plt.subplot(3, 3, 7)
features = [f[0] for f in sorted_features[:7]]
scores = [f[1] for f in sorted_features[:7]]
ax7.barh(features, scores, color='steelblue')
ax7.set_title('Feature Importance (EDA-Based)', fontweight='bold')
ax7.set_xlabel('Importance Score')

# 8. Thalassemia analysis
ax8 = plt.subplot(3, 3, 8)
thal_disease = df.groupby('thal')['target'].mean() * 100
thal_disease.plot(kind='bar', color='lightgreen', ax=ax8)
ax8.set_title('Disease Rate by Thalassemia', fontweight='bold')
ax8.set_xlabel('Thalassemia Type')
ax8.set_ylabel('Disease Rate (%)')
ax8.tick_params(axis='x', rotation=45)

# 9. Sex analysis
ax9 = plt.subplot(3, 3, 9)
sex_disease = df.groupby('sex')['target'].mean() * 100
sex_disease.plot(kind='bar', color=['lightblue', 'pink'], ax=ax9)
ax9.set_title('Disease Rate by Sex', fontweight='bold')
ax9.set_xlabel('Sex')
ax9.set_ylabel('Disease Rate (%)')

plt.suptitle('EDA Dashboard - Heart Disease Dataset', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('eda_dashboard.png', dpi=100, bbox_inches='tight')
print("Saved: eda_dashboard.png")

print("\n" + "="*80)
print("EDA COMPLETE")
print("="*80)