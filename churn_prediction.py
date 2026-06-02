"""
Customer Churn Prediction using Machine Learning
Complete implementation with Logistic Regression, Decision Tree, Random Forest, and SVM.
Dataset: IBM Telco Customer Churn (loaded from GitHub)
"""

# ============================
# 1. Import Required Libraries
# ============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import joblib

# Set visualization style
sns.set_style('whitegrid')

# ============================
# 2. Load Dataset
# ============================
print("="*50)
print("Loading dataset...")
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
df = pd.read_csv(url)
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(df.head())

# ============================
# 3. Data Cleaning
# ============================
print("\n" + "="*50)
print("Data Cleaning...")

# Check missing values
print("Missing values per column:")
print(df.isnull().sum())

# Convert TotalCharges to numeric (it may have empty strings)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Drop rows with missing TotalCharges (only a few)
df.dropna(subset=['TotalCharges'], inplace=True)

# Drop customerID (not useful for modeling)
df.drop('customerID', axis=1, inplace=True)

# Verify data types
print("\nData types after cleaning:")
print(df.dtypes)

# ============================
# 4. Exploratory Data Analysis (EDA)
# ============================
print("\n" + "="*50)
print("Performing EDA...")

# 4.1 Target variable distribution
plt.figure(figsize=(6,4))
sns.countplot(x='Churn', data=df)
plt.title('Churn Distribution')
plt.show()

# 4.2 Categorical features vs Churn (dynamic grid)
categorical_cols = df.select_dtypes(include=['object']).columns.drop('Churn')
n_cats = len(categorical_cols)

# Calculate grid dimensions (max 3 columns)
n_cols = 3
n_rows = (n_cats + n_cols - 1) // n_cols  # ceiling division

fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(15, 5 * n_rows))
axes = axes.flatten()  # flatten to 1D array for easy indexing

for i, col in enumerate(categorical_cols):
    churn_rate = pd.crosstab(df[col], df['Churn'], normalize='index')
    churn_rate.plot(kind='bar', stacked=True, ax=axes[i], legend=False)
    axes[i].set_title(f'Churn Rate by {col}')
    axes[i].set_ylabel('Proportion')
    axes[i].tick_params(axis='x', rotation=45)

# Hide any unused subplots
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

axes[0].legend(['No', 'Yes'], loc='upper right')
plt.tight_layout()
plt.show()

# 4.3 Numerical features vs Churn
numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15,5))
for i, col in enumerate(numerical_cols):
    sns.boxplot(x='Churn', y=col, data=df, ax=axes[i])
plt.tight_layout()
plt.show()

# 4.4 Correlation heatmap (convert Churn to numeric temporarily)
df_corr = df.copy()
df_corr['Churn'] = df_corr['Churn'].map({'Yes':1, 'No':0})
num_df = df_corr.select_dtypes(include=[np.number])
plt.figure(figsize=(10,8))
sns.heatmap(num_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.show()

print("EDA completed. Close plots to continue...")

# ============================
# 5. Feature Engineering & Preprocessing
# ============================
print("\n" + "="*50)
print("Preprocessing data...")

# Encode binary Yes/No columns (including target)
binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
le = LabelEncoder()
for col in binary_cols:
    df[col] = le.fit_transform(df[col])

# One-hot encode multi-category columns
multi_cat_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                  'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                  'Contract', 'PaymentMethod']
df = pd.get_dummies(df, columns=multi_cat_cols, drop_first=True)

# Scale numerical features
scaler = StandardScaler()
df[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.fit_transform(df[['tenure', 'MonthlyCharges', 'TotalCharges']])

print("Preprocessing completed.")
print(f"Final feature matrix shape: {df.shape}")

# ============================
# 6. Train-Test Split & Handle Imbalance (SMOTE)
# ============================
print("\n" + "="*50)
print("Splitting data and handling imbalance...")

X = df.drop('Churn', axis=1)
y = df['Churn']

print("Original class distribution:")
print(y.value_counts())

# Split (stratify to maintain class ratio)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Apply SMOTE to training data only
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

print("\nAfter SMOTE - training class distribution:")
print(pd.Series(y_train_res).value_counts())

# ============================
# 7. Model Building & Evaluation
# ============================
print("\n" + "="*50)
print("Training models...")

# Define models (including SVM)
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
    'SVM': SVC(random_state=42, probability=True)  # probability=True for potential calibration
}

# Store results
results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_res, y_train_res)
    y_pred = model.predict(X_test)
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    results[name] = {
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'Confusion Matrix': cm
    }
    
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

# ============================
# 8. Model Comparison
# ============================
print("\n" + "="*50)
print("Model Comparison:")

results_df = pd.DataFrame(results).T
print(results_df[['Accuracy', 'Precision', 'Recall', 'F1-Score']])

# Plot confusion matrices
fig, axes = plt.subplots(1, 4, figsize=(20,4))
for i, (name, result) in enumerate(results.items()):
    sns.heatmap(result['Confusion Matrix'], annot=True, fmt='d', cmap='Blues', ax=axes[i])
    axes[i].set_title(name)
    axes[i].set_xlabel('Predicted')
    axes[i].set_ylabel('Actual')
plt.tight_layout()
plt.show()

# ============================
# 9. Feature Importance (Random Forest)
# ============================
print("\n" + "="*50)
print("Feature Importance from Random Forest:")

rf_model = models['Random Forest']
importances = rf_model.feature_importances_
feature_names = X.columns

# Create a DataFrame for better visualization
feat_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feat_imp = feat_imp.sort_values('Importance', ascending=False).head(10)

plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Feature', data=feat_imp)
plt.title('Top 10 Feature Importances (Random Forest)')
plt.tight_layout()
plt.show()

print("\nTop 10 features:")
print(feat_imp.to_string(index=False))

# ============================
# 10. Save the Best Model and Scaler
# ============================
print("\n" + "="*50)
print("Saving best model (Random Forest) and scaler...")

joblib.dump(rf_model, 'churn_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Model saved as 'churn_model.pkl'")
print("Scaler saved as 'scaler.pkl'")
print("\nProject completed successfully!")