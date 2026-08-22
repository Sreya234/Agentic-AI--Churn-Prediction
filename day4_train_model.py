import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# 1. Load preprocessed data
df = pd.read_csv("churn_data.csv")

# 2. Safety Check: Drop customerID or any non-numeric columns if present
if 'customerID' in df.columns:
    df = df.drop(columns=['customerID'])

# 3. Separate Features (X) and Target (y)
X = df.drop(columns=['Churn'])
y = df['Churn']

# Ensure X only contains numeric data types
X = X.select_dtypes(include=['number', 'bool']).astype(float)

# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("--- TRAINING RANDOM FOREST MODEL ---")
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# 5. Evaluate Performance
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\n--- MODEL PERFORMANCE ---")
print(classification_report(y_test, y_pred))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")

# 6. Save Model & Features
joblib.dump(model, "churn_model.pkl")
joblib.dump(list(X.columns), "model_features.pkl")
print("\n✅ Model saved as 'churn_model.pkl'")

# 7. Generate Churn Predictions for Agent Phase
df_results = pd.read_csv("churn_data.csv")
df_results['Churn_Probability'] = model.predict_proba(X)[:, 1]
df_results['Risk_Level'] = df_results['Churn_Probability'].apply(
    lambda p: 'High Risk' if p >= 0.70 else ('Medium Risk' if p >= 0.40 else 'Low Risk')
)

df_results.to_csv("churn_predictions.csv", index=False)
print("✅ Saved final predictions to 'churn_predictions.csv'")

# High Risk Count
high_risk_count = (df_results['Risk_Level'] == 'High Risk').sum()
print(f"\n⚠️ Identified {high_risk_count} High Risk customers ready for AI Agent intervention!")