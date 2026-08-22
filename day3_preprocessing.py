import pandas as pd

# 1. Load data
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("--- STARTING PREPROCESSING ---")

# 2. Handle missing TotalCharges
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
median_total = df['TotalCharges'].median()
df['TotalCharges'] = df['TotalCharges'].fillna(median_total)
print(f"✅ Handled missing TotalCharges using median (${median_total:.2f})")

# 3. Drop customerID explicitly from the dataframe
if 'customerID' in df.columns:
    df = df.drop(columns=['customerID'])

# 4. Convert target column ('Churn' -> 1/0)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# 5. One-Hot Encode all non-numeric columns
categorical_cols = df.select_dtypes(exclude=['number']).columns
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# 6. Convert all True/False booleans to integers (1/0)
for col in df_encoded.columns:
    if df_encoded[col].dtype == 'bool':
        df_encoded[col] = df_encoded[col].astype(int)

print(f"✅ Processed dataset shape: {df_encoded.shape}")

# 7. Save preprocessed dataset
df_encoded.to_csv("churn_processed.csv", index=False)
print("✅ Saved clean data as 'churn_processed.csv'")
