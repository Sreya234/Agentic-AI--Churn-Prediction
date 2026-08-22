import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv(r"D:\agentic_churn_project\WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("--- 1. DATASET OVERVIEW ---")
print(df.info())

print("\n--- 2. CHURN DISTRIBUTION ---")
churn_counts = df['Churn'].value_counts()
churn_pct = df['Churn'].value_counts(normalize=True) * 100
print(f"No Churn: {churn_counts['No']} ({churn_pct['No']:.1f}%)")
print(f"Churn:    {churn_counts['Yes']} ({churn_pct['Yes']:.1f}%)")

print("\n--- 3. DATA CLEANING CHECK ---")
# The 'TotalCharges' column in Telco dataset often has hidden space characters
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
missing_total_charges = df['TotalCharges'].isnull().sum()
print(f"Missing/Blank values found in 'TotalCharges': {missing_total_charges}")

print("\n--- 4. GENERATING SUMMARY PLOTS ---")
# Plot Churn Distribution
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.countplot(x='Churn', data=df, palette='Set2')
plt.title("Overall Customer Churn Count")

plt.subplot(1, 2, 2)
sns.boxplot(x='Churn', y='tenure', data=df, palette='Set2')
plt.title("Tenure (Months) vs Churn Risk")

plt.tight_layout()
plt.savefig("eda_summary.png")
print("✅ Saved plot summary to 'eda_summary.png'")