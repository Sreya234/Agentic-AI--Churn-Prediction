import pandas as pd
import sklearn

print("✅ Environment setup successful!")
print(f"Pandas version: {pd.__version__}")
print(f"Scikit-Learn version: {sklearn.__version__}")

# Load dataset preview
try:
    df = pd.read_csv(r"D:\agentic_churn_project\WA_Fn-UseC_-Telco-Customer-Churn.csv")
    print(f"\n✅ Dataset loaded successfully! Shape: {df.shape}")
    print("\nFirst 3 records:")
    print(df[['customerID', 'tenure', 'MonthlyCharges', 'Churn']].head(3))
except FileNotFoundError:
    print("\n⚠️ 'WA_Fn-UseC_-Telco-Customer-Churn.csv' not found in the specified path. Please check the file path.")