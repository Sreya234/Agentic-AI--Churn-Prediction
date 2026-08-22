import os
import pandas as pd
from groq import Groq

# 1. Check API Key
api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    raise ValueError("⚠️ GROQ_API_KEY not found! Set it using: $env:GROQ_API_KEY='your_key'")

client = Groq(api_key=api_key)

# 2. Load Predictions from Day 4
df = pd.read_csv("churn_predictions.csv")

# Filter for High Risk customers (Probability >= 70%)
high_risk_df = df[df['Risk_Level'] == 'High Risk']

print("--- AI AGENT ACTIVATED ---")
print(f"Total High-Risk Customers Available: {len(high_risk_df)}")

# Pick the top 3 high-risk customers for demonstration
sample_customers = high_risk_df.head(3)

def generate_retention_plan(customer):
    prompt = f"""
    You are an expert Customer Retention AI Agent for a Telecom provider.
    Analyze this high-risk customer and suggest a specific retention action:

    - Customer ID: {customer['customerID']}
    - Tenure: {customer['tenure']} months
    - Monthly Charges: ${customer['MonthlyCharges']}
    - Contract Type: {customer['Contract']}
    - Payment Method: {customer['PaymentMethod']}
    - Churn Risk Probability: {customer['Churn_Probability']:.2%}

    Provide your answer in exactly this structure:
    1. Primary Risk Factor: (1 sentence explaining why they are at risk)
    2. Recommended Retention Offer: (A realistic discount, contract upgrade, or perk)
    3. Representative Script: (A 2-sentence script for a customer service rep)
    """

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=300
    )
    return response.choices[0].message.content

print("\nGenerating personalized retention strategies...\n")

for idx, (_, customer) in enumerate(sample_customers.iterrows(), 1):
    print("=" * 65)
    print(f"CUSTOMER {idx} | ID: {customer['customerID']} | Risk: {customer['Churn_Probability']:.1%}")
    print("=" * 65)
    
    plan = generate_retention_plan(customer)
    print(plan)
    print("\n")

print("✅ Day 5 AI Agent Execution Complete!")