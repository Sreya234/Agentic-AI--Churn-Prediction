import os
import pandas as pd
import litellm
from crewai import Agent, Task, Crew, LLM

# 1. THE FIX: Patch litellm.completion directly to strip cache_breakpoint from messages
litellm.drop_params = True

_original_completion = litellm.completion

def patched_completion(*args, **kwargs):
    if "messages" in kwargs and isinstance(kwargs["messages"], list):
        for msg in kwargs["messages"]:
            if isinstance(msg, dict):
                msg.pop("cache_breakpoint", None)
    return _original_completion(*args, **kwargs)

litellm.completion = patched_completion

# Suppress telemetry noise
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["CREWAI_TRACING_ENABLED"] = "false"

# 2. Check API Key
api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    raise ValueError("⚠️ GROQ_API_KEY environment variable not found!")

# 3. Configure Groq LLM with the updated model
llm = LLM(
    model="groq/openai/gpt-oss-120b",
    api_key=api_key,
    temperature=0.7
)

# 4. Define Retention Agent
retention_specialist = Agent(
    role="Customer Retention Specialist",
    goal="Prevent customer churn by analyzing account history and crafting tailored retention strategies.",
    backstory=(
        "You are an expert telecom customer success specialist. You analyze tenure, contract terms, "
        "and payment patterns to resolve customer pain points with tailored incentives."
    ),
    verbose=True,
    llm=llm
)

# 5. Day 8: Filter High-Risk Customers (> 0.70)
df = pd.read_csv("churn_predictions.csv")
high_risk_customers = df[df["Churn_Probability"] > 0.70]
print(f"🔍 Found {len(high_risk_customers)} high-risk customers (Churn Probability > 70%).\n")

customer = high_risk_customers.iloc[0]

# 6. Inject Context into Task
retention_task = Task(
    description=(
        f"Analyze high-risk customer {customer['customerID']}:\n"
        f"- Tenure: {customer['tenure']} months\n"
        f"- Monthly Charges: ${customer['MonthlyCharges']}\n"
        f"- Contract Type: {customer['Contract']}\n"
        f"- Payment Method: {customer['PaymentMethod']}\n"
        f"- Churn Probability: {customer['Churn_Probability']:.2%}\n\n"
        "Based on these specific risk factors, craft a detailed retention strategy explaining:\n"
        "1. The root cause of their churn risk.\n"
        "2. A custom incentive to address their specific pain points."
    ),
    expected_output="A tailored retention plan based on the customer's account profile.",
    agent=retention_specialist
)

# 7. Execute Crew
crew = Crew(agents=[retention_specialist], tasks=[retention_task])

if __name__ == "__main__":
    print("🚀 Running Day 8 Retention Task...\n")
    result = crew.kickoff()
    
    print("\n" + "=" * 60)
    print("📌 DAY 8 OUTPUT")
    print("=" * 60)
    print(result)