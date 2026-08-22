import os
import json
import random
import pandas as pd
import litellm
from pydantic import BaseModel, Field
from crewai import Agent, Task, Crew, LLM

# 1. Patch LiteLLM for Groq payload compatibility
litellm.drop_params = True

_original_completion = litellm.completion
def patched_completion(*args, **kwargs):
    if "messages" in kwargs and isinstance(kwargs["messages"], list):
        for msg in kwargs["messages"]:
            if isinstance(msg, dict):
                msg.pop("cache_breakpoint", None)
    return _original_completion(*args, **kwargs)

litellm.completion = patched_completion

os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["CREWAI_TRACING_ENABLED"] = "false"

# 2. Verify API Key
api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    raise ValueError("⚠️ GROQ_API_KEY environment variable not found!")

# 3. Configure Groq LLM
llm = LLM(
    model="groq/openai/gpt-oss-120b",
    api_key=api_key,
    temperature=0.7
)

# 4. Pydantic Schema
class RetentionPlan(BaseModel):
    primary_churn_driver: str = Field(..., description="Main reason for churn risk.")
    tailored_incentive: str = Field(..., description="Specific offer (e.g., $15 credit, speed upgrade).")
    outreach_message: str = Field(..., description="Personalized outreach message/email text.")

# 5. Define Retention Specialist Agent
retention_specialist = Agent(
    role="Customer Retention Specialist",
    goal="Prevent customer churn by analyzing account history and crafting tailored retention strategies.",
    backstory="You are an expert customer success specialist specializing in telecom retention.",
    verbose=False,
    llm=llm
)

# 6. Days 11–12: Feedback Simulation Logic (60% Acceptance Rate)
def simulate_customer_response(acceptance_rate: float = 0.60) -> bool:
    return random.random() < acceptance_rate

# 7. Day 10: Batch Loop Execution
def run_batch_closed_loop(csv_path: str = "churn_predictions.csv", batch_size: int = 5):
    df = pd.read_csv(csv_path)

    # Initialize tracking columns if missing
    for col in ["Retention_Attempted", "Offer_Accepted", "New_Risk_Level"]:
        if col not in df.columns:
            df[col] = False if col != "New_Risk_Level" else "High Risk"

    # Filter High-Risk Customers (> 0.70)
    high_risk_indices = df[df["Churn_Probability"] > 0.70].head(batch_size).index
    print(f"🚀 Starting automated execution for a batch of {len(high_risk_indices)} customers...\n")

    for idx in high_risk_indices:
        customer = df.loc[idx]
        cust_id = customer["customerID"]

        print(f"--------------------------------------------------")
        print(f" Processing Customer: {cust_id} | Risk: {customer['Churn_Probability']:.2%}")
        print(f"--------------------------------------------------")

        # Define Task dynamically per customer
        retention_task = Task(
            description=(
                f"Analyze high-risk customer {cust_id}:\n"
                f"- Tenure: {customer['tenure']} months\n"
                f"- Monthly Charges: ${customer['MonthlyCharges']}\n"
                f"- Contract Type: {customer['Contract']}\n"
                f"- Payment Method: {customer['PaymentMethod']}\n"
                f"- Churn Probability: {customer['Churn_Probability']:.2%}\n\n"
                "Generate a structured retention plan."
            ),
            expected_output="Structured JSON retention plan.",
            output_json=RetentionPlan,
            agent=retention_specialist
        )

        crew = Crew(agents=[retention_specialist], tasks=[retention_task])
        result = crew.kickoff()

        # Parse AI output
        plan = json.loads(result.raw)
        print(f"  [Driver]  : {plan['primary_churn_driver'][:75]}...")
        print(f"  [Offer]   : {plan['tailored_incentive']}")

        # Simulate response & update database records
        accepted = simulate_customer_response(0.60)
        
        df.loc[idx, "Retention_Attempted"] = True
        df.loc[idx, "Offer_Accepted"] = accepted
        df.loc[idx, "New_Risk_Level"] = "Low Risk" if accepted else "Lost Customer"

        outcome = "✅ ACCEPTED (Status: Low Risk)" if accepted else "❌ DECLINED (Status: Lost Customer)"
        print(f"  [Outcome] : {outcome}\n")

    # Save output dataset
    output_path = "churn_predictions_closed_loop.csv"
    df.to_csv(output_path, index=False)
    print("=" * 60)
    print(f"💾 Closed-loop execution complete! Results saved to '{output_path}'.")
    print("=" * 60)

if __name__ == "__main__":
    run_batch_closed_loop(batch_size=5)