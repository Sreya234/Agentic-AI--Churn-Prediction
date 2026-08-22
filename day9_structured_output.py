import os
import json
import pandas as pd
import litellm
from pydantic import BaseModel, Field
from crewai import Agent, Task, Crew, LLM

# 1. Patch LiteLLM to handle Groq formatting
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

# 2. Check API Key
api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    raise ValueError("⚠️ GROQ_API_KEY environment variable not found!")

# 3. Configure Groq LLM
llm = LLM(
    model="groq/openai/gpt-oss-120b",
    api_key=api_key,
    temperature=0.7
)

# 4. DAY 9: Define Pydantic Schema for Structured Output
class RetentionPlan(BaseModel):
    primary_churn_driver: str = Field(..., description="Main cause for churn risk based on user profile.")
    tailored_incentive: str = Field(..., description="Specific, cost-effective offer (e.g., $15 credit, tier upgrade).")
    outreach_message: str = Field(..., description="Personalized outreach message/email text to send to the customer.")

# 5. Define Agent
retention_specialist = Agent(
    role="Customer Retention Specialist",
    goal="Prevent customer churn by analyzing account history and crafting tailored retention strategies.",
    backstory="You are an expert customer success specialist specializing in telecom retention.",
    verbose=True,
    llm=llm
)

# 6. Load Customer Data
df = pd.read_csv("churn_predictions.csv")
customer = df[df["Churn_Probability"] > 0.70].iloc[0]

# 7. DAY 9: Enforce Structured Output in Task
retention_task = Task(
    description=(
        f"Analyze high-risk customer {customer['customerID']}:\n"
        f"- Tenure: {customer['tenure']} months\n"
        f"- Monthly Charges: ${customer['MonthlyCharges']}\n"
        f"- Contract Type: {customer['Contract']}\n"
        f"- Payment Method: {customer['PaymentMethod']}\n"
        f"- Churn Probability: {customer['Churn_Probability']:.2%}\n\n"
        "Generate a structured retention response containing the churn driver, incentive, and outreach message."
    ),
    expected_output="A structured JSON object with primary_churn_driver, tailored_incentive, and outreach_message.",
    output_json=RetentionPlan,
    agent=retention_specialist
)

crew = Crew(agents=[retention_specialist], tasks=[retention_task])

if __name__ == "__main__":
    print("🚀 Running Day 9 Task with Pydantic JSON enforcement...\n")
    result = crew.kickoff()
    
    # Parse and display JSON output
    plan_dict = json.loads(result.raw)
    
    print("\n" + "=" * 60)
    print("📌 DAY 9 STRUCTURED JSON OUTPUT")
    print("=" * 60)
    print(json.dumps(plan_dict, indent=2))