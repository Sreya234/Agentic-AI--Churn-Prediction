import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import litellm
from pydantic import BaseModel, Field
from crewai import Agent, Task, Crew, LLM

# --- LITELLM GROQ PATCH ---
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

# --- PYDANTIC SCHEMA ---
class RetentionPlan(BaseModel):
    primary_churn_driver: str = Field(..., description="Main reason for churn risk.")
    tailored_incentive: str = Field(..., description="Specific, tailored offer.")
    outreach_message: str = Field(..., description="Personalized email draft.")

# --- UI SETUP ---
st.set_page_config(page_title="AI Retention Dashboard", layout="wide")
st.title("🛡️ AI-Powered Customer Retention Dashboard")

# --- API KEY INPUT ---
api_key = st.sidebar.text_input("Groq API Key", type="password", value=os.environ.get("GROQ_API_KEY", ""))
if not api_key:
    st.warning("⚠️ Please enter your Groq API Key in the sidebar to proceed.")
    st.stop()

os.environ["GROQ_API_KEY"] = api_key

# --- LOAD DATA ---
@st.cache_data
def load_data():
    file_path = "churn_predictions_closed_loop.csv" if os.path.exists("churn_predictions_closed_loop.csv") else "churn_predictions.csv"
    if not os.path.exists(file_path):
        return None
    return pd.read_csv(file_path)

df = load_data()

if df is None:
    st.error("❌ No CSV found. Ensure 'churn_predictions.csv' is in the root folder.")
    st.stop()

# --- NAVIGATION VIA BUTTONS ---
if "active_view" not in st.session_state:
    st.session_state.active_view = "analytics"

col_btn1, col_btn2, _ = st.columns([1, 1, 3])

with col_btn1:
    if st.button("📊 Portfolio Analytics", use_container_width=True, type="primary" if st.session_state.active_view == "analytics" else "secondary"):
        st.session_state.active_view = "analytics"
        st.rerun()

with col_btn2:
    if st.button("🤖 Agent Intervention", use_container_width=True, type="primary" if st.session_state.active_view == "agent" else "secondary"):
        st.session_state.active_view = "agent"
        st.rerun()

st.divider()

# ==========================================
# VIEW 1: PORTFOLIO ANALYTICS
# ==========================================
if st.session_state.active_view == "analytics":
    st.subheader("Predictive Churn & Closed-Loop Analytics")
    
    total_customers = len(df)
    high_risk_count = len(df[df["Churn_Probability"] > 0.70])
    high_risk_pct = (high_risk_count / total_customers) * 100
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Customers Evaluated", f"{total_customers:,}")
    m2.metric("High Risk Targets (>70%)", f"{high_risk_count:,}")
    m3.metric("High Risk Ratio", f"{high_risk_pct:.1f}%")
    
    st.write("")

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    sns.histplot(df["Churn_Probability"], kde=True, ax=axes[0], color="#2b5c8f", bins=20)
    axes[0].axvline(0.70, color="red", linestyle="--", label="Agent Action Threshold (>0.70)")
    axes[0].set_title("Predictive Risk Score Distribution", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Churn Probability")
    axes[0].set_ylabel("Customer Count")
    axes[0].legend()

    if "Offer_Accepted" in df.columns and df["Retention_Attempted"].sum() > 0:
        attempted = df[df["Retention_Attempted"] == True]
        counts = attempted["Offer_Accepted"].value_counts().rename({True: "Accepted", False: "Declined"})
        counts.plot(kind="pie", autopct="%1.1f%%", ax=axes[1], colors=["#2ca02c", "#d62728"], startangle=90, explode=(0.05, 0))
        axes[1].set_title("Simulated Agent Conversion Rate", fontsize=11, fontweight="bold")
        axes[1].set_ylabel("")
    else:
        axes[1].text(0.5, 0.5, "No closed-loop execution data yet.\nRun batch execution to view results.", 
                     ha="center", va="center", fontsize=10)
        axes[1].set_title("Agent Conversion Analytics", fontsize=11, fontweight="bold")

    plt.tight_layout()
    st.pyplot(fig)

# ==========================================
# VIEW 2: AGENT INTERVENTION
# ==========================================
elif st.session_state.active_view == "agent":
    st.subheader("Select Target Account for AI Plan Generation")
    
    high_risk_df = df[df["Churn_Probability"] > 0.70]
    customer_ids = high_risk_df["customerID"].tolist()
    selected_id = st.selectbox("Select High-Risk Customer ID", customer_ids)
    
    customer = high_risk_df[high_risk_df["customerID"] == selected_id].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Churn Risk", f"{customer['Churn_Probability']:.2%}")
    c2.metric("Tenure", f"{customer['tenure']} months")
    c3.metric("Monthly Bill", f"${customer['MonthlyCharges']}")
    c4.metric("Contract Type", customer['Contract'])

    if st.button("🚀 Generate Retention Plan", type="primary"):
        with st.spinner("Executing CrewAI agent via Groq..."):
            llm = LLM(model="groq/openai/gpt-oss-120b", api_key=api_key, temperature=0.7)
            
            agent = Agent(
                role="Customer Retention Specialist",
                goal="Prevent customer churn with customized incentives.",
                backstory="Expert telecom customer success specialist.",
                verbose=False,
                llm=llm
            )
            
            task = Task(
                description=(
                    f"Analyze customer {selected_id} (Tenure: {customer['tenure']}m, "
                    f"Charges: ${customer['MonthlyCharges']}, Contract: {customer['Contract']}, "
                    f"Payment: {customer['PaymentMethod']}). Formulate a retention plan."
                ),
                expected_output="Structured JSON plan.",
                output_json=RetentionPlan,
                agent=agent
            )
            
            crew = Crew(agents=[agent], tasks=[task])
            result = crew.kickoff()
            
            try:
                plan = json.loads(result.raw)
                st.success("✅ Retention Plan Generated!")
                
                st.markdown("### 🔍 AI Analysis")
                st.info(f"**Primary Churn Driver:** {plan['primary_churn_driver']}")
                st.warning(f"**Tailored Incentive Offer:** {plan['tailored_incentive']}")
                
                st.markdown("### 📧 Ready-to-Send Outreach")
                st.code(plan['outreach_message'], language="text")
                
            except Exception as e:
                st.error(f"Failed to parse output: {e}\n\nRaw: {result.raw}")