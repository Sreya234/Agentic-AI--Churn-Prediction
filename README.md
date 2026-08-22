# 🤖 Agentic AI Customer Churn Prediction & Retention Platform

An end-to-end **Agentic AI Customer Retention System** that combines Machine Learning-based churn prediction with autonomous AI agents to identify high-risk customers and generate personalized retention strategies.

The system uses a **Random Forest model** to predict customer churn probability and **CrewAI + Groq LLMs** to analyze high-risk customers and generate structured retention plans.

---

## 🚀 Release v1.0.0

### 📌 Overview

**Agentic AI Customer Churn Prediction & Retention Platform** combines:

- Predictive Machine Learning
- AI-powered customer analysis
- Autonomous agent workflows
- Structured retention recommendations
- Customer response simulation
- Closed-loop feedback
- Interactive Streamlit dashboard

The platform moves beyond simply predicting churn by connecting **prediction → AI reasoning → retention action → customer response → feedback**.

---

## 🧠 System Architecture

```text
                    ┌─────────────────────┐
                    │   Customer Dataset  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Data Preprocessing │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Random Forest ML   │
                    │  Churn Prediction   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ High-Risk Filtering │
                    │ Probability > 70%  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    CrewAI Agent     │
                    │ Customer Retention  │
                    │    Specialist       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │     Groq LLM        │
                    │ openai/gpt-oss-120b │
                    └──────────┬──────────┘
                               │
                               ▼
              ┌─────────────────────────────────┐
              │ Structured Retention Strategy   │
              │                                 │
              │ • Churn Driver                  │
              │ • Tailored Incentive            │
              │ • Outreach Message              │
              └───────────────┬─────────────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │ Customer Response   │
                    │    Simulation       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Conversion / Outcome│
                    │      Tracking       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Feedback Loop     │
                    │ Improve Retention   │
                    └─────────────────────┘
