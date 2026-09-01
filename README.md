# 🤖 Agentic AI Customer Churn Prediction & Retention System

An end-to-end **Machine Learning + Agentic AI** system that predicts customer churn and automatically generates personalized retention strategies.

## 📌 Overview

The system combines a **Random Forest Classifier** with a **CrewAI retention agent powered by Groq (`openai/gpt-oss-120b`)**.

It identifies high-risk customers, generates personalized retention plans, simulates customer responses, and stores the outcomes in a closed-loop workflow.

## 🚀 Key Features

* 📊 Customer churn prediction using Random Forest
* ⚠️ High-risk customer identification (>70% churn probability)
* 🤖 Autonomous retention strategies using CrewAI + Groq
* 📋 Structured AI outputs using Pydantic
* 🔄 Closed-loop customer response simulation
* 📈 Interactive Streamlit dashboard
* 💾 Automated retention outcome persistence

## 📊 Results

* **7,043** customer records analyzed
* **1,836** high-risk customers identified
* **100%** JSON schema compliance
* **80.6%** simulated retention rate

## 🛠️ Tech Stack

**Python • Scikit-learn • Random Forest • Pandas • NumPy • CrewAI • Groq • Pydantic • Streamlit**

## 🔄 Workflow

```text
Customer Data
     ↓
Churn Prediction
     ↓
High-Risk Customers
     ↓
CrewAI Retention Agent
     ↓
Personalized Retention Plan
     ↓
Customer Response Simulation
     ↓
Closed-Loop Results
```

## ▶️ Run the Project

```bash
pip install -r requirements.txt
streamlit run app.py
```

Add your Groq API key to `.env`:

```env
GROQ_API_KEY=your_api_key
```

## 🎓 Project

**Final Capstone Project — Elevate Labs**

> **Predict → Analyze → Act → Learn**

The project demonstrates how traditional predictive ML can be combined with Agentic AI to transform churn prediction into automated customer retention.

## 👩‍💻 Author

**Sreya S.**
MSc Data Science & Business Analysis
