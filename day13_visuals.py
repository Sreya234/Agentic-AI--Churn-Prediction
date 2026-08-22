import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Chart 1: ML Model Risk Distribution (Day 4 Output)
try:
    df = pd.read_csv("churn_predictions_closed_loop.csv")
except FileNotFoundError:
    df = pd.read_csv("churn_predictions.csv")

sns.histplot(df["Churn_Probability"], kde=True, ax=axes[0], color="#2b5c8f", bins=20)
axes[0].axvline(0.70, color="red", linestyle="--", label="Agent Action Threshold (>0.70)")
axes[0].set_title("Predictive Model Risk Distribution", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Predicted Churn Probability")
axes[0].set_ylabel("Customer Count")
axes[0].legend()

# Chart 2: Closed-Loop Agent Conversion Rate (Days 11-12 Output)
if "Offer_Accepted" in df.columns and df["Retention_Attempted"].sum() > 0:
    attempted = df[df["Retention_Attempted"] == True]
    counts = attempted["Offer_Accepted"].value_counts().rename({True: "Accepted", False: "Declined"})
    
    colors = ["#2ca02c", "#d62728"]
    counts.plot(kind="pie", autopct="%1.1f%%", ax=axes[1], colors=colors, startangle=90, explode=(0.05, 0))
    axes[1].set_title("Agent Retention Success Rate (Batch)", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("")
else:
    axes[1].text(0.5, 0.5, "Run day10_closed_loop.py\nto generate conversion metrics.", 
                 ha="center", va="center", fontsize=11)
    axes[1].set_title("Agent Retention Success Rate", fontsize=12, fontweight="bold")

plt.tight_layout()
plt.savefig("phase3_retention_metrics.png", dpi=300)
print("📊 Visualizations saved successfully as 'phase3_retention_metrics.png'.")