import os

def run_step(step_name, script_name):
    print(f"\n==========================================")
    print(f"▶️ Executing {step_name} ({script_name})...")
    print(f"==========================================")
    exit_code = os.system(f"python {script_name}")
    if exit_code != 0:
        raise RuntimeError(f"❌ Execution failed at {step_name}.")

if __name__ == "__main__":
    print("🚀 Launching Master Retention System Pipeline...")
    
    # Days 10-12 Execution
    run_step("Day 10-12 Closed-Loop Pipeline", "day10_closed_loop.py")
    
    # Day 13 Visuals
    run_step("Day 13 Summary Visualizations", "day13_visuals.py")
    
    print("\n✅ Entire pipeline executed cleanly from start to finish.")