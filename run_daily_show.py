import subprocess
import os
import sys
import time
from datetime import datetime

# --- CONFIGURATION ---
# The filenames of your three distinct stages
SCRIPT_1_HARVEST = "email_harvest.py"
SCRIPT_2_PRODUCE = "main.py"
SCRIPT_3_DEPLOY  = "deploy_github.py"

def run_step(script_name, step_name):
    """Runs a python script and stops the whole show if it fails."""
    print(f"\n▶️  STARTING STEP: {step_name} ({script_name})...")
    
    if not os.path.exists(script_name):
        print(f"❌ ERROR: File '{script_name}' not found.")
        return False

    try:
        # Check=True will raise an error if the script crashes
        subprocess.run([sys.executable, script_name], check=True)
        print(f"✅ {step_name} COMPLETE.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {step_name} FAILED (Exit Code {e.returncode})")
        print("   ⛔ Stopping production line.")
        return False

def run_production_pipeline():
    start_time = datetime.now()
    print("="*60)
    print(f"🎬 DAILY BRIEFING PRODUCTION STARTED: {start_time.strftime('%H:%M:%S')}")
    print("="*60)

    # --- STEP 1: HARVEST EMAILS ---
    if not run_step(SCRIPT_1_HARVEST, "Email Harvesting"):
        return

    # --- STEP 2: GENERATE CONTENT ---
    # (Assuming main.py reads the emails and creates the audio)
    if not run_step(SCRIPT_2_PRODUCE, "AI Production"):
        return

    # --- STEP 3: DEPLOY TO WORLD ---
    if not run_step(SCRIPT_3_DEPLOY, "GitHub Deployment"):
        return

    # --- SUMMARY ---
    duration = datetime.now() - start_time
    print("\n" + "="*60)
    print(f"🎉 SUCCESS! SHOW IS LIVE.")
    print(f"⏱️  Total Run Time: {duration}")
    print("="*60)

if __name__ == "__main__":
    run_production_pipeline()