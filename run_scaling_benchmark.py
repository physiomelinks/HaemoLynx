import numpy as np
import subprocess
import time
import csv
from pathlib import Path

# Configuration
TEST_VOLUMES = np.arange(0.1, 1.1, 0.1)
BRANCHES = [
    {"name": "devel_dale", "label": "Original Pipeline"},
    {"name": "devel_dale_cb_pipeline_speedup", "label": "Optimized Pipeline (All Speedups)"}
]
OUTPUT_FILE = "benchmark_scaling_results.csv"

def run_benchmark():
    # Initialize CSV
    with open(OUTPUT_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Branch", "Sub_Volume", "Runtime_Seconds"])

    for branch in BRANCHES:
        branch_name = branch["name"]
        label = branch["label"]
        print(f"\n==================================================")
        print(f"SWITCHING TO BRANCH: {branch_name} ({label})")
        print(f"==================================================")
        
        # Checkout the branch
        checkout = subprocess.run(["git", "checkout", branch_name], capture_output=True, text=True)
        if checkout.returncode != 0:
            print(f"Failed to checkout {branch_name}:\n{checkout.stderr}")
            continue
            
        for vol in TEST_VOLUMES:
            print(f"\n--- Testing {label} at {vol*100:.0f}% Sub-Volume ---")
            
            start_time = time.time()
            
            # Run the pipeline script in a subprocess
            cmd = ["venv/bin/python", "examples/carotid_image_to_model.py", "--sub-volume", str(vol)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            end_time = time.time()
            duration = end_time - start_time
            
            if result.returncode == 0:
                print(f"Success! Runtime: {duration:.2f} seconds.")
                # Save immediately so data isn't lost if a larger run crashes
                with open(OUTPUT_FILE, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([label, vol, duration])
            else:
                print(f"FAILED or CRASHED. Returning inf and aborting remaining tests for {label}.")
                print(f"Error Log:\n{result.stderr[-1000:]}") # Print tail of error
                with open(OUTPUT_FILE, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([label, vol, float('inf')])
                break  # Skip the rest of the volumes for this branch

                    
    print("\nBENCHMARK COMPLETE. Returning to optimized branch...")
    subprocess.run(["git", "checkout", "devel_dale_cb_pipeline_speedup"])
    print(f"Results saved to {OUTPUT_FILE}")
    print("Run `venv/bin/python plot_benchmark.py` to generate the graph.")

if __name__ == "__main__":
    run_benchmark()