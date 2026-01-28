import json
import re
import matplotlib.pyplot as plt
import numpy as np

# --- CONFIG ---
RESULTS_FILE = "paper_results_disl.json"
GAS_FILE = "gas_estimates.json"
PREFIX_TO_REMOVE = "contracts_organized_disl/"

def main():
    # 1. Load Data
    print("Loading data...")
    gas_data = {}
    with open(GAS_FILE, 'r') as f:
        matches = re.findall(r'"([^"]+)":\s*(\d+)', f.read())
        for k, v in matches: gas_data[k] = int(v)

    with open(RESULTS_FILE, 'r') as f:
        content = f.read()

    # 2. Extract Savings per Bin
    # Bins: 0.80-0.85, 0.85-0.90, 0.90-0.95, 0.95-1.00
    bins = {
        '0.80-0.85': [],
        '0.85-0.90': [],
        '0.90-0.95': [],
        '0.95-1.00': []
    }
    
    pattern = re.compile(r'"(contracts_organized_disl/[^"]+)":\s*\[|\[\s*"([^"]+)",\s*"([^"]+)",\s*([\d\.]+)\s*\]')
    
    curr_uid, curr_gas = None, 0
    
    for m in pattern.finditer(content):
        if m.group(1): # Key
            curr_uid = m.group(1).replace(PREFIX_TO_REMOVE, "")
            curr_gas = gas_data.get(curr_uid, 0)
        elif m.group(2): # Match
            if curr_uid and curr_gas > 0:
                sig, path, score = m.group(2), m.group(3), float(m.group(4))
                
                if score < 0.80: continue
                
                match_uid = f"{path.replace(PREFIX_TO_REMOVE, '')}.{sig}"
                match_gas = gas_data.get(match_uid, 0)
                
                if match_gas > 0 and match_uid != curr_uid:
                    diff = abs(curr_gas - match_gas)
                    max_g = max(curr_gas, match_gas)
                    savings = (diff / max_g) * 100
                    
                    if savings < 1.0: continue # Ignore noise < 1%

                    # Assign to Bin
                    if 0.80 <= score < 0.85: bins['0.80-0.85'].append(savings)
                    elif 0.85 <= score < 0.90: bins['0.85-0.90'].append(savings)
                    elif 0.90 <= score < 0.95: bins['0.90-0.95'].append(savings)
                    elif 0.95 <= score <= 1.00: bins['0.95-1.00'].append(savings)

    # 3. Plotting
    labels = sorted(bins.keys())
    data = [bins[l] for l in labels]
    
    plt.figure(figsize=(8, 5))
    
    # Boxplot properties
    box = plt.boxplot(data, labels=labels, patch_artist=True, 
                      showfliers=False, # Hide extreme outliers to keep chart readable
                      medianprops=dict(color="black", linewidth=1.5))
    
    # Coloring
    colors = ['#A1C9F4', '#FFB482', '#8DE5A1', '#FF9F9B']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    plt.title("Distribution of Gas Savings by Semantic Similarity")
    plt.xlabel("Similarity Interval (WL Kernel)")
    plt.ylabel("Gas Savings (%)")
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Add counts above boxes
    for i, d in enumerate(data):
        plt.text(i+1, np.median(d) + 2, f"n={len(d)}", ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig("fig_savings_boxplot.pdf")
    print("Saved fig_savings_boxplot.pdf")

if __name__ == "__main__":
    main()