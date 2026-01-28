import json
import re
import matplotlib.pyplot as plt
import numpy as np
import itertools
import statsmodels

# --- CONFIGURATION ---
RQ1_FILE = "rq1_distribution.json"
RESULTS_FILE = "paper_results_disl.json"
GAS_FILE = "gas_estimates.json"
PREFIX_TO_REMOVE = "contracts_organized_disl/"
MIN_SIMILARITY_SCATTER = 0.80
MAX_SCATTER_POINTS = 300

def main():
    print("--- 1. Loading RQ1 Data (Distribution) ---")
    try:
        with open(RQ1_FILE, 'r') as f:
            rq1_scores = json.load(f)
        print(f"Loaded {len(rq1_scores)} sampled scores for Histogram.")
    except FileNotFoundError:
        print("Error: rq1_distribution.json not found.")
        return

    print("\n--- 2. Loading Results & Gas Data ---")
    
    # 1. Load Gas Data (Regex for robustness)
    gas_data = {}
    try:
        with open(GAS_FILE, 'r') as f:
            content = f.read()
            # Regex for "Key": 123
            matches = re.findall(r'"([^"]+)":\s*(\d+)', content)
            for k, v in matches:
                gas_data[k] = int(v)
        print(f"Loaded {len(gas_data)} gas estimates.")
    except FileNotFoundError:
        print("Gas file not found.")
        return

    # 2. Load Results (Stream Regex)
    # Pattern: [ "Sig", "Path", Score ]
    pattern = re.compile(r'"(contracts_organized_disl/[^"]+)":\s*\[|\[\s*"([^"]+)",\s*"([^"]+)",\s*([\d\.]+)\s*\]')
    
    max_savings_map = {}
    total_pairs_checked = 0
    optimization_candidates = 0
    significant_optimizations = 0
    sum_significant_savings = 0.0
    significant_savings = []
    positive_savings_count = 0
    sum_positive_savings = 0.0
    positive_savings = []
    
    current_query_uid = None
    current_query_gas = 0
    
    try:
        with open(RESULTS_FILE, 'r') as f:
            content = f.read()
        
        print("Processing results for Scatter Plot & Stats...")
        for m in pattern.finditer(content):
            if m.group(1): # New Key
                raw_key = m.group(1)
                uid = raw_key.replace(PREFIX_TO_REMOVE, "")
                if uid in gas_data:
                    current_query_uid = uid
                    current_query_gas = gas_data[uid]
                else:
                    current_query_uid = None
            
            elif m.group(2): # Match
                if current_query_uid and current_query_gas > 0:
                    sig, path, score = m.group(2), m.group(3), float(m.group(4))
                    
                    if score < MIN_SIMILARITY_SCATTER: continue
                    
                    match_filename = path.replace(PREFIX_TO_REMOVE, "")
                    match_uid = f"{match_filename}.{sig}"
                    
                    if match_uid in gas_data:
                        g2 = gas_data[match_uid]
                        if g2 > 0 and match_uid != current_query_uid:
                            
                            total_pairs_checked += 1
                            
                            diff = abs(current_query_gas - g2)
                            max_g = max(current_query_gas, g2)
                            savings_pct = (diff / max_g) * 100
                            
                            # Stats
                            if score >= 0.9:
                                optimization_candidates += 1
                                if savings_pct > 0 and savings_pct<90:
                                    significant_optimizations += 1
                                    sum_significant_savings += savings_pct
                                    significant_savings.append(savings_pct)

                                if savings_pct > 0 and savings_pct<90:
                                    positive_savings_count += 1
                                    sum_positive_savings += savings_pct
                                    positive_savings.append(savings_pct)
                            
                            # Frontier Map
                            if score not in max_savings_map and savings_pct < 90:
                                max_savings_map[score] = savings_pct
                            elif score in max_savings_map and savings_pct < 90:
                                if savings_pct > max_savings_map[score]:
                                    max_savings_map[score] = savings_pct
                                    
    except FileNotFoundError:
        print("Results file not found.")
        return

    # --- 3. PRINT FINAL STATS ---
    avg_savings_sig = 0
    std_savings_sig = 0
    if positive_savings_count > 0:
        avg_savings_sig = sum_positive_savings / positive_savings_count
        if len(positive_savings) > 1:
            std_savings_sig = np.std(positive_savings, ddof=1)

    print("\n" + "="*40)
    print("   QUANTITATIVE RESULTS (Copy to Paper)")
    print("="*40)
    print(f"Total RQ1 Samples (Histogram):      {len(rq1_scores)}")
    print(f"Pairs Analyzed for Optimization:    {total_pairs_checked}")
    print(f"Optimization Candidates (>0.9):    {optimization_candidates}")
    print(f"Significant Optimizations (>10%):   {significant_optimizations}")
    print(f"Average Savings (>0%):              {avg_savings_sig:.2f}%")
    print(f"Std Dev Savings (>0%):              {std_savings_sig:.2f}%")
    print("="*40 + "\n")

    # --- 4. PLOT 1: FULL HISTOGRAM (RQ1) ---
    plt.figure(figsize=(7, 4))
    
    # Plot the full distribution from 0.0 to 1.0
    plt.hist(rq1_scores, bins=50, range=(0.0, 1.0), color='#4C72B0', edgecolor='black', alpha=0.7, log=True)
    
    plt.xlabel("Cosine Similarity (WL Kernel)")
    plt.ylabel("Frequency (Log Scale)")
    
    # Add Threshold Line
    plt.axvline(x=0.9, color='red', linestyle='--', linewidth=1.5, label="Similarity Threshold (0.9)")
    
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig("fig_similarity_distribution.pdf")
    print("Saved 'fig_similarity_distribution.pdf'")

    # --- 5. PLOT 2: FRONTIER SCATTER (RQ2) ---
    sorted_scores = sorted(max_savings_map.keys())
    max_savings = [max_savings_map[s] for s in sorted_scores]

    plt.figure(figsize=(7, 4))
    x_full = np.array(sorted_scores)
    y_full = np.array(max_savings)

    # Subsample points for readability if needed.
    x_plot = x_full
    y_plot = y_full
    if len(x_full) > MAX_SCATTER_POINTS:
        step = int(np.ceil(len(x_full) / MAX_SCATTER_POINTS))
        x_plot = x_full[::step]
        y_plot = y_full[::step]

    # LOESS regression to show trend between similarity and gas savings.
    if len(x_full) >= 2:
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess
            loess_curve = lowess(y_full, x_full, frac=0.35, it=0, return_sorted=True)
            plt.plot(loess_curve[:, 0], loess_curve[:, 1], color='#C44E52', linewidth=2.0, label='LOESS Fit')
        except ImportError:
            coef = np.polyfit(x_full, y_full, 1)
            x_fit = np.linspace(x_full.min(), x_full.max(), 200)
            y_fit = np.polyval(coef, x_fit)
            plt.plot(x_fit, y_fit, color='#C44E52', linewidth=2.0, label='Linear Fit (statsmodels missing)')

    # Light points for context (optional but useful for density).
    plt.scatter(x_plot, y_plot, alpha=0.25, s=16, c='#4C72B0', rasterized=True)

    plt.xlabel("Semantic Similarity (WL Kernel)")
    plt.ylabel("Max Gas Cost Reduction (%)")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("fig_gas_max_savings.pdf", dpi=300)
    print("Saved 'fig_gas_max_savings.pdf'")

if __name__ == "__main__":
    main()
