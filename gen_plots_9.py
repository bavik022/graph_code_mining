import json
import matplotlib.pyplot as plt
import numpy as np

# --- CONFIGURATION ---
INPUT_FILE = "paper_results_disl.json"
OUTPUT_IMAGE = "similarity_histogram_gt_0.9.pdf"
FILTER_THRESHOLD = 0.90  # Only plot scores strictly greater than this
BINS = 50                # Number of bars in the histogram

def main():
    print(f"--- Loading data from {INPUT_FILE} ---")
    
    try:
        with open(INPUT_FILE, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {INPUT_FILE} not found. Please make sure it's in the same folder.")
        return

    # 1. Extract Scores
    high_scores = []
    total_matches = 0
    
    for query, matches in data.items():
        for m in matches:
            # Match structure: [Signature, FilePath, Score]
            score = float(m[2])
            total_matches += 1
            
            if score > FILTER_THRESHOLD:
                high_scores.append(score)

    if not high_scores:
        print(f"No scores found above {FILTER_THRESHOLD}. Check your data or threshold.")
        return

    print(f"Total Matches Checked: {total_matches}")
    print(f"Matches > {FILTER_THRESHOLD}: {len(high_scores)} ({len(high_scores)/total_matches:.1%})")

    # 2. Plotting
    plt.figure(figsize=(8, 5))
    
    # Create Histogram
    # We set the range from Threshold to 1.0 to focus on the high-similarity region
    plt.hist(high_scores, bins=BINS, range=(FILTER_THRESHOLD, 1.0), 
             color='#4C72B0', edgecolor='black', alpha=0.75, log=True)

    # 3. Styling
    plt.xlabel("Cosine Similarity Score", fontsize=12)
    plt.ylabel("Frequency (Count of Pairs)", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Add a vertical line for the mean of this high-score group
    mean_val = np.mean(high_scores)
    plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=1.5, label=f'Mean ({mean_val:.4f})')
    plt.legend()

    # 4. Save
    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE)
    print(f"Histogram saved to: {OUTPUT_IMAGE}")
    plt.show()

if __name__ == "__main__":
    main()