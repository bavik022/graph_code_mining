import re
import json

# --- CONFIGURATION ---
RESULTS_FILE = "paper_results_disl_test.json"
GAS_FILE = "gas_estimates_test.json"
PREFIX_TO_REMOVE = "test_contracts/"
MIN_SIMILARITY = 0.90  # High similarity to ensure they look like clones
MIN_SAVINGS = 1.0    # Significant gas difference to make the example obvious

def main():
    print("--- 1. Loading Gas Data ---")
    gas_data = {}
    try:
        with open(GAS_FILE, 'r') as f:
            content = f.read()
            # Robust Regex for "key": value
            matches = re.findall(r'"([^"]+)":\s*(\d+)', content)
            for k, v in matches:
                gas_data[k] = int(v)
        print(f"Loaded {len(gas_data)} gas estimates.")
    except FileNotFoundError:
        print(f"Error: {GAS_FILE} not found.")
        return

    print("--- 2. Scanning for Golden Examples ---")
    
    # Regex to stream the results file
    pattern = re.compile(r'"(contracts_organized_disl/[^"]+)":\s*\[|\[\s*"([^"]+)",\s*"([^"]+)",\s*([\d\.]+)\s*\]')
    
    candidates = []
    current_query_uid = None
    current_query_gas = 0
    
    try:
        with open(RESULTS_FILE, 'r') as f:
            content = f.read()
            
        for m in pattern.finditer(content):
            if m.group(1): # New Query Key
                raw_key = m.group(1)
                uid = raw_key.replace(PREFIX_TO_REMOVE, "")
                if uid in gas_data:
                    current_query_uid = uid
                    current_query_gas = gas_data[uid]
                else:
                    current_query_uid = None
            
            elif m.group(2): # Match Found
                if current_query_uid and current_query_gas > 0:
                    sig = m.group(2)
                    path = m.group(3)
                    score = float(m.group(4))
                    
                    if score < MIN_SIMILARITY: continue
                    
                    match_filename = path.replace(PREFIX_TO_REMOVE, "")
                    match_uid = f"{match_filename}.{sig}"
                    
                    if match_uid in gas_data:
                        g2 = gas_data[match_uid]
                        if g2 > 0 and match_uid != current_query_uid:
                            
                            diff = abs(current_query_gas - g2)
                            max_g = max(current_query_gas, g2)
                            savings_pct = (diff / max_g) * 100
                            
                            if savings_pct > MIN_SAVINGS:
                                # Determine which is Expensive vs Cheap
                                if current_query_gas > g2:
                                    expensive_uid = current_query_uid
                                    cheap_uid = match_uid
                                    expensive_gas = current_query_gas
                                    cheap_gas = g2
                                    # File paths for user
                                    expensive_file = raw_key.split('.sol')[0] + ".sol"
                                    cheap_file = path
                                else:
                                    expensive_uid = match_uid
                                    cheap_uid = current_query_uid
                                    expensive_gas = g2
                                    cheap_gas = current_query_gas
                                    expensive_file = path
                                    cheap_file = raw_key.split('.sol')[0] + ".sol"

                                candidates.append({
                                    'similarity': score,
                                    'savings': savings_pct,
                                    'expensive_gas': expensive_gas,
                                    'cheap_gas': cheap_gas,
                                    'expensive_func': expensive_uid.split('.')[-1],
                                    'cheap_func': cheap_uid.split('.')[-1],
                                    'file_expensive': expensive_file,
                                    'file_cheap': cheap_file
                                })
    except FileNotFoundError:
        print(f"Error: {RESULTS_FILE} not found.")
        return

    # Sort by Similarity (descending) to find the "Cleanest" clones
    candidates.sort(key=lambda x: x['similarity'], reverse=True)
    
    print(f"\nFound {len(candidates)} candidates. Here are the Top 10 to inspect:\n")
    
    for i, c in enumerate(candidates[:10]):
        print(f"=== CANDIDATE PAIR #{i+1} ===")
        print(f"Similarity: {c['similarity']:.4f} | Gas Savings: {c['savings']:.1f}%")
        print(f"\n[1] EXPENSIVE VERSION (Gas: {c['expensive_gas']})")
        print(f"    File: {c['file_expensive']}")
        print(f"    Func: {c['expensive_func']}")
        print(f"\n[2] OPTIMIZED VERSION (Gas: {c['cheap_gas']})")
        print(f"    File: {c['file_cheap']}")
        print(f"    Func: {c['cheap_func']}")
        print("-" * 60)

if __name__ == "__main__":
    main()