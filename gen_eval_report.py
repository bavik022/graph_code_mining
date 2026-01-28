import pickle
import json
import hashlib
import numpy as np
import networkx as nx
import os
import gc
import random
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize

# --- CONFIG ---
INPUT_CPG = "function_cpgs.pkl"
INPUT_GAS = "gas_estimates.json"
OUTPUT_REPORT = "evaluation_report_million.txt"
OUTPUT_JSON = "results_million.json"
RQ1_FILE = "rq1_distribution.json"
PARTIAL_SAVE_FILE = "candidates_partial.json"

WL_ITERATIONS = 2
SIMILARITY_THRESHOLD = 0.9
GAS_GAP_MIN = 0.10
GAS_GAP_MAX = 0.80

# Reduced Batch Size for Stability with 1M+ items
BATCH_SIZE = 200   
MAX_BATCHES = 20 # Run until you press Ctrl+C

# --- HELPERS ---
def wl_hash_graph(G, iterations=2):
    def stable_digest(s):
        return hashlib.blake2b(s.encode("utf-8"), digest_size=8).hexdigest()

    nodes = sorted(G.nodes(), key=lambda x: str(x))
    current = {n: str(G.nodes[n].get("kind", "OP")) for n in nodes}
    vocabulary = []
    
    for _ in range(iterations):
        signatures = {}
        for n in nodes:
            neighbors = sorted([str(current.get(nbr, "0")) for nbr in G.neighbors(n)])
            sig = f"{current[n]}|{','.join(neighbors)}"
            signatures[n] = sig
        
        new_labels = {n: stable_digest(signatures[n]) for n in nodes}
        current = new_labels
        vocabulary.extend(current.values())
        
    return " ".join(vocabulary)

def dict_to_nx(func_data):
    G = nx.DiGraph()
    if 'graph' not in func_data: return G
    for nid, attrs in func_data['graph']['nodes'].items():
        G.add_node(nid, kind=attrs.get('kind', 'UNK'))
    for edge in func_data['graph']['edges']:
        G.add_edge(edge['src'], edge['dst'])
    return G

# --- MAIN ---
def main():
    print("--- 1. Loading 1.15M Dataset ---")
    with open(INPUT_CPG, "rb") as f:
        dataset = pickle.load(f)
    functions = dataset.get("functions", [])
    
    with open(INPUT_GAS, "r") as f:
        gas_data = json.load(f)

    func_meta = []
    wl_corpus = []
    
    print("Generating WL Hashes...")
    for func in tqdm(functions):
        c_name = func['contract'].split('/')[-1]
        uid = f"{c_name}.{func['function_full_name']}"
        
        # Only process if we have gas info
        if uid not in gas_data: continue
        
        G = dict_to_nx(func)
        if len(G) < 2: continue
        
        wl_str = wl_hash_graph(G, iterations=WL_ITERATIONS)
        
        func_meta.append({
            'uid': uid,
            'gas': gas_data[uid],
            'code': func.get('source_code', '')
        })
        wl_corpus.append(wl_str)
        
    num_funcs = len(func_meta)
    print(f"Valid Functions with Gas Data: {num_funcs}")
    
    # Vectorize
    print("--- 2. Vectorizing ---")
    # min_df=2 is crucial here to keep features manageable
    vectorizer = CountVectorizer(token_pattern=r"\S+", min_df=2) 
    X = vectorizer.fit_transform(wl_corpus)
    X_norm = normalize(X)

    # --- MINING LOOP ---
    print(f"--- 3. Mining Best Matches ---")
    
    final_candidates = []
    rq1_sample = []
    processed_hashes = set() # For Deduplication
    
    # Determine Iterations
    iters = range(0, num_funcs, BATCH_SIZE)
    if MAX_BATCHES:
        iters = range(0, min(num_funcs, MAX_BATCHES * BATCH_SIZE), BATCH_SIZE)
    
    try:
        for i in tqdm(iters):
            # Check if entire batch is skippable (Optimization)
            # (Requires checking hashes ahead of time, but row-by-row is safer for now)
            
            end = min(i + BATCH_SIZE, num_funcs)
            batch = X_norm[i:end]
            
            # Compute Similarity (Batch vs 1.15M)
            # Result is (200 x 1,150,000) dense matrix
            sim_batch = (batch @ X_norm.T).toarray()
            
            # --- RQ1 SAMPLING (Random 0.1%) ---
            # We take fewer samples because the matrix is huge
            sample_points = np.random.choice(sim_batch.flatten(), size=100, replace=False)
            rq1_sample.extend([float(x) for x in sample_points])
            
            for local_row_idx in range(sim_batch.shape[0]):
                global_idx = i + local_row_idx
                
                # --- LAZY DEDUPLICATION (The Speed Saver) ---
                current_wl = wl_corpus[global_idx]
                if current_wl in processed_hashes:
                    continue 
                processed_hashes.add(current_wl)
                # --------------------------------------------

                scores = sim_batch[local_row_idx]
                
                # Filter > Threshold
                candidate_indices = np.where(scores > SIMILARITY_THRESHOLD)[0]
                if len(candidate_indices) == 0: continue

                best_cand = None
                best_metric = -1
                
                f1 = func_meta[global_idx]
                g1 = f1['gas']
                if g1 == 0: continue

                for match_idx in candidate_indices:
                    if match_idx == global_idx: continue
                    
                    f2 = func_meta[match_idx]
                    g2 = f2['gas']
                    if g2 == 0: continue
                    
                    diff = abs(g1 - g2)
                    max_g = max(g1, g2)
                    pct_diff = diff / max_g
                    
                    if not (GAS_GAP_MIN < pct_diff < GAS_GAP_MAX): continue
                        
                    sim_val = scores[match_idx]
                    metric = sim_val + (pct_diff * 0.01) 
                    
                    if metric > best_metric:
                        best_metric = metric
                        is_f1_expensive = g1 > g2
                        best_cand = {
                            'pair': (f1['uid'], f2['uid']),
                            'similarity': float(sim_val),
                            'savings_pct': float(pct_diff * 100),
                            'gas_expensive': int(max_g),
                            'gas_cheap': int(min(g1, g2)),
                            'code_expensive': f1['code'] if is_f1_expensive else f2['code'],
                            'code_cheap': f2['code'] if is_f1_expensive else f1['code']
                        }

                if best_cand:
                    final_candidates.append(best_cand)

            # CLEANUP
            del sim_batch
            gc.collect()
            
            # Incremental Save
            if len(final_candidates) > 0 and len(final_candidates) % 100 == 0:
                 with open(PARTIAL_SAVE_FILE, "w") as f:
                    json.dump(final_candidates, f)

    except KeyboardInterrupt:
        print("\nStopping early...")

    # Sort & Save
    final_candidates.sort(key=lambda x: x['savings_pct'], reverse=True)
    
    print(f"--- Saving Full Results ({len(final_candidates)} candidates) ---")
    with open(OUTPUT_JSON, "w") as f:
        json.dump(final_candidates, f, indent=2)

    print(f"--- Saving RQ1 Data ({len(rq1_sample)} points) ---")
    with open(RQ1_FILE, "w") as f:
        json.dump(rq1_sample, f)

    print(f"--- Writing Text Report ---")
    with open(OUTPUT_REPORT, "w") as f:
        f.write(f"EVALUATION REPORT (1M Scale)\n")
        f.write(f"Total Database Size: {num_funcs}\n")
        f.write(f"Unique Optimization Opportunities: {len(final_candidates)}\n\n")
        
        f.write("TOP 5 CASE STUDIES\n")
        for k, c in enumerate(final_candidates[:5]):
            f.write(f"\n[PAIR #{k+1}]\n")
            f.write(f"Expensive: {c['pair'][0]} (Gas: {c['gas_expensive']})\n")
            f.write(f"Cheap:     {c['pair'][1]} (Gas: {c['gas_cheap']})\n")
            f.write(f"Similarity: {c['similarity']:.4f} | Savings: {c['savings_pct']:.1f}%\n")
            f.write("\n--- CODE (Expensive) ---\n")
            f.write(c['code_expensive'].strip()[:500] + "...\n")
            f.write("-" * 50 + "\n")

    print("Done.")

if __name__ == "__main__":
    main()