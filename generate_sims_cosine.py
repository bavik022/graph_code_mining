import pickle
import json
import hashlib
import networkx as nx
import random
import os
from collections import Counter, defaultdict
from pathlib import Path
from tqdm import tqdm
import gen_function_ops
import math
from typing import Any, Callable, Hashable, Optional
import json
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import normalize
from collections import defaultdict


# --- IMPORTS FOR LABELING ---
try:
    from slither.slither import Slither
    from slither.detectors import all_detectors
except ImportError:
    print("CRITICAL ERROR: Slither is not installed.")
    print("Run: pip install slither-analyzer")
    exit(1)

# --- CONFIGURATION ---
CPG_DATASET_PATH = "function_cpgs_test.pkl"       # Your existing CPG file
RESULTS_FILE = "paper_results_disl_test.json"     # Where to save findings
LABEL_CACHE_FILE = "slither_label_cache.json" # To save time on restarts

# WL Kernel Settings
WL_ITERATIONS = 2
SIMILARITY_THRESHOLD = 0.9

# Gas Detectors to Flag "Bad" Functions
# If a function triggers ANY of these, it is labeled "Inefficient" (0)
GAS_PATTERNS = [
    'external-function',        # Should be external
    'const-state-vars',         # Should be constant
    'immutable-states',         # Should be immutable
    'dead-code',                # Unused code
    'solc-version',             # Old compiler version (usually gas heavy)
    'unchecked-lowlevel',       # Low level calls
    'shadowing-state',          # Shadowing state variables
    'calls-loop',               # External calls inside loops
    'costly-loop'               # Expensive operations in loops
]

# ==========================================
# PART 1: AUTO-LABELING (Slither Integration)
# ==========================================

def analyze_file_gas_costs(file_path):
    """
    Runs Slither on a single Solidity file and returns a dict:
    { "function_name": is_optimized (1 or 0) }
    """
    results = {}
    
    if not os.path.exists(file_path):
        return {}

    try:
        opt_version = gen_function_ops.get_optimal_compiler_version(file_path)
        gen_function_ops.set_solc_version(opt_version)
        # Initialize Slither on the file
        slither = Slither(str(file_path))
        #slither.register_detector_classes(all_detectors)
        
        # Run Detectors
        issues = slither.run_detectors()
        
        # 1. Default all functions to Optimized (1)
        for contract in slither.contracts:
            for function in contract.functions:
                # Store by name. Note: overloading might cause collisions, 
                # but for this scale, name matching is acceptable.
                results[function.name] = 1 
                results[function.full_name] = 1

        # 2. Mark inefficient functions as (0)
        for issue in issues:
            check_name = issue['check']
            if check_name in GAS_PATTERNS:
                for element in issue['elements']:
                    if element['type'] == 'function':
                        # Mark this function as inefficient
                        f_name = element['name']
                        print("f_name", f_name)
                        results[f_name] = 0
                        # Try to capture full name if available in source mapping
                        # (Slither elements are sometimes just strings)
                        
    except Exception as e:
        print("Error:", e)
        # Compilation errors are common in large datasets
        # We silently skip files that don't compile
        pass

    return results

def get_labels_for_dataset(cpg_functions):
    """
    Iterates through all CPGs, finds unique source files, and runs Slither.
    Uses caching to avoid re-running Slither on 47k files if script restarts.
    """
    print("\n--- PHASE 1: GENERATING GAS LABELS ---")
    
    # 1. Identify Unique Files
    unique_files = set()
    for entry in cpg_functions:
        unique_files.add(entry['contract'])
    
    print(f"Found {len(unique_files)} unique source files in CPG dataset.")
    
    # 2. Load Cache if exists
    file_labels = {}
    if os.path.exists(LABEL_CACHE_FILE):
        print("Loading cached labels...")
        with open(LABEL_CACHE_FILE, "r") as f:
            file_labels = json.load(f)
            
    # 3. Process missing files
    files_to_process = [f for f in unique_files if f not in file_labels]
    
    if files_to_process:
        print(f"Running Slither on {len(files_to_process)} new files...")
        for file_path in tqdm(files_to_process):
            # Run Slither
            labels = analyze_file_gas_costs(file_path)
            file_labels[file_path] = labels
            
        # Save cache
        with open(LABEL_CACHE_FILE, "w") as f:
            json.dump(file_labels, f)
            
    return file_labels

# ==========================================
# PART 2: STRUCTURAL ANALYSIS (WL Kernel)
# ==========================================

def cpg_to_networkx(cpg_dict):
    """Converts CPG dict to NetworkX, using structural labels only."""
    G = nx.Graph()
    
    # Add Nodes
    for nid, attrs in cpg_dict['graph']['nodes'].items():
        g_type = str(attrs.get('graph', 'unk'))
        
        # Custom Labeling for CFG Nodes to capture block size (Gas proxy)
        if g_type == 'cfg':
            k_type = str(attrs.get('kind', 'unk'))
            # "irs_count" tells us how many ops are in this block (e.g. empty vs heavy)
            cnt = str(attrs.get('irs_count', '0'))
            sig = f"cfg_{k_type}_{cnt}"
            
        else:
            # Standard labeling for AST and DFG
            t_type = str(attrs.get('type', ''))
            k_type = str(attrs.get('kind', ''))
            sig = f"{g_type}_{t_type}_{k_type}"

        G.add_node(nid, label=sig)
        
    # Add Edges
    for edge in cpg_dict['graph']['edges']:
        G.add_edge(edge['src'], edge['dst'])
        
    return G

def wl_hash_graph(
    G,
    iterations: int = 2,
    node_label_attr: str = "label",
    default_node_label: str = "0",
    directed_neighbors: str = "auto",   # "auto" | "in" | "out" | "both"
    edge_label_attr: Optional[str] = None,
    include_iteration_in_features: bool = True,
    digest_size: int = 16,              # bytes for blake2b digest (small & fast)
) -> Counter:

    # ---- Helpers ----
    def stable_digest(s: str) -> str:
        # blake2b is fast and stable; digest_size controls length.
        return hashlib.blake2b(s.encode("utf-8"), digest_size=digest_size).hexdigest()

    def get_node_label(n) -> str:
        v = G.nodes[n].get(node_label_attr, default_node_label)
        return str(v)

    def iter_neighbors(n):
        # Directed handling
        if directed_neighbors == "auto":
            is_directed = getattr(G, "is_directed", lambda: False)()
            mode = "both" if is_directed else "out"
        else:
            mode = directed_neighbors

        if mode == "out":
            return G.successors(n) if hasattr(G, "successors") else G.neighbors(n)
        if mode == "in":
            return G.predecessors(n) if hasattr(G, "predecessors") else G.neighbors(n)
        if mode == "both":
            if hasattr(G, "successors") and hasattr(G, "predecessors"):
                # union, but deterministic via sorted later
                return list(G.successors(n)) + list(G.predecessors(n))
            return G.neighbors(n)
        raise ValueError("directed_neighbors must be 'auto', 'in', 'out', or 'both'")

    def edge_label(u, v) -> str:
        if edge_label_attr is None:
            return ""
        data = G.get_edge_data(u, v, default={})
        # MultiDiGraph/MultiGraph: data can be dict-of-dicts
        if isinstance(data, dict) and any(isinstance(val, dict) for val in data.values()):
            # take all parallel edges, collect labels
            labels = []
            for _k, d in data.items():
                labels.append(str(d.get(edge_label_attr, "")))
            labels.sort()
            return "|".join(labels)
        return str(data.get(edge_label_attr, ""))

    # Deterministic node order (stringified to be stable across mixed node types)
    nodes = sorted(G.nodes(), key=lambda x: str(x))

    # ---- Iteration 0 labels ----
    current = {n: get_node_label(n) for n in nodes}

    # Standard WL feature map counts labels per iteration
    feats = Counter()
    for n in nodes:
        key = (0, current[n]) if include_iteration_in_features else current[n]
        feats[key] += 1

    # ---- WL refinement iterations ----
    for it in range(1, iterations + 1):
        # Build signatures
        signatures = {}
        for n in nodes:
            neigh = list(iter_neighbors(n))
            # determinism: sort neighbors by string, then build multiset of (edge_label, neighbor_label)
            parts = []
            for m in sorted(neigh, key=lambda x: str(x)):
                lbl = current.get(m, default_node_label)
                if edge_label_attr is None:
                    parts.append(lbl)
                else:
                    parts.append(f"{edge_label(n, m)}:{lbl}")

            # WL signature: own label + sorted multiset of neighbor contexts
            sig = f"{current[n]}|{','.join(sorted(parts))}"
            signatures[n] = sig

        # Compress signatures to new labels (hash)
        new = {n: stable_digest(signatures[n]) for n in nodes}
        current = new

        # Update feature counts
        for n in nodes:
            key = (it, current[n]) if include_iteration_in_features else current[n]
            feats[key] += 1

    return feats

def jaccard_similarity(vec1, vec2):
    keys1 = set(vec1.keys())
    keys2 = set(vec2.keys())
    intersection = keys1.intersection(keys2)
    union = keys1.union(keys2)
    if not union: return 0.0
    return len(intersection) / len(union)

def wl_cosine(phi1: Counter, phi2: Counter) -> float:
    # dot product
    dot = sum(v * phi2.get(k, 0) for k, v in phi1.items())
    n1 = math.sqrt(sum(v * v for v in phi1.values()))
    n2 = math.sqrt(sum(v * v for v in phi2.values()))
    return dot / (n1 * n2 + 1e-12)

if __name__ == "__main__":
    # 1. Load CPGs
    print(f"Loading CPG dataset: {CPG_DATASET_PATH}...")
    try:
        with open(CPG_DATASET_PATH, "rb") as f:
            data = pickle.load(f)
            functions = data['functions']
    except FileNotFoundError:
        print("Error: function_cpgs.pkl not found.")
        functions = []

    # 2. Get Labels (Integrated Step)
    # This runs Slither on the source files referenced in the pickle
    #file_labels_map = get_labels_for_dataset(functions)

    # 3. Build Embeddings
    print("\n--- PHASE 2: COMPUTING EMBEDDINGS ---")
    database = []

    for entry in tqdm(functions):
        # Look up Label
        f_full_name = entry['function_full_name']
        f_path = entry['contract']
        f_name = entry['function_full_name'] # e.g. "transfer"
        f_simple_name = entry['function_simple_name']          # e.g. "transfer(address,uint256)"
            
        # Filter: Skip tiny graphs (likely getters/setters)
        if entry['num_nodes'] < 5:
            continue

        # Convert and Hash
        G = cpg_to_networkx(entry)
        vec = wl_hash_graph(G, iterations=WL_ITERATIONS)
    
        database.append({
            "full_name": f_full_name,
            "contract": entry['contract'],
            "vector": vec,
            "path": f_path,
            "nodes": entry['num_nodes']
        })
        
    print(f"Successfully indexed {len(database)} labeled functions.")

    """sims = defaultdict(list)
    for i in tqdm(range(len(database))):
        func = database[i]['vector']
        func_name = database[i]['full_name']
        func_contract = database[i]['contract']
        for other in database:
            if wl_cosine(func, other['vector']) > 0.9:
                sims[f"{func_contract}.{func_name}"].append((other['full_name'], other['contract'], wl_cosine(func, other['vector'])))
    print("Saving results...")
    with open(RESULTS_FILE, "w") as f:
        json.dump(sims, f, indent=2)"""

    print(f"Processing {len(database)} items...")

    # DictVectorizer converts list of Counters -> Scipy Sparse Matrix (CSR)
    # It automatically handles the string keys (hashes) and maps them to integer indices.
    vectorizer = DictVectorizer(sparse=True, dtype=np.float32)
    feature_dicts = [d['vector'] for d in database[:10000]]
    X = vectorizer.fit_transform(feature_dicts)

    print(f"Vectorized shape: {X.shape} (Rows: Functions, Cols: Unique Subgraphs)")

    # Normalize
    # Cosine Similarity is Dot Product of L2-normalized vectors
    # This is much faster than computing cosine manually for every pair
    X_normalized = normalize(X, norm='l2', axis=1)

    # Compute Similarity Matrix
    # This performs N^2 comparisons using highly optimized C routines (BLAS)
    # Result is a sparse matrix if the overlap is sparse, or dense if many graphs share common subgraphs.
    similarity_matrix = X_normalized @ X_normalized.T

    # Extract Results > 0.9
    sims = defaultdict(list)

    # Get the indices of the upper triangle (to avoid duplicates and self-loops)
    # We convert to COO format to iterate efficiently over non-zero elements
    # Note: If your matrix is extremely dense (common '0' labels), this might still be large.
    sim_coo = similarity_matrix.tocoo()

    # Thresholding before iteration saves massive time
    mask = sim_coo.data > 0.9
    rows = sim_coo.row[mask]
    cols = sim_coo.col[mask]
    scores = sim_coo.data[mask]

    # Metadata lookup list for fast access
    meta = [(d['full_name'], d['contract']) for d in database]

    for src_idx, tgt_idx, score in tqdm(zip(rows, cols, scores), total=len(rows)):
        if src_idx == tgt_idx:
            continue # Skip self-similarity

        src_name, src_contract = meta[src_idx]
        tgt_name, tgt_contract = meta[tgt_idx]

        key = f"{src_contract}.{src_name}"
        # Because dot product is symmetric, we might want to store both directions
        # or just one depending on your use case.
        sims[key].append((tgt_name, tgt_contract, float(score)))

    print("Saving results...")
    with open(RESULTS_FILE, "w") as f:
        json.dump(sims, f, indent=2)
