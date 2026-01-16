import pickle
import json
import hashlib
import networkx as nx
import re
import os
from pathlib import Path
from tqdm import tqdm
from collections import Counter

# --- CONFIGURATION ---
CPG_FILE = "function_cpgs.pkl"
GAS_FILE = "gas_estimates.json"
OUTPUT_TEXT = "final_paper_snippets.txt"

SIMILARITY_THRESHOLD = 0.85
GAS_IMPROVEMENT_FACTOR = 2.0 
FILE_LOCATION_CACHE = {}

# --- 1. ROBUST FILE FINDER ---
def find_file(stored_path):
    if os.path.exists(stored_path): return stored_path
    filename = Path(stored_path).name
    if filename in FILE_LOCATION_CACHE: return FILE_LOCATION_CACHE[filename]
    # Search recursively
    for p in Path(".").rglob(filename):
        if p.is_file():
            FILE_LOCATION_CACHE[filename] = str(p)
            return str(p)
    return None

# --- 2. INTELLIGENT SOURCE EXTRACTION ---
def extract_function_body(content, func_name):
    """
    Finds 'function func_name(...) ... {' and extracts the body.
    Handles standard functions and constructors.
    """
    # Pattern 1: Standard function
    # matches: function myName ( or function myName(
    pattern_std = re.compile(rf'function\s+{re.escape(func_name)}\s*\(', re.MULTILINE)
    
    # Pattern 2: Constructor (no name)
    # matches: constructor ( or constructor(
    pattern_constr = re.compile(r'constructor\s*\(', re.MULTILINE)

    if func_name == 'constructor':
        match = pattern_constr.search(content)
    else:
        match = pattern_std.search(content)

    if not match:
        return None

    # Find the opening brace '{'
    start_index = match.start()
    open_brace_index = content.find('{', start_index)
    
    if open_brace_index == -1: return None

    # Brace Counting to extract full body
    balance = 1
    current_index = open_brace_index + 1
    
    while balance > 0 and current_index < len(content):
        char = content[current_index]
        if char == '{':
            balance += 1
        elif char == '}':
            balance -= 1
        current_index += 1
        
    return content[start_index:current_index]

def recover_source_code(file_path, full_name, simple_name):
    """
    Parses the function name robustly and extracts code.
    """
    real_path = find_file(file_path)
    if not real_path: return f"Error: File not found {file_path}"
    
    try:
        with open(real_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
        # --- NAME CLEANING STRATEGY ---
        target_name = None
        
        # Case A: Constructor
        if 'constructor' in full_name.lower() or 'constructor' in simple_name.lower():
            target_name = 'constructor'
            
        # Case B: Trust Simple Name (if it looks valid)
        elif simple_name and '(' not in simple_name and ',' not in simple_name and ')' not in simple_name:
            target_name = simple_name
            
        # Case C: Parse Full Name (Fallback)
        # Full name is usually: "ContractName.functionName(args)"
        elif full_name:
            # Remove arguments
            base = full_name.split('(')[0]
            # Get last part after dot
            if '.' in base:
                target_name = base.split('.')[-1]
            else:
                target_name = base
                
        if not target_name:
            return f"Error: Could not determine function name from '{full_name}'"

        # Attempt Extraction
        code = extract_function_body(content, target_name)
        
        if code:
            return code
        else:
            return f"Error: Regex could not find function '{target_name}' in text."

    except Exception as e:
        return f"Error extracting code: {e}"

# --- 3. STANDARD PIPELINE (No Changes Here) ---
def cpg_to_networkx(cpg_dict):
    G = nx.Graph()
    for nid, attrs in cpg_dict['graph']['nodes'].items():
        g_type = str(attrs.get('graph', 'unk'))
        if g_type == 'cfg':
            sig = f"cfg_{attrs.get('kind', 'unk')}_{attrs.get('irs_count', 0)}"
        else:
            t_type = str(attrs.get('type', ''))
            k_type = str(attrs.get('kind', ''))
            sig = f"{g_type}_{t_type}_{k_type}"
        G.add_node(nid, label=sig)
    for edge in cpg_dict['graph']['edges']:
        G.add_edge(edge['src'], edge['dst'])
    return G

def wl_hash(G, iterations=2):
    current_labels = {n: G.nodes[n]['label'] for n in G.nodes()}
    all_patterns = []
    all_patterns.extend(current_labels.values())
    for _ in range(iterations):
        new_labels = {}
        for node in G.nodes():
            neighbors = sorted([current_labels[n] for n in G.neighbors(node)])
            sig = current_labels[node] + "(" + ",".join(neighbors) + ")"
            new_labels[node] = hashlib.md5(sig.encode()).hexdigest()
        current_labels = new_labels
        all_patterns.extend(current_labels.values())
    return Counter(all_patterns)

def jaccard(c1, c2):
    intersection = sum((c1 & c2).values())
    union = sum((c1 | c2).values())
    return intersection / union if union else 0

def build_gas_lookup(gas_data):
    lookup = {}
    iterable = gas_data if isinstance(gas_data, list) else gas_data.items()
    for item in iterable:
        if isinstance(gas_data, list):
            raw_key = item['id'] if 'id' in item else item.get('function_full_name')
            score = item['estimated_gas_score'] if 'estimated_gas_score' in item else item.get('gas_score')
        else:
            raw_key, score = item
        try:
            if ".sol" in raw_key:
                filename_part = raw_key.split(".sol")[0] + ".sol"
            else:
                filename_part = raw_key
            func_part = raw_key.split(".")[-1]
            lookup[(Path(filename_part).name, func_part)] = score
        except: continue
    return lookup

def main():
    print("1. Loading Data...")
    with open(CPG_FILE, "rb") as f:
        cpg_data = pickle.load(f)['functions']
    with open(GAS_FILE, "r") as f:
        gas_data = json.load(f)
    gas_lookup = build_gas_lookup(gas_data)
    
    print("2. Indexing...")
    database = []
    for func in tqdm(cpg_data):
        fname = Path(func['contract']).name
        full_name = func.get('function_full_name', '')
        simple_name = func.get('function_simple_name', '')
        
        score = gas_lookup.get((fname, full_name))
        if not score: score = gas_lookup.get((fname, full_name + "()"))
        if not score: score = gas_lookup.get((fname, simple_name))
            
        if score is None or func['num_nodes'] < 10: continue

        G = cpg_to_networkx(func)
        vec = wl_hash(G)
        
        database.append({
            "id": f"{fname}::{full_name}",
            "vec": vec,
            "gas": score,
            "contract": func['contract'],
            "full_name": full_name,
            "simple_name": simple_name
        })

    print("3. Matching...")
    database.sort(key=lambda x: x['gas'], reverse=True)
    results = []
    
    # Analyze Top 500 Expensive functions
    for bad in tqdm(database[:500]):
        best_match = None
        best_sim = 0.0
        for candidate in database:
            if bad['contract'] == candidate['contract']: continue
            if bad['gas'] < candidate['gas'] * GAS_IMPROVEMENT_FACTOR: continue
            sim = jaccard(bad['vec'], candidate['vec'])
            if sim > SIMILARITY_THRESHOLD and sim > best_sim:
                best_sim = sim
                best_match = candidate
        
        if best_match:
            results.append({"similarity": best_sim, "inefficient": bad, "optimized": best_match})
            
    results.sort(key=lambda x: x['similarity'], reverse=True)
    
    print("\n=== TOP 5 MATCHES ===")
    output_text = []
    
    for i, res in enumerate(results[:5]):
        bad = res['inefficient']
        good = res['optimized']
        
        bad_code = recover_source_code(bad['contract'], bad['full_name'], bad['simple_name'])
        good_code = recover_source_code(good['contract'], good['full_name'], good['simple_name'])

        entry = f"""
#######################################################
MATCH #{i+1} [Similarity: {res['similarity']:.4f}]
Gas: {bad['gas']} -> {good['gas']}
Inefficient: {bad['full_name']} ({bad['contract']})
Optimized: {good['full_name']} ({good['contract']})

[INEFFICIENT CODE]
{bad_code}

[OPTIMIZED CODE]
{good_code}
#######################################################
"""
        print(entry)
        output_text.append(entry)

    with open(OUTPUT_TEXT, "w") as f:
        f.write("\n".join(output_text))

if __name__ == "__main__":
    main()