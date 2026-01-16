import re
import os
import subprocess
import solcx 
from solc_select import solc_select
import json
import evm_cfg_builder
from evm_cfg_builder.cfg import CFG
#from slither.slither import Slither
from Crypto.Hash import keccak
import difflib
import numpy as np
from tqdm import tqdm
import sklearn.model_selection as sk
import sklearn 
import random
from gensim.models import Word2Vec
import pickle
from numpy import dot
from numpy.linalg import norm
from collections import defaultdict
import sys
import io
import pandas as pd
import copy
import pickle
import statistics

def extract_pragma_statements(file_content):
    """
    Extract all pragma statements from the file content.
    """
    pragma_statements = re.findall(r'pragma solidity ([^;]+);', file_content)
    if not pragma_statements:
        raise ValueError("No Solidity pragma statements found.")
    return pragma_statements

def get_supported_versions():
    try:
        versions = solc_select.get_installable_versions()
    except Exception as e:
        raise RuntimeError(f"Failed to get supported Solidity versions: {e}")
    return [str(version) for version in versions]

def get_installed_versions():
    try:
        versions = solc_select.installed_versions()
    except Exception as e:
        raise RuntimeError(f"Failed to get installed Solidity versions: {e}")
    return [str(version) for version in versions]
    

def version_in_range(version, min_version, max_version, ops):
    """
    Check if a version is within the given range.
    """
    if ops[0] == '>' and ops[1] == '<':
        if min_version and version <= min_version:
            return False
        if max_version and version >= max_version:
            return False
    elif ops[0] == '>=' and ops[1] == '<':
        if min_version and version < min_version:
            return False
        if max_version and version >= max_version:
            return False
    elif ops[0] == '>' and ops[1] == '<=':
        if min_version and version <= min_version:
            return False
        if max_version and version > max_version:
            return False
    elif ops[0] == '>=' and ops[1] == '<=':
        if min_version and version < min_version:
            return False
        if max_version and version > max_version:
            return False
    elif ops[0] == '!' and version != min_version:
        return False
    elif ops[0] == '^' and (version >= max_version or version < min_version):
        return False
    return True

def parse_version_range(version_range):
    """
    Parse a version range or fixed version.
    """
    fixed_version_match = re.match(r'^\^?([0-9]+\.[0-9]+\.[0-9]+)$', version_range)
    if fixed_version_match:
        if fixed_version_match.group()[0] == '^':
            vers = fixed_version_match.group(1).split('.')
            vers[1] = int(vers[1]) + 1
            vers[2] = 0
            vers = '.'.join([str(ver) for ver in vers])
            return fixed_version_match.group(1), vers, ['^']
        return fixed_version_match.group(1), fixed_version_match.group(1), ['!']
    
    range_match = re.match(r'^(>=?([0-9]+\.[0-9]+\.[0-9]+))?\s*(<([0-9]+\.[0-9]+\.[0-9]+))?$', version_range)
    if not range_match:
        raise ValueError(f"Invalid version range format: {version_range}")
    
    min_version = range_match.group(2)
    max_version = range_match.group(4)
    ops = range_match.group().replace(range_match.group(2), '').replace(range_match.group(4), '').split(' ')
    return min_version, max_version, ops

def determine_optimal_version(pragma_statements, supported_versions):
    """
    Determine the optimal Solidity compiler version that satisfies all pragma constraints.
    """
    min_versions = []
    max_versions = []
    operators = []

    for pragma in pragma_statements:
        min_version, max_version, ops = parse_version_range(pragma)
        if min_version:
            min_versions.append(min_version)
        if max_version:
            max_versions.append(max_version)
        if ops:
            operators.append(ops)

    valid_versions = []

    for version in supported_versions:
        flag = True
        for min_version, max_version, ops in zip(min_versions, max_versions, operators):
            if not version_in_range(version, min_version, max_version, ops):
                flag = False
        if flag:
            valid_versions.append(version)
    #print(min_versions)
    #print(max_versions)
    return max(valid_versions, key=lambda v: list(map(int, v.split('.'))))

def set_solc_version(version):
    """
    Set the global Solidity compiler version using solc-select.
    """
    try:
        solc_select.switch_global_version(version, True)
    except Exception as e:
        print("Error in setting Solidity compiler version: ", e)
        
def get_optimal_compiler_version(contract_file):
    try:
        with open(contract_file, 'r') as file:
            file_content = file.read()
        
        pragma_statements = extract_pragma_statements(file_content)
        print(f"Solidity pragma statements found: {pragma_statements}")
        
        supported_versions = get_supported_versions()
        installed_versions = get_installed_versions()

        supported_versions = supported_versions + installed_versions
        
        optimal_version = determine_optimal_version(pragma_statements, supported_versions)
        #print(f"Solidity compiler version: {optimal_version}")
        return optimal_version
    
    except Exception as e:
        print(f"An error occurred: {e}")

def read_contract_files(filename):
    f1 = open(filename, 'r')
    code = f1.read()
    #print("Deciding on the optimal compiler version...")
    compiler_version = get_optimal_compiler_version(filename)
    return compiler_version, code

def compile_contracts(code, compiler_version):
    #print(compiler_version, compiler_version_opt)
    set_solc_version(compiler_version)
    #print("Compiling first contract ...")
    if compiler_version >= '0.6.0':
        compl = solcx.compile_source(code, output_values=['abi', 'bin', 'bin-runtime', 'ast', 'srcmap-runtime', 'opcodes'], optimize=False, no_optimize_yul=True)
    else:
        compl = solcx.compile_source(code, output_values=['abi', 'bin', 'bin-runtime', 'ast', 'srcmap-runtime', 'opcodes'], optimize=False)
    return compl

def get_function_ops(function):
    ops = []
    for basic_block in sorted(function.basic_blocks, key=lambda x: x.start.pc):
        # Each basic block has a start and end instruction
        # instructions are pyevmasm.Instruction objects
        #print(f"\t- @{hex(basic_block.start.pc)}-{hex(basic_block.end.pc)}")
        for ins in basic_block.instructions:
            ops.append((ins.name, ins.fee))
    return ops

def keccak256(text):
        k = keccak.new(digest_bits=256)
        k.update(text.encode())
        return k.hexdigest()

def get_function_selector(function_signature):
    return keccak256(function_signature)[:8]  # First 4 bytes (8 hex characters)''

def find_function_selectors_in_bytecode(bytecode, abi_json):
    """
    Finds and returns the occurrences of function selectors in the given bytecode based on the ABI.
    Also returns the indices at which these selectors are found.

    :param bytecode: The compiled bytecode of the smart contract as a hex string.
    :param abi_json: A string containing the ABI in JSON format.
    :return: A dictionary where keys are function signatures and values are tuples of their selectors and indices found in the bytecode.
    """

    abi = json.loads(abi_json)
    selectors_found = {}

    for item in abi:
        if item['type'] == 'function':
            # Create function signature
            input_types = ','.join([input['type'] for input in item['inputs']])
            signature = f"{item['name']}({input_types})"
            
            # Calculate selector
            selector = get_function_selector(signature)

            # Find all occurrences of the selector in the bytecode
            indices = [i for i in range(len(bytecode)) if bytecode.startswith(selector, i)]

            if indices:
                selectors_found[signature] = (selector, indices)

    return selectors_found

def _get_function_evm(cfg, function_name, function_hash):
    for function_evm in cfg.functions:
        # Match function hash
        if function_evm.name[:2] == "0x" and function_evm.name == function_hash:
            return function_evm
        # Match function name
        if function_evm.name[:2] != "0x" and function_evm.name.split("(")[0] == function_name:
            return function_evm
    return None

def get_function_ops_gas(cfg, ls_selectors):
    fops = {}
    for sel in ls_selectors:
        #print("0x" + sel[1].lstrip('0'))
        fevm = _get_function_evm(cfg, sel[0], "0x" + sel[1].lstrip('0'))
        if fevm:
            ops = get_function_ops(fevm)
            fops["0x" + sel[1].lstrip('0')] = {
                "name": sel[0], 
                "ops": ops
            }
    
    return fops

'''def get_function_source_maps(contr, file, fops):
    sl = Slither(file)
    contracts = sl.contracts
    for contract in contracts:
        if contract.name == contr:
            fns = contract.functions
    for fn in fns:
        selector = get_function_selector(fn.full_name)
        key = "0x" + selector.lstrip('0')
        if key in fops:
            fops[key]['source'] = str(fn.source_mapping.content)
            fops[key]['source_map_lines'] = str(fn.source_mapping.lines)
    return fops'''

def get_function_insights(f1, contract):
    print("Reading contract files...")
    compiler_version, code = read_contract_files(f1)
    print("Compiling contract files...")
    compl = compile_contracts(code, compiler_version)
    print("Analyzing compiled contracts...")
    contr_key = "<stdin>:"+contract[0].upper() + contract[1:]
    if compl[contr_key]:
        bin = compl[contr_key]['bin']
        bin_runtime = compl[contr_key]['bin-runtime']
        abi =json.dumps(compl[contr_key]['abi'])
        selectors_in_bytecode = find_function_selectors_in_bytecode(bin, abi)
        ls_selectors = []
        for k in selectors_in_bytecode:
            ls_selectors.append((k, selectors_in_bytecode[k][0], selectors_in_bytecode[k][1]))
        cfg = CFG(bin_runtime)
        fops = get_function_ops_gas(cfg, ls_selectors)
        #fops = get_function_source_maps(contract, f1, fops)
        return fops
    raise Exception("Invalid contract name.")
    exit()

def generate_function_comparator(fops, fops_opt):
    common_keys = set(fops).intersection(fops_opt)
    comparator = {}
    for key in common_keys:
        if "source" in fops[key]:
            comparator[key] = {
                'name': fops[key]['name'],
                'gas': sum(n for _, n in fops[key]['ops']),
                'gas_opt': sum(n for _, n in fops_opt[key]['ops']),
                'source': fops[key]['source'],
                'source_opt': fops_opt[key]['source']
            }
        else:
            comparator[key] = {
                'name': fops[key]['name'],
                'gas': sum(n for _, n in fops[key]['ops']),
                'gas_opt': sum(n for _, n in fops_opt[key]['ops']),
            }
    return comparator

def get_function_embeddings(fops):
    w2v_model = Word2Vec.load('opcode_w2v_mac.model')
    wv = w2v_model.wv
    keys = list(fops.keys())
    for i in range(len(keys)):
        key = keys[i]
        ops = fops[key]['ops']
        opcode_vectors = np.array([wv[op[0]] for op in ops])
        fops[key]['embeds'] = opcode_vectors
    return fops

def analyze_functions(contracts, ancons):
    functions = defaultdict(list)
    error_list = []
    for i in tqdm(range(len(contracts))):
        confile = contracts[i]
        con = 'contracts_organized_disl/' + confile
        ancon = ancons[confile]
        for contract in ancon['contracts']:
            try:
                sys.stdout = io.StringIO()
                fops = get_function_insights(con, contract)
                sys.stdout = sys.__stdout__
                fops = get_function_embeddings(fops)
                for fn in fops:
                    fops[fn]['contract_name'] = contract
                    fops[fn]['filename'] = confile
                    functions[fn].append(fops[fn])
            except Exception as e:
                error_list.append({
                    'contract': contract,
                    'error': e
                })
                continue
    return functions, error_list

def generate_comparison(fops1, fops2):
    d = difflib.Differ()
    ops1 = fops1['ops']
    ops2 = fops2['ops']
    gas1 = sum(n for _, n in ops1)
    gas2 = sum(n for _, n in ops2)
    if gas1 > gas2:
        text = 'function2 is more efficient than function1'
        if 'source' in fops1 and 'source' in fops2 and (gas1-gas2)/gas1 < 0.95:
            dence = d.compare(fops1['source'].splitlines(), fops2['source'].splitlines())
            sm = difflib.SequenceMatcher(None, fops1['source'], fops2['source']).ratio()
            if sm == 1.0:
                return None
            difftext = '\n'.join(list(dence))
            #else:
            #    diff = difflib.ndiff([n for n, _ in fops1['ops']], [n for n, _ in fops2['ops']])
            #    difftext = '\n'.join(diff)
            dictreturn = {
                'function1': fops1['name'],
                'function2': fops2['name'], 
                'filename1': fops1['filename'],
                'filename2': fops2['filename'],
                'contract1': fops1['contract_name'],
                'contract2': fops2['contract_name'],
                'gas1': gas1,
                'gas2': gas2,
                'ops1': ops1,
                'ops2': ops2,
                'comment': text,
                'difftext': difftext,
                'gasdifference': gas1 - gas2,
                'percentage_difference': (gas1 - gas2)/gas1
            }
            #print(dictreturn)
            return dictreturn

def prepare_dataset(functions):
    dataset = []
    for function in functions:
        fns = functions[function]
        for fn in fns:
            mean_embed = np.mean(fn['embeds'], axis = 0)
            fn['embeds'] = mean_embed
            fn['signature'] = function
            dataset.append(fn)
    return dataset

def find_optimal_function(fops, dataset):
    max_gas_diff = 0
    max_sim = 0
    max_gas_diff_dict = {}
    max_sim_dict = {}
    embeds = fops['embeds']
    for data in dataset:
        embeds_opt = data['embeds']
        cos_sim = dot(embeds, embeds_opt)/(norm(embeds)*norm(embeds_opt))
        if fops['signature'] == data['signature']:
            dictreturn = generate_comparison(fops, data)
            if dictreturn:
                if dictreturn['gasdifference'] > max_gas_diff:
                    max_gas_diff = dictreturn['gasdifference']
                    max_gas_diff_dict = copy.deepcopy(dictreturn)
                if cos_sim > max_sim:
                    max_sim = cos_sim
                    max_sim_dict = copy.deepcopy(dictreturn)
    return max_gas_diff_dict, max_sim_dict

def prepare_stats(df):
    count = 0
    temp_contract = ''
    temp_file = ''
    notsim = []
    list_diff = []
    list_sim = []
    list_same = []
    sum_percentage_sim = 0
    sum_percentage_diff = 0
    sum_percentage_same = 0
    count_sim = 0
    count_diff = 0
    for idx, row in df.iterrows():
        if row['type'] == 'max_gas_diff':
            temp_contract = row['contract2']
            temp_file = row['filename2']
            sum_percentage_diff += row['percentage_difference']
            list_diff.append(row['percentage_difference'])
            count_diff += 1
        if row['type'] == 'max_sim':
            sum_percentage_sim += row['percentage_difference']
            list_sim.append(row['percentage_difference'])
            count_sim += 1
            if row['contract2'] == temp_contract and row['filename2'] == temp_file:
                count += 1
                sum_percentage_same += row['percentage_difference']
                list_same.append(row['percentage_difference'])
            else:
                notsim.append((row['filename1'], row['contract1'], row['function1']))

if __name__ == '__main__':
    print("Reading processed functions...")
    ancons = pickle.load(open('split_contracts.pkl', 'rb'))
    contracts = os.listdir('contracts_organized_disl')[:25000]
    print("Analyzing the functions...")
    function_list = analyze_functions(contracts, ancons)
    pickle.dump(function_list, open('processed_functions.pkl', 'wb'))
    print("Preparing the dataset...")
    dataset = prepare_dataset(function_list)
    pickle.dump(dataset, open('dataset_mean_embeds.pkl', 'wb'))
    #dataset = pickle.load(open('dataset_mean_embeds.pkl', 'rb'))
    print("Analyzing the dataset...")
    df = pd.DataFrame(columns = ['function1', 'function2', 'type', 'filename1', 'filename2', 'contract1', 'contract2', 'gas1', 'gas2', 'ops1', 'ops2', 'comment', 'difftext', 'gasdifference', 'percentage_difference'])
    for i in tqdm(range(len(dataset))):
        data = dataset[i]
        max_gas_diff_dict, max_sim_dict = find_optimal_function(data, dataset)
        if max_gas_diff_dict and max_sim_dict:
            max_gas_diff_dict['type'] = 'max_gas_diff'
            max_sim_dict['type'] = 'max_sim'
            df.loc[len(df)] = max_gas_diff_dict
            df.loc[len(df)] = max_sim_dict
    print("Writing comparions to an Excel file...")
    df.to_excel('gas_comparison_with_signatures.xlsx', sheet_name='Sheet1')
    print("Generating statistics...")
    count = 0
    temp_contract = ''
    temp_file = ''
    notsim = []
    list_diff = []
    list_sim = []
    list_same = []
    sum_percentage_sim = 0
    sum_percentage_diff = 0
    sum_percentage_same = 0
    count_sim = 0
    count_diff = 0
    for idx, row in df.iterrows():
        if row['type'] == 'max_gas_diff':
            temp_contract = row['contract2']
            temp_file = row['filename2']
            sum_percentage_diff += row['percentage_difference']
            list_diff.append(row['percentage_difference'])
            count_diff += 1
        if row['type'] == 'max_sim':
            sum_percentage_sim += row['percentage_difference']
            list_sim.append(row['percentage_difference'])
            count_sim += 1
            if row['contract2'] == temp_contract and row['filename2'] == temp_file:
                count += 1
                sum_percentage_same += row['percentage_difference']
                list_same.append(row['percentage_difference'])
            else:
                notsim.append((row['filename1'], row['contract1'], row['function1']))
    print("-----------------------------------------------")
    print("Comparions with maximum gas difference:")
    print("Mean percentage of gas cost reduction: ", str((sum_percentage_diff/count_diff)*100))
    print("Standard deviation of percentage of gas cost reduction: ", str(statistics.stdev(list_diff)))
    print("Mode percentage of gas cost reduction: ", str(statistics.mode(list_diff)))
    print("-----------------------------------------------")
    print("Comparions with maximum similarity:")
    print("Mean percentage of gas cost reduction: ", str((sum_percentage_sim/count_sim)*100))
    print("Standard deviation of percentage of gas cost reduction: ", str(statistics.stdev(list_sim)))
    print("Mode percentage of gas cost reduction: ", str(statistics.mode(list_sim)))
    print("-----------------------------------------------")
    print("Comparions where the maximum similarity and maximum gas difference outputs are the same:")
    print("Mean percentage of gas cost reduction: ", str((sum_percentage_same/count)*100))
    print("Standard deviation of percentage of gas cost reduction: ", str(statistics.stdev(list_same)))
    print("Mode percentage of gas cost reduction: ", str(statistics.mode(list_same)))
    print("-----------------------------------------------")
