#!/usr/bin/env python3
"""Generate per-function CPGs for all Solidity files in contracts_organized_disl."""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from tqdm import tqdm

from slither.exceptions import SlitherError

from slither_dfg_builder import (
    SlitherDFGBuilder,
    _select_ast_function_node,
    _simple_function_name,
    build_gnn_inputs,
)

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Walk Solidity contracts, build per-function CPGs, and emit GNN-ready tensors.",
    )
    parser.add_argument(
        "--contracts-dir",
        type=Path,
        default=Path("contracts_organized_disl"),
        help="Directory containing Solidity sources (default: contracts_organized_disl).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("function_cpgs.pkl"),
        help="Where to store the dataset (default: function_cpgs.pkl).",
    )
    parser.add_argument(
        "--format",
        choices=("pickle", "json"),
        default="pickle",
        help="Serialization format for the dataset (default: pickle).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional upper bound for how many contracts to process.",
    )
    parser.add_argument(
        "--make-undirected",
        action="store_true",
        help="Duplicate edges in reverse order so the graph becomes undirected.",
    )
    parser.add_argument(
        "--solc-version",
        help="Force a specific solc version for all contracts (skips pragma detection).",
    )
    parser.add_argument(
        "--solc-binary",
        help="Use a specific solc binary path instead of solc-select.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level (default: INFO).",
    )
    return parser.parse_args()


def collect_contract_files(root: Path) -> List[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Contracts directory not found: {root}")
    return sorted(p for p in root.rglob("*.sol") if p.is_file())


def build_ast_child_map(ast_edges: Iterable[Dict[str, Any]]) -> Dict[str, List[str]]:
    mapping: Dict[str, List[str]] = defaultdict(list)
    for edge in ast_edges:
        src = edge.get("src")
        dst = edge.get("dst")
        if src and dst:
            mapping[src].append(dst)
    return mapping


def collect_ast_subtree(root_id: Optional[str], child_map: Dict[str, List[str]]) -> Set[str]:
    if root_id is None:
        return set()
    nodes: Set[str] = set()
    queue: deque[str] = deque([root_id])
    while queue:
        current = queue.popleft()
        if current in nodes:
            continue
        nodes.add(current)
        for child in child_map.get(current, []):
            queue.append(child)
    return nodes


def build_function_graph(
    function_name: str,
    contract_graph: Dict[str, Any],
    child_map: Dict[str, List[str]],
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    combined_nodes = contract_graph["combined"]["nodes"]
    combined_edges = contract_graph["combined"]["edges"]
    ast_nodes = contract_graph["ast"]["nodes"]
    ast_map = contract_graph["ast"]["function_map"]
    dfg_graph = contract_graph["dfg"].get(function_name)
    cfg_graph = contract_graph["cfg"].get(function_name)
    if dfg_graph is None:
        return None, None

    node_ids: Set[str] = set()

    def add_scope_nodes(prefix: str, graph: Optional[Dict[str, Any]]) -> None:
        if graph is None:
            return
        scope_id = f"{prefix}::{function_name}:scope"
        if scope_id in combined_nodes:
            node_ids.add(scope_id)
        for node_key in graph.get("nodes", {}).keys():
            node_id = f"{prefix}::{function_name}::{node_key}"
            if node_id in combined_nodes:
                node_ids.add(node_id)

    add_scope_nodes("dfg", dfg_graph)
    add_scope_nodes("cfg", cfg_graph)

    # Add AST subtree rooted at the selected function definition.
    ast_root = None
    simple_name = _simple_function_name(function_name)
    ast_candidates = ast_map.get(simple_name, [])
    if ast_candidates:
        ast_root = _select_ast_function_node(function_name, ast_candidates, ast_nodes)
        node_ids.update(collect_ast_subtree(ast_root, child_map))

    # Ensure AST nodes that individual DFG nodes map to are also covered (plus their children).
    mapped_ast_roots: Set[str] = set()
    for edge in combined_edges:
        if edge.get("kind") == "maps_to_ast" and edge.get("src") in node_ids:
            mapped = edge.get("dst")
            if mapped:
                mapped_ast_roots.add(mapped)
    for mapped_root in mapped_ast_roots:
        node_ids.update(collect_ast_subtree(mapped_root, child_map))

    if not node_ids:
        return None, ast_root

    nodes = {nid: dict(combined_nodes[nid]) for nid in node_ids if nid in combined_nodes}
    if not nodes:
        return None, ast_root

    edges: List[Dict[str, Any]] = []
    for edge in combined_edges:
        src = edge.get("src")
        dst = edge.get("dst")
        if src in node_ids and dst in node_ids:
            edges.append(dict(edge))

    return {"nodes": nodes, "edges": edges}, ast_root


def extract_function_cpgs(
    contract_path: Path,
    contract_graph: Dict[str, Any],
    make_undirected: bool,
) -> List[Dict[str, Any]]:
    child_map = build_ast_child_map(contract_graph["ast"]["edges"])
    functions = []
    for function_name in sorted(contract_graph["dfg"].keys()):
        fn_graph, ast_root = build_function_graph(function_name, contract_graph, child_map)
        if not fn_graph:
            LOGGER.debug("Skipping %s:%s – no usable graph nodes", contract_path, function_name)
            continue
        gnn_inputs = build_gnn_inputs(fn_graph, make_undirected=make_undirected)
        functions.append(
            {
                "contract": str(contract_path),
                "function_full_name": function_name,
                "function_simple_name": _simple_function_name(function_name),
                "ast_root": ast_root,
                "graph": fn_graph,
                "gnn": gnn_inputs,
                "num_nodes": len(fn_graph["nodes"]),
                "num_edges": len(fn_graph["edges"]),
            }
        )
    return functions


def save_dataset(dataset: Dict[str, Any], output_path: Path, fmt: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "pickle":
        with output_path.open("wb") as handle:
            pickle.dump(dataset, handle)
    else:
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(dataset, handle)


def main() -> None:
    args = parse_args()

    template_name = "logging"
    fh = logging.FileHandler(filename="logs.log")
    logger = logging.getLogger(template_name)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
        handlers=[fh],
    )

    contract_files = collect_contract_files(args.contracts_dir)
    if args.limit:
        contract_files = contract_files[: args.limit]
    LOGGER.info("Discovered %d Solidity contracts", len(contract_files))

    builder_kwargs: Dict[str, Any] = {}
    if args.solc_version:
        builder_kwargs["solc_version"] = args.solc_version
    if args.solc_binary:
        builder_kwargs["solc_binary"] = args.solc_binary
    builder = SlitherDFGBuilder(**builder_kwargs)

    dataset: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "contracts_dir": str(args.contracts_dir),
        "solc_version": args.solc_version,
        "solc_binary": args.solc_binary,
        "make_undirected": args.make_undirected,
        "stats": {
            "contracts_total": len(contract_files),
            "contracts_succeeded": 0,
            "contracts_failed": 0,
            "functions_total": 0,
        },
        "failures": [],
        "functions": [],
    }

    for index, contract_path in tqdm(enumerate(contract_files, start=1), total=len(contract_files)):
        LOGGER.info("Processing %s (%d/%d)", contract_path, index, len(contract_files))
        try:
            contract_graph = builder.build_with_ast(contract_path)
        except (SlitherError, RuntimeError, FileNotFoundError) as exc:
            LOGGER.error("Failed to process %s: %s", contract_path, exc)
            dataset["stats"]["contracts_failed"] += 1
            dataset["failures"].append({"contract": str(contract_path), "error": str(exc)})
            continue
        except Exception as exc:  # pragma: no cover - defensive guardrail
            LOGGER.exception("Unexpected failure while processing %s", contract_path)
            dataset["stats"]["contracts_failed"] += 1
            dataset["failures"].append({"contract": str(contract_path), "error": str(exc)})
            continue

        function_graphs = extract_function_cpgs(contract_path, contract_graph, args.make_undirected)
        if not function_graphs:
            LOGGER.warning("No functions extracted for %s", contract_path)
        dataset["functions"].extend(function_graphs)
        dataset["stats"]["contracts_succeeded"] += 1
        dataset["stats"]["functions_total"] += len(function_graphs)

    save_dataset(dataset, args.output, args.format)
    LOGGER.info(
        "Finished. Contracts succeeded: %d, failed: %d, functions: %d. Output => %s",
        dataset["stats"]["contracts_succeeded"],
        dataset["stats"]["contracts_failed"],
        dataset["stats"]["functions_total"],
        args.output,
    )


if __name__ == "__main__":
    main()
