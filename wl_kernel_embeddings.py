#!/usr/bin/env python3
"""Compute Weisfeiler-Lehman kernel embeddings for function graphs using GraKeL."""

from __future__ import annotations

import argparse
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

try:
    from grakel import Graph
    from grakel.kernels import VertexHistogram, WeisfeilerLehman
except ImportError as exc:
    raise SystemExit(
        "GraKeL is required for this script."
    ) from exc


def _load_dataset(path: Path) -> MutableMapping[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Input dataset not found: {path}")
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    with path.open("rb") as handle:
        return pickle.load(handle)


def _save_output(obj: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        serializable = dict(obj)
        serializable["embeddings"] = serializable["embeddings"].tolist()
        with path.open("w", encoding="utf-8") as handle:
            json.dump(serializable, handle)
        return
    with path.open("wb") as handle:
        pickle.dump(obj, handle)


def _select_node_label(data: Mapping[str, object], preferences: Sequence[str]) -> str:
    for key in preferences:
        value = data.get(key)
        if value:
            return str(value)
    parts: List[str] = []
    if data.get("graph"):
        parts.append(f"g={data['graph']}")
    if data.get("kind"):
        parts.append(f"k={data['kind']}")
    if data.get("type"):
        parts.append(f"t={data['type']}")
    if data.get("label"):
        parts.append(f"l={data['label']}")
    return "|".join(parts) if parts else "unknown"


def _build_grakel_graph(
    fn_entry: Mapping[str, object],
    *,
    node_attr_order: Sequence[str],
    edge_attr: Optional[str],
    undirected: bool,
) -> Graph:
    graph_data = fn_entry["graph"]
    node_ids = sorted(graph_data["nodes"].keys())
    
    # If a graph has zero nodes, return an empty Graph object safely
    if not node_ids:
        return Graph({}, node_labels={})

    node_to_idx: Dict[str, int] = {nid: i for i, nid in enumerate(node_ids)}

    node_labels: Dict[int, str] = {
        idx: _select_node_label(graph_data["nodes"][nid], node_attr_order)
        for nid, idx in node_to_idx.items()
    }

    # FIX: Use an Adjacency Dictionary instead of an edge list.
    # This ensures isolated nodes are registered correctly.
    adj: Dict[int, List[int]] = {i: [] for i in range(len(node_ids))}
    edge_labels: Dict[Tuple[int, int], str] = {}

    for edge in graph_data["edges"]:
        src_id = edge.get("src")
        dst_id = edge.get("dst")
        
        # safely look up indices (skip edges pointing to missing nodes)
        src = node_to_idx.get(src_id)
        dst = node_to_idx.get(dst_id)
        
        if src is None or dst is None:
            continue
            
        # Add edge to adjacency dict
        adj[src].append(dst)
        
        # Capture edge label if needed
        if edge_attr:
            label = edge.get(edge_attr)
            if label:
                edge_labels[(src, dst)] = str(label)
        
        # Handle Undirected Logic
        if undirected:
            # Add reverse connection
            adj[dst].append(src)
            # Duplicate edge label for reverse direction
            if edge_attr and (src, dst) in edge_labels:
                edge_labels[(dst, src)] = edge_labels[(src, dst)]

    # Initialize Graph with the dictionary
    if edge_attr and edge_labels:
        return Graph(adj, node_labels=node_labels, edge_labels=edge_labels)
    
    return Graph(adj, node_labels=node_labels)

def _build_graphs(
    dataset: MutableMapping[str, object],
    *,
    node_attr_order: Sequence[str],
    edge_attr: Optional[str],
    undirected: bool,
) -> List[Graph]:
    graphs: List[Graph] = []
    for fn in dataset.get("functions", []):
        graphs.append(
            _build_grakel_graph(
                fn,
                node_attr_order=node_attr_order,
                edge_attr=edge_attr,
                undirected=undirected,
            )
        )
    return graphs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply the GraKeL WL kernel to function graphs and emit embeddings.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("function_cpgs.pkl"),
        help="Path to the function graph dataset (pickle or JSON).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("function_wl_embeddings.pkl"),
        help="Where to store the embeddings (pickle recommended).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Number of WL refinement rounds.",
    )
    parser.add_argument(
        "--node-attrs",
        nargs="+",
        default=["label", "kind", "type"],
        help="Ordered list of node attributes to prefer for WL node labels.",
    )
    parser.add_argument(
        "--edge-attr",
        default="kind",
        help="Edge attribute to include (set to '' to ignore edge labels).",
    )
    parser.add_argument(
        "--undirected",
        action="store_true",
        help="Treat graphs as undirected for the kernel.",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable WL kernel normalization.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = _load_dataset(args.input)
    edge_attr = args.edge_attr or None

    graphs = _build_graphs(
        dataset,
        node_attr_order=args.node_attrs,
        edge_attr=edge_attr,
        undirected=args.undirected,
    )

    if not graphs:
        raise SystemExit("No graphs found in the dataset.")

    wl_kernel = WeisfeilerLehman(
        n_iter=args.iterations,
        normalize=not args.no_normalize,
        base_graph_kernel=VertexHistogram,
    )
    kernel_matrix: np.ndarray = wl_kernel.fit_transform(graphs)

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source_dataset": str(args.input),
        "iterations": args.iterations,
        "node_attrs": list(args.node_attrs),
        "edge_attr": edge_attr,
        "undirected": args.undirected,
        "normalized": not args.no_normalize,
        "functions": dataset.get("functions", []),
        "embeddings": kernel_matrix,
        "graphs": graphs,
    }
    _save_output(output, args.output)


if __name__ == "__main__":
    main()
