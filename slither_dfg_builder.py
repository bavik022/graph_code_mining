"""Utilities to build a data-flow graph (DFG) from Slither IR."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import gen_function_ops
from solc_select import solc_select
from slither.slither import Slither
from slither.core.declarations import Contract, FunctionContract
from slither.core.cfg.node import Node as SlitherNode
from slither.slithir.variables.temporary import TemporaryVariable
from slither.slithir.variables.reference import ReferenceVariable
from slither.slithir.variables.tuple import TupleVariable
from slither.slithir.operations import Operation
from slither.slithir.variables.constant import Constant
from slither.core.variables.local_variable import LocalVariable
from slither.core.variables.state_variable import StateVariable
from slither.core.variables.variable import Variable
from slither.core.declarations.solidity_variables import (
    SolidityVariable,
    SolidityVariableComposed,
)

try:
    from tree_sitter import Language, Parser
    import tree_sitter_solidity as ts_solidity
except ImportError:  # pragma: no cover - optional dependency resolution
    Language = None  # type: ignore
    Parser = None  # type: ignore
    ts_solidity = None  # type: ignore

_TS_SOLIDITY_LANGUAGE: Optional[Language] = None
_TS_SOLIDITY_PARSER: Optional[Parser] = None

LOGGER = logging.getLogger(__name__)


def _get_solidity_parser() -> Parser:
    if Parser is None or ts_solidity is None:
        raise ImportError(
            "tree_sitter and tree_sitter_solidity are required for AST construction."
        )
    global _TS_SOLIDITY_LANGUAGE, _TS_SOLIDITY_PARSER
    if _TS_SOLIDITY_LANGUAGE is None:
        _TS_SOLIDITY_LANGUAGE = Language(ts_solidity.language())
    if _TS_SOLIDITY_PARSER is None:
        _TS_SOLIDITY_PARSER = Parser(_TS_SOLIDITY_LANGUAGE)
    return _TS_SOLIDITY_PARSER


def build_ast_graph(
    source_code: str, source_path: Optional[Path] = None
) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]], Dict[str, List[str]]]:
    """Build a tree-sitter AST graph for the given Solidity source."""
    source_bytes = source_code.encode("utf8")
    parser = _get_solidity_parser()
    tree = parser.parse(source_bytes)
    root = tree.root_node

    ast_nodes: Dict[str, Dict[str, Any]] = {}
    ast_edges: List[Dict[str, Any]] = []
    function_name_to_ast_ids: Dict[str, List[str]] = {}

    node_id_counter = 0

    def next_id() -> str:
        nonlocal node_id_counter
        node_id_counter += 1
        return f"ast:{node_id_counter}"

    if source_path is not None:
        file_abs = str(source_path.resolve())
        file_short = source_path.as_posix()
    else:
        file_abs = None
        file_short = None

    def text_snippet(n) -> str:
        byte_slice = source_bytes[n.start_byte : n.end_byte]
        snippet = byte_slice.decode("utf8", errors="replace")
        return snippet if len(snippet) <= 160 else snippet[:157] + "..."

    def visit(node, parent_id: Optional[str], depth: int) -> None:
        nid = next_id()
        start_row, start_col = node.start_point
        end_row, end_col = node.end_point
        ast_nodes[nid] = {
            "graph": "ast",
            "kind": "ast",
            "type": node.type,
            "start": [start_row, start_col],
            "end": [end_row, end_col],
            "start_byte": node.start_byte,
            "end_byte": node.end_byte,
            "code": text_snippet(node),
            "depth": depth,
        }
        if file_abs is not None:
            ast_nodes[nid]["file"] = file_abs
            ast_nodes[nid]["filename_short"] = file_short

        if parent_id is not None:
            ast_edges.append(
                {"src": parent_id, "dst": nid, "kind": "ast_child", "graph": "ast"}
            )

        if node.type == "function_definition":
            func_name = None
            for child in node.children:
                if child.type == "identifier":
                    func_name = source_code[child.start_byte : child.end_byte]
                    break
            if func_name is None:
                for child in node.children:
                    if child.type in {"constructor", "fallback", "receive"}:
                        func_name = child.type
                        break
            if func_name:
                function_name_to_ast_ids.setdefault(func_name, []).append(nid)

        for child in node.named_children:
            visit(child, nid, depth + 1)

    visit(root, None, 0)
    return ast_nodes, ast_edges, function_name_to_ast_ids


def _simple_function_name(function_full_name: str) -> str:
    tail = function_full_name.split(".")[-1]
    return tail.split("(")[0]


def _select_ast_function_node(
    function_full_name: str,
    candidate_ids: List[str],
    ast_nodes: Dict[str, Dict[str, Any]],
) -> Optional[str]:
    if not candidate_ids:
        return None
    signature = function_full_name.split(".")[-1]
    for ast_id in candidate_ids:
        snippet = ast_nodes.get(ast_id, {}).get("code", "")
        if signature in snippet:
            return ast_id
    return candidate_ids[0]


def _select_ast_node_for_lines(
    ast_nodes: Dict[str, Dict[str, Any]],
    start_line: int,
    end_line: int,
    filename_short: Optional[str],
) -> Optional[str]:
    best_choice: Optional[Tuple[int, int]] = None
    best_ast_id: Optional[str] = None
    for ast_id, info in ast_nodes.items():
        if info.get("type") == "source_unit":
            continue
        if filename_short and info.get("filename_short") not in (None, filename_short):
            continue
        node_start = info.get("start", [0, 0])[0]
        node_end = info.get("end", [0, 0])[0]
        if node_start <= start_line <= node_end and node_start <= end_line <= node_end:
            span = node_end - node_start
            depth = info.get("depth", 0)
            candidate = (span, -depth)
            if best_choice is None or candidate < best_choice:
                best_choice = candidate
                best_ast_id = ast_id
    return best_ast_id

def _map_dfg_nodes_to_ast_edges(
    ast_nodes: Dict[str, Dict[str, Any]],
    dfg_nodes: Dict[str, Dict[str, Any]],
    id_map: Dict[str, str],
    function_full_name: str,
) -> List[Dict[str, Any]]:
    mapping_edges: List[Dict[str, Any]] = []
    for node_key, node in dfg_nodes.items():
        src_id = id_map.get(node_key)
        if not src_id:
            continue
        source_meta = node.get("source")
        if not source_meta:
            continue
        lines = source_meta.get("lines")
        if not lines:
            continue
        start_line = min(lines) - 1
        end_line = max(lines) - 1
        filename_short = source_meta.get("filename_short")
        ast_id = _select_ast_node_for_lines(ast_nodes, start_line, end_line, filename_short)
        if ast_id:
            mapping_edges.append(
                {
                    "src": src_id,
                    "dst": ast_id,
                    "kind": "maps_to_ast",
                    "graph": "mapping",
                    "lines": lines,
                    "dfg_function": function_full_name,
                    "defs": node.get("defs"),
                    "uses": node.get("uses"),
                }
            )
    return mapping_edges


def _version_sort_key(version: str) -> Tuple[int, int, int]:
    core = version.split("+", 1)[0].split("-", 1)[0]
    parts = []
    for chunk in core.split("."):
        try:
            parts.append(int(chunk))
        except ValueError:
            parts.append(0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts[:3])


def _combine_graphs(
    ast_nodes: Dict[str, Dict[str, Any]],
    ast_edges: List[Dict[str, Any]],
    ast_function_map: Dict[str, List[str]],
    cfg_graphs: Dict[str, Dict[str, Any]],
    dfg_graphs: Dict[str, Dict[str, Any]],
) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    combined_nodes: Dict[str, Dict[str, Any]] = {
        node_id: dict(node_data) for node_id, node_data in ast_nodes.items()
    }
    combined_edges: List[Dict[str, Any]] = [dict(edge) for edge in ast_edges]

    for function_full_name, graph in cfg_graphs.items():
        scope_id = f"cfg::{function_full_name}:scope"
        combined_nodes[scope_id] = {
            "graph": "cfg",
            "kind": "cfg_function",
            "function": function_full_name,
        }

        simple_name = _simple_function_name(function_full_name)
        ast_candidates = ast_function_map.get(simple_name, [])
        ast_scope = _select_ast_function_node(function_full_name, ast_candidates, ast_nodes)
        if ast_scope:
            combined_edges.append(
                {
                    "src": ast_scope,
                    "dst": scope_id,
                    "kind": "maps_to_cfg",
                    "graph": "mapping",
                }
            )

        node_id_map: Dict[str, str] = {}
        for node_key, node_data in graph.get("nodes", {}).items():
            new_id = f"cfg::{function_full_name}::{node_key}"
            node_id_map[node_key] = new_id
            combined_nodes[new_id] = {
                **node_data,
                "graph": "cfg",
                "cfg_function": function_full_name,
            }
            combined_edges.append(
                {
                    "src": scope_id,
                    "dst": new_id,
                    "kind": "contains",
                    "graph": "cfg",
                }
            )

        for edge in graph.get("edges", []):
            src = node_id_map.get(edge.get("src"))
            dst = node_id_map.get(edge.get("dst"))
            if src and dst:
                combined_edges.append(
                    {
                        "src": src,
                        "dst": dst,
                        "kind": edge.get("kind", "cfg"),
                        "graph": "cfg",
                    }
                )

    for function_full_name, graph in dfg_graphs.items():
        scope_id = f"dfg::{function_full_name}:scope"
        combined_nodes[scope_id] = {
            "graph": "dfg",
            "kind": "dfg_function",
            "function": function_full_name,
        }

        simple_name = _simple_function_name(function_full_name)
        ast_candidates = ast_function_map.get(simple_name, [])
        ast_scope = _select_ast_function_node(function_full_name, ast_candidates, ast_nodes)
        if ast_scope:
            combined_edges.append(
                {
                    "src": ast_scope,
                    "dst": scope_id,
                    "kind": "maps_to_dfg",
                    "graph": "mapping",
                }
            )

        node_id_map: Dict[str, str] = {}
        for node_key, node_data in graph.get("nodes", {}).items():
            new_id = f"dfg::{function_full_name}::{node_key}"
            node_id_map[node_key] = new_id
            combined_nodes[new_id] = {
                **node_data,
                "graph": "dfg",
                "dfg_function": function_full_name,
            }
            combined_edges.append(
                {
                    "src": scope_id,
                    "dst": new_id,
                    "kind": "contains",
                    "graph": "dfg",
                }
            )

        for edge in graph.get("edges", []):
            src = node_id_map.get(edge.get("src"))
            dst = node_id_map.get(edge.get("dst"))
            if src and dst:
                combined_edges.append(
                    {
                        "src": src,
                        "dst": dst,
                        "kind": edge.get("kind", "dfg"),
                        "graph": "dfg",
                        "var": edge.get("var"),
                    }
                )

        combined_edges.extend(
            _map_dfg_nodes_to_ast_edges(ast_nodes, graph.get("nodes", {}), node_id_map, function_full_name)
        )

    return combined_nodes, combined_edges


def build_gnn_inputs(
    combined_graph: Dict[str, Any],
    *,
    make_undirected: bool = False,
) -> Dict[str, Any]:
    """Create numeric node features and an adjacency list suitable for GNN libraries."""

    nodes = combined_graph["nodes"]
    edges = combined_graph["edges"]

    node_ids = sorted(nodes.keys())
    node_index = {nid: idx for idx, nid in enumerate(node_ids)}

    graph_types = sorted({nodes[nid].get("graph", "unknown") for nid in node_ids})
    ast_types = sorted({nodes[nid].get("type") for nid in node_ids if nodes[nid].get("graph") == "ast"})
    dfg_kinds = sorted({nodes[nid].get("kind") for nid in node_ids if nodes[nid].get("graph") == "dfg"})

    max_depth = max((nodes[nid].get("depth", 0) for nid in node_ids if nodes[nid].get("graph") == "ast"), default=0)

    feature_names: List[str] = []
    feature_names.extend([f"graph::{gt}" for gt in graph_types])
    feature_names.extend([f"ast_type::{tp}" for tp in ast_types])
    feature_names.extend([f"dfg_kind::{kd}" for kd in dfg_kinds])
    numeric_features = [
        "depth_norm",
        "span_lines",
        "span_bytes",
        "label_len",
        "num_defs",
        "num_uses",
        "has_source",
        "is_scope",
        "is_parameter",
    ]
    feature_names.extend(numeric_features)

    feature_rows: List[List[float]] = []
    for nid in node_ids:
        node = nodes[nid]
        row: List[float] = []

        # Graph type one-hot
        node_graph = node.get("graph", "unknown")
        row.extend(1.0 if node_graph == gt else 0.0 for gt in graph_types)

        # AST type one-hot
        node_type = node.get("type") if node_graph == "ast" else None
        row.extend(1.0 if node_type == tp else 0.0 for tp in ast_types)

        # DFG kind one-hot
        node_kind = node.get("kind") if node_graph == "dfg" else None
        row.extend(1.0 if node_kind == kd else 0.0 for kd in dfg_kinds)

        # Numeric features
        if node_graph == "ast":
            depth = node.get("depth", 0)
            span_lines = float(node.get("end", [0, 0])[0] - node.get("start", [0, 0])[0] + 1)
            span_bytes = float(node.get("end_byte", 0) - node.get("start_byte", 0))
            label_len = float(len(node.get("code", "")))
        else:
            depth = 0.0
            span_lines = 0.0
            span_bytes = 0.0
            label_len = float(len(node.get("label", "")))
        depth_norm = depth / max_depth if max_depth > 0 else 0.0

        defs = node.get("defs") or []
        uses = node.get("uses") or []
        num_defs = float(len(defs))
        num_uses = float(len(uses))
        has_source = 1.0 if node.get("source") else 0.0
        is_scope = 1.0 if node.get("kind") == "dfg_function" else 0.0
        is_parameter = 1.0 if node.get("kind") == "parameter" else 0.0

        row.extend(
            [
                float(depth_norm),
                span_lines,
                span_bytes,
                label_len,
                num_defs,
                num_uses,
                has_source,
                is_scope,
                is_parameter,
            ]
        )

        feature_rows.append(row)

    edge_src: List[int] = []
    edge_dst: List[int] = []
    edge_types: List[str] = []

    for edge in edges:
        src = node_index.get(edge["src"]) if isinstance(edge, dict) else None
        dst = node_index.get(edge["dst"]) if isinstance(edge, dict) else None
        if src is None or dst is None:
            continue
        edge_src.append(src)
        edge_dst.append(dst)
        edge_types.append(edge.get("kind", "unknown") if isinstance(edge, dict) else "unknown")
        if make_undirected:
            edge_src.append(dst)
            edge_dst.append(src)
            edge_types.append(edge_types[-1])

    edge_index = [edge_src, edge_dst]

    edge_type_vocab = sorted(set(edge_types))
    edge_type_index = [edge_type_vocab.index(kind) for kind in edge_types]

    return {
        "node_index": node_index,
        "feature_names": feature_names,
        "node_features": feature_rows,
        "edge_index": edge_index,
        "edge_types": edge_types,
        "edge_type_index": edge_type_index,
        "vocab": {
            "graph_types": graph_types,
            "ast_types": ast_types,
            "dfg_kinds": dfg_kinds,
            "edge_types": edge_type_vocab,
        },
    }


@dataclass
class DFGNode:
    node_id: str
    label: str
    kind: str
    defs: List[str] = field(default_factory=list)
    uses: List[str] = field(default_factory=list)
    source: Optional[Dict[str, object]] = None
    raw_ir: Optional[str] = None


@dataclass
class DFGEdge:
    src: str
    dst: str
    var: str
    kind: str


class SlitherDFGBuilder:
    """Build a per-function data-flow graph from Slither IR."""

    def __init__(self, solc_binary: Optional[str] = None, solc_version: Optional[str] = None) -> None:
        self._solc_binary = solc_binary
        self._solc_version = solc_version

    def build(self, source_path: str | Path, contract_name: Optional[str] = None,
              functions: Optional[Iterable[str]] = None) -> Dict[str, Dict[str, object]]:
        source_path = Path(source_path)
        if not source_path.exists():
            raise FileNotFoundError(f"Solidity source not found: {source_path}")

        sl = self._load_slither(source_path)
        target_functions = self._get_target_functions(sl, contract_name, functions, source_path)

        results: Dict[str, Dict[str, object]] = {}
        for fn in target_functions:
            graph = self._build_function_dfg(fn)
            key = getattr(fn, "canonical_name", fn.full_name)
            results[key] = graph
        return results

    def build_with_ast(
        self,
        source_path: str | Path,
        contract_name: Optional[str] = None,
        functions: Optional[Iterable[str]] = None,
    ) -> Dict[str, Any]:
        path = Path(source_path)
        if not path.exists():
            raise FileNotFoundError(f"Solidity source not found: {path}")

        source_code = path.read_text()
        ast_nodes, ast_edges, ast_function_map = build_ast_graph(source_code, path)

        sl = self._load_slither(path)
        target_functions = self._get_target_functions(sl, contract_name, functions, path)

        dfg_graphs: Dict[str, Dict[str, Any]] = {}
        cfg_graphs: Dict[str, Dict[str, Any]] = {}
        for fn in target_functions:
            key = getattr(fn, "canonical_name", fn.full_name)
            dfg_graphs[key] = self._build_function_dfg(fn)
            cfg_graphs[key] = self._build_function_cfg(fn)

        combined_nodes, combined_edges = _combine_graphs(
            ast_nodes, ast_edges, ast_function_map, cfg_graphs, dfg_graphs
        )

        return {
            "ast": {
                "nodes": ast_nodes,
                "edges": ast_edges,
                "function_map": ast_function_map,
            },
            "dfg": dfg_graphs,
            "cfg": cfg_graphs,
            "combined": {"nodes": combined_nodes, "edges": combined_edges},
        }

    def _load_slither(self, source_path: Path) -> Slither:
        solc_bin = self._ensure_solc(source_path)
        return Slither(str(source_path), solc=solc_bin)

    def _get_target_functions(
        self,
        sl: Slither,
        contract_name: Optional[str],
        functions: Optional[Iterable[str]],
        source_path: Path,
    ) -> List[FunctionContract]:
        if contract_name:
            contracts = sl.get_contract_from_name(contract_name)
            if not contracts:
                raise ValueError(f"Contract '{contract_name}' not found in {source_path}")
        else:
            contracts = sl.contracts

        function_filters: Optional[set[str]] = None
        if functions is not None:
            function_filters = {fn for fn in functions}

        targets: List[FunctionContract] = []
        for contract in contracts:
            for fn in contract.functions:
                if function_filters:
                    if (
                        fn.full_name not in function_filters
                        and fn.name not in function_filters
                        and getattr(fn, "canonical_name", fn.full_name) not in function_filters
                    ):
                        continue
                targets.append(fn)
        return targets

    def _ensure_solc(self, source_path: Path) -> str:
        if self._solc_binary:
            return self._solc_binary

        version = self._solc_version
        if version is None:
            version = self._deduce_solc_version(source_path)
        solc_select.switch_global_version(version, True, silent=True)
        binary = solc_select.artifact_path(version)
        if not binary.exists():
            raise FileNotFoundError(
                f"solc {version} is not installed. Run 'solc-select install {version}' first."
            )
        return str(binary)

    def _deduce_solc_version(self, source_path: Path) -> str:
        pragma_statements = gen_function_ops.extract_pragma_statements(source_path.read_text())
        try:
            supported = gen_function_ops.get_supported_versions()
        except Exception as exc:  # pragma: no cover - network failures
            LOGGER.warning(
                "Unable to fetch installable solc versions (offline mode assumed): %s", exc
            )
            supported = []
        installed = gen_function_ops.get_installed_versions()
        candidates = [ver for ver in supported + installed if ver]
        if not candidates:
            raise RuntimeError(
                "No Solidity compiler versions are available. Install one via solc-select."
            )
        try:
            version = gen_function_ops.determine_optimal_version(pragma_statements, candidates)
        except Exception as exc:
            if not installed:
                raise
            version = max(installed, key=_version_sort_key)
            LOGGER.warning(
                "Falling back to installed solc '%s' for %s after failing to honor pragma %s: %s",
                version,
                source_path,
                pragma_statements,
                exc,
            )
        return version

    def _build_function_dfg(self, fn: FunctionContract) -> Dict[str, object]:
        nodes: Dict[str, DFGNode] = {}
        edges: List[DFGEdge] = []
        last_defs: Dict[str, str] = {}
        source_nodes: Dict[str, str] = {}
        parameter_ids: set[str] = set()

        def ensure_source_node(var_id: str, meta: Dict[str, object], kind: str) -> str:
            """Create (or reuse) a synthetic node for external definitions."""
            node_id = source_nodes.get(var_id)
            if node_id is None:
                node_id = f"source::{var_id}"
                nodes[node_id] = DFGNode(
                    node_id=node_id,
                    label=meta.get("label", var_id),
                    kind=kind,
                    defs=[var_id],
                    uses=[],
                    source=None,
                )
                source_nodes[var_id] = node_id
                last_defs[var_id] = node_id
            return node_id

        # Treat function parameters as already-defined sources so their uses get connected.
        for param in fn.parameters:
            meta = self._describe_variable(param)
            parameter_ids.add(meta["id"])
            ensure_source_node(meta["id"], meta, "parameter")

        ir_index = 0
        # Iterate through the IRs following Slither's dominance ordering.
        for node in fn.nodes_ordered_dominators:
            for ir in node.irs:
                node_id = f"{fn.canonical_name}:{ir_index}"
                ir_index += 1
                defs_meta = [self._describe_variable(var) for var in self._iter_defs(ir)]
                uses_meta = [self._describe_variable(var) for var in getattr(ir, "read", [])]
                source_info = self._source_mapping(node)
                nodes[node_id] = DFGNode(
                    node_id=node_id,
                    label=str(ir),
                    kind=ir.__class__.__name__,
                    defs=[meta["id"] for meta in defs_meta],
                    uses=[meta["id"] for meta in uses_meta],
                    source=source_info,
                    raw_ir=str(ir),
                )

                for meta in uses_meta:
                    var_id = meta["id"]
                    origin_kind = "parameter" if var_id in parameter_ids else meta.get("origin", "external")
                    src_node = last_defs.get(var_id)
                    edge_kind = "def_use"
                    if src_node is None:
                        src_node = ensure_source_node(var_id, meta, origin_kind)
                        edge_kind = f"{origin_kind}_use"
                    elif source_nodes.get(var_id) == src_node:
                        edge_kind = f"{origin_kind}_use"
                    edges.append(DFGEdge(src=src_node, dst=node_id, var=var_id, kind=edge_kind))

                for meta in defs_meta:
                    last_defs[meta["id"]] = node_id

        return {
            "function": getattr(fn, "canonical_name", fn.full_name),
            "nodes": {node_id: node.__dict__ for node_id, node in nodes.items()},
            "edges": [edge.__dict__ for edge in edges],
        }

    def _build_function_cfg(self, fn: FunctionContract) -> Dict[str, Any]:
        nodes: Dict[str, Dict[str, Any]] = {}
        edges: List[Dict[str, Any]] = []

        for node in fn.nodes:
            key = str(node.node_id)
            source_meta = None
            src = node.source_mapping
            if src:
                filename = getattr(src, "filename", None)
                source_meta = {
                    "lines": getattr(src, "lines", None),
                    "file": getattr(filename, "absolute", str(filename)) if filename else None,
                    "filename_short": getattr(filename, "short", None),
                }
            irs = [str(ir) for ir in node.irs]
            nodes[key] = {
                "kind": node.type.name if hasattr(node.type, "name") else str(node.type),
                "type": node.type.name if hasattr(node.type, "name") else str(node.type),
                "expression": str(node.expression) if node.expression else None,
                "irs": irs,
                "irs_count": len(irs),
                "source": source_meta,
            }

        for node in fn.nodes:
            for son in node.sons:
                edges.append(
                    {
                        "src": str(node.node_id),
                        "dst": str(son.node_id),
                        "kind": "cfg",
                    }
                )

        return {
            "function": getattr(fn, "canonical_name", fn.full_name),
            "nodes": nodes,
            "edges": edges,
        }

    def _iter_defs(self, ir: Operation) -> Iterable[Variable]:
        lvalue = getattr(ir, "lvalue", None)
        if lvalue is None:
            return []
        if isinstance(lvalue, (list, tuple, set)):
            return [lv for lv in lvalue if isinstance(lv, Variable)]
        if isinstance(lvalue, Variable):
            return [lvalue]
        return []

    def _describe_variable(self, var: Variable) -> Dict[str, object]:
        type_name = type(var).__name__
        base_id = self._variable_id(var)
        meta: Dict[str, object] = {
            "id": base_id,
            "type": type_name,
            "origin": self._variable_origin(var),
            "label": self._variable_label(var),
        }
        name = getattr(var, "name", None)
        if name:
            meta["name"] = name
        canonical = getattr(var, "canonical_name", None)
        if canonical:
            meta["canonical_name"] = canonical
        value = getattr(var, "value", None)
        if value is not None:
            meta["value"] = value
        var_type = getattr(var, "type", None)
        if var_type is not None:
            meta["var_type"] = str(var_type)
        return meta

    def _variable_id(self, var: Variable) -> str:
        type_name = type(var).__name__
        canonical = getattr(var, "canonical_name", None)
        if canonical:
            core = canonical
        elif getattr(var, "name", None):
            core = var.name
        elif isinstance(var, Constant):
            core = str(var.value)
        else:
            core = str(var)
        return f"{type_name}|{core}"

    def _variable_label(self, var: Variable) -> str:
        if isinstance(var, Constant):
            return f"const {var.value}"
        if getattr(var, "name", None):
            return var.name
        return str(var)

    def _variable_origin(self, var: Variable) -> str:
        if isinstance(var, Constant):
            return "constant"
        if isinstance(var, LocalVariable):
            return "local"
        if isinstance(var, StateVariable):
            return "state"
        if isinstance(var, TemporaryVariable):
            return "temporary"
        if isinstance(var, ReferenceVariable):
            return "reference"
        if isinstance(var, TupleVariable):
            return "tuple"
        if isinstance(var, (SolidityVariable, SolidityVariableComposed)):
            return "solidity_builtin"
        if isinstance(var, Contract):
            return "contract"
        return "external"

    def _source_mapping(self, node: SlitherNode) -> Optional[Dict[str, object]]:
        src = node.source_mapping
        if not src:
            return None
        expression = getattr(node, "expression", None)
        info: Dict[str, object] = {
            "expression": str(expression) if expression is not None else None,
            "lines": getattr(src, "lines", None),
        }
        filename = getattr(src, "filename", None)
        if filename is not None:
            info["file"] = getattr(filename, "absolute", str(filename))
            info["filename_short"] = getattr(filename, "short", None)
        return info
