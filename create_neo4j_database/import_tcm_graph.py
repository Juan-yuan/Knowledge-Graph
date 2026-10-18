"""Load extracted graph JSON files into Neo4j. Schema and project settings come from .env."""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config import Config
from common.path_utils import get_file_path

_conf = Config()

PROJECT = _conf.require("TCM_NEO4J_PROJECT")
ENTITY_LABELS = set(_conf.require_json("TCM_IMPORT_ENTITY_LABELS"))
ATTRIBUTE_KEYS = tuple(_conf.require_json("TCM_IMPORT_ATTRIBUTE_KEYS"))
RELATION_SCHEMA = {
    key: tuple(value)
    for key, value in _conf.require_json("TCM_IMPORT_RELATION_SCHEMA").items()
}

BATCH_SIZE = 500

DEFAULT_DATA_DIRS = [
    get_file_path("extract_information/medicinal_herbs"),
    get_file_path("extract_information/herbal_formulas"),
]

NodeKey = Tuple[str, str]  # (label, name)
RelKey = Tuple[str, str, str, str, str]  # (src_label, src_name, rel_type, tgt_label, tgt_name)


def load_extract_dicts(data_dirs: Iterable[str]) -> List[dict]:
    """Read every *.json file and return its extract_dict, tagged with the file name."""
    docs = []
    for data_dir in data_dirs:
        for path in sorted(Path(data_dir).glob("*.json")):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            extract = data.get("extract_dict") or {}
            docs.append({
                "file": path.name,
                "entities": extract.get("entities") or [],
                "relations": extract.get("relations") or [],
            })
    return docs


def _clean_name(name) -> str:
    return name.strip() if isinstance(name, str) else ""


def _clean_attributes(entity: dict) -> dict:
    """Merge attribute blocks into one dict, dropping null/empty values."""
    attrs = {}
    for key in ATTRIBUTE_KEYS:
        for prop, value in (entity.get(key) or {}).items():
            if value is None or value == "" or value == []:
                continue
            attrs[prop] = value
    return attrs


def _pick_label(candidates: Optional[Set[str]], expected: str) -> Optional[str]:
    if not candidates:
        return None
    if expected in candidates:
        return expected
    if len(candidates) == 1:
        return next(iter(candidates))
    return sorted(candidates)[0]


def build_graph(docs: List[dict]) -> Tuple[Dict[NodeKey, dict], Set[RelKey]]:
    """
    Turn extract dicts into de-duplicated nodes and relations.

    Nodes are keyed by (label, name). Attributes from several files are merged;
    the occurrence with the most attributes wins on conflicts. Relation endpoints
    are resolved via local entities, then global entities, then RELATION_SCHEMA.
    """
    occurrences: Dict[NodeKey, List[dict]] = defaultdict(list)
    global_labels: Dict[str, Set[str]] = defaultdict(set)

    for doc in docs:
        for entity in doc["entities"]:
            name, label = _clean_name(entity.get("name")), entity.get("type")
            if not name or label not in ENTITY_LABELS:
                continue
            occurrences[(label, name)].append(_clean_attributes(entity))
            global_labels[name].add(label)

    nodes: Dict[NodeKey, dict] = {}
    for key, attr_list in occurrences.items():
        merged = {}
        for attrs in sorted(attr_list, key=len):  # richest occurrence applied last
            merged.update(attrs)
        nodes[key] = merged

    relations: Set[RelKey] = set()
    for doc in docs:
        local_labels: Dict[str, Set[str]] = defaultdict(set)
        for entity in doc["entities"]:
            if entity.get("type") in ENTITY_LABELS:
                local_labels[_clean_name(entity.get("name"))].add(entity["type"])

        for rel in doc["relations"]:
            rel_type = rel.get("type")
            src, tgt = _clean_name(rel.get("source")), _clean_name(rel.get("target"))
            if rel_type not in RELATION_SCHEMA or not src or not tgt:
                continue
            exp_src, exp_tgt = RELATION_SCHEMA[rel_type]
            src_label = (_pick_label(local_labels.get(src), exp_src)
                         or _pick_label(global_labels.get(src), exp_src)
                         or exp_src)
            tgt_label = (_pick_label(local_labels.get(tgt), exp_tgt)
                         or _pick_label(global_labels.get(tgt), exp_tgt)
                         or exp_tgt)

            if rel_type == "FROM_SOURCE" and src_label == "Source" and tgt_label != "Source":
                src, tgt, src_label, tgt_label = tgt, src, tgt_label, src_label

            for key in ((src_label, src), (tgt_label, tgt)):
                nodes.setdefault(key, {})
            relations.add((src_label, src, rel_type, tgt_label, tgt))

    return nodes, relations


def _batches(rows: List[dict], size: int = BATCH_SIZE):
    for i in range(0, len(rows), size):
        yield rows[i:i + size]


def build_constraint_queries(labels: Iterable[str]) -> List[str]:
    return [
        f"CREATE CONSTRAINT {label.lower()}_name_unique IF NOT EXISTS "
        f"FOR (n:{label}) REQUIRE n.name IS UNIQUE"
        for label in sorted(labels)
    ]


def build_write_queries(nodes: Dict[NodeKey, dict],
                        relations: Set[RelKey]) -> List[Tuple[str, dict]]:
    """Build parameterised UNWIND batches for Neo4jClient.run_multiple_cypher."""
    queries: List[Tuple[str, dict]] = []

    rows_by_label: Dict[str, List[dict]] = defaultdict(list)
    for (label, name), props in sorted(nodes.items()):
        rows_by_label[label].append({"name": name, "props": props})
    for label, rows in sorted(rows_by_label.items()):
        query = (
            "UNWIND $rows AS row "
            f"MERGE (n:{label} {{name: row.name}}) "
            "SET n += row.props, n.project = $project"
        )
        for batch in _batches(rows):
            queries.append((query, {"rows": batch, "project": PROJECT}))

    rows_by_pattern: Dict[Tuple[str, str, str], List[dict]] = defaultdict(list)
    for src_label, src, rel_type, tgt_label, tgt in sorted(relations):
        rows_by_pattern[(src_label, rel_type, tgt_label)].append({"source": src, "target": tgt})
    for (src_label, rel_type, tgt_label), rows in sorted(rows_by_pattern.items()):
        query = (
            "UNWIND $rows AS row "
            f"MATCH (a:{src_label} {{name: row.source}}) "
            f"MATCH (b:{tgt_label} {{name: row.target}}) "
            f"MERGE (a)-[:{rel_type}]->(b)"
        )
        for batch in _batches(rows):
            queries.append((query, {"rows": batch}))

    return queries


def import_graph(client, data_dirs: Iterable[str] = DEFAULT_DATA_DIRS,
                 clear: bool = False) -> Tuple[Dict[NodeKey, dict], Set[RelKey]]:
    docs = load_extract_dicts(data_dirs)
    nodes, relations = build_graph(docs)
    print(f"Loaded {len(docs)} file(s): {len(nodes)} node(s), {len(relations)} relation(s).")

    if clear:
        client.clear_database()

    for query in build_constraint_queries({label for label, _ in nodes}):
        client.run_cypher(query)
    client.run_multiple_cypher(build_write_queries(nodes, relations))
    return nodes, relations


def print_summary(client) -> None:
    for row in client.run_cypher(
        "MATCH (n {project: $project}) RETURN head(labels(n)) AS label, count(*) AS count "
        "ORDER BY label", {"project": PROJECT}
    ):
        print(f"  ({row['label']}): {row['count']}")
    for row in client.run_cypher(
        "MATCH ({project: $project})-[r]->() RETURN type(r) AS type, count(*) AS count "
        "ORDER BY type", {"project": PROJECT}
    ):
        print(f"  [{row['type']}]: {row['count']}")


def main():
    parser = argparse.ArgumentParser(
        description="Import extracted graph JSON into Neo4j.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--clear", action="store_true",
                        help="delete ALL nodes, relationships, indexes and constraints first")
    parser.add_argument("--data-dir", action="append", dest="data_dirs",
                        help="folder of extraction JSON files (repeatable)")
    args = parser.parse_args()

    from common.neo4j_manager import neo4j_client

    try:
        import_graph(neo4j_client, args.data_dirs or DEFAULT_DATA_DIRS, clear=args.clear)
        print("Graph now contains:")
        print_summary(neo4j_client)
    finally:
        neo4j_client.close()


if __name__ == "__main__":
    main()
