"""Dataset-scoped observed schema intersected with the application's read contract."""
from dataclasses import dataclass, replace
import json
import re

from app.services.cypher_compiler import CompiledCypher
from app.services.cypher_direction import ALLOWED_ENDPOINTS, observed_arrow, permitted, relationship_patterns
from app.services.errors import LLMConfigurationError


COMMON = {"id": "STRING", "dataset": "STRING", "source": "STRING"}
ALLOWED_PROPERTIES = {
    "Product": {**COMMON, "name": "STRING"},
    "Supplier": {**COMMON, "name": "STRING"},
    "Category": {**COMMON, "name": "STRING"},
    "Order": {**COMMON, "order_date": "DATE"},
    "OrderLine": {**COMMON, "units": "INTEGER", "net_amount_micros": "INTEGER"},
}
MAX_SCHEMA_ROWS = 64


def metadata_queries(dataset: str) -> tuple[CompiledCypher, CompiledCypher]:
    # Only trusted labels/property names enter the query text. No values or
    # database-wide procedures are returned to the application or the LLM.
    labels = json.dumps(sorted(ALLOWED_PROPERTIES))
    properties = json.dumps(sorted({p for props in ALLOWED_PROPERTIES.values() for p in props}))
    relations = json.dumps(sorted(ALLOWED_ENDPOINTS))
    nodes = (
        "MATCH (n:BusinessNode) WHERE n.dataset = $dataset "
        f"RETURN DISTINCT [label IN labels(n) WHERE label IN {labels}] AS labels, "
        f"[key IN keys(n) WHERE key IN {properties} | "
        "{property: key, type: valueType(n[key])}] AS properties LIMIT $limit"
    )
    edges = (
        "MATCH (a:BusinessNode)-[r]->(b:BusinessNode) "
        "WHERE a.dataset = $dataset AND b.dataset = $dataset "
        f"AND type(r) IN {relations} "
        f"RETURN DISTINCT [label IN labels(a) WHERE label IN {labels}] AS sources, "
        f"type(r) AS relation, [label IN labels(b) WHERE label IN {labels}] AS targets LIMIT $limit"
    )
    return tuple(CompiledCypher(query, {"dataset": dataset, "limit": MAX_SCHEMA_ROWS + 1},
                                MAX_SCHEMA_ROWS, "Product", "records", "none")
                 for query in (nodes, edges))


@dataclass(frozen=True)
class ObservedGraphSchema:
    dataset: str
    properties: dict[str, dict[str, str]]
    edges: frozenset[tuple[str, str, str]]

    @classmethod
    def from_rows(cls, dataset, node_rows, edge_rows):
        if max(len(node_rows), len(edge_rows)) > MAX_SCHEMA_ROWS:
            raise LLMConfigurationError("Neo4j schema metadata exceeded its bound.")
        properties = {}
        for row in node_rows:
            for label in row["labels"]:
                if label not in ALLOWED_PROPERTIES:
                    continue
                visible = properties.setdefault(label, {})
                for item in row["properties"]:
                    name = item["property"]
                    expected = ALLOWED_PROPERTIES[label].get(name)
                    if expected is None:
                        continue
                    if item["type"] != expected + " NOT NULL":
                        raise LLMConfigurationError("Neo4j property types conflict with the read contract.")
                    visible[name] = expected
        if not properties or any(not {"id", "dataset"} <= set(p) for p in properties.values()):
            raise LLMConfigurationError("Neo4j has no usable dataset-scoped schema.")
        edges = frozenset((source, row["relation"], target)
                          for row in edge_rows for source in row["sources"] for target in row["targets"]
                          if permitted(source, row["relation"], target)
                          and source in properties and target in properties)
        return cls(dataset, properties, edges)

    def prompt_text(self) -> str:
        # Exclude the dataset value and all record values from the model context.
        return (
            "Observed schema for the configured Neo4j dataset, restricted to allowed fields. "
            "Only listed labels/properties/relationships are available. Missing structures "
            "may be absent in this snapshot, not universally nonexistent. "
            "Properties are observed types, not guarantees that every record has a value. "
            "Each relationship is [source_label, type, target_label] in its actual stored direction. "
            "Follow these directions, not arrow conventions from examples or names. "
            "If both directions exist for the same allowed pair/type, that traversal is ambiguous. "
            "OrderLine.net_amount_micros, when present, is discounted net revenue in millionths "
            "of currency; units is the purchased quantity.\n"
            + json.dumps({"nodes": self.properties, "relationships": sorted(self.edges)}, sort_keys=True)
        )

    def validate_compiled(self, compiled: CompiledCypher) -> None:
        # This checks only the server compiler's restricted syntax, NOT arbitrary
        # model-authored Cypher. Compiler equality and EXPLAIN remain mandatory.
        if compiled.parameters.get("dataset") != self.dataset:
            raise ValueError("Schema belongs to a different dataset")
        aliases = dict(re.findall(r"\((\w+):(\w+)\)", compiled.cypher))
        if not aliases or any(label not in self.properties for label in aliases.values()):
            raise ValueError("Query label absent from observed schema")
        for alias, prop in re.findall(r"\b(\w+)\.(\w+)\b", compiled.cypher):
            if alias in aliases and prop not in self.properties[aliases[alias]]:
                raise ValueError("Query property absent from observed schema")
        for match in relationship_patterns(compiled.cypher):
            relation = match["forward"] or match["reverse"]
            expected = observed_arrow(match["left_label"], relation, match["right_label"], self.edges)
            if match["arrow"] != expected:
                raise ValueError("Query direction differs from observed Neo4j relationship")

    def orient_template(self, compiled: CompiledCypher) -> CompiledCypher:
        """Change only trusted template arrow tokens, never its filters or parameters."""
        query = compiled.cypher
        for match in reversed(relationship_patterns(query)):
            relation = match["forward"] or match["reverse"]
            arrow = observed_arrow(match["left_label"], relation, match["right_label"], self.edges)
            start, end = match.span("arrow")
            query = query[:start] + arrow + query[end:]
        return replace(compiled, cypher=query)
