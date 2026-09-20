"""Prepare a complete, public business graph; exclude personal/contact columns."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path


@dataclass(frozen=True)
class BusinessGraph:
    nodes: list[dict]
    edges: list[dict]
    source_counts: dict[str, int]
    source_hashes: dict[str, str]

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.source_hashes, sort_keys=True).encode()).hexdigest()


def prepare_business_graph(directory: str | Path) -> BusinessGraph:
    root = Path(directory)
    specs = {
        "Products": {"ProductID", "ProductName", "SupplierID", "CategoryID"},
        "Suppliers": {"SupplierID", "CompanyName"},
        "Categories": {"CategoryID", "CategoryName"},
        "Orders": {"OrderID", "OrderDate"},
        "_Order_Details": {"OrderID", "ProductID", "UnitPrice", "Quantity", "Discount"},
    }
    tables, hashes = {}, {}
    for table, fields in specs.items():
        path = root / f"{table}.csv"
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not fields.issubset(reader.fieldnames or []):
                raise ValueError(f"Missing columns: {table}")
            tables[table] = [{key: (row.get(key) or "").strip() for key in fields} for row in reader]
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    nodes, edges, ids = [], [], {}
    for table, label, key, name in [
        ("Products", "Product", "ProductID", "ProductName"),
        ("Suppliers", "Supplier", "SupplierID", "CompanyName"),
        ("Categories", "Category", "CategoryID", "CategoryName"),
        ("Orders", "Order", "OrderID", None),
    ]:
        ids[label] = set()
        for row in tables[table]:
            for field, value in row.items():
                if field.endswith("ID"):
                    if not value.isascii() or not value.isdigit() or int(value) <= 0:
                        raise ValueError("Invalid identifier")
                    row[field] = str(int(value))
            identifier = f"{label}:{row[key]}"
            if identifier in ids[label]:
                raise ValueError("Duplicate identifier")
            ids[label].add(identifier)
            node = {"id": identifier, "label": label, "source": f"{table}.csv", "source_sha256": hashes[f"{table}.csv"]}
            if name:
                value = row[name]
                if not value or len(value) > 200 or any(ord(char) < 32 for char in value):
                    raise ValueError("Invalid public name")
                node["name"] = value
            else:
                node["order_date"] = datetime.fromisoformat(row["OrderDate"]).date().isoformat()
            nodes.append(node)
    for row in tables["Products"]:
        for label, field, relation in [("Supplier", "SupplierID", "SUPPLIED_BY"), ("Category", "CategoryID", "BELONGS_TO")]:
            target = f"{label}:{row[field]}"
            if target not in ids[label]:
                raise ValueError("Unresolved catalog foreign key")
            edges.append({"source": f"Product:{row['ProductID']}", "target": target, "type": relation})
    line_ids, total_micros, total_units = set(), 0, 0
    for row in tables["_Order_Details"]:
        for key in ("OrderID", "ProductID", "Quantity"):
            if not row[key].isascii() or not row[key].isdigit() or int(row[key]) <= 0:
                raise ValueError("Invalid order line identifier or quantity")
            row[key] = str(int(row[key]))
        order, product = f"Order:{row['OrderID']}", f"Product:{row['ProductID']}"
        identifier = f"OrderLine:{row['OrderID']}:{row['ProductID']}"
        if identifier in line_ids or order not in ids["Order"] or product not in ids["Product"]:
            raise ValueError("Duplicate order line or unresolved foreign key")
        line_ids.add(identifier)
        try:
            price, discount = Decimal(row["UnitPrice"]), Decimal(row["Discount"])
            if not price.is_finite() or not discount.is_finite() or price < 0 or not 0 <= discount <= 1:
                raise ValueError("Invalid price/discount")
            amount = price * int(row["Quantity"]) * (1 - discount) * 1_000_000
            if amount != amount.to_integral_value():
                raise ValueError("Revenue requires more than six decimal places")
            total_micros += int(amount)
            total_units += int(row["Quantity"])
            if total_micros >= 2**63 or total_units >= 2**63:
                raise ValueError("Dataset exceeds integer aggregation capacity")
        except InvalidOperation:
            raise ValueError("Invalid decimal") from None
        nodes.append({"id": identifier, "label": "OrderLine", "units": int(row["Quantity"]),
                      "net_amount_micros": int(amount), "source": "_Order_Details.csv",
                      "source_sha256": hashes["_Order_Details.csv"]})
        edges.extend([{"source": order, "target": identifier, "type": "CONTAINS"},
                      {"source": identifier, "target": product, "type": "OF_PRODUCT"}])
    return BusinessGraph(nodes, edges, {table: len(rows) for table, rows in tables.items()}, hashes)


def load_business_graph(driver, *, database: str, dataset: str, graph: BusinessGraph) -> dict:
    """Atomic create-only load. Never merge into or delete an existing snapshot."""
    from app.services.cypher_compiler import EDGES, SCHEMA_VERSION
    labels = {"Product", "Supplier", "Category", "Order", "OrderLine"}
    if not dataset or not graph.nodes or not {node["label"] for node in graph.nodes}.issubset(labels):
        raise ValueError("Invalid dataset")
    def write(tx):
        existing = tx.run("MATCH (n {dataset: $dataset}) RETURN count(n) AS count", dataset=dataset).single()["count"]
        if existing:
            raise ValueError("Dataset already exists; use a new dataset name. Nothing was replaced.")
        for label in sorted(labels):
            rows = [{key: value for key, value in node.items() if key != "label"} for node in graph.nodes if node["label"] == label]
            query = f"UNWIND $rows AS row CREATE (n:BusinessNode:{label}) SET n = row, n.dataset = $dataset"
            if label == "Order":
                query += ", n.order_date = date(row.order_date)"
            tx.run(query, rows=rows, dataset=dataset).consume()
        for source, relation, target in sorted(EDGES):
            rows = [edge for edge in graph.edges if edge["type"] == relation]
            tx.run(f"UNWIND $rows AS row MATCH (a:{source} {{id: row.source, dataset: $dataset}}), "
                   f"(b:{target} {{id: row.target, dataset: $dataset}}) CREATE (a)-[:{relation}]->(b)",
                   rows=rows, dataset=dataset).consume()
        counts = tx.run("MATCH (n:BusinessNode {dataset: $dataset}) OPTIONAL MATCH (n)-[r]->(m:BusinessNode {dataset: $dataset}) "
                        "RETURN count(DISTINCT n) AS nodes, count(r) AS edges", dataset=dataset).single().data()
        if counts != {"nodes": len(graph.nodes), "edges": len(graph.edges)}:
            raise ValueError("Import counts failed; transaction rolled back")
        tx.run("CREATE (s:CatalogSnapshot {dataset: $dataset, schema_version: $version, node_count: $nodes, "
               "edge_count: $edges, source_fingerprint: $fingerprint})", dataset=dataset, version=SCHEMA_VERSION,
               fingerprint=graph.fingerprint, **counts).consume()
        return counts
    with driver.session(database=database) as session:
        # Composite uniqueness also protects against concurrent duplicate imports.
        session.run("CREATE CONSTRAINT business_node_identity IF NOT EXISTS FOR (n:BusinessNode) REQUIRE (n.dataset, n.id) IS UNIQUE").consume()
        session.run("CREATE CONSTRAINT business_snapshot_identity IF NOT EXISTS FOR (s:CatalogSnapshot) REQUIRE s.dataset IS UNIQUE").consume()
        return session.execute_write(write)
