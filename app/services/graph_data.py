"""Local-only, allowlisted catalog preparation. Not an LLM GraphRAG index."""
from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path


@dataclass(kw_only=True, frozen=True)
class PreparedGraphData:
    nodes: list[dict]
    relationships: list[dict]
    documents: dict[str, str]
    source_counts: dict[str, int]
    source_hashes: dict[str, str]
    review_report: dict | None = None


def prepare_catalog(data_dir: str | Path, *, limit: int = 20) -> PreparedGraphData:
    """Join explicit foreign keys; never infer policy or support relationships."""
    if limit < 1 or limit > 1000:
        raise ValueError("limit must be between 1 and 1000")
    root = Path(data_dir)
    tables, hashes = {}, {}
    specs = {
        "Products": ("ProductID", {"ProductID", "ProductName", "SupplierID", "CategoryID"}),
        "Suppliers": ("SupplierID", {"SupplierID", "CompanyName"}),
        "Categories": ("CategoryID", {"CategoryID", "CategoryName"}),
    }
    for name, (key, fields) in specs.items():
        path = root / f"{name}.csv"
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if not fields.issubset(reader.fieldnames or []):
                raise ValueError(f"Missing catalog columns in {name}")
            rows = {}
            for row in reader:
                # Discard every non-allowlisted column immediately. Contact
                # details, prices, inventory, noisy descriptions are excluded.
                item = {field: (row.get(field) or "").strip() for field in fields}
                if any(not value or len(value) > 200 or any(ord(char) < 32 for char in value)
                       for value in item.values()):
                    raise ValueError(f"Blank, oversized, or multiline catalog field in {name}")
                for field, value in item.items():
                    if field.endswith("ID") and (not value.isascii() or not value.isdigit() or int(value) < 1):
                        raise ValueError(f"Invalid catalog identifier in {name}")
                    if field.endswith("ID"):
                        item[field] = str(int(value))
                if item[key] in rows:
                    raise ValueError(f"Duplicate identifier in {name}")
                rows[item[key]] = item
        tables[name] = rows
    # Validate the full dataset before sampling. Broken references must not
    # silently become invented supplier/category nodes.
    for product in tables["Products"].values():
        if product["SupplierID"] not in tables["Suppliers"] or product["CategoryID"] not in tables["Categories"]:
            raise ValueError("Product has an unresolved supplier/category foreign key")
    nodes, relationships, documents = {}, [], {}
    selected = sorted(tables["Products"].values(), key=lambda row: int(row["ProductID"]))[:limit]
    for product in selected:
        supplier = tables["Suppliers"][product["SupplierID"]]
        category = tables["Categories"][product["CategoryID"]]
        pid, sid, cid = (f"Product:{product['ProductID']}",
                         f"Supplier:{supplier['SupplierID']}", f"Category:{category['CategoryID']}")
        for identifier, label, name, source in [
            (pid, "Product", product["ProductName"], "Products.csv"),
            (sid, "Supplier", supplier["CompanyName"], "Suppliers.csv"),
            (cid, "Category", category["CategoryName"], "Categories.csv"),
        ]:
            nodes[identifier] = {"id": identifier, "label": label, "name": name,
                                 "source": source, "source_sha256": hashes[source]}
        relationships.extend([
            {"source": pid, "type": "SUPPLIED_BY", "target": sid, "source_column": "Products.SupplierID"},
            {"source": pid, "type": "BELONGS_TO", "target": cid, "source_column": "Products.CategoryID"},
        ])
        documents[f"product-{product['ProductID']}.txt"] = (
            f"Catalog source: Products.csv; product key: {pid}.\n"
            f"Product {product['ProductName']} ({pid}) is supplied by "
            f"{supplier['CompanyName']} ({sid}).\n"
            f"Product {product['ProductName']} ({pid}) belongs to category "
            f"{category['CategoryName']} ({cid}).\n"
            "These are catalog relationships, not evidence of current availability, "
            "return eligibility, customer experience, or revenue.\n"
        )
    return PreparedGraphData(
        nodes=list(nodes.values()), relationships=relationships, documents=documents,
        source_counts={name: len(rows) for name, rows in tables.items()}, source_hashes=hashes,
    )


def export_catalog(prepared: PreparedGraphData, destination: str | Path) -> dict:
    """Create a new review workspace. Refuse to replace existing artifacts."""
    target = Path(destination)
    target.mkdir(parents=True, exist_ok=False)
    (target / "input").mkdir()
    for filename, text in prepared.documents.items():
        (target / "input" / filename).write_text(text, encoding="utf-8")
    manifest = {
        "kind": "catalog_preparation_only", "index_built": False,
        "nodes": len(prepared.nodes), "relationships": len(prepared.relationships),
        "documents": len(prepared.documents), "source_counts": prepared.source_counts,
        "source_sha256": prepared.source_hashes,
        "excluded": ["contact_details", "customer_records", "employees", "orders", "reviews",
                     "unreviewed_descriptions", "price_and_stock"],
        "review_required_before_cloud_indexing": True,
        "review_preparation": prepared.review_report,
    }
    if prepared.review_report is not None:
        manifest["excluded"].remove("reviews")
        manifest["excluded"].append("customer_linkage_in_reviews")
        (target / "review_sources.json").write_text(
            json.dumps(prepared.review_report["document_sources"], indent=2) + "\n", encoding="utf-8",
        )
    for filename, value in [("nodes.json", prepared.nodes), ("relationships.json", prepared.relationships),
                            ("manifest.json", manifest)]:
        (target / filename).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return manifest
