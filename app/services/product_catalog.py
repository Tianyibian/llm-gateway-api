from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
import re


@dataclass(frozen=True, kw_only=True)
class ProductRecord:
    """An immutable, keyword-only product value loaded from the CSV catalog."""

    product_id: int
    product_name: str
    category: str
    supplier: str
    quantity_per_unit: str
    unit_price: float
    units_in_stock: int
    units_on_order: int
    discontinued: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class ProductCatalog:
    """Small in-memory lexical search over the local demonstration CSV files."""

    _TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
    _STOP_WORDS = {
        "a",
        "about",
        "all",
        "and",
        "are",
        "available",
        "can",
        "catalog",
        "do",
        "does",
        "for",
        "have",
        "i",
        "in",
        "information",
        "is",
        "me",
        "of",
        "on",
        "please",
        "product",
        "products",
        "show",
        "tell",
        "the",
        "this",
        "what",
        "which",
        "with",
        "you",
    }

    def __init__(self, data_dir: str | Path) -> None:
        directory = Path(data_dir)
        if not directory.is_absolute():
            directory = Path(__file__).resolve().parents[2] / directory
        self.data_dir = directory
        self.source_name = "Business_data/Products.csv"
        self._products = self._load_products()

    @staticmethod
    def _read_lookup(path: Path, key: str, value: str) -> dict[int, str]:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            return {
                int(row[key]): row[value].strip()
                for row in csv.DictReader(handle)
            }

    def _load_products(self) -> list[ProductRecord]:
        categories = self._read_lookup(
            self.data_dir / "Categories.csv",
            "CategoryID",
            "CategoryName",
        )
        suppliers = self._read_lookup(
            self.data_dir / "Suppliers.csv",
            "SupplierID",
            "CompanyName",
        )
        with (self.data_dir / "Products.csv").open(
            encoding="utf-8-sig",
            newline="",
        ) as handle:
            return [
                ProductRecord(
                    product_id=int(row["ProductID"]),
                    product_name=row["ProductName"].strip(),
                    category=categories.get(int(row["CategoryID"]), "Unknown"),
                    supplier=suppliers.get(int(row["SupplierID"]), "Unknown"),
                    quantity_per_unit=row["QuantityPerUnit"].strip(),
                    unit_price=float(row["UnitPrice"]),
                    units_in_stock=int(row["UnitsInStock"]),
                    units_on_order=int(row["UnitsOnOrder"]),
                    discontinued=bool(int(row["Discontinued"])),
                )
                for row in csv.DictReader(handle)
            ]

    @classmethod
    def _tokens(cls, text: str) -> set[str]:
        tokens: set[str] = set()
        for token in cls._TOKEN_PATTERN.findall(text.lower()):
            if token in cls._STOP_WORDS:
                continue
            tokens.add(token)
            if len(token) > 4 and token.endswith("s"):
                tokens.add(token[:-1])
        return tokens

    def search(self, query: str, *, limit: int = 5) -> list[ProductRecord]:
        query_text = " ".join(query.lower().split())
        query_tokens = self._tokens(query_text)
        if not query_tokens:
            return []

        mentioned_categories = {
            category
            for category in {product.category for product in self._products}
            if self._tokens(category) <= query_tokens
        }
        mentioned_suppliers = {
            supplier
            for supplier in {product.supplier for product in self._products}
            if self._tokens(supplier) <= query_tokens
        }

        scored: list[tuple[int, ProductRecord]] = []
        for product in self._products:
            if mentioned_categories and product.category not in mentioned_categories:
                continue
            if mentioned_suppliers and product.supplier not in mentioned_suppliers:
                continue
            name = product.product_name.lower()
            name_tokens = self._tokens(name)
            category_tokens = self._tokens(product.category)
            supplier_tokens = self._tokens(product.supplier)

            score = 0
            if name in query_text:
                score += 100
            score += 12 * len(query_tokens & name_tokens)
            score += 7 * len(query_tokens & category_tokens)
            score += 5 * len(query_tokens & supplier_tokens)
            if score:
                scored.append((score, product))

        scored.sort(key=lambda item: (-item[0], item[1].product_name))
        return [product for _, product in scored[:limit]]
