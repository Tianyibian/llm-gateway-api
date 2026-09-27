import csv

import pytest

from app.services.graph_data import prepare_catalog, export_catalog
from app.services.merged_review_data import prepare_merged_reviews
from graphrag_runtime.worker import prepare, check_inputs
from tests.test_graph_data import catalog


def block(content="Easy pairing.", customer="PRIVATE-CUSTOMER", product="Acme Sensor", category="Sensor"):
    return (
        f"Customer ID: {customer}\nCustomer Company: SECRET COMPANY\n"
        "Customer Location: PRIVATE CITY\n"
        f"Product Info: {product} (Category: {category})\nManufacturer: Acme Supply\n"
        "Product Price: 9876.54\nRating: 4.8 stars\nReview Date: 2026-06-08\n"
        f'Review Content: "{content}"'
    )


def merged(path, rows):
    source = path / "merged_reviews.csv"
    with source.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "text", "token_count", "CategoryName"])
        writer.writerows(rows)
    return source


def run(path, rows, **kwargs):
    # Deterministic counting seam for unit tests, not the production tokenizer.
    return prepare_merged_reviews(merged(path, rows), prepare_catalog(path), count_tokens=len, **kwargs)


def test_allowlist_canonical_binding_and_source_locators(catalog):
    result = run(catalog, [["first", block(product="AcmeSensor"), 1, "Sensor"]])
    text = next(iter(result.documents.values()))
    assert "Product:1" in text and "4.8 out of 5" in text
    for secret in ["PRIVATE-CUSTOMER", "SECRET COMPANY", "PRIVATE CITY", "9876.54", "Manufacturer:"]:
        assert secret not in str(result)
    source = next(iter(result.review_report["document_sources"].values()))["reviews"][0]
    assert source["source_locator"] == "MergedReview:first:1"
    assert result.review_report["original_review_ids_available"] is False
    assert result.review_report["largest_document_tokens"] == len(text)


def test_actual_budget_not_legacy_count_and_no_truncation(catalog):
    result = run(catalog, [[1, block("X" * 900), 1, "Sensor"]], max_tokens=500)
    assert not result.documents
    assert result.review_report["skipped"] == {"oversized_review_requires_review": 1}


def test_group_size_and_all_provenance(catalog):
    rows = [[i, block(customer=f"c{i}"), 99999, "Sensor"] for i in range(6)]
    result = run(catalog, rows, group_size=1, limit=None)
    assert len(result.documents) == 6
    assert result.review_report["selected_review_count"] == 6
    assert result == run(catalog, rows, group_size=1, limit=None)


def test_deduplicate_only_identical_source_blocks(catalog):
    result = run(catalog, [[1, block() + "<ROW_SEP>" + block(), 1, "Sensor"],
                           [2, block(customer="another"), 1, "Sensor"]])
    assert result.review_report["selected_review_count"] == 2
    assert result.review_report["skipped"] == {"duplicate_source_block": 1}


@pytest.mark.parametrize("content", ["", "Email me at person@example.com", "Call +1 (212) 555-0189"])
def test_empty_or_contact_content_excluded(catalog, content):
    result = run(catalog, [[1, block(content), 1, "Sensor"]])
    assert result.review_report["selected_review_count"] == 0


@pytest.mark.parametrize("text,category", [
    (block(product="Unknown"), "Sensor"), (block(), "Unknown"),
    (block().replace("Acme Supply", "Other"), "Sensor"),
    (block().replace("4.8 stars", "8 stars"), "Sensor"),
    (block().replace("2026-06-08", "yesterday"), "Sensor"),
    (block() + "\nProduct Info: other", "Sensor"),
    (block().replace('"Easy pairing."', "unquoted"), "Sensor"),
])
def test_malformed_or_conflicting_source_fails_closed(catalog, text, category):
    with pytest.raises(ValueError):
        run(catalog, [[1, text, 1, category]])


def test_duplicate_csv_id_rejected(catalog):
    with pytest.raises(ValueError, match="duplicate"):
        run(catalog, [[1, block(), 1, "Sensor"], [1, block(), 1, "Sensor"]])


def test_round_robin_covers_products_before_second_review(catalog):
    with (catalog / "Products.csv").open("a") as handle:
        handle.write("2,Acme Plug,7,2,10\n")
    rows = [[i, block(customer=f"c{i}"), 1, "Sensor"] for i in range(4)]
    rows.append([5, block(product="Acme Plug"), 1, "Sensor"])
    result = run(catalog, rows, limit=2)
    assert result.review_report["selected_product_count"] == 2
    assert result.review_report["eligible_not_selected_due_to_limit"] == 3


def test_group_token_limit_counts_separator(catalog):
    rows = [[i, block(customer=f"c{i}"), 1, "Sensor"] for i in range(7)]
    result = run(catalog, rows, limit=None, group_size=5, max_tokens=800)
    assert all(len(text) <= 800 for text in result.documents.values())
    assert sum(len(source["reviews"]) for source in result.review_report["document_sources"].values()) == 7


def test_preparation_preserves_provenance_and_existing_workspace(catalog):
    result = run(catalog, [[1, block(), 1, "Sensor"]])
    seed, root = catalog / "seed", catalog / "runtime"
    export_catalog(result, seed)
    prepare(root, seed)
    assert check_inputs(root)["provenance_sha256"]["review_sources.json"]
    with pytest.raises(FileExistsError):
        prepare(root, seed)
    (root / "review_sources.json").write_text("{}")
    with pytest.raises(ValueError, match="provenance"):
        check_inputs(root)


def test_explicit_corpus_limits_required(catalog):
    result = run(catalog, [[i, block(customer=str(i)), 1, "Sensor"] for i in range(65)], group_size=1)
    seed, root = catalog / "seed", catalog / "runtime"
    export_catalog(result, seed)
    with pytest.raises(ValueError, match="limit exceeded"):
        prepare(root, seed)
    assert not root.exists()
    prepare(root, seed, max_documents=65)
    assert len(check_inputs(root)["documents"]) == 65


def test_original_review_id_disambiguates_same_name_products(catalog):
    with (catalog / "Products.csv").open("a") as handle:
        handle.write("2,Acme Sensor,7,2,10\n")
    original = catalog / "Reviews.csv"
    original.write_text("ReviewID,ProductID,CustomerID,Rating,ReviewDate,ReviewText\n42,2,PRIVATE-CUSTOMER,4.8,2026-06-08,Easy pairing.\n")
    result = run(catalog, [[1, block(), 1, "Sensor"]], original_reviews=original)
    text = next(iter(result.documents.values()))
    assert "Product:2" in text and "Review:42" in text
    assert "PRIVATE-CUSTOMER" not in str(result)
    with pytest.raises(ValueError, match="Ambiguous"):
        run(catalog, [[1, block(), 1, "Sensor"]])
    with pytest.raises(ValueError, match="uniquely match"):
        run(catalog, [[1, block("Changed content"), 1, "Sensor"]], original_reviews=original)
