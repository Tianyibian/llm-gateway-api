import csv

import pytest

from app.services.graph_data import prepare_catalog, export_catalog
from app.services.graph_review_data import prepare_review_corpus
from tests.test_graph_data import catalog


def reviews(path, texts):
    with (path / "Reviews.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ReviewID", "ProductID", "CustomerID", "Rating", "ReviewDate", "ReviewText"])
        for index, text in enumerate(texts, 1):
            writer.writerow([index, 1, "SECRET-CUSTOMER", 4, "2025-01-02", text])


def test_reviews_keep_every_source_and_no_customer_columns(catalog):
    reviews(catalog, ["Pairing was easy.", "The battery life is short."])
    data = prepare_review_corpus(catalog, prepare_catalog(catalog))
    assert data.review_report["selected_review_count"] == 2
    assert data.review_report["review_document_count"] == 1
    assert "SECRET-CUSTOMER" not in str(data)
    source = next(iter(data.review_report["document_sources"].values()))
    assert source["review_ids"] == ["Review:1", "Review:2"]
    review_text = next(text for filename, text in data.documents.items() if filename.startswith("reviews-"))
    assert "unverified opinion" in review_text and "Supplier:" in review_text
    assert "Manufacturer" not in review_text


def test_group_size_is_applied_even_when_grouping_by_category(catalog):
    reviews(catalog, ["Works well."] * 11)
    data = prepare_review_corpus(catalog, prepare_catalog(catalog), group_size=5)
    groups = data.review_report["document_sources"].values()
    assert sorted(len(group["review_ids"]) for group in groups) == [1, 5, 5]


def test_character_budget_is_hard_and_does_not_truncate_reviews(catalog):
    reviews(catalog, ["The battery is good. " * 8] * 4 + ["x" * 2000])
    data = prepare_review_corpus(catalog, prepare_catalog(catalog), max_chars=700)
    assert data.review_report["selected_review_count"] == 4
    assert data.review_report["skipped"]["oversized_review_requires_review"] == 1
    assert all(len(text) <= 700 for filename, text in data.documents.items() if filename.startswith("reviews-"))


@pytest.mark.parametrize("text", ["Contact me at person@example.com", "Call +1 (212) 555-0189", "Visit https://private.example"])
def test_obvious_contact_like_text_is_quarantined_not_claimed_anonymized(catalog, text):
    reviews(catalog, [text])
    data = prepare_review_corpus(catalog, prepare_catalog(catalog))
    assert data.review_report["selected_review_count"] == 0
    assert data.review_report["human_privacy_and_quality_review_required"] is True


def test_review_preparation_is_stable_and_limit_is_explicit(catalog):
    reviews(catalog, ["Works well."] * 10)
    data = prepare_review_corpus(catalog, prepare_catalog(catalog), limit=3)
    assert data == prepare_review_corpus(catalog, prepare_catalog(catalog), limit=3)
    assert data.review_report["selected_review_count"] == 3
    assert data.review_report["eligible_not_selected_due_to_limit"] == 7
    manifest = export_catalog(data, catalog / "review-seed")
    assert manifest["review_preparation"]["budget_unit"] == "characters_not_model_tokens"
    assert (catalog / "review-seed/review_sources.json").exists()


def test_fractional_ratings_are_valid_and_not_rounded(catalog):
    reviews(catalog, ["The sensor works well."])
    path = catalog / "Reviews.csv"
    path.write_text(path.read_text().replace(",4,2025-01-02", ",4.8,2025-01-02"))
    data = prepare_review_corpus(catalog, prepare_catalog(catalog))
    assert data.review_report["selected_review_count"] == 1
    assert any("Rating: 4.8 out of 5" in text for text in data.documents.values())


def test_orphan_review_product_reference_is_rejected(catalog):
    reviews(catalog, ["Works well."])
    path = catalog / "Reviews.csv"
    path.write_text(path.read_text().replace("1,1,SECRET", "1,999,SECRET"))
    with pytest.raises(ValueError, match="foreign key"):
        prepare_review_corpus(catalog, prepare_catalog(catalog))
