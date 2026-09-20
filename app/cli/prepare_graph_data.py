"""Prepare a small, local catalog graph seed and GraphRAG text input for review."""
import argparse
import json

from app.services.graph_data import export_catalog, prepare_catalog
from app.services.graph_review_data import prepare_review_corpus


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="Business_data")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--include-reviews", action="store_true", help="Prepare review text locally; human review required before cloud indexing")
    parser.add_argument("--review-limit", type=int, default=100)
    parser.add_argument("--review-group-size", type=int, default=5)
    parser.add_argument("--review-max-chars", type=int, default=6000)
    parser.add_argument("--output", required=True, help="New directory; existing directories are never overwritten")
    args = parser.parse_args()
    prepared = prepare_catalog(args.data_dir, limit=args.limit)
    if args.include_reviews:
        prepared = prepare_review_corpus(args.data_dir, prepared, limit=args.review_limit,
                                         group_size=args.review_group_size, max_chars=args.review_max_chars)
    manifest = export_catalog(prepared, args.output)
    if manifest["review_preparation"] is not None:
        manifest = {**manifest, "review_preparation": {
            key: value for key, value in manifest["review_preparation"].items() if key != "document_sources"
        }}
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
