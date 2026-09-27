"""Prepare token-bounded review documents locally; this does not call an LLM."""
import argparse
import json
from pathlib import Path

from app.services.graph_data import export_catalog, prepare_catalog
from app.services.merged_review_data import prepare_merged_reviews


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="Business_data/merged_reviews.csv")
    parser.add_argument("--data-dir", default="Business_data")
    parser.add_argument("--output", required=True, help="New directory; existing outputs are never overwritten")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--limit", type=int, default=100)
    selection.add_argument("--all", action="store_true", help="Prepare every eligible review without sampling")
    parser.add_argument("--group-size", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=1000)
    args = parser.parse_args()
    # Use the same tokenizer as the pinned official GraphRAG chunker. This CLI
    # runs in .venv-graphrag; the FastAPI environment needs no new dependency.
    from graphrag.tokenizer.get_tokenizer import get_tokenizer
    tokenizer = get_tokenizer(encoding_model="o200k_base")
    prepared = prepare_merged_reviews(
        args.source, prepare_catalog(args.data_dir, limit=1000),
        count_tokens=lambda text: len(tokenizer.encode(text)),
        limit=None if args.all else args.limit, group_size=args.group_size, max_tokens=args.max_tokens,
        original_reviews=Path(args.data_dir) / "Reviews.csv",
    )
    manifest = export_catalog(prepared, args.output)
    report = {key: value for key, value in manifest["review_preparation"].items() if key != "document_sources"}
    print(json.dumps({"output": args.output, "index_built": False, **report}, indent=2))


if __name__ == "__main__":
    main()
