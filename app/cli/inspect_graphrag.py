"""Preview GraphRAG Parquet tables without modifying the index or calling an LLM."""
from __future__ import annotations

import argparse
from pathlib import Path


PREVIEW_COLUMNS = {
    "documents": ["id", "title", "text"],
    "text_units": ["id", "text", "document_ids"],
    "entities": ["title", "type", "description"],
    "relationships": ["source", "target", "description", "weight"],
    "communities": ["community", "level", "title", "size"],
    "community_reports": ["community", "title", "summary"],
}


def preview(root: str | Path, table: str, *, limit: int = 5,
            columns: list[str] | None = None, schema_only: bool = False) -> str:
    if table not in PREVIEW_COLUMNS or not 1 <= limit <= 100:
        raise ValueError("Choose a known GraphRAG table and a limit from 1 to 100")
    import pyarrow.parquet as pq

    # ParquetFile only reads. Never save a DataFrame back into the index.
    with pq.ParquetFile(Path(root) / "output" / f"{table}.parquet") as source:
        names = source.schema_arrow.names
        header = f"{table}: {source.metadata.num_rows} rows\nColumns: {', '.join(names)}"
        if schema_only:
            return header + "\n" + str(source.schema_arrow)
        selected = columns if columns is not None else [c for c in PREVIEW_COLUMNS[table] if c in names]
        if not selected or any(c not in names for c in selected):
            raise ValueError("Unknown or empty column selection; inspect --schema first")
        batch = next(source.iter_batches(batch_size=limit, columns=selected), None)
        if batch is None:
            return header + "\nNo rows."
        return header + "\n" + batch.to_pandas().to_string(index=False, max_colwidth=100)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="GraphRAG workspace containing output/")
    parser.add_argument("--table", choices=PREVIEW_COLUMNS, default="entities")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--columns", nargs="+", help="Optional exact column names")
    parser.add_argument("--schema", action="store_true", help="Show column types without loading data rows")
    args = parser.parse_args()
    try:
        print(preview(args.root, args.table, limit=args.limit,
                      columns=args.columns, schema_only=args.schema))
    except ImportError:
        parser.exit(1, "Use .venv-graphrag/bin/python; PyArrow and pandas are required.\n")
    except (OSError, ValueError):
        parser.exit(1, "Cannot preview this table. Check the path, file integrity, limit and column names.\n")


if __name__ == "__main__":
    main()
