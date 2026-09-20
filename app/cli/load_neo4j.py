"""Validate local CSV data; import only with explicit --execute and loader credentials."""
import argparse
import json

from app.core.config import Settings
from app.services.neo4j_data import load_business_graph, prepare_business_graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    settings = Settings()
    graph = prepare_business_graph(settings.business_data_dir)
    report = {"source_counts": graph.source_counts, "nodes": len(graph.nodes), "edges": len(graph.edges), "imported": False}
    if args.execute:
        if not settings.neo4j_load_user or not settings.neo4j_load_password:
            parser.error("Set dedicated NEO4J_LOAD_USER and NEO4J_LOAD_PASSWORD locally first.")
        from neo4j import GraphDatabase
        try:
            with GraphDatabase.driver(settings.neo4j_uri, auth=(settings.neo4j_load_user, settings.neo4j_load_password.get_secret_value())) as driver:
                load_business_graph(driver, database=settings.neo4j_database, dataset=settings.neo4j_dataset, graph=graph)
        except Exception:
            parser.exit(1, "Neo4j import failed; no snapshot was published. Check local connectivity and loader permissions.\n")
        report["imported"] = True
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
