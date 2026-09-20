"""Map GraphRAG's actual retrieved context back to persisted artifact IDs."""
import re
from collections import Counter


def normalize_id(value):
    return str(int(value)) if isinstance(value, float) and value.is_integer() else str(value)


def context_rows(context, key):
    if isinstance(context, dict):
        rows = context.get(key, [])
        if isinstance(rows, list):
            yield from (row for row in rows if isinstance(row, dict) and "id" in row)
        for name, child in context.items():
            if name != key and isinstance(child, (dict, list)):
                yield from context_rows(child, key)
    elif isinstance(context, list):
        for child in context:
            yield from context_rows(child, key)


def candidates(context, tables, answer, *, limit=5):
    entities = tables["entities"]
    reports = {normalize_id(row["community"]): row for row in tables["community_reports"] if row.get("level") == 0}
    sources = {normalize_id(row["human_readable_id"]): row for row in tables["text_units"]}
    allowed_types = {kind.casefold(): kind for kind in ["Product", "Supplier", "Category", "SupportTopic"]}
    output, seen = [], set()
    # Prefer primary text-unit evidence when present, then derived reports.
    for key, lookup, field, kind, label in [
        ("sources", sources, "text", "text_unit", "Sources"),
        ("reports", reports, "full_content", "community_report", "Reports"),
    ]:
        rows = list(context_rows(context, key))
        cited = Counter()
        for block in re.findall(rf"{label}\s*\(([^)]*)\)", str(answer)):
            cited.update(re.findall(r"\d+", block))
        rows.sort(key=lambda row: -cited[normalize_id(row["id"])])
        for row in rows:
            source = lookup.get(normalize_id(row["id"]))
            if not source:
                continue
            source_id = f"microsoft_graphrag:{kind}:{source['id']}"
            if source_id in seen:
                continue
            seen.add(source_id)
            prefix = "Indexed review/catalog excerpt: " if kind == "text_unit" else "Generated community report from the sampled corpus: "
            original = str(source.get(field) or source.get("summary") or "")
            if not original.strip():
                continue
            # Truncation is explicit, never presented as a complete document.
            budget = 2000 - len(prefix)
            text = prefix + (original if len(original) <= budget else original[:budget - 16] + " [excerpt ends]")
            names = []
            for entity in entities:
                name = str(entity.get("title", ""))
                entity_type = allowed_types.get(str(entity.get("type", "")).casefold())
                if name and len(name) <= 200 and entity_type and name.casefold() in text.casefold():
                    names.append({"text": name, "entity_type": entity_type})
                if len(names) == 12:
                    break
            output.append({"source_id": source_id, "text": text, "entities": names,
                           "source_document_ids": [source["document_id"]] if source.get("document_id") else [],
                           "context_id": normalize_id(row["id"]), "source_kind": kind})
            if len(output) >= limit:
                return output
    return output


def validate_citations(answer, context):
    """Validate reference IDs, not the semantic truth of generated claims."""
    known = {label: {normalize_id(row["id"]) for row in context_rows(context, key)}
             for label, key in [("Sources", "sources"), ("Reports", "reports"),
                                ("Entities", "entities"), ("Relationships", "relationships"), ("Claims", "claims")]}
    blocks = re.findall(r"\[Data:\s*(.*?)\]", str(answer), flags=re.S)
    invalid = []
    for block in blocks:
        for part in block.split(";"):
            match = re.fullmatch(r"\s*(Sources|Reports|Entities|Relationships|Claims)\s*\(([^)]*)\)\s*", part)
            if not match:
                invalid.append("malformed_reference")
                continue
            label, values = match.groups()
            ids = [item.strip() for item in values.split(",") if item.strip() != "+more"]
            invalid.extend(f"{label}:{item}" for item in ids if item not in known[label])
    return {"valid": bool(blocks) and not invalid, "invalid_references": invalid,
            "checked": "reference_ids_only_not_claim_entailment"}
