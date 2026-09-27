"""Bounded, database-grounded lexical resolution; no model-authored entity IDs."""
from dataclasses import dataclass
import re
import unicodedata


def tokens(text: str) -> tuple[str, ...]:
    words = re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", text).casefold())
    # Conservative regular English plural equivalence, not stemming arbitrary
    # prefixes or a product-specific alias table. Irregular synonyms need curation.
    def singular(word):
        if len(word) > 4 and word.endswith("ies"):
            return word[:-3] + "y"
        if len(word) > 4 and word.endswith(("ches", "shes", "xes", "zes")):
            return word[:-2]
        if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
            return word[:-1]
        return word
    return tuple(singular(word) for word in words)


@dataclass(frozen=True)
class EntityBinding:
    filter_index: int
    surface: str
    label: str
    entity_id: str
    name: str
    dataset: str


def candidates(surface: str, rows: list[dict]) -> list[dict]:
    """Exact full names win; otherwise match complete normalized token sequences."""
    wanted = tokens(surface)
    if not wanted:
        return []
    exact = [row for row in rows if tokens(row["name"]) == wanted]
    if exact:
        return exact
    return [row for row in rows if any(
        tokens(row["name"])[i:i + len(wanted)] == wanted
        for i in range(len(tokens(row["name"])) - len(wanted) + 1)
    )]
