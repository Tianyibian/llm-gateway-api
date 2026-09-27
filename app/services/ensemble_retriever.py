"""Small-corpus BM25 and reciprocal-rank fusion, independent of LLM providers."""
from collections import Counter
from math import log
import re
import unicodedata


def tokenize(text: str) -> list[str]:
    """Normalize words/model numbers; split CJK characters without translation."""
    text = unicodedata.normalize("NFKC", text).casefold()
    return re.findall(r"[\u3400-\u9fff]|[^\W_\u3400-\u9fff]+", text)


def bm25_rank(query: str, documents: dict[str, str], *, limit: int) -> list[tuple[str, float]]:
    """Okapi BM25 with positive IDF, k1=1.5 and b=0.75; zero matches are excluded."""
    terms = set(tokenize(query))
    counts = {key: Counter(tokenize(text)) for key, text in documents.items()}
    lengths = {key: sum(count.values()) for key, count in counts.items()}
    average = sum(lengths.values()) / len(counts) if counts else 0
    if not terms or not average:
        return []
    frequencies = Counter(term for count in counts.values() for term in terms if term in count)
    scores = []
    for key, count in counts.items():
        score = 0.0
        for term in sorted(terms & count.keys()):
            frequency = count[term]
            idf = log(1 + (len(counts) - frequencies[term] + 0.5) / (frequencies[term] + 0.5))
            score += idf * frequency * 2.5 / (frequency + 1.5 * (0.25 + 0.75 * lengths[key] / average))
        if score > 0:
            scores.append((key, score))
    return sorted(scores, key=lambda item: (-item[1], item[0]))[:limit]


def reciprocal_rank_fusion(*rankings: list[str], limit: int, constant: int = 60) -> list[tuple[str, float]]:
    """Equal-weight RRF. Deduplicate within and across lists by stable chunk ID."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, key in enumerate(dict.fromkeys(ranking), start=1):
            scores[key] = scores.get(key, 0.0) + 1.0 / (constant + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]
