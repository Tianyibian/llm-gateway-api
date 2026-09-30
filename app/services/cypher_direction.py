"""Use observed Neo4j triples for direction; endpoint permissions stay explicit."""
import re


# Permission to traverse a relationship does not prescribe its stored direction.
ALLOWED_ENDPOINTS = {
    "SUPPLIED_BY": frozenset({"Product", "Supplier"}),
    "BELONGS_TO": frozenset({"Product", "Category"}),
    "CONTAINS": frozenset({"Order", "OrderLine"}),
    "OF_PRODUCT": frozenset({"OrderLine", "Product"}),
}


def permitted(source, relation, target):
    return ALLOWED_ENDPOINTS.get(relation) == frozenset({source, target})


def observed_arrow(left, relation, right, observed_edges):
    if not permitted(left, relation, right):
        raise ValueError("Relationship endpoints are outside the read contract")
    forward = (left, relation, right) in observed_edges
    reverse = (right, relation, left) in observed_edges
    if forward == reverse:
        raise ValueError("Relationship direction is missing or ambiguous in Neo4j")
    return f"-[:{relation}]->" if forward else f"<-[:{relation}]-"


# Only matches the server compiler/templates' labeled, one-hop patterns.
# Lookahead retains overlapping links in chained patterns (including <- links).
PATTERN = re.compile(
    r"(?=\((?P<left>\w+):(?P<left_label>\w+)\)"
    r"(?P<arrow>-\[:(?P<forward>\w+)\]->|<-\[:(?P<reverse>\w+)\]-)"
    r"\((?P<right>\w+):(?P<right_label>\w+)\))"
)


def relationship_patterns(cypher):
    matches = list(PATTERN.finditer(cypher))
    if len(matches) != cypher.count("[:"):
        raise ValueError("Unsupported relationship syntax in server-compiled query")
    return matches
