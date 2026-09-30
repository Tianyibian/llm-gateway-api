"""Live-schema boundaries with deterministic metadata and model doubles."""
import asyncio
from dataclasses import replace
from unittest.mock import AsyncMock

import pytest

from app.services.errors import LLMConfigurationError
from app.services.neo4j_schema import (
    ALLOWED_PROPERTIES, MAX_SCHEMA_ROWS, ObservedGraphSchema, metadata_queries,
)
from tests.test_neo4j_queries import service
from tests.test_cypher_subgraph import executor_with_session


def node(label="Product", **extra):
    props = {**ALLOWED_PROPERTIES[label], **extra}
    return {"labels": [label], "properties": [
        {"property": name, "type": kind + " NOT NULL"} for name, kind in props.items()
    ]}


def test_metadata_never_selects_record_values_or_global_procedures():
    for query in metadata_queries("isolated"):
        assert query.parameters["dataset"] == "isolated"
        assert "$dataset" in query.cypher and "LIMIT $limit" in query.cypher
        assert "CALL " not in query.cypher and "RETURN n" not in query.cypher
        assert "CREATE" not in query.cypher and "DELETE" not in query.cypher


def test_unknown_labels_properties_and_edges_are_not_exposed():
    schema = ObservedGraphSchema.from_rows("test", [
        node(email="STRING", password="STRING"), node("Supplier"),
        {"labels": ["Customer"], "properties": [{"property": "secret", "type": "STRING NOT NULL"}]},
    ], [
        {"sources": ["Product"], "relation": "SUPPLIED_BY", "targets": ["Supplier"]},
        {"sources": ["Product"], "relation": "PRIVATE", "targets": ["Supplier"]},
    ])
    text = schema.prompt_text()
    assert all(value not in text for value in ("email", "password", "Customer", "secret", "PRIVATE"))
    assert "SUPPLIED_BY" in text and "STRING" in text
    assert "OrderLine" not in schema.properties


@pytest.mark.parametrize("rows", [[], [node()] * (MAX_SCHEMA_ROWS + 1),
                                     [{"labels": ["Product"], "properties": []}]])
def test_empty_incomplete_or_overflow_metadata_fails_closed(rows):
    with pytest.raises(LLMConfigurationError):
        ObservedGraphSchema.from_rows("test", rows, [])


def test_wrong_or_mixed_property_type_fails_closed():
    with pytest.raises(LLMConfigurationError):
        ObservedGraphSchema.from_rows("test", [node(), node(name="INTEGER")], [])


@pytest.mark.parametrize("strategy", ["template", "text_to_cypher"])
def test_model_stages_receive_same_observed_schema(strategy):
    svc, selector, chain, executor = service()
    result = asyncio.run(svc.query("Who supplies Acme Sensor?", strategy=strategy))
    assert result.status == "complete"
    actual = executor.schema.return_value.prompt_text()
    model = selector if strategy == "template" else chain
    assert model.ainvoke.await_args.args[0]["schema"] == actual
    assert svc.reviewer.ainvoke.await_args.args[0]["schema"] == actual
    executor.schema.assert_awaited_once()


@pytest.mark.parametrize("strategy", ["template", "text_to_cypher"])
@pytest.mark.parametrize("missing", ["label", "property", "edge"])
def test_missing_structure_stops_before_review_and_data_execution(strategy, missing):
    svc, _, _, executor = service()
    schema = executor.schema.return_value
    properties = {label: dict(props) for label, props in schema.properties.items()}
    if missing == "label":
        properties.pop("Supplier")
    elif missing == "property":
        properties["Supplier"].pop("name")
    executor.schema.return_value = replace(schema, properties=properties,
        edges=frozenset() if missing == "edge" else schema.edges)
    result = asyncio.run(svc.query("Who supplies Acme Sensor?", strategy=strategy))
    assert result.status == "rejected"
    svc.reviewer.ainvoke.assert_not_awaited()
    assert executor.run.await_count == 1  # Snapshot check only.


@pytest.mark.parametrize("failure", ["offline", "wrong_dataset"])
def test_schema_failure_does_not_fall_back_to_static_prompt(failure):
    svc, selector, chain, executor = service()
    if failure == "offline":
        executor.schema.side_effect = RuntimeError("private connection details")
    else:
        executor.schema.return_value = replace(executor.schema.return_value, dataset="other")
    result = asyncio.run(svc.query("Who supplies Acme Sensor?"))
    assert result.status == "unavailable"
    assert "private" not in result.model_dump_json()
    selector.ainvoke.assert_not_awaited()
    chain.ainvoke.assert_not_awaited()


def test_schema_is_refreshed_per_query_not_shared_or_cached():
    svc, _, _, executor = service()
    schema = executor.schema.return_value
    executor.schema.side_effect = [schema, replace(schema, edges=frozenset())]
    executor.run.side_effect = None
    executor.run.return_value = ([], {})
    assert asyncio.run(svc.query("Who supplies Acme Sensor?")).status == "complete"
    assert asyncio.run(svc.query("Who supplies Acme Sensor?")).status == "rejected"
    assert executor.schema.await_count == 2


def test_executor_uses_guarded_dataset_scoped_metadata_queries():
    executor, _ = executor_with_session("r")
    executor.run = AsyncMock(side_effect=[([node()], {}), ([], {})])
    schema = asyncio.run(executor.schema())
    assert schema.dataset == "test" and set(schema.properties) == {"Product"}
    for call in executor.run.await_args_list:
        assert call.args[0].parameters["dataset"] == "test"
