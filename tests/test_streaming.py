from decimal import Decimal
import json

import pytest

from app.services.streaming import encode_sse


def test_snowflake_decimal_rows_are_json_serializable_without_precision_loss():
    result = encode_sse("sources", {"rows":[{"revenue":Decimal("1234567890123456.78"),"units_sold":3}]})
    payload = json.loads(result.split("data: ",1)[1])
    assert payload == {"rows":[{"revenue":"1234567890123456.78", "units_sold":3}]}


@pytest.mark.parametrize("value", [Decimal("NaN"), Decimal("Infinity"), object()])
def test_sse_does_not_stringify_unsupported_objects(value):
    with pytest.raises(TypeError):
        encode_sse("sources", {"value":value})
