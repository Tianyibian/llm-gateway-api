from __future__ import annotations

import json
from decimal import Decimal
from typing import Any


def encode_sse(event: str, data: Any) -> str:
    def encode_exact_number(value):
        if isinstance(value, Decimal) and value.is_finite():
            # Snowflake monetary values remain exact in JSON; consumers may
            # format these numeric strings without an intermediate float.
            return format(value, "f")
        raise TypeError(f"Unsupported SSE value type: {type(value).__name__}")
    payload = json.dumps(data, ensure_ascii=False, default=encode_exact_number)
    return f"event: {event}\ndata: {payload}\n\n"
