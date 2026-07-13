"""Direct wire round-trips for the sandbox codec.

Covers the paths the broker round-trip tests never reach: the non-tabular
``kind: result`` JSON branch, the unknown-kind guard, and every
error-reconstruction branch of ``raise_decoded_error``.
"""

from __future__ import annotations

import pandas as pd
import pytest
from parsimony.errors import ConnectorError, RateLimitError
from parsimony.result import Result

from parsimony_agents.execution.sandbox.connector_rpc import (
    decode_result,
    encode_error,
    encode_result,
    raise_decoded_error,
)


def test_plain_result_round_trips_as_json() -> None:
    original = Result(raw={"name": "Alice", "items": [1, 2, 3]})
    meta, blob = encode_result(original)
    assert meta["kind"] == "result"
    assert blob == b""
    decoded = decode_result(meta, blob)
    assert decoded.raw == {"name": "Alice", "items": [1, 2, 3]}


def test_tabular_result_round_trips_as_arrow() -> None:
    frame = pd.DataFrame({"date": ["2020-01-01", "2020-01-02"], "value": [1.0, 2.0]})
    original = Result(raw=frame)
    meta, blob = encode_result(original)
    assert meta["kind"] == "tabular"
    assert blob  # the table crosses out of band
    decoded = decode_result(meta, blob)
    assert decoded.is_tabular
    pd.testing.assert_frame_equal(decoded.raw, frame)


def test_decode_unknown_kind_raises() -> None:
    with pytest.raises(ValueError, match="unknown result wire kind"):
        decode_result({"kind": "mystery"}, b"")


def test_known_connector_error_subclass_reconstructs() -> None:
    err = encode_error(RateLimitError("fred", 60.0))
    with pytest.raises(RateLimitError) as ei:
        raise_decoded_error(err)
    # The rendered message (with the agent directive) survives; subclass-only
    # ctor attributes like retry_after are deliberately not carried.
    assert "rate limit hit" in str(ei.value)
    assert ei.value.provider == "fred"


def test_unknown_connector_error_subclass_falls_back_to_base() -> None:
    err = {
        "type": "FutureConnectorError",  # not a class in parsimony.errors
        "message": "novel failure",
        "provider": "boj",
        "connector_error": True,
    }
    with pytest.raises(ConnectorError) as ei:
        raise_decoded_error(err)
    assert "novel failure" in str(ei.value)


def test_non_connector_error_becomes_runtime_error() -> None:
    err = encode_error(ValueError("plain failure"))
    with pytest.raises(RuntimeError, match="ValueError: plain failure"):
        raise_decoded_error(err)
