"""One connector call crossing the kernel boundary: codec, broker, transport.

The kernel-side :class:`SocketConnectorTransport` sends ``connector_invoke``
over the duplex RPC; the supervisor-side :class:`ConnectorBroker` — which holds
the **bound** connectors, so credentials live only in the trusted process —
runs ``connector.fn`` and returns the framework-built :class:`Result`. The
broker is the only component that performs connector network egress, making a
bound connector the sole outbound path once the kernel substrate denies
everything else.

The shared codec is both halves' wire language: a tabular result crosses as
Arrow IPC (out of band, in the frame's binary blob, never pickle); a
non-tabular result crosses as JSON. Connector domain errors (``ConnectorError``
and friends) travel as structured response data — not RPC-level failures — and
are reconstructed kernel-side so the agent sees the right typed exception with
its directive intact.
"""

from __future__ import annotations

__all__ = [
    "ConnectorBroker",
    "SocketConnectorTransport",
    "decode_result",
    "encode_error",
    "encode_result",
    "raise_decoded_error",
]

import json
from collections.abc import Mapping, Sequence
from typing import Any

import pyarrow as pa
from parsimony import errors as _errors
from parsimony.connector import Connectors
from parsimony.result import Result, TabularResult

from parsimony_agents.execution.sandbox.protocol import RpcEndpoint, RpcError

# -- codec --------------------------------------------------------------------


def encode_result(result: Result) -> tuple[dict[str, Any], bytes]:
    """Return ``(meta, blob)`` — tabular results put the table in *blob* as Arrow IPC."""
    if isinstance(result, TabularResult):
        table = result.to_arrow()
        sink = pa.BufferOutputStream()
        with pa.ipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table)
        return {"kind": "tabular"}, sink.getvalue().to_pybytes()
    return {"kind": "result", "result": result.model_dump(mode="json")}, b""


def decode_result(meta: dict[str, Any], blob: bytes) -> Result:
    kind = meta.get("kind")
    if kind == "tabular":
        with pa.ipc.open_stream(pa.py_buffer(blob)) as reader:
            table = reader.read_all()
        return TabularResult.from_arrow(table)
    if kind == "result":
        return Result.model_validate(meta["result"])
    raise ValueError(f"unknown result wire kind: {kind!r}")


def encode_error(exc: Exception) -> dict[str, Any]:
    """Structured payload for a connector failure the broker sends back as data.

    The message crosses to the (untrusted) kernel verbatim — that is what the
    agent needs to react to the failure. Connector authors must not embed
    secrets in exception messages; parsimony's transport already scrubs
    sensitive query params from the errors it raises.
    """
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "provider": getattr(exc, "provider", None),
        "connector_error": isinstance(exc, _errors.ConnectorError),
    }


def raise_decoded_error(err: dict[str, Any]) -> None:
    """Re-raise a broker-marshalled connector error as the right kernel-side exception.

    Reconstructs the original :class:`ConnectorError` subclass (so the
    ``ExceptionObject`` renderer takes the no-traceback ``str(exc)`` path and the
    embedded agent directive survives). Subclass-specific attributes used only to
    *build* the default message (``status_code``, ``retry_after``) are not carried
    — the final message text already is.
    """
    etype = err.get("type", "RuntimeError")
    message = err.get("message", "")
    provider = err.get("provider") or "unknown"
    if err.get("connector_error"):
        cls = getattr(_errors, etype, None)
        if isinstance(cls, type) and issubclass(cls, _errors.ConnectorError):
            inst = cls.__new__(cls)
            _errors.ConnectorError.__init__(inst, message, provider=provider)
            raise inst
        raise _errors.ConnectorError(message, provider=provider)
    raise RuntimeError(f"{etype}: {message}" if etype != "RuntimeError" else message)


# -- supervisor side: the broker ------------------------------------------------


class ConnectorBroker:
    """Runs bound connectors on behalf of the kernel's proxies.

    ``bundles`` maps a binding name (the local the kernel exposes, e.g.
    ``"connectors"``) to its bound :class:`Connectors`. Construct it on the
    supervisor side from the host's credentialed connectors; pass
    :meth:`handle` as the RPC endpoint's request handler.
    """

    def __init__(self, bundles: dict[str, Connectors]) -> None:
        self._bundles = bundles

    def set_bundles(self, bundles: dict[str, Connectors]) -> None:
        """Replace the bound connectors in place.

        Mutates so an already-wired RPC endpoint (which holds :meth:`handle` as a
        bound method) keeps serving against the latest set after the host calls
        ``set_connectors`` mid-session.
        """
        self._bundles = bundles

    async def handle(self, method: str, params: dict[str, Any], blob: bytes) -> tuple[dict[str, Any], bytes]:
        if method != "connector_invoke":
            raise RpcError("UnknownMethod", f"broker has no method {method!r}")
        try:
            # encode_result stays inside the try: a Result that fails wire
            # serialization must reach the agent as a typed, named error — not
            # as an opaque transport failure on a connector that works
            # in-process.
            meta, out_blob = encode_result(await self._invoke(params))
        except Exception as exc:  # noqa: BLE001 - connector domain errors travel as data
            return {"ok": False, "error": encode_error(exc)}, b""
        return {"ok": True, **meta}, out_blob

    async def _invoke(self, params: dict[str, Any]):  # noqa: ANN202
        binding = params["binding"]
        name = params["name"]
        args = params.get("args") or []
        kwargs = params.get("kwargs") or {}
        bundle = self._bundles.get(binding)
        if bundle is None:
            raise KeyError(f"no connector binding {binding!r}")
        connector = bundle[name]  # KeyError with available names if missing
        return await connector(*args, **kwargs)


# -- kernel side: the transport -------------------------------------------------


def _check_json_args(name: str, args: Sequence[Any], kwargs: Mapping[str, Any]) -> None:
    """Raise a clear, connector-named TypeError for non-JSON-native arguments.

    In-process calls accept any Python object; over the RPC the arguments must
    survive JSON. Failing here — naming the connector and the offending
    parameter — beats the bare ``TypeError`` the framing layer would raise.
    """
    for label, value in (*((f"argument {i}", v) for i, v in enumerate(args)), *kwargs.items()):
        try:
            json.dumps(value)
        except (TypeError, ValueError):
            raise TypeError(
                f"connector {name!r}: parameter {label!r} of type {type(value).__name__} is not "
                f"JSON-serializable; under the sandboxed executor connector arguments must be "
                f"JSON-native (str/int/float/bool/None/list/dict — e.g. pass dates as ISO strings)"
            ) from None


class SocketConnectorTransport:
    """Dispatches connector calls for one binding to the broker over RPC.

    Satisfies :class:`~parsimony.capability.ConnectorTransport`. Holds no
    credential and opens no network socket of its own — calling it sends a
    ``connector_invoke`` over the supervisor connection. This is the kernel
    half of "a bound connector is the only way out".
    """

    __slots__ = ("_binding", "_rpc")

    def __init__(self, binding: str, rpc: RpcEndpoint) -> None:
        self._binding = binding
        self._rpc = rpc

    async def invoke(self, name: str, args: Sequence[Any], kwargs: Mapping[str, Any]) -> Result:
        _check_json_args(name, args, kwargs)
        resp, blob = await self._rpc.call(
            "connector_invoke",
            {"binding": self._binding, "name": name, "args": list(args), "kwargs": dict(kwargs)},
        )
        if not resp.get("ok", False):
            raise_decoded_error(resp.get("error") or {})
        return decode_result(resp, blob)
