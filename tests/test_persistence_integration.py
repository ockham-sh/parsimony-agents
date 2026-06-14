"""End-to-end: a return_notebook → return_dataset flow persists to .ockham/ and is reusable.

Drives the real in-process CodeExecutor through the agent loop with a scripted
LLM (no network). Proves the Phase-1 fix: deliverables actually land on disk, so
return_dataset's "recipe published?" lineage check passes and a follow-up turn
discovers the artifact in <turn_artifacts>.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from parsimony_agents.agent.agent import Agent
from parsimony_agents.agent.local_store import list_local_artifacts
from parsimony_agents.identity import notebook_content_sha

_NB_CODE = (
    '"""Produce a tiny unemployment frame."""\n'
    "import pandas as pd\n"
    "result_df = pd.DataFrame({'date': ['2020-01-01', '2020-02-01'], 'value': [3.6, 3.5]})\n"
)

# One LLM turn per inner list: (tool_call_id, tool_name, arguments_dict).
_SCRIPT = [
    [("call-nb", "return_notebook", {"path": "notebooks/unrate.py", "code": _NB_CODE, "execute": True})],
    [
        (
            "call-ds",
            "return_dataset",
            {
                "dataset_variable_name": "result_df",
                "title": "US Unemployment Rate",
                "description": "Monthly unemployment.",
                "notes": [],
                "live_name": "unrate",
            },
        )
    ],
    [("call-done", "return_done", {"summary": "Published the unemployment dataset."})],
]


def _message_for(turn_calls: list[tuple[str, str, dict]]) -> SimpleNamespace:
    tcs = [
        SimpleNamespace(
            id=cid,
            type="function",
            function=SimpleNamespace(name=name, arguments=json.dumps(args)),
        )
        for cid, name, args in turn_calls
    ]
    msg = SimpleNamespace(role="assistant", content=None, reasoning_content=None, tool_calls=tcs)
    msg.model_dump = lambda mode="json": {  # noqa: ARG005
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": tc.id, "type": tc.type, "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in tcs
        ],
    }
    return msg


def _stream_for(turn_calls: list[tuple[str, str, dict]]):
    delta = SimpleNamespace(
        content=None,
        reasoning_content=None,
        tool_calls=[SimpleNamespace(id=cid, function=SimpleNamespace(name=name)) for cid, name, _ in turn_calls],
    )
    chunk = SimpleNamespace(choices=[SimpleNamespace(delta=delta)])

    class _S:
        def __aiter__(self):
            async def _gen():
                yield chunk

            return _gen()

    return _S()


@pytest.mark.anyio
async def test_publish_persists_to_ockham_and_is_discoverable(monkeypatch: pytest.MonkeyPatch) -> None:
    turn = {"i": 0}

    async def _fake_acompletion(*args, **kwargs):  # noqa: ANN002, ANN003
        return _stream_for(_SCRIPT[turn["i"]])

    def _fake_stream_chunk_builder(chunks, messages):  # noqa: ANN001
        del chunks, messages
        msg = _message_for(_SCRIPT[turn["i"]])
        turn["i"] += 1
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    import parsimony_agents.agent.agent as agent_module

    monkeypatch.setattr(agent_module.litellm, "acompletion", _fake_acompletion)
    monkeypatch.setattr(agent_module.litellm, "stream_chunk_builder", _fake_stream_chunk_builder)

    agent = Agent(model="test-model")
    cwd = Path(agent.code_executor.cwd)

    events = [e async for e in agent.run("publish the unemployment dataset")]

    # No lineage/persist failure surfaced as a tool result.
    blob = " ".join(str(getattr(e, "result", "")) for e in events)
    assert "has not been published yet" not in blob, "return_dataset lineage check failed"
    assert "could not be saved" not in blob, "persist failed"

    # return_dataset produced a successful return event.
    return_events = [
        e
        for e in events
        if getattr(e, "tool_type", None) == "return"
        and getattr(e, "completed", False)
        and getattr(e, "tool_name", None) == "return_dataset"
    ]
    seen_tools = [getattr(e, "tool_name", None) for e in events]
    assert len(return_events) == 1, f"no successful return_dataset event; tools={seen_tools}"

    # Persisted to disk: notebook recipe + dataset triplet.
    assert (cwd / ".ockham" / "notebooks").is_dir()
    ds_dir = cwd / ".ockham" / "datasets"
    assert ds_dir.is_dir()
    curations = list(ds_dir.rglob("curation.json"))
    assert curations, "no dataset curation.json written"
    cur = json.loads(curations[0].read_text())
    assert cur["live_name"] == "unrate"
    assert (curations[0].parent / "log.jsonl").is_file()
    assert list(curations[0].parent.glob("*.parquet")), "no dataset snapshot written"

    # Discoverable via the standalone list surface.
    rows = list_local_artifacts(cwd, None, "dataset", 20)
    assert [r["live_name"] for r in rows] == ["unrate"]


@pytest.mark.anyio
async def test_persist_failure_surfaces_error_and_skips_mint(monkeypatch: pytest.MonkeyPatch) -> None:
    turn = {"i": 0}

    async def _fake_acompletion(*args, **kwargs):  # noqa: ANN002, ANN003
        return _stream_for(_SCRIPT[turn["i"]])

    def _fake_stream_chunk_builder(chunks, messages):  # noqa: ANN001
        del chunks, messages
        msg = _message_for(_SCRIPT[turn["i"]])
        turn["i"] += 1
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    import parsimony_agents.agent.agent as agent_module

    monkeypatch.setattr(agent_module.litellm, "acompletion", _fake_acompletion)
    monkeypatch.setattr(agent_module.litellm, "stream_chunk_builder", _fake_stream_chunk_builder)

    agent = Agent(model="test-model")
    cwd = Path(agent.code_executor.cwd)

    # Fail ONLY the dataset snapshot write — the notebook must persist so the
    # lineage check passes and we exercise the dataset persist-FAILURE branch
    # (not the "recipe not published" branch).
    orig_write = agent.code_executor.write_workspace_file

    async def _failing_write(path: str, data: bytes) -> None:
        if "/datasets/" in path and path.endswith(".parquet"):
            raise OSError("simulated disk failure")
        return await orig_write(path, data)

    monkeypatch.setattr(agent.code_executor, "write_workspace_file", _failing_write)

    events = [e async for e in agent.run("publish the unemployment dataset")]

    # The error is surfaced as the tool-result message the LLM reads
    # (ToolResultObserved.llm_content), not the success ToolEvent.
    blob = " ".join(str(getattr(e, "llm_content", "")) + str(getattr(e, "result", "")) for e in events)
    assert "could not be saved" in blob, "persist failure was not surfaced as a tool result"
    # No successful return_dataset event, and nothing on disk under datasets/.
    rds = [
        e
        for e in events
        if getattr(e, "tool_type", None) == "return"
        and getattr(e, "completed", False)
        and getattr(e, "tool_name", None) == "return_dataset"
    ]
    assert rds == [], "return_dataset must not report success when persist failed"
    ds_dir = cwd / ".ockham" / "datasets"
    assert not ds_dir.exists() or not list(ds_dir.rglob("*.parquet")), "no dataset snapshot should exist"


@pytest.mark.anyio
async def test_notebook_snapshot_persisted_before_code_event_is_yielded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Persist-before-yield contract for notebooks (regression).

    A workspace host consumes the loop's events and reads each artifact's
    snapshot back the instant it receives the event (the ``yield`` hands control
    to the consumer). So when the ``return_notebook`` code event is yielded, the
    snapshot must ALREADY be on disk — if the hook persisted *after* the yield,
    the host would read a not-yet-written file. This asserts the file exists at
    the moment the event arrives, not merely after the loop fully drains.
    """
    script = [
        [("call-nb", "return_notebook", {"path": "notebooks/unrate.py", "code": _NB_CODE, "execute": True})],
        [("call-done", "return_done", {"summary": "Published the notebook."})],
    ]
    turn = {"i": 0}

    async def _fake_acompletion(*args, **kwargs):  # noqa: ANN002, ANN003
        return _stream_for(script[turn["i"]])

    def _fake_stream_chunk_builder(chunks, messages):  # noqa: ANN001
        del chunks, messages
        msg = _message_for(script[turn["i"]])
        turn["i"] += 1
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    import parsimony_agents.agent.agent as agent_module

    monkeypatch.setattr(agent_module.litellm, "acompletion", _fake_acompletion)
    monkeypatch.setattr(agent_module.litellm, "stream_chunk_builder", _fake_stream_chunk_builder)

    agent = Agent(model="test-model")
    cwd = Path(agent.code_executor.cwd)
    csha = notebook_content_sha(_NB_CODE)
    snap = cwd / ".ockham" / "notebooks" / "unrate" / f"{csha}.py"

    saw_code_event = False
    async for e in agent.run("publish the notebook"):
        if getattr(e, "tool_type", None) == "code" and getattr(e, "completed", False):
            saw_code_event = True
            assert snap.is_file(), (
                "notebook snapshot must be on disk when the code event is yielded "
                "(host reads it back during event processing)"
            )

    assert saw_code_event, "no completed code event was emitted"


def _run_report_script(monkeypatch: pytest.MonkeyPatch, markdown: str):
    """Wire a scripted LLM that publishes one return_report then return_done."""
    script = [
        [
            (
                "call-report",
                "return_report",
                {
                    "title": "Quarterly Review",
                    "markdown": markdown,
                    "description": "A quarterly summary.",
                    "notes": [],
                    "live_name": "quarterly",
                },
            )
        ],
        [("call-done", "return_done", {"summary": "Published the report."})],
    ]
    turn = {"i": 0}

    async def _fake_acompletion(*args, **kwargs):  # noqa: ANN002, ANN003
        return _stream_for(script[turn["i"]])

    def _fake_stream_chunk_builder(chunks, messages):  # noqa: ANN001
        del chunks, messages
        msg = _message_for(script[turn["i"]])
        turn["i"] += 1
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    import parsimony_agents.agent.agent as agent_module

    monkeypatch.setattr(agent_module.litellm, "acompletion", _fake_acompletion)
    monkeypatch.setattr(agent_module.litellm, "stream_chunk_builder", _fake_stream_chunk_builder)


def _ctx_with_validator(validator):  # noqa: ANN001
    from parsimony_agents.agent.models import AgentContext, AgentMessage
    from parsimony_agents.messages import Text

    return AgentContext(
        messages=[AgentMessage(role="system", content=Text(content="placeholder"))],
        session_id="report-validator-test",
        report_validator=validator,
    )


_BAD_REPORT_MD = "Some prose.\n\n```{python}\nimport os\nos.system('id')\n```\n"
_CLEAN_REPORT_MD = "Some prose.\n\nA second paragraph with no active content.\n"


@pytest.mark.anyio
async def test_report_validator_rejects_unsafe_body_before_persist(monkeypatch: pytest.MonkeyPatch) -> None:
    """A host-injected report_validator gates return_report at WRITE time.

    The framework calls ``ctx.report_validator(markdown, pin_map_keys=...)`` before
    persisting (inside ``persist_artifact``), so an unsafe body is rejected —
    nothing reaches the workspace tree (every read/render path then trusts the
    snapshot) and the failure is surfaced to the LLM. This is the host trust
    boundary, enforced at the single write chokepoint.
    """
    _run_report_script(monkeypatch, _BAD_REPORT_MD)

    def _reject(md: str, *, pin_map_keys: frozenset[str] | None = None) -> None:
        if "{python}" in md:
            raise ValueError("executable Quarto fence is not allowed")

    agent = Agent(model="test-model")
    cwd = Path(agent.code_executor.cwd)
    events = [e async for e in agent.run("publish the report", ctx=_ctx_with_validator(_reject))]

    blob = " ".join(str(getattr(e, "llm_content", "")) + str(getattr(e, "result", "")) for e in events)
    assert "FAILED validation" in blob, "validation rejection was not surfaced to the LLM"

    # Nothing unsafe on disk: validation runs BEFORE any write, so not just the
    # snapshot but the curation sidecar and log are all absent (no partial write).
    reports_dir = cwd / ".ockham" / "reports"
    assert not reports_dir.exists() or not list(reports_dir.rglob("*.qmd")), "an unsafe report must never be persisted"
    assert not reports_dir.exists() or not list(reports_dir.rglob("curation.json")), "no curation may be written"
    assert not reports_dir.exists() or not list(reports_dir.rglob("log.jsonl")), "no log may be written"
    # No successful return_report event.
    ok = [
        e
        for e in events
        if getattr(e, "tool_type", None) == "return"
        and getattr(e, "completed", False)
        and getattr(e, "tool_name", None) == "return_report"
    ]
    assert ok == [], "return_report must not report success when validation fails"


@pytest.mark.anyio
async def test_report_validator_accepts_clean_body(monkeypatch: pytest.MonkeyPatch) -> None:
    """A clean report passes the injected validator and persists normally."""
    _run_report_script(monkeypatch, _CLEAN_REPORT_MD)

    calls: list[str] = []

    def _accept(md: str, *, pin_map_keys: frozenset[str] | None = None) -> None:
        calls.append(md)  # validator ran; raises nothing

    agent = Agent(model="test-model")
    cwd = Path(agent.code_executor.cwd)
    events = [e async for e in agent.run("publish the report", ctx=_ctx_with_validator(_accept))]

    assert calls, "the injected report_validator was not invoked"
    blob = " ".join(str(getattr(e, "llm_content", "")) + str(getattr(e, "result", "")) for e in events)
    assert "FAILED validation" not in blob
    reports_dir = cwd / ".ockham" / "reports"
    assert list(reports_dir.rglob("*.qmd")), "a clean report must persist its snapshot"
    cur = list(reports_dir.rglob("curation.json"))
    assert cur and json.loads(cur[0].read_text())["live_name"] == "quarterly"


@pytest.mark.anyio
async def test_followup_turn_sees_persisted_artifact_in_session_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    turn = {"i": 0}

    async def _fake_acompletion(*args, **kwargs):  # noqa: ANN002, ANN003
        return _stream_for(_SCRIPT[turn["i"]])

    def _fake_stream_chunk_builder(chunks, messages):  # noqa: ANN001
        del chunks, messages
        msg = _message_for(_SCRIPT[turn["i"]])
        turn["i"] += 1
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    import parsimony_agents.agent.agent as agent_module

    monkeypatch.setattr(agent_module.litellm, "acompletion", _fake_acompletion)
    monkeypatch.setattr(agent_module.litellm, "stream_chunk_builder", _fake_stream_chunk_builder)

    agent = Agent(model="test-model")

    # Turn 1: publish.
    ctx = None
    async for e in agent.run("publish", ctx=ctx):
        if getattr(e, "type", None) == "state_snapshot":
            ctx = e.context

    # Turn 2: a fresh run rebuilds session_state from disk before iterating.
    turn["i"] = 2  # script straight to return_done
    ctx2 = None
    async for e in agent.run("now reuse it", ctx=ctx):
        if getattr(e, "type", None) == "state_snapshot":
            ctx2 = e.context

    assert ctx2 is not None and ctx2.session_state is not None
    live_names = {a.live_name for a in ctx2.session_state.workspace_artifacts}
    assert "unrate" in live_names, f"persisted dataset not in turn-2 <turn_artifacts>: {live_names}"


@pytest.mark.anyio
async def test_notebook_persist_failure_skips_event_and_warns(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the notebook snapshot write fails, no code event is emitted and the LLM is warned.

    Covers the ``emit_code_event=False`` branch in the code-tool case: the host
    reads the snapshot back on the code event, so when persistence fails the event
    must be suppressed (the host would read a missing file) and a WARNING prepended
    to the kernel output so the agent can self-correct.
    """
    script = [
        [("call-nb", "return_notebook", {"path": "notebooks/unrate.py", "code": _NB_CODE, "execute": True})],
        [("call-done", "return_done", {"summary": "done"})],
    ]
    turn = {"i": 0}

    async def _fake_acompletion(*args, **kwargs):  # noqa: ANN002, ANN003
        return _stream_for(script[turn["i"]])

    def _fake_stream_chunk_builder(chunks, messages):  # noqa: ANN001
        del chunks, messages
        msg = _message_for(script[turn["i"]])
        turn["i"] += 1
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    import parsimony_agents.agent.agent as agent_module

    monkeypatch.setattr(agent_module.litellm, "acompletion", _fake_acompletion)
    monkeypatch.setattr(agent_module.litellm, "stream_chunk_builder", _fake_stream_chunk_builder)

    agent = Agent(model="test-model")
    cwd = Path(agent.code_executor.cwd)

    orig_write = agent.code_executor.write_workspace_file

    async def _failing_write(path: str, data: bytes) -> None:
        if "/notebooks/" in path and path.endswith(".py"):
            raise OSError("simulated notebook snapshot failure")
        return await orig_write(path, data)

    monkeypatch.setattr(agent.code_executor, "write_workspace_file", _failing_write)

    events = [e async for e in agent.run("publish the notebook")]

    blob = " ".join(str(getattr(e, "llm_content", "")) + str(getattr(e, "result", "")) for e in events)
    assert "could not be saved" in blob, "notebook persist-failure warning was not surfaced to the LLM"
    completed_code = [e for e in events if getattr(e, "tool_type", None) == "code" and getattr(e, "completed", False)]
    assert completed_code == [], "no completed code event may be emitted when the snapshot did not persist"
    nb_dir = cwd / ".ockham" / "notebooks"
    assert not nb_dir.exists() or not list(nb_dir.rglob("*.py")), "no notebook snapshot should exist on failure"


@pytest.mark.anyio
async def test_return_artifact_persisted_before_return_event_is_yielded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Persist-before-yield contract for return artifacts (dataset), mirroring the notebook guard.

    The host reads a return artifact's snapshot back the instant it receives the
    ``tool_type='return'`` event, so the full triplet (snapshot + curation + log)
    must be on disk at yield time — not merely after the loop drains.
    """
    turn = {"i": 0}

    async def _fake_acompletion(*args, **kwargs):  # noqa: ANN002, ANN003
        return _stream_for(_SCRIPT[turn["i"]])

    def _fake_stream_chunk_builder(chunks, messages):  # noqa: ANN001
        del chunks, messages
        msg = _message_for(_SCRIPT[turn["i"]])
        turn["i"] += 1
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    import parsimony_agents.agent.agent as agent_module

    monkeypatch.setattr(agent_module.litellm, "acompletion", _fake_acompletion)
    monkeypatch.setattr(agent_module.litellm, "stream_chunk_builder", _fake_stream_chunk_builder)

    agent = Agent(model="test-model")
    cwd = Path(agent.code_executor.cwd)
    ds_dir = cwd / ".ockham" / "datasets"

    saw_return = False
    async for e in agent.run("publish the unemployment dataset"):
        if (
            getattr(e, "tool_type", None) == "return"
            and getattr(e, "completed", False)
            and getattr(e, "tool_name", None) == "return_dataset"
        ):
            saw_return = True
            assert list(ds_dir.rglob("*.parquet")), "dataset snapshot must exist when the return event is yielded"
            assert list(ds_dir.rglob("curation.json")), "dataset curation must exist at yield time"
            assert list(ds_dir.rglob("log.jsonl")), "dataset log must exist at yield time"

    assert saw_return, "no completed return_dataset event was emitted"


@pytest.mark.anyio
async def test_standalone_report_publishes_without_validator(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no report_validator injected (standalone), a clean report persists normally.

    Guards against the chokepoint accidentally requiring a validator: persist must
    proceed when ``report_validator`` is None.
    """
    _run_report_script(monkeypatch, _CLEAN_REPORT_MD)

    agent = Agent(model="test-model")
    cwd = Path(agent.code_executor.cwd)
    # No ctx => default AgentContext with report_validator=None.
    events = [e async for e in agent.run("publish the report")]

    blob = " ".join(str(getattr(e, "llm_content", "")) + str(getattr(e, "result", "")) for e in events)
    assert "FAILED validation" not in blob
    reports_dir = cwd / ".ockham" / "reports"
    assert list(reports_dir.rglob("*.qmd")), "a clean report must persist even with no validator"


@pytest.mark.anyio
async def test_resume_applies_host_report_validator_via_configure_ctx(monkeypatch: pytest.MonkeyPatch) -> None:
    """A resumed turn is held to the host trust boundary (the security regression).

    ``Agent.resume`` rebuilds the ctx from the suspension record, dropping the
    host's ``exclude=True`` seams — so the host re-applies them via ``configure_ctx``.
    This proves the validator set in ``configure_ctx`` actually runs on the resumed
    turn: a ``return_report`` with an executable fence published AFTER resume is
    rejected, exactly as it would be on a fresh turn.
    """
    # Turn 1 (run): ask_user → suspends. Turns 2,3 (resume): bad report, then done.
    script = [
        [("c1", "ask_user", {"question": "which dataset?"})],
        [
            (
                "c2",
                "return_report",
                {"title": "R", "markdown": _BAD_REPORT_MD, "description": "d", "notes": [], "live_name": "r"},
            )
        ],
        [("c3", "return_done", {"summary": "done"})],
    ]
    turn = {"i": 0}

    async def _fake_acompletion(*args, **kwargs):  # noqa: ANN002, ANN003
        return _stream_for(script[turn["i"]])

    def _fake_stream_chunk_builder(chunks, messages):  # noqa: ANN001
        del chunks, messages
        msg = _message_for(script[turn["i"]])
        turn["i"] += 1
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    import parsimony_agents.agent.agent as agent_module
    from parsimony_agents.agent.events import UserInputRequested

    monkeypatch.setattr(agent_module.litellm, "acompletion", _fake_acompletion)
    monkeypatch.setattr(agent_module.litellm, "stream_chunk_builder", _fake_stream_chunk_builder)

    agent = Agent(model="test-model")
    cwd = Path(agent.code_executor.cwd)

    events1 = [e async for e in agent.run("start")]
    suspensions = [e for e in events1 if isinstance(e, UserInputRequested)]
    assert len(suspensions) == 1, "turn 1 should suspend on ask_user"
    record = suspensions[0].suspension_record

    rejected: list[str] = []

    async def _configure(ctx) -> None:  # noqa: ANN001
        def _reject(body: str, *, pin_map_keys: frozenset[str] | None = None) -> None:
            if "{python}" in body:
                rejected.append(body)
                raise ValueError("executable fence blocked on resume")

        ctx.report_validator = _reject

    events2 = [e async for e in agent.resume(record, "use dataset A", configure_ctx=_configure)]

    assert rejected, "the configure_ctx validator was not applied on resume"
    blob = " ".join(str(getattr(e, "llm_content", "")) + str(getattr(e, "result", "")) for e in events2)
    assert "FAILED validation" in blob, "resumed report rejection was not surfaced to the LLM"
    reports_dir = cwd / ".ockham" / "reports"
    assert not reports_dir.exists() or not list(reports_dir.rglob("*.qmd")), "unsafe resumed report must not persist"
