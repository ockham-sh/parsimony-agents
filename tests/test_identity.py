"""Tests for content-addressed identity primitives.

Pin canonicalization: same inputs must always produce the same hash,
regardless of call-site ordering or process boundaries.
"""

from __future__ import annotations

import pytest

from parsimony_agents.identity import (
    SNAPSHOT_KINDS,
    ArtifactRef,
    chart_logical_id,
    content_sha,
    data_object_logical_id,
    dataset_logical_id,
    notebook_content_sha,
    notebook_logical_id,
    object_pool_path,
    report_logical_id,
    slug_from_title,
)

# ---------------------------------------------------------------------------
# ArtifactRef
# ---------------------------------------------------------------------------


def test_artifact_ref_workspace_path_for_each_kind() -> None:
    assert (
        ArtifactRef(kind="notebook", logical_id="lid", content_sha="csha").workspace_file_path
        == ".ockham/notebooks/lid/csha.py"
    )
    assert (
        ArtifactRef(kind="data_object", logical_id="csha", content_sha="csha").workspace_file_path
        == object_pool_path("csha")
    )
    assert (
        ArtifactRef(kind="dataset", logical_id="lid", content_sha="csha").workspace_file_path
        == ".ockham/datasets/lid/csha.parquet"
    )
    assert (
        ArtifactRef(kind="chart", logical_id="lid", content_sha="csha").workspace_file_path
        == ".ockham/charts/lid/csha.vl.json"
    )
    assert (
        ArtifactRef(kind="report", logical_id="lid", content_sha="csha").workspace_file_path
        == ".ockham/reports/lid/csha.qmd"
    )


def test_artifact_ref_notebook_logical_id_can_differ_from_content_sha() -> None:
    """Notebook logical_id is the live_name (path-derived) — independent of bytes."""
    ref = ArtifactRef(kind="notebook", logical_id="us_macro_data", content_sha="some-csha")
    assert ref.logical_id != ref.content_sha
    assert ref.workspace_file_path == ".ockham/notebooks/us_macro_data/some-csha.py"


def test_artifact_ref_rejects_empty_fields() -> None:
    with pytest.raises(ValueError, match="logical_id must be non-empty"):
        ArtifactRef(kind="dataset", logical_id="", content_sha="csha")
    with pytest.raises(ValueError, match="content_sha must be non-empty"):
        ArtifactRef(kind="dataset", logical_id="lid", content_sha="")


def test_artifact_ref_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError, match="unsupported kind"):
        ArtifactRef(kind="bogus", logical_id="lid", content_sha="csha")  # type: ignore[arg-type]


def test_artifact_ref_round_trip_dict() -> None:
    ref = ArtifactRef(kind="dataset", logical_id="lid", content_sha="csha")
    again = ArtifactRef.from_dict(ref.to_dict())
    assert again == ref


def test_artifact_ref_round_trips_workspace_file_path() -> None:
    """``from_workspace_file_path`` must invert ``workspace_file_path`` for every kind."""
    cases = [
        ArtifactRef(kind="notebook", logical_id="lid", content_sha="csha"),
        ArtifactRef(kind="data_object", logical_id="csha", content_sha="csha"),
        ArtifactRef(kind="dataset", logical_id="lid", content_sha="csha"),
        ArtifactRef(kind="chart", logical_id="lid", content_sha="csha"),
        ArtifactRef(kind="report", logical_id="lid", content_sha="csha"),
    ]
    for ref in cases:
        parsed = ArtifactRef.from_workspace_file_path(ref.workspace_file_path)
        if ref.kind == "data_object":
            assert parsed is not None
            assert parsed.kind == "data_object"
            assert parsed.content_sha == ref.content_sha
        else:
            assert parsed == ref


def test_artifact_ref_from_workspace_file_path_rejects_non_canonical() -> None:
    """Paths outside the canonical layout return ``None`` (no kind guessing)."""
    assert ArtifactRef.from_workspace_file_path("notebooks/working.py") is None
    assert ArtifactRef.from_workspace_file_path("data/file.parquet") is None
    assert ArtifactRef.from_workspace_file_path(".ockham/datasets/onlydir.parquet") is None
    assert ArtifactRef.from_workspace_file_path(".ockham/notebooks/lid/csha.txt") is None
    assert ArtifactRef.from_workspace_file_path(".ockham/widgets/lid/csha.parquet") is None
    # Old flat notebook layout no longer parses — refactor migration is one-way.
    assert ArtifactRef.from_workspace_file_path(".ockham/notebooks/abc.py") is None


def test_notebook_logical_id_is_path_basename() -> None:
    assert notebook_logical_id("notebooks/us_cpi.py") == "us_cpi"
    assert notebook_logical_id("notebooks/foo_bar.py") == "foo_bar"


def test_notebook_logical_id_rejects_subdirectories() -> None:
    with pytest.raises(ValueError, match="flat"):
        notebook_logical_id("notebooks/sub/foo.py")


def test_notebook_logical_id_rejects_non_notebook_paths() -> None:
    with pytest.raises(ValueError, match="must start with"):
        notebook_logical_id("data/foo.parquet")


def test_notebook_logical_id_rejects_non_py_extensions() -> None:
    with pytest.raises(ValueError, match="must end with"):
        notebook_logical_id("notebooks/foo.ipynb")


def test_artifact_ref_xml_attrs_format() -> None:
    """XML attribute fragment is the single source of truth for the wire format."""
    ref = ArtifactRef(kind="dataset", logical_id="lid", content_sha="csha")
    assert ref.to_xml_attrs() == 'kind="dataset" logical_id="lid" content_sha="csha"'
    assert (
        ref.to_self_closing_tag("notebook_ref")
        == '<notebook_ref kind="dataset" logical_id="lid" content_sha="csha"/>'
    )
    assert ref.to_self_closing_tag().startswith("<ref ")


def test_snapshot_kinds_tuple_matches_literal() -> None:
    assert set(SNAPSHOT_KINDS) == {"notebook", "data_object", "dataset", "chart", "report"}


# ---------------------------------------------------------------------------
# content_sha
# ---------------------------------------------------------------------------


def test_content_sha_deterministic() -> None:
    assert content_sha(b"hello") == content_sha(b"hello")


def test_content_sha_distinguishes_bytes() -> None:
    assert content_sha(b"hello") != content_sha(b"world")


def test_content_sha_is_hex_64() -> None:
    digest = content_sha(b"abc")
    assert len(digest) == 64
    int(digest, 16)  # parses cleanly as hex


# ---------------------------------------------------------------------------
# notebook_content_sha
# ---------------------------------------------------------------------------


def test_notebook_content_sha_strips_trailing_whitespace() -> None:
    a = notebook_content_sha("import x\n")
    b = notebook_content_sha("import x")
    c = notebook_content_sha("import x\n\n  ")
    assert a == b == c


def test_notebook_content_sha_distinguishes_content() -> None:
    assert notebook_content_sha("import x") != notebook_content_sha("import y")


# ---------------------------------------------------------------------------
# data_object_logical_id
# ---------------------------------------------------------------------------


class _FakeProvenance:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def model_dump(self, *, mode: str, exclude: set[str]) -> dict:
        return {k: v for k, v in self._payload.items() if k not in exclude}


def test_data_object_logical_id_excludes_fetched_at() -> None:
    a = _FakeProvenance({"source": "fred", "params": {"id": "GDP"}, "fetched_at": "2026-01-01"})
    b = _FakeProvenance({"source": "fred", "params": {"id": "GDP"}, "fetched_at": "2026-05-01"})
    assert data_object_logical_id(a) == data_object_logical_id(b)


def test_data_object_logical_id_distinguishes_params() -> None:
    a = _FakeProvenance({"source": "fred", "params": {"id": "GDP"}})
    b = _FakeProvenance({"source": "fred", "params": {"id": "UNRATE"}})
    assert data_object_logical_id(a) != data_object_logical_id(b)


def test_data_object_logical_id_ignores_properties() -> None:
    a = _FakeProvenance({"source": "fred", "params": {"id": "GDP"}, "properties": {"series_url": "a"}})
    b = _FakeProvenance({"source": "fred", "params": {"id": "GDP"}, "properties": {"series_url": "b"}})
    assert data_object_logical_id(a) == data_object_logical_id(b)


# ---------------------------------------------------------------------------
# dataset_logical_id
# ---------------------------------------------------------------------------


def _nb_ref(name: str, content_sha: str | None = None) -> ArtifactRef:
    """Build a notebook ArtifactRef.

    Default ``content_sha=None`` makes ``content_sha == logical_id`` —
    fine for tests that don't care about the distinction. Passing both
    independently exercises the post-R1 invariant that dataset/chart
    identity hashes notebook ``logical_id`` only, never ``content_sha``.
    """
    return ArtifactRef(
        kind="notebook", logical_id=name, content_sha=content_sha or name
    )


def _do_ref(lid: str, csha: str) -> ArtifactRef:
    return ArtifactRef(kind="data_object", logical_id=lid, content_sha=csha)


def test_dataset_logical_id_ordering_invariance() -> None:
    nb = [_nb_ref("aaa"), _nb_ref("bbb")]
    nb_swapped = [_nb_ref("bbb"), _nb_ref("aaa")]
    sources = [_do_ref("L1", "C1"), _do_ref("L2", "C2")]
    sources_swapped = [_do_ref("L2", "C2"), _do_ref("L1", "C1")]
    a = dataset_logical_id(notebook_refs=nb, variable_name="df", source_refs=sources)
    b = dataset_logical_id(
        notebook_refs=nb_swapped, variable_name="df", source_refs=sources_swapped
    )
    assert a == b


def test_dataset_logical_id_distinguishes_inputs() -> None:
    nb = [_nb_ref("aaa")]
    s1 = [_do_ref("L1", "C1")]
    s2 = [_do_ref("L2", "C1")]
    assert dataset_logical_id(
        notebook_refs=nb, variable_name="df", source_refs=s1
    ) != dataset_logical_id(notebook_refs=nb, variable_name="df", source_refs=s2)


def test_dataset_logical_id_uses_source_logical_id_not_content_sha() -> None:
    nb = [_nb_ref("aaa")]
    a = [_do_ref("L1", "C1")]
    b = [_do_ref("L1", "C99")]  # same logical_id, different content_sha
    assert dataset_logical_id(
        notebook_refs=nb, variable_name="df", source_refs=a
    ) == dataset_logical_id(notebook_refs=nb, variable_name="df", source_refs=b)


def test_dataset_logical_id_uses_notebook_logical_id_not_content_sha() -> None:
    """Notebook edits don't fork dataset identity (R1).

    Same notebook ``logical_id`` + different ``content_sha`` (a
    cosmetic notebook edit between publishes) must produce the same
    ``dataset_logical_id`` — refresh appends a new ``content_sha``
    snapshot under the unchanged ``logical_id`` instead of forking.
    """
    src = [_do_ref("L1", "C1")]
    a = [_nb_ref("us_macro", content_sha="csha-v1")]
    b = [_nb_ref("us_macro", content_sha="csha-v2")]
    assert dataset_logical_id(
        notebook_refs=a, variable_name="df", source_refs=src
    ) == dataset_logical_id(notebook_refs=b, variable_name="df", source_refs=src)


def test_dataset_logical_id_distinguishes_notebook_logical_id() -> None:
    """Different notebooks (different logical_ids) → different datasets."""
    src = [_do_ref("L1", "C1")]
    a = [_nb_ref("us_macro")]
    b = [_nb_ref("eu_macro")]
    assert dataset_logical_id(
        notebook_refs=a, variable_name="df", source_refs=src
    ) != dataset_logical_id(notebook_refs=b, variable_name="df", source_refs=src)


def test_dataset_logical_id_rejects_empty_variable_name() -> None:
    with pytest.raises(ValueError, match="variable_name"):
        dataset_logical_id(notebook_refs=[], variable_name="", source_refs=[])


# ---------------------------------------------------------------------------
# chart_logical_id
# ---------------------------------------------------------------------------


def _ds_ref(lid: str, csha: str) -> ArtifactRef:
    return ArtifactRef(kind="dataset", logical_id=lid, content_sha=csha)


def test_chart_logical_id_ordering_invariance() -> None:
    nb = _nb_ref("nbsha")
    ds_a = [_ds_ref("dlid1", "dcsha1"), _ds_ref("dlid2", "dcsha2")]
    ds_b = [_ds_ref("dlid2", "dcsha2"), _ds_ref("dlid1", "dcsha1")]
    a = chart_logical_id(
        notebook_ref=nb,
        chart_variable_name="ch",
        source_dataset_refs=ds_a,
        source_refs=[],
    )
    b = chart_logical_id(
        notebook_ref=nb,
        chart_variable_name="ch",
        source_dataset_refs=ds_b,
        source_refs=[],
    )
    assert a == b


def test_chart_logical_id_distinguishes_notebook_logical_id() -> None:
    a = chart_logical_id(
        notebook_ref=_nb_ref("nb1"),
        chart_variable_name="ch",
        source_dataset_refs=[],
        source_refs=[],
    )
    b = chart_logical_id(
        notebook_ref=_nb_ref("nb2"),
        chart_variable_name="ch",
        source_dataset_refs=[],
        source_refs=[],
    )
    assert a != b


def test_chart_logical_id_uses_notebook_logical_id_not_content_sha() -> None:
    """Notebook edits don't fork chart identity (R1)."""
    a = chart_logical_id(
        notebook_ref=_nb_ref("us_macro", content_sha="csha-v1"),
        chart_variable_name="ch",
        source_dataset_refs=[],
        source_refs=[],
    )
    b = chart_logical_id(
        notebook_ref=_nb_ref("us_macro", content_sha="csha-v2"),
        chart_variable_name="ch",
        source_dataset_refs=[],
        source_refs=[],
    )
    assert a == b


def test_chart_logical_id_rejects_non_notebook_ref() -> None:
    with pytest.raises(ValueError, match="kind='notebook'"):
        chart_logical_id(
            notebook_ref=_ds_ref("lid", "csha"),
            chart_variable_name="ch",
            source_dataset_refs=[],
            source_refs=[],
        )


# ---------------------------------------------------------------------------
# report_logical_id
# ---------------------------------------------------------------------------


def test_report_logical_id_ordering_invariance() -> None:
    refs_a = [_ds_ref("a", "x"), _ds_ref("b", "y")]
    refs_b = [_ds_ref("b", "y"), _ds_ref("a", "x")]
    assert report_logical_id(embedded_refs=refs_a, title="Q4 review") == report_logical_id(
        embedded_refs=refs_b, title="Q4 review"
    )


def test_report_logical_id_title_participates() -> None:
    refs = [_ds_ref("a", "x")]
    assert report_logical_id(embedded_refs=refs, title="Q4 review") != report_logical_id(
        embedded_refs=refs, title="Q4 review v2"
    )


def test_report_logical_id_uses_logical_id_not_content_sha() -> None:
    a = [_ds_ref("L1", "C1")]
    b = [_ds_ref("L1", "C99")]
    assert report_logical_id(embedded_refs=a, title="t") == report_logical_id(
        embedded_refs=b, title="t"
    )


def test_report_logical_id_rejects_empty_title() -> None:
    with pytest.raises(ValueError, match="title"):
        report_logical_id(embedded_refs=[], title="")


# ---------------------------------------------------------------------------
# slug_from_title re-export
# ---------------------------------------------------------------------------


def test_slug_from_title_reexported() -> None:
    assert slug_from_title("Hello World") == "hello_world"
