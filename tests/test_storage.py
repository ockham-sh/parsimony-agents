"""Tests for :mod:`parsimony_agents.storage`."""

from __future__ import annotations

import asyncio
from pathlib import Path

from parsimony_agents.storage import LocalFileStorage


def _run(coro):
    return asyncio.run(coro)


def test_read_write_delete_exists(tmp_path: Path) -> None:
    storage = LocalFileStorage(tmp_path)

    async def body() -> None:
        await storage.write("a/b/c.txt", b"hello")
        assert await storage.exists("a/b/c.txt")
        assert await storage.read("a/b/c.txt") == b"hello"
        await storage.delete("a/b/c.txt")
        assert not await storage.exists("a/b/c.txt")

    _run(body())


def test_append_creates_and_extends(tmp_path: Path) -> None:
    storage = LocalFileStorage(tmp_path)

    async def body() -> None:
        await storage.append("logs/t.jsonl", b'{"a":1}\n')
        await storage.append("logs/t.jsonl", b'{"a":2}\n')
        assert await storage.read("logs/t.jsonl") == b'{"a":1}\n{"a":2}\n'

    _run(body())


def test_list_keys_includes_dot_path_components(tmp_path: Path) -> None:
    """Storage returns everything; visibility filtering is the caller's job."""
    storage = LocalFileStorage(tmp_path)

    async def body() -> None:
        (tmp_path / ".ockham").mkdir(parents=True)
        (tmp_path / ".ockham" / "secret.json").write_bytes(b"{}")
        await storage.write("visible.txt", b"x")
        keys = set(await storage.list_keys())
        assert "visible.txt" in keys
        assert ".ockham/secret.json" in keys

    _run(body())


def test_list_keys_prefix(tmp_path: Path) -> None:
    storage = LocalFileStorage(tmp_path)

    async def body() -> None:
        await storage.write("data/x.parquet", b"p")
        await storage.write("other/y.txt", b"t")
        assert set(await storage.list_keys("data")) == {"data/x.parquet"}

    _run(body())


def test_delete_prefix_removes_subtree(tmp_path: Path) -> None:
    storage = LocalFileStorage(tmp_path)

    async def body() -> None:
        await storage.write("ws/.ockham/meta.json", b"{}")
        await storage.write("ws/data/x.parquet", b"p")
        await storage.write("other/keep.txt", b"k")
        await storage.delete_prefix("ws")
        assert await storage.list_keys() == ["other/keep.txt"]

    _run(body())


def test_delete_prefix_missing_is_silent(tmp_path: Path) -> None:
    storage = LocalFileStorage(tmp_path)

    async def body() -> None:
        await storage.delete_prefix("does/not/exist")

    _run(body())


def test_materialize_prefix_returns_canonical_path(tmp_path: Path) -> None:
    storage = LocalFileStorage(tmp_path)

    async def body() -> None:
        await storage.write("ws/data/x.parquet", b"p")
        target = await storage.materialize_prefix("ws")
        assert target == (tmp_path / "ws").resolve()
        assert (target / "data" / "x.parquet").read_bytes() == b"p"

    _run(body())


def test_materialize_prefix_creates_directory(tmp_path: Path) -> None:
    storage = LocalFileStorage(tmp_path)

    async def body() -> None:
        target = await storage.materialize_prefix("ws/new")
        assert target.is_dir()

    _run(body())


def test_sync_back_is_noop(tmp_path: Path) -> None:
    """For local storage the materialized dir IS the canonical store."""
    storage = LocalFileStorage(tmp_path)

    async def body() -> None:
        target = await storage.materialize_prefix("ws")
        (target / "user.txt").write_bytes(b"u")
        await storage.sync_back(target, "ws")
        assert await storage.read("ws/user.txt") == b"u"

    _run(body())
