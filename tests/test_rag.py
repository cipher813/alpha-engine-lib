"""Tests for the rag submodule.

The rag submodule consolidates code that used to live in both
alpha-engine-research/rag/ and alpha-engine-data/rag/. These tests verify
that imports work and re-exports resolve correctly. Live database
operations are out of scope here — those are integration-tested in the
consumer repos against a real Neon pgvector instance.
"""

from __future__ import annotations

import importlib

import pytest


def test_top_level_imports_resolve():
    """All advertised re-exports should be importable from the top level."""
    from alpha_engine_lib.rag import (
        get_connection,
        is_available,
        embed_texts,
        retrieve,
        ingest_document,
        document_exists,
    )

    # Verify the re-exports are callables (or at minimum, attributes — we
    # don't invoke them here because that requires a live database)
    for name, obj in [
        ("get_connection", get_connection),
        ("is_available", is_available),
        ("embed_texts", embed_texts),
        ("retrieve", retrieve),
        ("ingest_document", ingest_document),
        ("document_exists", document_exists),
    ]:
        assert callable(obj), f"{name} should be callable"


def test_submodules_importable():
    """Each submodule of alpha_engine_lib.rag should import cleanly."""
    for sub in ("db", "embeddings", "retrieval"):
        mod = importlib.import_module(f"alpha_engine_lib.rag.{sub}")
        assert mod is not None


def test_schema_sql_packaged():
    """schema.sql ships as package data so consumers can locate it."""
    import importlib.resources as ir

    files = ir.files("alpha_engine_lib.rag")
    schema_path = files / "schema.sql"
    assert schema_path.is_file(), "schema.sql should be packaged with alpha_engine_lib.rag"

    content = schema_path.read_text()
    assert "CREATE" in content.upper(), "schema.sql should contain DDL"


def test_is_available_safe_when_db_unreachable(monkeypatch):
    """is_available() must never raise — it's a probe, not an assertion."""
    from alpha_engine_lib.rag import is_available

    # Force RAG_DATABASE_URL to a guaranteed-unreachable target. The probe
    # should swallow the connection error and return False.
    monkeypatch.setenv("RAG_DATABASE_URL", "postgresql://nope:nope@localhost:1/nope")
    result = is_available()
    assert result is False
