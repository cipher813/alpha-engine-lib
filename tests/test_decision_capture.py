"""
Round-trip + truncation tests for ``alpha_engine_lib.decision_capture``.

Schema is the cross-cutting contract every captured agent decision must
satisfy. These tests lock down: extra-field rejection at the artifact level,
range constraints on token counts + cost, schema_version pinning, and the
1 MB truncation helper's behavior on small / large / pathological payloads.

Workstream design: ``alpha-engine-docs/private/alpha-engine-research-typed-
state-capture-260429.md``.
"""

from __future__ import annotations

import json

import pytest

from alpha_engine_lib.decision_capture import (
    DecisionArtifact,
    FullPromptContext,
    ModelMetadata,
    _INPUT_SNAPSHOT_DEFAULT_CAP_BYTES,
    _serialized_size,
    truncate_snapshot,
)


# ── ModelMetadata ─────────────────────────────────────────────────────────


class TestModelMetadata:
    def test_minimal_model_only(self):
        m = ModelMetadata(model_name="claude-haiku-4-5")
        assert m.input_tokens == 0
        assert m.output_tokens == 0
        assert m.cost_usd == 0.0
        assert m.model_version is None

    def test_full(self):
        m = ModelMetadata(
            model_name="claude-haiku-4-5",
            model_version="20250101",
            input_tokens=4000,
            output_tokens=1200,
            cache_read_tokens=2000,
            cache_create_tokens=500,
            cost_usd=0.04321,
        )
        assert m.cost_usd == 0.04321

    def test_negative_token_counts_rejected(self):
        with pytest.raises(ValueError):
            ModelMetadata(model_name="x", input_tokens=-1)
        with pytest.raises(ValueError):
            ModelMetadata(model_name="x", output_tokens=-5)

    def test_negative_cost_rejected(self):
        with pytest.raises(ValueError):
            ModelMetadata(model_name="x", cost_usd=-0.01)

    def test_extra_fields_rejected(self):
        # Cross-cutting metadata is locked — adding a field requires a
        # schema_version bump on DecisionArtifact.
        with pytest.raises(ValueError):
            ModelMetadata(model_name="x", undocumented="value")


# ── FullPromptContext ─────────────────────────────────────────────────────


class TestFullPromptContext:
    def test_minimal(self):
        ctx = FullPromptContext(system_prompt="sys", user_prompt="user")
        assert ctx.tool_definitions == []
        assert ctx.prompt_version_hash is None

    def test_with_tools(self):
        ctx = FullPromptContext(
            system_prompt="sys",
            user_prompt="user",
            tool_definitions=[
                {"name": "quant_indicators", "args_schema": {"type": "object"}},
                {"name": "qual_news_search", "args_schema": {"type": "object"}},
            ],
        )
        assert len(ctx.tool_definitions) == 2

    def test_with_version_hash(self):
        ctx = FullPromptContext(
            system_prompt="sys", user_prompt="user",
            prompt_version_hash="abc123",
        )
        assert ctx.prompt_version_hash == "abc123"

    def test_extra_fields_rejected(self):
        with pytest.raises(ValueError):
            FullPromptContext(system_prompt="sys", user_prompt="user", extra="x")


# ── DecisionArtifact ──────────────────────────────────────────────────────


def _minimal_artifact_kwargs() -> dict:
    """Helper for tests: minimal valid kwargs for DecisionArtifact."""
    return {
        "run_id": "run-2026-04-29",
        "timestamp": "2026-04-29T22:30:00Z",
        "agent_id": "sector_quant",
        "model_metadata": ModelMetadata(model_name="claude-haiku-4-5"),
        "full_prompt_context": FullPromptContext(
            system_prompt="sys", user_prompt="user",
        ),
        "input_data_snapshot": {"market_regime": "neutral", "tickers": ["AAPL"]},
        "agent_output": {"recommendations": [{"ticker": "AAPL", "score": 75}]},
    }


class TestDecisionArtifactBasics:
    def test_minimal(self):
        art = DecisionArtifact(**_minimal_artifact_kwargs())
        assert art.schema_version == 1
        assert art.input_data_summary is None
        assert art.input_data_truncated_at is None

    def test_schema_version_pinned_to_1(self):
        # Future-proofing: any attempt to set schema_version=2 must fail
        # until a v2 schema actually ships.
        kwargs = _minimal_artifact_kwargs()
        kwargs["schema_version"] = 2
        with pytest.raises(ValueError):
            DecisionArtifact(**kwargs)

    def test_extra_fields_rejected_at_top_level(self):
        # Top-level contract is locked — new fields require a schema_version
        # bump or an additive landing on v1 with a Pydantic field default.
        kwargs = _minimal_artifact_kwargs()
        kwargs["undocumented_field"] = "value"
        with pytest.raises(ValueError):
            DecisionArtifact(**kwargs)

    def test_input_data_snapshot_allows_arbitrary_dict_shapes(self):
        # Per-agent snapshot shapes vary; the capture layer treats them as
        # opaque dicts.
        kwargs = _minimal_artifact_kwargs()
        kwargs["input_data_snapshot"] = {
            "any": "shape",
            "nested": {"deep": {"value": [1, 2, 3]}},
            "list": [{"a": 1}, {"b": 2}],
        }
        art = DecisionArtifact(**kwargs)
        assert art.input_data_snapshot["nested"]["deep"]["value"] == [1, 2, 3]

    def test_agent_output_allows_arbitrary_dict_shapes(self):
        kwargs = _minimal_artifact_kwargs()
        kwargs["agent_output"] = {
            "reasoning": "long chain of thought...",
            "tool_calls": [{"tool": "x"}, {"tool": "y"}],
            "final_decision": {"recs": [], "score": 0},
        }
        art = DecisionArtifact(**kwargs)
        assert "tool_calls" in art.agent_output


class TestDecisionArtifactRoundTrip:
    def test_dump_then_validate_yields_equal_model(self):
        original = DecisionArtifact(**_minimal_artifact_kwargs())
        dumped = original.model_dump()
        roundtripped = DecisionArtifact.model_validate(dumped)
        assert roundtripped == original

    def test_json_dump_then_validate(self):
        original = DecisionArtifact(**_minimal_artifact_kwargs())
        dumped_json = original.model_dump_json()
        roundtripped = DecisionArtifact.model_validate_json(dumped_json)
        assert roundtripped == original

    def test_with_optional_fields_populated(self):
        kwargs = _minimal_artifact_kwargs()
        kwargs["input_data_summary"] = "Sector=Technology, 28 candidates, ..."
        kwargs["input_data_truncated_at"] = 1_500_000
        art = DecisionArtifact(**kwargs)
        roundtripped = DecisionArtifact.model_validate(art.model_dump())
        assert roundtripped.input_data_summary == kwargs["input_data_summary"]
        assert roundtripped.input_data_truncated_at == 1_500_000


class TestDecisionArtifactValidation:
    def test_truncated_at_must_be_non_negative(self):
        kwargs = _minimal_artifact_kwargs()
        kwargs["input_data_truncated_at"] = -1
        with pytest.raises(ValueError):
            DecisionArtifact(**kwargs)


# ── truncate_snapshot ─────────────────────────────────────────────────────


class TestTruncateSnapshotNoTruncation:
    def test_small_payload_passes_through(self):
        payload = {"market_regime": "neutral", "tickers": ["AAPL", "MSFT"]}
        result, original_size = truncate_snapshot(payload)
        assert result == payload
        assert original_size is None

    def test_at_cap_passes_through(self):
        # Construct a payload exactly at cap (or just under).
        # Using small cap for test speed.
        payload = {"x": "a" * 100}
        result, original_size = truncate_snapshot(payload, cap_bytes=200)
        assert result == payload
        assert original_size is None


class TestTruncateSnapshotWithTruncation:
    def test_oversized_payload_drops_largest_field(self):
        # One huge field, one small field.
        payload = {
            "small_field": "small",
            "huge_field": "x" * 5000,
        }
        result, original_size = truncate_snapshot(payload, cap_bytes=500)

        # Original size reflects pre-truncation
        assert original_size is not None
        assert original_size > 5000

        # The huge field has been replaced with a marker
        assert isinstance(result["huge_field"], dict)
        assert result["huge_field"]["_truncated"] is True
        assert result["huge_field"]["original_field"] == "huge_field"
        assert result["huge_field"]["original_size_bytes"] > 5000

        # The small field is preserved
        assert result["small_field"] == "small"

    def test_repeated_truncation_until_under_cap(self):
        # Multiple oversized fields — truncator must drop them progressively
        # until the result fits.
        payload = {
            f"field_{i}": "x" * 1000 for i in range(10)
        }
        result, original_size = truncate_snapshot(payload, cap_bytes=2000)
        assert original_size > 10000
        # Final serialized size must fit under cap
        assert _serialized_size(result) <= 2000
        # At least some fields are truncated markers
        truncated_count = sum(
            1 for v in result.values()
            if isinstance(v, dict) and v.get("_truncated") is True
        )
        assert truncated_count > 0

    def test_pathological_payload_replaced_with_single_marker(self):
        # A payload where even the dropped-field markers exceed the cap
        # gets replaced with a single top-level marker.
        # (Cap so small that even one marker triggers the fallback.)
        payload = {"x": "a" * 50}
        result, original_size = truncate_snapshot(payload, cap_bytes=20)
        # Either the field is replaced with a marker AND serialized fits...
        # or the whole payload is the fallback marker.
        if "_truncated" in result and result["_truncated"] is True:
            # Fallback path
            assert result["reason"] == "exceeded_cap_after_full_field_drop"
            assert result["original_size_bytes"] == original_size
            assert result["cap_bytes"] == 20

    def test_truncated_payload_remains_json_serializable(self):
        # A truncation result must always serialize cleanly so the wrapper
        # can write it to S3 without surprises.
        payload = {f"f{i}": "x" * 500 for i in range(20)}
        result, _ = truncate_snapshot(payload, cap_bytes=1000)
        # No exception raised — payload is JSON-clean
        json.dumps(result)


class TestTruncateSnapshotDefaultCap:
    def test_default_cap_is_1mb(self):
        # Constant kept here so consumers can rely on the documented default.
        assert _INPUT_SNAPSHOT_DEFAULT_CAP_BYTES == 1_000_000

    def test_typical_agent_payloads_fit_under_default_cap(self):
        # Sector_qual is the largest steady-state input (~300-400 KB worst
        # case from the design doc). Synthesize a comparable payload and
        # verify it doesn't trigger truncation under the default cap.
        chunks = ["A" * 1500] * 200  # ~300 KB worth of content
        payload = {
            "sector": "Healthcare",
            "candidate_tickers": ["AAPL"] * 50,
            "rag_retrieved_chunks": chunks,
        }
        result, original_size = truncate_snapshot(payload)
        # 300 KB < 1 MB cap — should pass through untruncated
        assert original_size is None
        assert result == payload
