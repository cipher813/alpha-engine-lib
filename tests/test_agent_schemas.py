"""Tests for the lifted agent_schemas submodule.

Coverage strategy: each schema gets a "happy path" test confirming it
accepts a valid LLM-style payload, plus a regression test for any
non-trivial validator (clamp, JSON-string-as-list parser). The
agent_id → schema dispatch map gets full coverage so replay tooling
can rely on it.

Schemas are intentionally permissive (extra="allow") to tolerate
forward-compatible drift from the LLM, so most fields don't have
hard validation; the ones that DO (sector_modifiers clamp,
RubricDimensionScore.score range, CIORawOutput.decisions min_length)
each get an explicit test.
"""

from __future__ import annotations

import logging

import pytest
from pydantic import ValidationError


# ── Quant analyst ────────────────────────────────────────────────────────


class TestQuantAnalystOutput:
    def test_accepts_typical_payload(self):
        from alpha_engine_lib.agent_schemas import QuantAnalystOutput

        out = QuantAnalystOutput(
            ranked_picks=[
                {"ticker": "NVDA", "quant_score": 88, "rationale": "AI tailwind"},
                {"ticker": "AAPL", "quant_score": 75, "rationale": "FCF strong"},
            ]
        )
        assert len(out.ranked_picks) == 2
        assert out.ranked_picks[0].ticker == "NVDA"

    def test_quant_score_range_enforced(self):
        from alpha_engine_lib.agent_schemas import QuantPick

        with pytest.raises(ValidationError):
            QuantPick(ticker="X", quant_score=150)

    def test_extra_fields_allowed(self):
        from alpha_engine_lib.agent_schemas import QuantAnalystOutput

        # Forward-compat: LLM may emit additional keys.
        out = QuantAnalystOutput(
            ranked_picks=[],
            future_field="ok",  # type: ignore[call-arg]
        )
        assert out.ranked_picks == []


# ── Qual analyst ─────────────────────────────────────────────────────────


class TestQualAnalystOutput:
    def test_accepts_assessments_plus_additional_candidate(self):
        from alpha_engine_lib.agent_schemas import QualAnalystOutput

        out = QualAnalystOutput(
            assessments=[
                {"ticker": "PFE", "qual_score": 70, "bull_case": "pipeline"},
            ],
            additional_candidate={"ticker": "MRK", "bull_case": "oncology"},
        )
        assert out.additional_candidate is not None
        assert out.additional_candidate.ticker == "MRK"


# ── Peer review ──────────────────────────────────────────────────────────


class TestJointSelectionOutput:
    """The two-pass flow's Pass 1 schema. Ticker-list + team-rationale
    only — per-ticker rationale moves to Pass 2 (one bounded
    JointFinalizationDecision call per selected ticker)."""

    def test_accepts_typical_payload(self):
        from alpha_engine_lib.agent_schemas import JointSelectionOutput

        out = JointSelectionOutput(
            selected_tickers=["NVDA", "PLTR", "RKLB"],
            team_rationale="Asymmetric high-R/R slate, AI-infrastructure tilt.",
        )
        assert len(out.selected_tickers) == 3
        assert out.selected_tickers[0] == "NVDA"

    def test_empty_selection_is_valid(self):
        """Edge case: agent emits an empty selection (no candidates clear
        the gate). Schema must accept; downstream gate decides whether
        empty is a hard-fail or graceful no-op."""
        from alpha_engine_lib.agent_schemas import JointSelectionOutput

        out = JointSelectionOutput()
        assert out.selected_tickers == []
        assert out.team_rationale == ""

    def test_extra_fields_allowed(self):
        """``extra='allow'`` lets the LLM emit forward-compatible fields
        (e.g. ``confidence``) without breaking validation."""
        from alpha_engine_lib.agent_schemas import JointSelectionOutput

        out = JointSelectionOutput(
            selected_tickers=["NVDA"],
            team_rationale="",
            confidence=0.85,
        )
        assert out.selected_tickers == ["NVDA"]


class TestJointFinalizationOutput:
    def test_accepts_typical_payload(self):
        from alpha_engine_lib.agent_schemas import JointFinalizationOutput

        out = JointFinalizationOutput(
            selected_decisions=[
                {"ticker": "JPM", "rationale": "Strong NIM"},
            ],
            team_rationale="Sector concentration controlled.",
        )
        assert len(out.selected_decisions) == 1

    def test_string_as_list_parser_recovers(self, caplog):
        """Regression test for the 2026-05-03 SF Sonnet failure mode
        where ``selected_decisions`` was returned as a JSON-encoded
        string. The validator parse-and-continues + emits a WARNING."""
        from alpha_engine_lib.agent_schemas import JointFinalizationOutput

        encoded = '[{"ticker": "JPM", "rationale": "x"}]'
        with caplog.at_level(logging.WARNING):
            out = JointFinalizationOutput(selected_decisions=encoded)
        assert len(out.selected_decisions) == 1
        assert out.selected_decisions[0].ticker == "JPM"
        assert any("JSON-string" in m for m in caplog.messages)

    def test_invalid_json_string_falls_through_to_pydantic_error(self):
        from alpha_engine_lib.agent_schemas import JointFinalizationOutput

        with pytest.raises(ValidationError):
            JointFinalizationOutput(selected_decisions="this isn't json")


class TestQuantAcceptanceVerdict:
    def test_accepts_minimal_payload(self):
        from alpha_engine_lib.agent_schemas import QuantAcceptanceVerdict

        out = QuantAcceptanceVerdict(accept=True, reason="strong tech score")
        assert out.accept is True


# ── Macro economist + critic ─────────────────────────────────────────────


class TestMacroEconomistRawOutput:
    def test_accepts_typical_payload(self):
        from alpha_engine_lib.agent_schemas import MacroEconomistRawOutput

        out = MacroEconomistRawOutput(
            report_md="Full regime narrative",
            market_regime="bull",
            sector_modifiers={"technology": 1.15, "financials": 1.0},
        )
        assert out.market_regime == "bull"
        assert out.sector_modifiers["technology"] == 1.15

    def test_sector_modifier_clamp_rejects_out_of_band(self):
        from alpha_engine_lib.agent_schemas import MacroEconomistRawOutput

        with pytest.raises(ValidationError):
            MacroEconomistRawOutput(sector_modifiers={"technology": 1.5})  # >1.30

        with pytest.raises(ValidationError):
            MacroEconomistRawOutput(sector_modifiers={"technology": 0.5})  # <0.70

    def test_regime_literal_enforced(self):
        from alpha_engine_lib.agent_schemas import MacroEconomistRawOutput

        with pytest.raises(ValidationError):
            MacroEconomistRawOutput(market_regime="exuberant")


class TestMacroCriticOutput:
    def test_accept_action(self):
        from alpha_engine_lib.agent_schemas import MacroCriticOutput

        out = MacroCriticOutput(action="accept", critique="looks sound")
        assert out.action == "accept"
        assert out.suggested_regime is None

    def test_revise_with_suggested_regime(self):
        from alpha_engine_lib.agent_schemas import MacroCriticOutput

        out = MacroCriticOutput(
            action="revise", critique="too bullish", suggested_regime="neutral",
        )
        assert out.suggested_regime == "neutral"


# ── Held-stock thesis update ─────────────────────────────────────────────


class TestHeldThesisUpdateLLMOutput:
    def test_no_score_fields(self):
        from alpha_engine_lib.agent_schemas import HeldThesisUpdateLLMOutput

        # Schema intentionally has no final_score / qual_score /
        # quant_score — the held-stock LLM update path must NOT
        # overwrite prior_scores. This test pins the contract.
        out = HeldThesisUpdateLLMOutput(
            bull_case="Services growth", conviction=70,
        )
        assert not hasattr(out, "final_score")
        assert out.conviction == 70


# ── CIO ──────────────────────────────────────────────────────────────────


class TestCIORawOutput:
    def test_accepts_decisions_with_advance(self):
        from alpha_engine_lib.agent_schemas import CIORawOutput

        out = CIORawOutput(
            decisions=[
                {
                    "ticker": "NVDA", "decision": "ADVANCE",
                    "rank": 1, "conviction": 85, "rationale": "RR 2.5",
                },
            ]
        )
        assert len(out.decisions) == 1
        assert out.decisions[0].decision == "ADVANCE"

    def test_min_length_rejects_empty_decisions(self):
        """2026-05-02 PR B regression: Sonnet emitted decisions=[] when
        the prompt's per-candidate cue was stripped. min_length=1
        defends at the schema layer."""
        from alpha_engine_lib.agent_schemas import CIORawOutput

        with pytest.raises(ValidationError):
            CIORawOutput(decisions=[])

    def test_default_factory_also_validates(self):
        """validate_default=True ensures the min_length=1 constraint
        fires when decisions is omitted entirely (default_factory=list
        path), not just when the caller explicitly passes []."""
        from alpha_engine_lib.agent_schemas import CIORawOutput

        with pytest.raises(ValidationError):
            CIORawOutput()

    def test_decision_literal_enforced(self):
        from alpha_engine_lib.agent_schemas import CIORawOutput

        with pytest.raises(ValidationError):
            CIORawOutput(decisions=[
                {"ticker": "X", "decision": "MAYBE"},  # not in literal
            ])

    def test_rule_tags_optional_default_none(self):
        """Legacy artifacts emitted by prompts < v1.3.0 omit rule_tags
        entirely. Schema must default to None so loading historical
        captures keeps working."""
        from alpha_engine_lib.agent_schemas import CIORawOutput

        out = CIORawOutput(decisions=[
            {"ticker": "NVDA", "decision": "ADVANCE",
             "rank": 1, "conviction": 85, "rationale": "RR 2.5"},
        ])
        assert out.decisions[0].rule_tags is None

    def test_rule_tags_accepts_single_tag(self):
        from alpha_engine_lib.agent_schemas import CIORawOutput

        out = CIORawOutput(decisions=[
            {"ticker": "MCD", "decision": "REJECT",
             "rationale": "Qual<50", "rule_tags": ["qual_veto"]},
        ])
        assert out.decisions[0].rule_tags == ["qual_veto"]

    def test_rule_tags_accepts_multiple_tags(self):
        """Real-world example: REJECT MCD because BOTH qual<50 AND
        Consumer Discretionary is underweight. Multi-tag is the
        common case for REJECTS."""
        from alpha_engine_lib.agent_schemas import CIORawOutput

        out = CIORawOutput(decisions=[
            {"ticker": "MCD", "decision": "REJECT",
             "rationale": "Qual<50 + sector underweight",
             "rule_tags": ["qual_veto", "macro_alignment"]},
        ])
        assert out.decisions[0].rule_tags == ["qual_veto", "macro_alignment"]

    def test_rule_tags_rejects_unknown_literal(self):
        """Vocabulary is closed — unknown tags must fail validation
        rather than silently accumulate as freeform strings."""
        from alpha_engine_lib.agent_schemas import CIORawOutput

        with pytest.raises(ValidationError):
            CIORawOutput(decisions=[
                {"ticker": "X", "decision": "REJECT",
                 "rule_tags": ["made_up_tag"]},
            ])

    def test_rule_tag_vocabulary_is_nine_tags(self):
        """Locked vocabulary — adding/removing a tag is a deliberate
        prompt-version + analysis-layer change, not an accident."""
        from alpha_engine_lib.agent_schemas import CIORuleTagLiteral
        from typing import get_args

        tags = set(get_args(CIORuleTagLiteral))
        assert tags == {
            "qual_veto", "quant_veto", "dual_score_floor",
            "rr_asymmetry", "macro_alignment", "portfolio_fit",
            "catalyst_specificity", "prior_continuity", "other",
        }


# ── LLM-as-judge eval ────────────────────────────────────────────────────


class TestRubricEvalLLMOutput:
    def test_accepts_typical_payload(self):
        from alpha_engine_lib.agent_schemas import RubricEvalLLMOutput

        out = RubricEvalLLMOutput(
            dimension_scores=[
                {
                    "dimension": "numerical_grounding",
                    "score": 4,
                    "reasoning": "Cited specific multiples.",
                },
            ],
            overall_reasoning="Strong on numerics; rationale could be deeper.",
        )
        assert len(out.dimension_scores) == 1
        assert out.dimension_scores[0].score == 4

    def test_score_range_enforced(self):
        from alpha_engine_lib.agent_schemas import RubricDimensionScore

        for invalid in (0, 6, -1, 10):
            with pytest.raises(ValidationError):
                RubricDimensionScore(
                    dimension="x", score=invalid, reasoning="r",
                )

    def test_string_as_list_parser_recovers(self, caplog):
        """Same regression class as JointFinalizationOutput — Haiku
        occasionally returns dimension_scores as a JSON-string."""
        from alpha_engine_lib.agent_schemas import RubricEvalLLMOutput

        encoded = (
            '[{"dimension": "x", "score": 3, "reasoning": "r"}]'
        )
        with caplog.at_level(logging.WARNING):
            out = RubricEvalLLMOutput(
                dimension_scores=encoded, overall_reasoning="ok",
            )
        assert len(out.dimension_scores) == 1
        assert any("JSON-string" in m for m in caplog.messages)


# ── agent_id dispatch ────────────────────────────────────────────────────


class TestSchemaDispatch:
    @pytest.mark.parametrize("agent_id,expected_name", [
        ("sector_quant", "QuantAnalystOutput"),
        ("sector_quant:technology", "QuantAnalystOutput"),
        ("sector_qual:healthcare", "QualAnalystOutput"),
        ("sector_peer_review:financials", "JointFinalizationOutput"),
        ("macro_economist", "MacroEconomistRawOutput"),
        ("ic_cio", "CIORawOutput"),
        ("thesis_update:AAPL", "HeldThesisUpdateLLMOutput"),
    ])
    def test_resolve_known_agent_ids(self, agent_id, expected_name):
        from alpha_engine_lib.agent_schemas import resolve_schema_for_agent

        cls = resolve_schema_for_agent(agent_id)
        assert cls is not None
        assert cls.__name__ == expected_name

    def test_unknown_agent_returns_none(self):
        from alpha_engine_lib.agent_schemas import resolve_schema_for_agent

        assert resolve_schema_for_agent("brand_new_agent") is None
        assert resolve_schema_for_agent("") is None
        assert resolve_schema_for_agent(None) is None  # type: ignore[arg-type]

    def test_dispatch_map_covers_six_canonical_families(self):
        from alpha_engine_lib.agent_schemas import SCHEMA_BY_AGENT_ID_BASE

        # Pin the canonical family list so additions surface in review.
        assert set(SCHEMA_BY_AGENT_ID_BASE.keys()) == {
            "sector_quant",
            "sector_qual",
            "sector_peer_review",
            "macro_economist",
            "ic_cio",
            "thesis_update",
        }
