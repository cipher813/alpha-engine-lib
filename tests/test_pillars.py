"""Tests for the pillars schema module — canonical 6-pillar attractiveness
shapes used by alpha-engine-research's Qual Analyst via tool-use forced
output.

Coverage strategy mirrors ``test_agent_schemas``: each schema gets a
happy-path test, plus a regression test for any non-trivial validator
(score range, durability cap, primary-vs-secondary moat collision, evidence
trimming). The ``PILLARS`` tuple ↔ ``PillarLiteral`` consistency check is
the structural invariant — both must be the same vocabulary.
"""

from __future__ import annotations

from typing import get_args

import pytest
from pydantic import ValidationError


# ── Vocabulary invariants ────────────────────────────────────────────────


class TestPillarVocabulary:
    def test_pillars_tuple_has_canonical_six(self):
        from alpha_engine_lib.pillars import PILLARS

        assert PILLARS == (
            "quality",
            "value",
            "momentum",
            "growth",
            "stewardship",
            "defensiveness",
        )

    def test_pillar_literal_matches_pillars_tuple(self):
        """``PillarLiteral`` and ``PILLARS`` must enumerate the same values
        in the same order. If one changes, the other must change in
        lockstep."""
        from alpha_engine_lib.pillars import PILLARS, PillarLiteral

        assert get_args(PillarLiteral) == PILLARS

    def test_moat_types_include_all_six_archetypes_plus_none(self):
        from alpha_engine_lib.pillars import MoatType

        assert set(get_args(MoatType)) == {
            "network_effects",
            "switching_costs",
            "cost_advantage",
            "intangibles",
            "efficient_scale",
            "process_power",
            "none",
        }


# ── MoatAssessment ───────────────────────────────────────────────────────


class TestMoatAssessment:
    def test_accepts_typical_wide_moat_payload(self):
        from alpha_engine_lib.pillars import MoatAssessment

        moat = MoatAssessment(
            primary_type="process_power",
            secondary_types=["intangibles"],
            width="wide",
            durability_years=25,
            trend="stable",
            evidence=[
                "TSMC's leading-edge node (3nm) commands 60%+ market share per 2026 Q1 10-Q",
                "Capex barrier estimated at $20B+ per fab per 10-K risk factors section",
            ],
        )
        assert moat.primary_type == "process_power"
        assert moat.width == "wide"
        assert len(moat.evidence) == 2

    def test_accepts_no_moat_default(self):
        """The honest default — most stocks have no identifiable moat."""
        from alpha_engine_lib.pillars import MoatAssessment

        moat = MoatAssessment(
            primary_type="none",
            width="none",
            durability_years=0,
            trend="stable",
        )
        assert moat.primary_type == "none"
        assert moat.secondary_types == []
        assert moat.evidence == []

    def test_durability_upper_bound_50_years(self):
        from alpha_engine_lib.pillars import MoatAssessment

        with pytest.raises(ValidationError):
            MoatAssessment(
                primary_type="network_effects",
                width="wide",
                durability_years=51,  # > cap
                trend="stable",
            )

    def test_durability_lower_bound_zero(self):
        from alpha_engine_lib.pillars import MoatAssessment

        with pytest.raises(ValidationError):
            MoatAssessment(
                primary_type="none",
                width="none",
                durability_years=-1,
                trend="stable",
            )

    def test_primary_must_not_appear_in_secondary(self):
        """LLM failure mode: agents sometimes restate primary in secondary
        for emphasis."""
        from alpha_engine_lib.pillars import MoatAssessment

        with pytest.raises(ValidationError, match="primary_type"):
            MoatAssessment(
                primary_type="network_effects",
                secondary_types=["network_effects", "switching_costs"],
                width="wide",
                durability_years=20,
                trend="stable",
            )

    def test_secondary_types_must_be_unique(self):
        from alpha_engine_lib.pillars import MoatAssessment

        with pytest.raises(ValidationError, match="unique"):
            MoatAssessment(
                primary_type="cost_advantage",
                secondary_types=["intangibles", "intangibles"],
                width="narrow",
                durability_years=12,
                trend="stable",
            )

    def test_evidence_strings_trimmed_and_empties_dropped(self):
        """LLM occasionally emits trailing whitespace + empty strings from
        format-token confusion."""
        from alpha_engine_lib.pillars import MoatAssessment

        moat = MoatAssessment(
            primary_type="efficient_scale",
            width="narrow",
            durability_years=15,
            trend="widening",
            evidence=["  regional landfill network  ", "", "   ", "permit moat"],
        )
        assert moat.evidence == ["regional landfill network", "permit moat"]

    def test_extra_fields_allowed_for_forward_compat(self):
        from alpha_engine_lib.pillars import MoatAssessment

        moat = MoatAssessment(
            primary_type="intangibles",
            width="wide",
            durability_years=30,
            trend="stable",
            future_field="ok",  # type: ignore[call-arg]
        )
        assert moat.primary_type == "intangibles"


# ── PillarSubscore ───────────────────────────────────────────────────────


class TestPillarSubscore:
    def test_accepts_typical_qual_only_emission(self):
        """At LLM emission time, only qual fields are populated; quant
        component arrives later from the composite scoring layer."""
        from alpha_engine_lib.pillars import PillarSubscore

        sub = PillarSubscore(
            pillar="quality",
            score=82,
            confidence="high",
            qual_component=82,
            evidence=["ROE > 25% sustained 5y", "wide moat from process power"],
        )
        assert sub.pillar == "quality"
        assert sub.score == 82
        assert sub.quant_component is None
        assert sub.qual_component == 82

    def test_accepts_blended_emission_with_both_components(self):
        from alpha_engine_lib.pillars import PillarSubscore

        sub = PillarSubscore(
            pillar="momentum",
            score=68,  # blended
            confidence="medium",
            quant_component=72.3,
            qual_component=64,
        )
        assert sub.quant_component == pytest.approx(72.3)
        assert sub.qual_component == 64

    def test_score_range_enforced(self):
        from alpha_engine_lib.pillars import PillarSubscore

        with pytest.raises(ValidationError):
            PillarSubscore(pillar="value", score=150, confidence="medium")

        with pytest.raises(ValidationError):
            PillarSubscore(pillar="value", score=-5, confidence="medium")

    def test_qual_component_range_enforced(self):
        from alpha_engine_lib.pillars import PillarSubscore

        with pytest.raises(ValidationError):
            PillarSubscore(
                pillar="growth",
                score=50,
                confidence="low",
                qual_component=200,
            )

    def test_confidence_literal_enforced(self):
        from alpha_engine_lib.pillars import PillarSubscore

        with pytest.raises(ValidationError):
            PillarSubscore(
                pillar="defensiveness",
                score=50,
                confidence="certain",  # not in {low, medium, high}
            )

    def test_pillar_literal_enforced(self):
        from alpha_engine_lib.pillars import PillarSubscore

        with pytest.raises(ValidationError):
            PillarSubscore(
                pillar="liquidity",  # not a canonical pillar
                score=50,
                confidence="medium",
            )

    def test_evidence_strings_trimmed(self):
        from alpha_engine_lib.pillars import PillarSubscore

        sub = PillarSubscore(
            pillar="stewardship",
            score=60,
            confidence="medium",
            evidence=["", "  buyback at 15x P/E  ", "   "],
        )
        assert sub.evidence == ["buyback at 15x P/E"]


# ── QualitativePillarAssessment ──────────────────────────────────────────


def _make_subscore(pillar, score=70, confidence="medium"):
    from alpha_engine_lib.pillars import PillarSubscore

    return PillarSubscore(pillar=pillar, score=score, confidence=confidence)


def _make_full_assessment(**overrides):
    """Helper: build a full 6-pillar + moat assessment with sensible defaults.

    Tests override individual fields via kwargs. The default payload is the
    "moderately attractive across the board" stock — score 70 on every
    pillar, narrow moat, zero catalyst modulation."""
    from alpha_engine_lib.pillars import (
        MoatAssessment,
        QualitativePillarAssessment,
    )

    payload = {
        "quality": _make_subscore("quality"),
        "quality_moat": MoatAssessment(
            primary_type="cost_advantage",
            width="narrow",
            durability_years=12,
            trend="stable",
        ),
        "value": _make_subscore("value"),
        "momentum": _make_subscore("momentum"),
        "growth": _make_subscore("growth"),
        "stewardship": _make_subscore("stewardship"),
        "defensiveness": _make_subscore("defensiveness"),
    }
    payload.update(overrides)
    return QualitativePillarAssessment(**payload)


class TestQualitativePillarAssessment:
    def test_accepts_typical_full_payload(self):
        assessment = _make_full_assessment()
        subscores = assessment.pillar_subscores()
        assert set(subscores.keys()) == {
            "quality",
            "value",
            "momentum",
            "growth",
            "stewardship",
            "defensiveness",
        }
        # Iteration order matches PILLARS canonical ordering
        from alpha_engine_lib.pillars import PILLARS

        assert tuple(subscores.keys()) == PILLARS

    def test_catalyst_horizon_modulation_default_zero(self):
        assessment = _make_full_assessment()
        assert assessment.catalyst_horizon_modulation == 0

    def test_catalyst_horizon_modulation_bounds(self):
        from alpha_engine_lib.pillars import QualitativePillarAssessment

        with pytest.raises(ValidationError):
            _make_full_assessment(catalyst_horizon_modulation=25)

        with pytest.raises(ValidationError):
            _make_full_assessment(catalyst_horizon_modulation=-25)

        # Accepts boundary values
        a = _make_full_assessment(catalyst_horizon_modulation=20)
        assert a.catalyst_horizon_modulation == 20
        b = _make_full_assessment(catalyst_horizon_modulation=-20)
        assert b.catalyst_horizon_modulation == -20

    def test_derive_legacy_qual_score_equal_weight_mean(self):
        """Translation layer for Phase 2 soak — legacy composite needs
        a scalar."""
        assessment = _make_full_assessment(
            quality=_make_subscore("quality", score=90),
            value=_make_subscore("value", score=60),
            momentum=_make_subscore("momentum", score=70),
            growth=_make_subscore("growth", score=80),
            stewardship=_make_subscore("stewardship", score=50),
            defensiveness=_make_subscore("defensiveness", score=70),
        )
        # Mean of (90, 60, 70, 80, 50, 70) = 420 / 6 = 70
        assert assessment.derive_legacy_qual_score() == 70

    def test_derive_legacy_qual_score_returns_int_in_range(self):
        assessment = _make_full_assessment(
            quality=_make_subscore("quality", score=100),
            value=_make_subscore("value", score=100),
            momentum=_make_subscore("momentum", score=100),
            growth=_make_subscore("growth", score=100),
            stewardship=_make_subscore("stewardship", score=100),
            defensiveness=_make_subscore("defensiveness", score=100),
        )
        result = assessment.derive_legacy_qual_score()
        assert isinstance(result, int)
        assert 0 <= result <= 100
        assert result == 100

    def test_derive_legacy_qual_score_rounds(self):
        """Mean of (1, 1, 1, 1, 1, 0) = 5/6 ≈ 0.833 → rounds to 1."""
        assessment = _make_full_assessment(
            quality=_make_subscore("quality", score=1),
            value=_make_subscore("value", score=1),
            momentum=_make_subscore("momentum", score=1),
            growth=_make_subscore("growth", score=1),
            stewardship=_make_subscore("stewardship", score=1),
            defensiveness=_make_subscore("defensiveness", score=0),
        )
        assert assessment.derive_legacy_qual_score() == 1

    def test_quality_moat_embedded(self):
        """The Quality pillar's qualitative core — moat — is embedded as a
        first-class field, not buried in evidence strings."""
        from alpha_engine_lib.pillars import MoatAssessment

        assessment = _make_full_assessment(
            quality_moat=MoatAssessment(
                primary_type="network_effects",
                secondary_types=["switching_costs"],
                width="wide",
                durability_years=25,
                trend="widening",
                evidence=["card-network two-sided market dynamics"],
            )
        )
        assert assessment.quality_moat.primary_type == "network_effects"
        assert assessment.quality_moat.width == "wide"
        assert assessment.quality_moat.trend == "widening"

    def test_missing_required_pillar_rejected(self):
        from alpha_engine_lib.pillars import (
            MoatAssessment,
            QualitativePillarAssessment,
        )

        with pytest.raises(ValidationError):
            # Missing 'defensiveness' field.
            QualitativePillarAssessment(  # type: ignore[call-arg]
                quality=_make_subscore("quality"),
                quality_moat=MoatAssessment(
                    primary_type="none",
                    width="none",
                    durability_years=0,
                    trend="stable",
                ),
                value=_make_subscore("value"),
                momentum=_make_subscore("momentum"),
                growth=_make_subscore("growth"),
                stewardship=_make_subscore("stewardship"),
            )

    def test_extra_fields_allowed_for_forward_compat(self):
        """LLM may emit additional fields as the prompt evolves."""
        assessment = _make_full_assessment(
            future_field="ok",  # type: ignore[call-arg]
        )
        # Doesn't raise; extra field is allowed.
        assert assessment.pillar_subscores()["quality"].pillar == "quality"
