"""
tests/unit/mas/test_evaluation_result_schema.py
==================================================
Unit tests for EvaluationResult, MetricDetail, and AQUSADetail schemas
(Data Contracts §11).
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from gdd_userstory_mas.schemas.evaluation_result import (
    AQUSADetail,
    EvaluationResult,
    MetricDetail,
)


# ─────────────────────────────────────────────────────────────────────────────
# MetricDetail
# ─────────────────────────────────────────────────────────────────────────────

class TestMetricDetail:

    def test_basic_construction(self):
        m = MetricDetail(
            metric_name="aqusa_score",
            formula="passing / total × 100",
            numerator=8.0,
            denominator=10.0,
            value=80.0,
            interpretation="80% passed.",
            limitation="Heuristic-based.",
        )
        assert m.value == 80.0

    def test_null_value(self):
        m = MetricDetail(
            metric_name="coverage",
            formula="covered / total × 100",
            value=None,
        )
        assert m.value is None

    def test_frozen(self):
        m = MetricDetail(metric_name="aqusa", formula="x / y")
        with pytest.raises(ValidationError):
            m.metric_name = "changed"


# ─────────────────────────────────────────────────────────────────────────────
# AQUSADetail
# ─────────────────────────────────────────────────────────────────────────────

class TestAQUSADetail:

    def test_construction(self):
        d = AQUSADetail(
            total_stories=10,
            role_empty_count=0,
            action_empty_count=1,
            benefit_empty_count=0,
            format_violated_count=2,
            ambiguity_heuristic_count=1,
            stories_passing_all=7,
        )
        assert d.stories_passing_all == 7

    def test_all_zeros(self):
        d = AQUSADetail(total_stories=5, stories_passing_all=5)
        assert d.role_empty_count == 0


# ─────────────────────────────────────────────────────────────────────────────
# EvaluationResult — full metrics
# ─────────────────────────────────────────────────────────────────────────────

def _full_result(**kwargs) -> dict:
    defaults = dict(
        evaluation_id="eval__run1",
        run_id="run1",
        document_id="doc1",
        pipeline_type="MAS",
        total_stories=10,
        hallucination_rate=10.0,
        redundancy_rate=20.0,
        aqusa_score=80.0,
        coverage=75.0,
        diversity=0.6,
        null_reasons={},
        consistency_notes=[],
    )
    defaults.update(kwargs)
    return defaults


class TestEvaluationResultFull:

    def test_valid_construction(self):
        r = EvaluationResult(**_full_result())
        assert r.aqusa_score == 80.0
        assert r.pipeline_type == "MAS"

    def test_baseline_pipeline(self):
        r = EvaluationResult(**_full_result(
            pipeline_type="Baseline",
            redundancy_rate=None,
            null_reasons={"redundancy_rate": "Not applicable for Baseline."},
        ))
        assert r.pipeline_type == "Baseline"
        assert r.redundancy_rate is None

    def test_zero_stories(self):
        r = EvaluationResult(**_full_result(total_stories=0))
        assert r.total_stories == 0

    def test_frozen(self):
        r = EvaluationResult(**_full_result())
        with pytest.raises(ValidationError):
            r.aqusa_score = 50.0


# ─────────────────────────────────────────────────────────────────────────────
# EvaluationResult — null metrics require reasons (VAL-029)
# ─────────────────────────────────────────────────────────────────────────────

class TestValR029:

    def test_null_without_reason_fails(self):
        with pytest.raises(ValidationError, match="VAL-029"):
            EvaluationResult(**_full_result(
                hallucination_rate=None,
                null_reasons={},  # Missing reason for hallucination_rate
            ))

    def test_null_with_reason_passes(self):
        r = EvaluationResult(**_full_result(
            hallucination_rate=None,
            null_reasons={"hallucination_rate": "Source chunks not provided."},
        ))
        assert r.hallucination_rate is None

    def test_all_null_with_all_reasons(self):
        r = EvaluationResult(**_full_result(
            hallucination_rate=None,
            redundancy_rate=None,
            aqusa_score=None,
            coverage=None,
            diversity=None,
            null_reasons={
                "hallucination_rate": "No chunks.",
                "redundancy_rate": "Baseline.",
                "aqusa_score": "No stories.",
                "coverage": "No chunks.",
                "diversity": "No stories.",
            },
        ))
        assert all(
            getattr(r, m) is None
            for m in ["hallucination_rate", "redundancy_rate",
                      "aqusa_score", "coverage", "diversity"]
        )


# ─────────────────────────────────────────────────────────────────────────────
# EvaluationResult — field validation
# ─────────────────────────────────────────────────────────────────────────────

class TestFieldValidation:

    def test_empty_evaluation_id(self):
        with pytest.raises(ValidationError):
            EvaluationResult(**_full_result(evaluation_id=""))

    def test_empty_run_id(self):
        with pytest.raises(ValidationError):
            EvaluationResult(**_full_result(run_id=""))

    def test_empty_document_id(self):
        with pytest.raises(ValidationError):
            EvaluationResult(**_full_result(document_id=""))

    def test_invalid_pipeline_type(self):
        with pytest.raises(ValidationError):
            EvaluationResult(**_full_result(pipeline_type="SingleAgent"))

    def test_hallucination_rate_above_100(self):
        with pytest.raises(ValidationError):
            EvaluationResult(**_full_result(hallucination_rate=101.0))

    def test_hallucination_rate_below_0(self):
        with pytest.raises(ValidationError):
            EvaluationResult(**_full_result(hallucination_rate=-1.0))

    def test_diversity_above_1(self):
        with pytest.raises(ValidationError):
            EvaluationResult(**_full_result(diversity=1.5))

    def test_diversity_below_0(self):
        with pytest.raises(ValidationError):
            EvaluationResult(**_full_result(diversity=-0.1))

    def test_negative_total_stories(self):
        with pytest.raises(ValidationError):
            EvaluationResult(**_full_result(total_stories=-1))

    def test_boundary_values(self):
        r = EvaluationResult(**_full_result(
            hallucination_rate=0.0,
            redundancy_rate=100.0,
            aqusa_score=100.0,
            coverage=0.0,
            diversity=0.0,
        ))
        assert r.hallucination_rate == 0.0
        assert r.aqusa_score == 100.0
