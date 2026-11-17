import pytest
from core.fairness_metrics import (
    calculate_group_distributions,
    calculate_cohens_d,
    calculate_fairness_metrics
)

def test_calculate_group_distributions():
    candidates = [
        {"candidate_id": "C1", "score": 90.0, "audit_group": "Group A"},
        {"candidate_id": "C2", "score": 80.0, "audit_group": "Group A"},
        {"candidate_id": "C3", "score": 70.0, "audit_group": "Group B"},
        {"candidate_id": "C4", "score": 60.0, "audit_group": "Group B"}
    ]
    stats = calculate_group_distributions(candidates)
    assert stats["Group A"]["count"] == 2
    assert stats["Group A"]["mean_score"] == 85.0
    assert stats["Group B"]["count"] == 2
    assert stats["Group B"]["mean_score"] == 65.0

def test_calculate_cohens_d():
    g1 = [90.0, 85.0, 95.0, 88.0]
    g2 = [70.0, 72.0, 68.0, 74.0]
    d = calculate_cohens_d(g1, g2)
    assert d > 1.0  # Large effect size

    # Identical groups -> d == 0
    d_zero = calculate_cohens_d(g1, g1)
    assert d_zero == 0.0

def test_fairness_metrics_four_fifths_compliant():
    # 4 candidates: 2 Group A (1 selected), 2 Group B (1 selected)
    # Selection rates: 50% vs 50% -> DIR = 1.0 (Compliant)
    candidates = [
        {"candidate_id": "C1", "score": 90.0, "rank": 1, "audit_group": "Group A"},
        {"candidate_id": "C2", "score": 85.0, "rank": 2, "audit_group": "Group B"},
        {"candidate_id": "C3", "score": 75.0, "rank": 3, "audit_group": "Group A"},
        {"candidate_id": "C4", "score": 70.0, "rank": 4, "audit_group": "Group B"}
    ]
    metrics = calculate_fairness_metrics(candidates, top_k=2)
    assert metrics["four_fifths_compliant"] is True
    assert metrics["disparate_impact_ratio"] == 1.0
    assert metrics["demographic_parity_diff"] == 0.0
    assert metrics["status"] == "COMPLIANT"

def test_fairness_metrics_adverse_impact_violation():
    # Top 2 selected are both Group A. Group B selection rate = 0%.
    # DIR = 0.0 (< 0.80) -> Alert
    candidates = [
        {"candidate_id": "C1", "score": 95.0, "rank": 1, "audit_group": "Group A"},
        {"candidate_id": "C2", "score": 90.0, "rank": 2, "audit_group": "Group A"},
        {"candidate_id": "C3", "score": 60.0, "rank": 3, "audit_group": "Group B"},
        {"candidate_id": "C4", "score": 55.0, "rank": 4, "audit_group": "Group B"}
    ]
    metrics = calculate_fairness_metrics(candidates, top_k=2)
    assert metrics["four_fifths_compliant"] is False
    assert metrics["disparate_impact_ratio"] == 0.0
    assert metrics["status"] == "ADVERSE_IMPACT_ALERT"
    assert "exposure_disparity" in metrics
    assert metrics["exposure_disparity"]["exposure_ratio"] < 0.70
