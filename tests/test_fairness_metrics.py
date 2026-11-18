import pytest
from core.fairness_metrics import (
    calculate_group_distributions,
    calculate_cohens_d,
    calculate_theil_index,
    calculate_fairness_metrics,
    calculate_exposure_disparity,
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

def test_theil_inequality_index():
    equal_exp = [1.0, 1.0, 1.0, 1.0]
    assert calculate_theil_index(equal_exp) == 0.0

    unequal_exp = [1.0, 0.63, 0.5, 0.43, 0.38]
    theil_val = calculate_theil_index(unequal_exp)
    assert theil_val > 0.0

def test_fairness_metrics_meets_threshold():
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
    assert metrics["status"] == "MEETS_THRESHOLD"
    assert "eod_methodology_note" in metrics
    assert "uncertainty_warnings" in metrics
    assert "disclaimer" in metrics

def test_fairness_metrics_adverse_impact_violation():
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

def test_fairness_metrics_multi_group_nyc_ll144():
    candidates = [
        {"candidate_id": "C1", "score": 92.0, "rank": 1, "audit_group": "Cohort A"},
        {"candidate_id": "C2", "score": 88.0, "rank": 2, "audit_group": "Cohort B"},
        {"candidate_id": "C3", "score": 84.0, "rank": 3, "audit_group": "Cohort C"},
        {"candidate_id": "C4", "score": 80.0, "rank": 4, "audit_group": "Cohort A"},
        {"candidate_id": "C5", "score": 76.0, "rank": 5, "audit_group": "Cohort B"},
        {"candidate_id": "C6", "score": 72.0, "rank": 6, "audit_group": "Cohort C"}
    ]
    metrics = calculate_fairness_metrics(candidates, top_k=3)
    assert "group_audit_table" in metrics
    table = metrics["group_audit_table"]
    assert len(table) == 3
    assert metrics["disparate_impact_ratio"] == 1.0
    assert metrics["four_fifths_compliant"] is True

def test_eod_with_ground_truth_labels():
    candidates = [
        {"candidate_id": "C1", "score": 90.0, "rank": 1, "audit_group": "Group A"},
        {"candidate_id": "C2", "score": 85.0, "rank": 2, "audit_group": "Group B"},
        {"candidate_id": "C3", "score": 75.0, "rank": 3, "audit_group": "Group A"},
        {"candidate_id": "C4", "score": 70.0, "rank": 4, "audit_group": "Group B"}
    ]
    labels = {"C1": True, "C2": True, "C3": False, "C4": True}
    metrics = calculate_fairness_metrics(candidates, top_k=2, ground_truth_labels=labels)
    assert metrics["eod_valid"] is True
    assert "independently provided" in metrics["eod_methodology_note"]

def test_insufficient_groups():
    candidates = [
        {"candidate_id": "C1", "score": 90.0, "rank": 1, "audit_group": "Group A"},
        {"candidate_id": "C2", "score": 85.0, "rank": 2, "audit_group": "Group A"},
    ]
    metrics = calculate_fairness_metrics(candidates)
    assert metrics["status"] == "INSUFFICIENT_GROUPS"
