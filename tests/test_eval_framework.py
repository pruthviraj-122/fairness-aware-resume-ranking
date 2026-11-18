import pytest
from evaluation.eval_framework import (
    SyntheticCandidateGenerator,
    RankingEvaluator,
    kendall_tau,
)


def test_synthetic_candidate_generator():
    generator = SyntheticCandidateGenerator(seed=42)
    jd, candidates, gt_labels = generator.generate_cohort(n=25)

    assert len(candidates) == 25
    assert len(gt_labels) == 25
    assert "Senior Backend Software Engineer" in jd

    for c in candidates:
        assert "candidate_id" in c
        assert "name" in c
        assert "qualifications" in c
        assert "audit_group" in c
        assert c["candidate_id"] in gt_labels
        assert isinstance(gt_labels[c["candidate_id"]], bool)


def test_ranking_evaluator_cohort_run():
    evaluator = RankingEvaluator(random_seed=42)
    jd, candidates, gt_labels = evaluator.generator.generate_cohort(n=20)

    result = evaluator.evaluate_cohort(
        jd, candidates, top_k=10, ground_truth_labels=gt_labels, run_counterfactual=False
    )

    assert result["candidate_count"] == 20
    assert result["top_k"] == 10
    assert result["ground_truth_labels_used"] is True
    assert "models" in result

    models = result["models"]
    assert "tfidf_baseline" in models
    assert "hybrid_ranking" in models
    assert "fair_reranked_moderate" in models

    for m_key in ["tfidf_baseline", "hybrid_ranking", "fair_reranked_moderate"]:
        m = models[m_key]
        assert "disparate_impact_ratio" in m
        assert "demographic_parity_diff" in m
        assert "precision_at_k" in m
        assert "recall_at_k" in m
        assert "ndcg_at_k" in m


def test_kendall_tau_identical_and_reversed():
    ranks_a = [1, 2, 3, 4, 5]
    ranks_identical = [1, 2, 3, 4, 5]
    ranks_reversed = [5, 4, 3, 2, 1]

    tau_id = kendall_tau(ranks_a, ranks_identical)
    assert tau_id == 1.0

    tau_rev = kendall_tau(ranks_a, ranks_reversed)
    assert tau_rev == -1.0
