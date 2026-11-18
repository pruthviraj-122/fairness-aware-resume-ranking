import pytest
from core.ranking_engine import (
    compute_skill_scores,
    compute_experience_alignment,
    compute_education_alignment,
    score_candidates_advanced,
    normalize_weights,
    fair_rerank,
    calculate_dcg,
    calculate_ndcg,
    calculate_precision_at_k,
    SEMANTIC_AVAILABLE,
)

def test_compute_skill_scores():
    job_skills = {"python", "docker", "kubernetes", "aws", "postgresql"}
    candidate_skills = {"python", "docker", "postgresql", "redis"}

    res = compute_skill_scores(job_skills, candidate_skills)
    assert res["skill_match_score"] > 50.0
    assert "python" in res["matched_skills"]
    assert "docker" in res["matched_skills"]
    assert "kubernetes" in res["missing_skills"]
    assert "redis" in res["bonus_skills"]
    assert res["coverage_ratio"] == 0.6  # 3/5
    assert "categorized_matched" in res

def test_compute_skill_scores_empty_job_skills():
    res = compute_skill_scores(set(), {"python", "docker"})
    assert res["skill_match_score"] == 50.0
    assert res["coverage_ratio"] == 1.0

def test_compute_experience_alignment():
    # 5 years required, 6 years candidate
    score, desc = compute_experience_alignment(5.0, 6.0)
    assert score >= 85.0
    assert "Meets requirement" in desc

    # 5 years required, 2 years candidate
    score_gap, desc_gap = compute_experience_alignment(5.0, 2.0)
    assert score_gap < 70.0
    assert "gap" in desc_gap.lower()

def test_compute_education_alignment():
    # Candidate meets required tier
    score, desc = compute_education_alignment(3, 4)
    assert score == 100.0

    # Candidate has unknown education (tier 0) -> no points auto-awarded
    score_unknown, desc_unknown = compute_education_alignment(3, 0)
    assert score_unknown == 0.0
    assert "not specified" in desc_unknown.lower()

def test_normalize_weights():
    w = normalize_weights({"skills": 50, "semantic": 20, "experience": 15, "education": 15})
    assert w["skills"] == 0.5
    assert w["semantic"] == 0.2
    assert round(sum(w.values()), 2) == 1.0

    # Zero or invalid weights -> fallback to defaults
    w_default = normalize_weights(None)
    assert w_default["skills"] == 0.40
    assert w_default["semantic"] == 0.30

    w_all_zeros = normalize_weights({"skills": 0, "semantic": 0, "experience": 0, "education": 0})
    assert w_all_zeros["skills"] == 0.40

def test_score_candidates_advanced_bounds_and_ranking():
    jd = "Need a Python backend engineer with Docker, Kubernetes, and PostgreSQL. Bachelor's required."
    candidates = [
        {
            "candidate_id": "C1",
            "name": "Expert Python Developer",
            "qualifications": "Senior Python engineer with 6 years experience in Docker, Kubernetes, and PostgreSQL. Bachelor's in CS.",
            "audit_group": "Group A"
        },
        {
            "candidate_id": "C2",
            "name": "Junior Developer",
            "qualifications": "Junior web designer familiar with HTML and CSS with 1 year experience.",
            "audit_group": "Group B"
        }
    ]

    ranked = score_candidates_advanced(jd, candidates)
    assert len(ranked) == 2
    assert ranked[0]["candidate_id"] == "C1"
    assert ranked[0]["score"] > ranked[1]["score"]
    assert 0.0 <= ranked[0]["score"] <= 100.0
    assert 0.0 <= ranked[1]["score"] <= 100.0
    assert ranked[0]["rank"] == 1
    assert ranked[1]["rank"] == 2
    assert "aspect_scores" in ranked[0]
    assert "skills_extracted" in ranked[0]

def test_score_candidates_with_custom_weights():
    jd = "Python engineer with PostgreSQL."
    candidates = [
        {
            "candidate_id": "C1",
            "name": "Skill Heavy",
            "qualifications": "Python and PostgreSQL specialist with 1 year experience.",
            "audit_group": "Group A"
        },
        {
            "candidate_id": "C2",
            "name": "Experience Heavy",
            "qualifications": "10 years experience building general software. Familiar with Python.",
            "audit_group": "Group B"
        }
    ]

    w_skill_heavy = {"skills": 0.70, "semantic": 0.10, "experience": 0.10, "education": 0.10}
    ranked = score_candidates_advanced(jd, candidates, weights=w_skill_heavy)
    assert ranked[0]["weights_used"]["skills"] == 0.70

def test_deterministic_tie_breaking():
    jd = "Software developer."
    candidates = [
        {"candidate_id": "C2", "name": "B", "qualifications": "Python developer with 3 years experience."},
        {"candidate_id": "C1", "name": "A", "qualifications": "Python developer with 3 years experience."}
    ]
    ranked = score_candidates_advanced(jd, candidates)
    assert len(ranked) == 2
    assert ranked[0]["rank"] == 1
    assert ranked[1]["rank"] == 2

def test_fair_rerank_preserves_candidates_and_corrects_quota():
    candidates = [
        {"candidate_id": "C1", "name": "A1", "score": 90.0, "rank": 1, "audit_group": "Group A"},
        {"candidate_id": "C2", "name": "A2", "score": 88.0, "rank": 2, "audit_group": "Group A"},
        {"candidate_id": "C3", "name": "B1", "score": 85.0, "rank": 3, "audit_group": "Group B"},
        {"candidate_id": "C4", "name": "B2", "score": 82.0, "rank": 4, "audit_group": "Group B"}
    ]

    result = fair_rerank(candidates, group_key="audit_group")
    reranked = result["reranked"]
    
    assert len(reranked) == 4
    assert {c["candidate_id"] for c in reranked} == {"C1", "C2", "C3", "C4"}
    assert result["utility_retention_pct"] >= 90.0
    assert "rank_deltas" in result
    assert "avg_rank_shift" in result
    assert "assumptions" in result

def test_fair_rerank_single_candidate():
    res = fair_rerank([{"candidate_id": "C1", "score": 80.0, "rank": 1}])
    assert len(res["reranked"]) == 1
    assert res["swaps_count"] == 0

def test_ndcg_and_precision_calculation():
    standard = [
        {"candidate_id": "C1", "score": 90.0},
        {"candidate_id": "C2", "score": 80.0},
        {"candidate_id": "C3", "score": 70.0},
    ]
    comparison = [
        {"candidate_id": "C2", "score": 80.0},
        {"candidate_id": "C1", "score": 90.0},
        {"candidate_id": "C3", "score": 70.0},
    ]

    ndcg_res = calculate_ndcg(standard, comparison, k=3)
    assert "ndcg_at_k" in ndcg_res
    assert 0.9 <= ndcg_res["ndcg_at_k"] <= 1.0

    prec = calculate_precision_at_k(standard, k=2, threshold=75.0)
    assert prec == 1.0

def test_semantic_flag_present():
    assert isinstance(SEMANTIC_AVAILABLE, bool)
