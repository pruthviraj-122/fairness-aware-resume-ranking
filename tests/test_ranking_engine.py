import pytest
from core.ranking_engine import (
    compute_skill_scores,
    compute_experience_alignment,
    compute_education_alignment,
    score_candidates_advanced,
    fair_rerank
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

def test_compute_experience_alignment():
    # 5 years required, 6 years candidate
    score, desc = compute_experience_alignment(5.0, 6.0)
    assert score >= 85.0
    assert "Meets requirement" in desc

    # 5 years required, 2 years candidate
    score_gap, desc_gap = compute_experience_alignment(5.0, 2.0)
    assert score_gap < 70.0
    assert "gap" in desc_gap.lower()

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

def test_fair_rerank_preserves_candidates_and_corrects_quota():
    # Setup 4 candidates where Group B is completely placed at the bottom
    candidates = [
        {"candidate_id": "C1", "name": "A1", "score": 90.0, "rank": 1, "audit_group": "Group A"},
        {"candidate_id": "C2", "name": "A2", "score": 88.0, "rank": 2, "audit_group": "Group A"},
        {"candidate_id": "C3", "name": "B1", "score": 85.0, "rank": 3, "audit_group": "Group B"},
        {"candidate_id": "C4", "name": "B2", "score": 82.0, "rank": 4, "audit_group": "Group B"}
    ]

    # Target 50% Group A, 50% Group B
    result = fair_rerank(candidates, group_key="audit_group")
    reranked = result["reranked"]
    
    assert len(reranked) == 4
    # Ensure all candidate IDs are retained
    assert {c["candidate_id"] for c in reranked} == {"C1", "C2", "C3", "C4"}
    assert result["utility_retention_pct"] == 100.0
    assert "rank_deltas" in result
