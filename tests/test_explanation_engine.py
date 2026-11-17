import pytest
from core.explanation_engine import (
    generate_candidate_explanation,
    generate_pairwise_comparison
)

def test_generate_candidate_explanation():
    cand = {
        "candidate_id": "C1",
        "name": "Jane Developer",
        "score": 87.5,
        "tier_badge": "Top Fit",
        "aspect_scores": {
            "skill_match": 92.0,
            "semantic_relevance": 85.0,
            "experience_fit": 88.0,
            "education_fit": 80.0
        },
        "skills_extracted": {
            "matched": ["python", "docker", "postgresql"],
            "missing": ["kubernetes"],
            "bonus": ["redis"],
            "total_count": 4,
            "coverage_pct": 75.0
        },
        "experience_details": {
            "years_found": 6.0,
            "alignment_desc": "Meets requirement (6.0 yrs vs 5.0 required)"
        },
        "education_details": {
            "level": "Master's Degree",
            "alignment_desc": "Meets or exceeds education requirement"
        },
        "top_terms": ["python", "docker", "postgresql"]
    }

    explanation = generate_candidate_explanation(cand)
    assert explanation["candidate_id"] == "C1"
    assert "87.5%" in explanation["narrative_summary"]
    assert len(explanation["strengths"]) > 0
    assert len(explanation["skill_gaps"]) > 0
    assert "kubernetes" in explanation["skill_gaps"][0]
    assert "Audited: Name" in explanation["demographic_isolation"]

def test_generate_pairwise_comparison():
    cand_a = {
        "candidate_id": "C1",
        "name": "Candidate A",
        "score": 88.0,
        "aspect_scores": {
            "skill_match": 95.0,
            "semantic_relevance": 82.0,
            "experience_fit": 85.0,
            "education_fit": 90.0
        },
        "skills_extracted": {
            "matched": ["python", "docker", "kubernetes", "postgresql"]
        },
        "experience_details": {"years_found": 6.0}
    }
    cand_b = {
        "candidate_id": "C2",
        "name": "Candidate B",
        "score": 76.0,
        "aspect_scores": {
            "skill_match": 70.0,
            "semantic_relevance": 78.0,
            "experience_fit": 80.0,
            "education_fit": 80.0
        },
        "skills_extracted": {
            "matched": ["python", "docker"]
        },
        "experience_details": {"years_found": 4.0}
    }

    comp = generate_pairwise_comparison(cand_a, cand_b)
    assert comp["higher_candidate"] == "Candidate A"
    assert comp["score_difference"] == 12.0
    assert "Candidate A is ranked above Candidate B" in comp["verdict_text"]
    assert "kubernetes" in comp["skills_unique_to_a"]
