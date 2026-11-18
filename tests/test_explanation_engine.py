import pytest
from core.explanation_engine import (
    generate_candidate_explanation,
    generate_pairwise_comparison,
    calculate_next_tier_guidance
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
        "weights_used": {
            "skills": 0.40,
            "semantic": 0.30,
            "experience": 0.15,
            "education": 0.15
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
            "required_years": 5.0,
            "alignment_desc": "Meets requirement (6.0 yrs vs 5.0 required)"
        },
        "education_details": {
            "level": "Master's Degree",
            "tier": 4,
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
    assert "point_contributions" in explanation
    assert "skills" in explanation["point_contributions"]
    assert explanation["point_contributions"]["skills"]["points_earned"] > 30.0

def test_next_tier_guidance_for_qualified_tier():
    cand = {
        "score": 58.0,
        "tier_badge": "Qualified",
        "weights_used": {"skills": 0.40, "semantic": 0.30, "experience": 0.15, "education": 0.15},
        "skills_extracted": {"missing": ["kubernetes", "aws", "docker"]},
        "experience_details": {"required_years": 5.0, "years_found": 3.0},
        "education_details": {"tier": 3}
    }
    guidance = calculate_next_tier_guidance(cand)
    assert guidance["current_tier"] == "Qualified"
    assert guidance["target_tier"] == "Strong Match"
    assert guidance["points_needed"] == 7.0
    assert len(guidance["actionable_recommendations"]) > 0
    assert any("kubernetes" in r for r in guidance["actionable_recommendations"])

def test_next_tier_guidance_for_top_fit():
    cand = {
        "score": 92.0,
        "tier_badge": "Top Fit",
        "skills_extracted": {"missing": []},
        "experience_details": {},
        "education_details": {}
    }
    guidance = calculate_next_tier_guidance(cand)
    assert guidance["current_tier"] == "Top Fit"
    assert guidance["target_tier"] is None
    assert guidance["points_needed"] == 0.0

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
    assert "python" in comp["shared_skills"]

def test_pairwise_comparison_equal_scores():
    cand_a = {"candidate_id": "C1", "name": "A", "score": 80.0, "aspect_scores": {}}
    cand_b = {"candidate_id": "C2", "name": "B", "score": 80.0, "aspect_scores": {}}
    comp = generate_pairwise_comparison(cand_a, cand_b)
    assert comp["score_difference"] == 0.0
    assert "identical overall score" in comp["verdict_text"]
