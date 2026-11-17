import pytest
from core.skill_taxonomy import (
    extract_skills_from_text,
    extract_years_of_experience,
    extract_education_level,
    get_skill_category,
    SYNONYM_TO_CANONICAL
)

def test_extract_skills_canonical_and_synonyms():
    text = "We use Python3, k8s, Docker, Postgres and PyTorch for our microservices."
    skills = extract_skills_from_text(text)
    
    assert "python" in skills
    assert "kubernetes" in skills
    assert "docker" in skills
    assert "postgresql" in skills
    assert "pytorch" in skills
    assert "microservices" in skills

def test_extract_skills_word_boundary_isolation():
    # 'go' should not match inside 'algorithm' or 'category'
    text = "This algorithm categorizes items without error."
    skills = extract_skills_from_text(text)
    assert "go" not in skills

    # 'go' by itself should match
    text_with_go = "Experience programming in Go and Rust."
    skills_with_go = extract_skills_from_text(text_with_go)
    assert "go" in skills_with_go
    assert "rust" in skills_with_go

def test_extract_years_of_experience():
    assert extract_years_of_experience("5+ years of experience in backend development") == 5.0
    assert extract_years_of_experience("Requires 3-5 years experience") == 5.0
    assert extract_years_of_experience("Over 7 years working with distributed databases") == 7.0
    assert extract_years_of_experience("Fresh graduate with no industry tenure") == 0.0

def test_extract_education_level():
    phd_info = extract_education_level("Ph.D in Artificial Intelligence from MIT")
    assert phd_info["tier"] == 5
    assert "PhD" in phd_info["level"]

    ms_info = extract_education_level("Master of Science in Software Engineering")
    assert ms_info["tier"] == 4

    bs_info = extract_education_level("B.Tech in Computer Science and Engineering")
    assert bs_info["tier"] == 3

def test_get_skill_category():
    assert get_skill_category("python") == "Languages"
    assert get_skill_category("react") == "Frameworks & Web"
    assert get_skill_category("docker") == "Cloud & DevOps"
    assert get_skill_category("pytorch") == "AI & Data Science"
    assert get_skill_category("postgresql") == "Databases & Big Data"
