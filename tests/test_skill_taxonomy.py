import pytest
from core.skill_taxonomy import (
    extract_skills_from_text,
    extract_skills_detailed,
    extract_years_of_experience,
    extract_education_level,
    extract_education_details,
    extract_employment_periods,
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

def test_extract_skills_detailed():
    text = "Senior Engineer with 5 years experience in Python, GraphQL, and Docker."
    mandatory = {"python", "kubernetes"}
    detailed = extract_skills_detailed(text, mandatory_skills=mandatory)

    assert "python" in detailed["found"]
    assert "graphql" in detailed["found"]
    assert "docker" in detailed["found"]
    assert "python" in detailed["mandatory_matched"]
    assert "kubernetes" in detailed["mandatory_missing"]
    assert "python" in detailed["evidence"]
    assert "docker" in detailed["evidence"]

def test_new_skills_recognized():
    text = "Engineers proficient in GraphQL, gRPC, Prometheus, Grafana, and RAG pipelines."
    skills = extract_skills_from_text(text)
    assert "graphql" in skills
    assert "grpc" in skills
    assert "prometheus" in skills
    assert "grafana" in skills
    assert "rag" in skills

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

    unknown_info = extract_education_level("Seasoned self-taught developer without credentials")
    # self-taught gets tier 1 (Bootcamp/Certificate/Self-taught)
    assert unknown_info["tier"] in (0, 1)

    empty_info = extract_education_level("Just worked on software projects.")
    assert empty_info["tier"] == 0
    assert empty_info["level"] == "Unknown"

def test_extract_education_details():
    text = "Master of Science in Computer Science from Stanford University"
    details = extract_education_details(text)
    assert details["tier"] == 4
    assert "Computer Science" in details["field"]
    assert "Stanford" in details["institution"]

def test_extract_employment_periods():
    text = "Lead Developer at Tech Corp (2018 - 2022). Prior Software Engineer (2015 - 2018)."
    periods = extract_employment_periods(text)
    assert len(periods) == 2
    assert periods[0]["start_year"] == 2018
    assert periods[0]["end_year"] == 2022
    assert periods[0]["duration_years"] == 4

def test_get_skill_category():
    assert get_skill_category("python") == "Languages"
    assert get_skill_category("react") == "Frameworks & Web"
    assert get_skill_category("docker") == "Cloud & DevOps"
    assert get_skill_category("pytorch") == "AI & Data Science"
    assert get_skill_category("postgresql") == "Databases & Big Data"
    assert get_skill_category("prometheus") == "Monitoring"
    assert get_skill_category("graphql") == "APIs & Protocols"
