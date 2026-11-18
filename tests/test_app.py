import io
import json
import pytest
from app import app, validate_file_extension

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

def test_get_index(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"FairHire" in response.data

def test_security_headers(client):
    response = client.get("/")
    assert response.headers.get("X-Content-Type-Options") == "nosniff"
    assert response.headers.get("X-Frame-Options") == "SAMEORIGIN"
    assert "Content-Security-Policy" in response.headers

def test_api_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["status"] == "ok"
    assert data["version"] == "3.0"

def test_validate_file_extension():
    assert validate_file_extension("resume.pdf") is True
    assert validate_file_extension("cv.docx") is True
    assert validate_file_extension("notes.txt") is True
    assert validate_file_extension("malicious.exe") is False
    assert validate_file_extension("script.py") is False
    assert validate_file_extension("") is False

def test_post_sample_dataset_load(client):
    response = client.post("/", data={
        "action": "load_sample",
        "sample_preset": "software_engineer"
    })
    assert response.status_code == 200
    assert b"Senior Backend Software Engineer" in response.data

def test_post_sample_dataset_multi_group_cohort(client):
    response = client.post("/", data={
        "action": "load_sample",
        "sample_preset": "fullstack_cloud_cohort"
    })
    assert response.status_code == 200
    assert b"Full Stack Cloud Platform Engineer" in response.data

def test_post_candidate_ranking_and_audit(client):
    response = client.post("/", data={
        "job_description": "Need a Python developer with Docker and PostgreSQL.",
        "candidate_name[]": ["Alex Smith", "Beth Jones"],
        "qualifications[]": [
            "Senior Python developer with 5 years experience in Docker and PostgreSQL.",
            "Junior developer with Python and Flask experience."
        ],
        "audit_group[]": ["Group A", "Group B"],
        "weight_skills": "50",
        "weight_semantic": "25",
        "weight_experience": "15",
        "weight_education": "10",
        "top_k": "1"
    })
    assert response.status_code == 200
    assert b"Ranked Candidates" in response.data
    assert b"Fairness Metrics Suite" in response.data

def test_api_sample_datasets(client):
    response = client.get("/api/sample-datasets")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["status"] == "success"
    assert "software_engineer" in data["datasets"]
    assert "fullstack_cloud_cohort" in data["datasets"]

def test_api_rank_endpoint(client):
    payload = {
        "job_description": "Looking for machine learning engineer with PyTorch and Python.",
        "candidates": [
            {
                "candidate_id": "C1",
                "name": "Jordan",
                "qualifications": "ML engineer with 4 years in PyTorch, Python, and Scikit-Learn.",
                "audit_group": "Group A"
            },
            {
                "candidate_id": "C2",
                "name": "Taylor",
                "qualifications": "Software engineer with Python and Flask.",
                "audit_group": "Group B"
            }
        ],
        "ranking_mode": "standard",
        "top_k": 1,
        "weights": {"skills": 0.50, "semantic": 0.20, "experience": 0.15, "education": 0.15}
    }
    response = client.post("/api/rank", json=payload)
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["status"] == "success"
    assert len(data["standard_ranking"]) == 2
    assert "fairness_metrics" in data
    assert "counterfactual_audit" in data

def test_api_rank_missing_job_description(client):
    response = client.post("/api/rank", json={"candidates": [{"name": "A", "qualifications": "Q"}]})
    assert response.status_code == 400
    data = json.loads(response.data)
    assert data["status"] == "error"

def test_api_rank_insufficient_candidates(client):
    response = client.post("/api/rank", json={
        "job_description": "Python dev",
        "candidates": [{"name": "A", "qualifications": "Python"}]
    })
    assert response.status_code == 400

def test_api_pairwise_compare(client):
    payload = {
        "candidate_a": {
            "candidate_id": "C1",
            "name": "Dev One",
            "score": 85.0,
            "aspect_scores": {"skill_match": 90.0, "semantic_relevance": 80.0, "experience_fit": 85.0, "education_fit": 80.0},
            "skills_extracted": {"matched": ["python", "docker"]}
        },
        "candidate_b": {
            "candidate_id": "C2",
            "name": "Dev Two",
            "score": 75.0,
            "aspect_scores": {"skill_match": 70.0, "semantic_relevance": 75.0, "experience_fit": 75.0, "education_fit": 80.0},
            "skills_extracted": {"matched": ["python"]}
        }
    }
    response = client.post("/api/pairwise-compare", json=payload)
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["status"] == "success"
    assert "comparison" in data
    assert data["comparison"]["higher_candidate"] == "Dev One"

def test_api_pairwise_compare_invalid(client):
    response = client.post("/api/pairwise-compare", json={})
    assert response.status_code == 400

def test_api_parse_resume_pii_protected(client):
    txt_content = b"Candidate Name: Dr. Jane Foster\nSkills: Python, PyTorch\nExperience: 5 years in production."
    data = {
        "file": (io.BytesIO(txt_content), "resume.txt")
    }
    response = client.post("/api/parse-resume", data=data, content_type="multipart/form-data")
    assert response.status_code == 200
    res_data = json.loads(response.data)
    assert res_data["status"] == "success"
    # PII protection check: raw full_text is redacted, text_preview & text_length returned
    assert "full_text" not in res_data
    assert "text_preview" in res_data
    assert res_data["text_length"] > 0
    assert "python" in res_data["skills"]

def test_api_parse_resume_disallowed_extension(client):
    data = {"file": (io.BytesIO(b"binary"), "executable.exe")}
    response = client.post("/api/parse-resume", data=data, content_type="multipart/form-data")
    assert response.status_code == 400

def test_api_export_audit_and_csv(client):
    # Perform evaluation
    client.post("/", data={
        "job_description": "Python Docker PostgreSQL developer",
        "candidate_name[]": ["Candidate 1", "Candidate 2"],
        "qualifications[]": ["Python Docker PostgreSQL with 5 years experience.", "Python basic with 1 year."],
        "audit_group[]": ["Group A", "Group B"]
    })

    # Test JSON export
    resp_json = client.get("/api/export-audit")
    assert resp_json.status_code == 200
    assert resp_json.mimetype == "application/json"
    audit_data = json.loads(resp_json.data)
    assert "ranking_results" in audit_data

    # Test CSV export
    resp_csv = client.get("/api/export-ranking-csv")
    assert resp_csv.status_code == 200
    assert resp_csv.mimetype == "text/csv"
    assert b"Candidate ID" in resp_csv.data
    assert b"Candidate 1" in resp_csv.data
