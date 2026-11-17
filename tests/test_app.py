import io
import json
import pytest
from app import app

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

def test_get_index(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"FairHire" in response.data

def test_post_sample_dataset_load(client):
    response = client.post("/", data={
        "action": "load_sample",
        "sample_preset": "software_engineer"
    })
    assert response.status_code == 200
    assert b"Senior Backend Software Engineer" in response.data

def test_post_candidate_ranking_and_audit(client):
    response = client.post("/", data={
        "job_description": "Need a Python developer with Docker and PostgreSQL.",
        "candidate_name[]": ["Alex Smith", "Beth Jones"],
        "qualifications[]": [
            "Senior Python developer with 5 years experience in Docker and PostgreSQL.",
            "Junior developer with Python and Flask experience."
        ],
        "audit_group[]": ["Group A", "Group B"]
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
        "ranking_mode": "standard"
    }
    response = client.post("/api/rank", json=payload)
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["status"] == "success"
    assert len(data["standard_ranking"]) == 2
    assert "fairness_metrics" in data
    assert "counterfactual_audit" in data

def test_api_parse_resume_txt(client):
    txt_content = b"Candidate Name: Dr. Jane Foster\nSkills: Python, PyTorch\nExperience: 5 years"
    data = {
        "file": (io.BytesIO(txt_content), "resume.txt")
    }
    response = client.post("/api/parse-resume", data=data, content_type="multipart/form-data")
    assert response.status_code == 200
    res_data = json.loads(response.data)
    assert res_data["status"] == "success"
    assert "Python" in res_data["full_text"]

def test_api_export_audit(client):
    # First trigger a post to populate LATEST_AUDIT_REPORT
    client.post("/", data={
        "job_description": "Python Docker PostgreSQL developer",
        "candidate_name[]": ["Candidate 1", "Candidate 2"],
        "qualifications[]": ["Python Docker PostgreSQL", "Python basic"],
        "audit_group[]": ["Group A", "Group B"]
    })
    response = client.get("/api/export-audit")
    assert response.status_code == 200
    assert response.mimetype == "application/json"
    audit_data = json.loads(response.data)
    assert "ranking_results" in audit_data
    assert "fairness_metrics" in audit_data
