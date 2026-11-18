import pytest
from core.fairness_audit import (
    perturb_text_demographics,
    perturb_candidate_name,
    detect_proxy_biases,
    run_counterfactual_audit
)

def test_perturb_text_demographics_swaps():
    original = "He is an experienced engineer and led his team. He is also a member of Society of Women Engineers."
    perturbed, subs = perturb_text_demographics(original)
    
    assert "She is" in perturbed or "she is" in perturbed
    assert "her team" in perturbed
    assert "engineering honors society" in perturbed.lower()
    assert len(subs) > 0

def test_perturb_text_single_pass_no_ping_pong():
    """Verify that feminized pronouns are substituted to masculine without being reversed back."""
    original = "She is a software architect and her code is maintainable. She mentored her colleagues."
    perturbed, subs = perturb_text_demographics(original)
    
    assert "He is a software architect" in perturbed
    assert "his code is maintainable" in perturbed
    assert "He mentored his colleagues" in perturbed
    assert "She" not in perturbed
    assert "her" not in perturbed

def test_perturb_candidate_name():
    name_a, swapped_a = perturb_candidate_name("David Chen")
    assert swapped_a is True
    assert name_a.startswith("Sarah")

    name_b, swapped_b = perturb_candidate_name("Unknown Person")
    assert swapped_b is True
    assert "Counterfactual Variant" in name_b

def test_detect_proxy_biases():
    resume_with_proxies = (
        "Graduated in 1985 with 30+ years of experience.\n"
        "Active member of Society of Women Engineers and Alpha Phi Alpha.\n"
        "Took 6 months parental leave in 2021.\n"
        "Native speaker requiring H1B visa sponsorship.\n"
        "Requested reasonable accommodation for wheelchair accessible workspace."
    )
    proxies = detect_proxy_biases(resume_with_proxies)
    assert len(proxies) >= 4
    categories = [p["category"] for p in proxies]
    assert any("Age" in c for c in categories)
    assert any("Gender" in c for c in categories)
    assert any("Family" in c for c in categories)
    assert any("National Origin" in c for c in categories)
    assert any("Disability" in c for c in categories)

def test_run_counterfactual_audit_invariance():
    jd = "Seeking a senior Python developer with PostgreSQL and Docker experience."
    candidates = [
        {
            "candidate_id": "C1",
            "name": "Alex",
            "audit_group": "Group A",
            "qualifications": "He has 6 years of Python, Docker, and PostgreSQL experience."
        },
        {
            "candidate_id": "C2",
            "name": "Jordan",
            "audit_group": "Group B",
            "qualifications": "She has 5 years of Python and PostgreSQL experience."
        }
    ]

    audit = run_counterfactual_audit(jd, candidates)
    assert "metrics" in audit
    assert "verdict" in audit
    assert "audit_limitations" in audit
    assert audit["metrics"]["total_candidates_audited"] == 2
    assert audit["metrics"]["name_isolation_tested"] is True
    assert audit["metrics"]["max_score_delta"] <= 0.5
    assert "score_consistency_pct" in audit["metrics"]
    assert len(audit["individual_audits"]) == 2

def test_counterfactual_audit_empty_candidates():
    result = run_counterfactual_audit("Some JD", [])
    assert "error" in result
