import pytest
from core.fairness_audit import (
    perturb_text_demographics,
    detect_proxy_biases,
    run_counterfactual_audit
)

def test_perturb_text_demographics_swaps():
    original = "He is an experienced engineer and led his team. He is also a member of Society of Women Engineers."
    perturbed, subs = perturb_text_demographics(original)
    
    # Verify pronoun swaps
    assert "She is" in perturbed or "she is" in perturbed
    assert "her team" in perturbed or "hers team" in perturbed
    assert len(subs) > 0

def test_detect_proxy_biases():
    resume_with_proxies = (
        "Graduated in 1985 with 30+ years of experience.\n"
        "Active member of Society of Women Engineers and Alpha Phi Alpha.\n"
        "Took 6 months parental leave in 2021."
    )
    proxies = detect_proxy_biases(resume_with_proxies)
    assert len(proxies) >= 2
    categories = [p["category"] for p in proxies]
    assert any("Age" in c for c in categories)
    assert any("Gender" in c for c in categories)

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
    assert audit["metrics"]["total_candidates_audited"] == 2
    # The advanced ranking engine isolates skills/technical tokens, so score shifts are minimal
    assert audit["metrics"]["max_score_delta"] <= 0.5
    assert len(audit["individual_audits"]) == 2
