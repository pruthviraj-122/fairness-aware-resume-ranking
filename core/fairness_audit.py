"""
FairHire AI — Advanced Non-Trivial Counterfactual Fairness & Proxy Bias Audit
Replaces the trivial group-tag swap with rigorous:
1. Counterfactual resume text perturbation (gendered pronouns, honorifics, affiliations)
2. Invariance & rank-flip sensitivity measurement
3. Protected-attribute proxy bias detection
4. Demographic masking (blind auditing)
"""

import copy
import re
from typing import Dict, List, Set, Tuple
from core.ranking_engine import score_candidates_advanced

# Directional demographic word mappings for counterfactual perturbation
DEMOGRAPHIC_SWAPS: Dict[str, str] = {
    # Pronouns
    "he": "she", "she": "he",
    "him": "her", "her": "him",
    "his": "hers", "hers": "his",
    "himself": "herself", "herself": "himself",
    
    # Honorifics & titles
    "mr": "ms", "ms": "mr", "mrs": "mr",
    "mr.": "ms.", "ms.": "mr.", "mrs.": "mr.",
    "male": "female", "female": "male",
    "man": "woman", "woman": "man",
    "men": "women", "women": "men",
    "boy": "girl", "girl": "boy",
    "father": "mother", "mother": "father",
    "son": "daughter", "daughter": "son",
    
    # Organizations & Associations
    "fraternity": "sorority", "sorority": "fraternity",
    "society of women engineers": "engineering honors society",
    "women in technology": "technology association",
    "women in computer science": "computer science society",
    "women in stem": "stem professional network",
    "men's leadership": "executive leadership",
    "women's leadership": "executive leadership",
}

# Proxy indicators that may signal protected attributes (Age, Gender, Family status, Origin)
PROXY_PATTERNS = {
    "Age Proxy (Early Graduation / Seniority Language)": [
        r"\b(?:graduated\s+in\s+|class\s+of\s+)(?:19[6789]\d)\b",
        r"\b(?:2[5-9]|[3-5]\d)\+?\s*years\s+of\s+experience\b",
        r"\b(?:seasoned\s+veteran|over\s+30\s+years|retired)\b"
    ],
    "Gender / Affinity Organization Proxy": [
        r"\b(?:society\s+of\s+women\s+engineers|swe|women\s+who\s+code|women\s+in\s+tech(?:nology)?)\b",
        r"\b(?:girls\s+who\s+code|women\s+in\s+stem|women\s+in\s+computer\s+science)\b",
        r"\b(?:sorority|fraternity|alpha\s+phi\s+alpha|delta\s+sigma\s+theta)\b",
        r"\b(?:men's|women's)\s+(?:basketball|soccer|rowing|lacrosse|track)\b"
    ],
    "Family / Parental Status Proxy": [
        r"\b(?:maternity\s+leave|paternity\s+leave|family\s+caregiver|parental\s+leave)\b"
    ],
    "National Origin / Visa Proxy": [
        r"\b(?:native\s+speaker|mother\s+tongue|h1-?b\s+visa|requires\s+sponsorship)\b"
    ]
}

def perturb_text_demographics(text: str) -> Tuple[str, List[Dict[str, str]]]:
    """
    Substitutes demographic markers (pronouns, titles, organizations) with counterfactual counterparts.
    Returns the perturbed text and an audit log of substitutions made.
    """
    if not text:
        return text, []

    perturbed = text
    substitutions_made = []

    # Sort swap phrases by length descending so longer phrases match first
    sorted_phrases = sorted(DEMOGRAPHIC_SWAPS.keys(), key=lambda s: len(s), reverse=True)

    for target in sorted_phrases:
        replacement = DEMOGRAPHIC_SWAPS[target]
        escaped = re.escape(target)
        pattern = rf"\b{escaped}\b"

        # Case-insensitive search with case preservation
        def replace_callback(match):
            original = match.group(0)
            if original.isupper():
                rep = replacement.upper()
            elif original.istitle():
                rep = replacement.title()
            else:
                rep = replacement.lower()
            substitutions_made.append({
                "original": original,
                "replacement": rep
            })
            return rep

        perturbed, count = re.subn(pattern, replace_callback, perturbed, flags=re.IGNORECASE)

    return perturbed, substitutions_made

def detect_proxy_biases(text: str) -> List[Dict[str, any]]:
    """
    Scans candidate resume text for proxy signals that correlate with protected classes.
    """
    detected_proxies = []
    text_lower = text.lower() if text else ""

    for category, patterns in PROXY_PATTERNS.items():
        found_matches = []
        for pat in patterns:
            matches = re.findall(pat, text_lower)
            if matches:
                found_matches.extend(matches)
        if found_matches:
            detected_proxies.append({
                "category": category,
                "flagged_terms": list(set(found_matches)),
                "recommendation": "Consider blind review or redacting non-qualification identity proxies."
            })

    return detected_proxies

def run_counterfactual_audit(
    job_description: str,
    candidates: List[Dict[str, any]]
) -> Dict[str, any]:
    """
    Conducts a rigorous counterfactual fairness audit:
    1. Evaluates baseline candidate ranking.
    2. Perturbs demographic markers within qualifications (pronouns, titles, associations).
    3. Evaluates counterfactual candidate ranking on identical qualifications.
    4. Measures score deltas, rank flips, maximum disparity, and individual fairness.
    5. Flags detected proxy biases.
    """
    if not candidates:
        return {"error": "No candidates provided for audit."}

    # 1. Baseline scoring
    baseline_results = score_candidates_advanced(job_description, candidates)
    baseline_scores = {c["candidate_id"]: c["score"] for c in baseline_results}
    baseline_ranks = {c["candidate_id"]: c["rank"] for c in baseline_results}

    # 2. Generate counterfactual variants with perturbed resume text
    perturbed_candidates = []
    candidate_substitutions: Dict[str, List[Dict[str, str]]] = {}
    candidate_proxies: Dict[str, List[Dict[str, any]]] = {}

    for c in candidates:
        cf_c = copy.deepcopy(c)
        perturbed_text, subs = perturb_text_demographics(c.get("qualifications", ""))
        cf_c["qualifications"] = perturbed_text
        # Invert audit group tag as secondary check
        cf_c["audit_group"] = "Group B" if c.get("audit_group") == "Group A" else "Group A"
        perturbed_candidates.append(cf_c)

        candidate_substitutions[c["candidate_id"]] = subs
        candidate_proxies[c["candidate_id"]] = detect_proxy_biases(c.get("qualifications", ""))

    # 3. Counterfactual scoring
    cf_results = score_candidates_advanced(job_description, perturbed_candidates)
    cf_scores = {c["candidate_id"]: c["score"] for c in cf_results}
    cf_ranks = {c["candidate_id"]: c["rank"] for c in cf_results}

    # 4. Invariance & Disparity metrics
    individual_disparities = []
    rank_flips = 0
    max_score_delta = 0.0
    total_delta = 0.0

    for c in baseline_results:
        cid = c["candidate_id"]
        b_score = baseline_scores[cid]
        c_score = cf_scores[cid]
        delta = round(c_score - b_score, 4)
        abs_delta = abs(delta)

        b_rank = baseline_ranks[cid]
        c_rank = cf_ranks[cid]
        rank_shift = c_rank - b_rank
        if rank_shift != 0:
            rank_flips += 1

        if abs_delta > max_score_delta:
            max_score_delta = abs_delta
        total_delta += abs_delta

        individual_disparities.append({
            "candidate_id": cid,
            "candidate_name": c.get("name", cid),
            "baseline_score": b_score,
            "counterfactual_score": c_score,
            "score_delta": delta,
            "baseline_rank": b_rank,
            "counterfactual_rank": c_rank,
            "rank_shift": rank_shift,
            "substitutions_tested": len(candidate_substitutions.get(cid, [])),
            "proxies_detected": len(candidate_proxies.get(cid, []))
        })

    num_candidates = len(candidates)
    mean_abs_delta = round(total_delta / num_candidates, 4) if num_candidates > 0 else 0.0
    max_score_delta = round(max_score_delta, 4)

    # Invariance check: score delta under 0.25 percentage points and zero rank flips
    is_counterfactually_fair = (max_score_delta <= 0.25) and (rank_flips == 0)

    # Invariance rate (% of candidates with <= 0.1 delta)
    invariant_count = sum(1 for d in individual_disparities if abs(d["score_delta"]) <= 0.1)
    invariance_rate_pct = round((invariant_count / num_candidates) * 100, 1)

    # Proxy summary
    total_proxies_found = sum(len(p) for p in candidate_proxies.values())

    if is_counterfactually_fair:
        verdict = "PASS — COUNTERFACTUALLY INVARIANT"
        verdict_summary = (
            f"The ranking model demonstrated individual counterfactual fairness. "
            f"Perturbing gender markers, pronouns, and demographic organizations across candidates "
            f"resulted in zero rank flips and a maximum score shift of only {max_score_delta} points."
        )
    else:
        verdict = "WARNING — SENSITIVITY DETECTED"
        verdict_summary = (
            f"Potential demographic sensitivity detected: {rank_flips} candidate rank flips observed "
            f"with a maximum score shift of {max_score_delta} points when demographic markers were perturbed."
        )

    return {
        "verdict": verdict,
        "is_counterfactually_fair": is_counterfactually_fair,
        "summary": verdict_summary,
        "metrics": {
            "max_score_delta": max_score_delta,
            "mean_abs_delta": mean_abs_delta,
            "rank_flips_count": rank_flips,
            "invariance_rate_pct": invariance_rate_pct,
            "total_candidates_audited": num_candidates,
            "total_proxies_flagged": total_proxies_found
        },
        "individual_audits": individual_disparities,
        "proxy_alerts": [
            {"candidate_id": cid, "name": next((c["name"] for c in candidates if c["candidate_id"] == cid), cid), "proxies": p}
            for cid, p in candidate_proxies.items() if p
        ]
    }
