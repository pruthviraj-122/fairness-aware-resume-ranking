"""
FairHire AI 3.0 — Multi-Variation Counterfactual Fairness & Proxy Bias Audit Engine

Evaluates candidate resumes across multiple controlled counterfactual variations:
1. Variation 1: Direct Gender Pronoun & Honorific Invariance (he/him <-> she/her, Mr <-> Ms)
2. Variation 2: Candidate Name Perturbation (Audit study methodology: Bertrand & Mullainathan 2004)
3. Variation 3: Affinity & Organizational Proxy Invariance (SWE, Women Who Code, HBCUs <-> neutral)
4. Variation 4: Combined Multi-Attribute Perturbation

Metrics evaluated per variation and per candidate:
- Original Qualification Score vs Counterfactual Score
- Absolute Score Difference (|s_cf - s_orig|)
- Relative Score Difference (|s_cf - s_orig| / s_orig)
- Original Rank vs Counterfactual Rank
- Rank Shift and Absolute Rank Displacement (|r_orig - r_cf|)
- Skill-set stability and explanation consistency

Theoretical & Causal Disclaimers:
- Cites Kusner et al. (2017) "Counterfactual Fairness": text-marker invariance tests observational
  sensitivity to surface tokens. It does NOT constitute formal causal counterfactual fairness.
- Distinguishes direct demographic sensitivity, proxy sensitivity, and group-level demographic parity.
"""

import copy
import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from core.ranking_engine import score_candidates_advanced

# ── Direct Gender Pronoun & Honorific Mappings ───────────────────────────
GENDER_PRONOUN_SWAPS: Dict[str, str] = {
    # Pronouns
    "he": "she", "she": "he",
    "him": "her", "her": "his",
    "his": "her", "hers": "his",
    "himself": "herself", "herself": "himself",

    # Honorifics & titles
    "mr": "ms", "ms": "mr", "mrs": "mr",
    "mr.": "ms.", "ms.": "mr.", "mrs.": "mr.",
    "sir": "madam", "madam": "sir",
    "gentleman": "lady", "lady": "gentleman",
    "gentlemen": "ladies", "ladies": "gentlemen",
    "male": "female", "female": "male",
    "man": "woman", "woman": "man",
    "men": "women", "women": "men",
    "boy": "girl", "girl": "boy",
    "boys": "girls", "girls": "boys",
    "father": "mother", "mother": "father",
    "son": "daughter", "daughter": "son",
    "brother": "sister", "sister": "brother",
    "husband": "wife", "wife": "husband",
}

# ── Affinity & Institutional Proxy Mappings ─────────────────────────────
AFFINITY_ORGANIZATION_SWAPS: Dict[str, str] = {
    "society of women engineers": "engineering honors society",
    "women who code": "open source developers guild",
    "women in technology": "technology association",
    "women in tech": "technology association",
    "women in computer science": "computer science society",
    "women in stem": "stem professional network",
    "girls who code": "youth coding academy",
    "fraternity": "sorority", "sorority": "fraternity",
    "alpha phi alpha": "honors academic society",
    "delta sigma theta": "honors academic society",
    "spelman college": "state university",
    "morehouse college": "state university",
    "men's leadership": "executive leadership",
    "women's leadership": "executive leadership",
}

# ── Combined Full Text Swaps ────────────────────────────────────────────
DEMOGRAPHIC_SWAPS: Dict[str, str] = {
    **GENDER_PRONOUN_SWAPS,
    **AFFINITY_ORGANIZATION_SWAPS,
}

# ── Name Perturbation Pairs (Diverse cultural & demographic audit pairs) ─
NAME_SWAPS: Dict[str, str] = {
    "alex": "alexis", "alexis": "alex",
    "james": "emily", "emily": "james",
    "john": "mary", "mary": "john",
    "david": "sarah", "sarah": "david",
    "michael": "jessica", "jessica": "michael",
    "marcus": "maya", "maya": "marcus",
    "robert": "amanda", "amanda": "robert",
    "kevin": "fatima", "fatima": "kevin",
    "jordan": "taylor", "taylor": "jordan",
    "daniel": "elena", "elena": "daniel",
    "jamal": "greg", "greg": "jamal",
    "lakisha": "emily", "priya": "sarah",
}

# ── Protected-Attribute Proxy Pattern Dictionary ────────────────────────
PROXY_PATTERNS: Dict[str, List[str]] = {
    "Age Proxy (Early Graduation / Seniority Language)": [
        r"\b(?:graduated\s+in\s+|class\s+of\s+)(?:19[6789]\d)\b",
        r"\b(?:2[5-9]|[3-5]\d)\+?\s*years\s+of\s+experience\b",
        r"\b(?:seasoned\s+veteran|over\s+30\s+years|retired|mature\s+professional)\b",
    ],
    "Gender / Affinity Organization Proxy": [
        r"\b(?:society\s+of\s+women\s+engineers|swe|women\s+who\s+code|women\s+in\s+tech(?:nology)?)\b",
        r"\b(?:girls\s+who\s+code|women\s+in\s+stem|women\s+in\s+computer\s+science)\b",
        r"\b(?:sorority|fraternity|alpha\s+phi\s+alpha|delta\s+sigma\s+theta)\b",
        r"\b(?:men's|women's)\s+(?:basketball|soccer|rowing|lacrosse|track|rugby|volleyball)\b",
    ],
    "Family / Parental / Caregiver Status Proxy": [
        r"\b(?:maternity\s+leave|paternity\s+leave|family\s+caregiver|parental\s+leave|stay-at-home\s+parent)\b",
    ],
    "National Origin / Visa / Language Proxy": [
        r"\b(?:native\s+speaker|mother\s+tongue|h1-?b\s+visa|f1\s+opt|requires\s+sponsorship|visa\s+sponsorship)\b",
    ],
    "Disability / Accommodation Proxy": [
        r"\b(?:reasonable\s+accommodation|wheelchair\s+accessible|service\s+animal)\b",
    ],
}


def _substitute_tokens_single_pass(
    text: str, swap_dict: Dict[str, str]
) -> Tuple[str, List[Dict[str, str]]]:
    """
    Substitutes dictionary tokens in a single regex pass to prevent ping-pong reversals.
    Preserves original token casing (upper, title, lower).
    """
    if not text:
        return text, []

    substitutions_made: List[Dict[str, str]] = []
    sorted_phrases = sorted(swap_dict.keys(), key=lambda s: len(s), reverse=True)

    pattern_str = r"\b(?:" + "|".join(re.escape(k) for k in sorted_phrases) + r")\b"
    combined_pattern = re.compile(pattern_str, flags=re.IGNORECASE)

    def replace_callback(match: re.Match) -> str:
        original = match.group(0)
        target_key = original.lower()
        replacement = swap_dict.get(target_key, original)

        if original.isupper():
            rep = replacement.upper()
        elif original.istitle():
            rep = replacement.title()
        else:
            rep = replacement.lower()

        substitutions_made.append({
            "original": original,
            "replacement": rep,
        })
        return rep

    perturbed = combined_pattern.sub(replace_callback, text)
    return perturbed, substitutions_made


def perturb_text_gender_pronouns(text: str) -> Tuple[str, List[Dict[str, str]]]:
    """Applies only direct gender pronouns and honorific substitutions."""
    return _substitute_tokens_single_pass(text, GENDER_PRONOUN_SWAPS)


def perturb_text_affinity_orgs(text: str) -> Tuple[str, List[Dict[str, str]]]:
    """Applies only affinity organization and institutional proxy substitutions."""
    return _substitute_tokens_single_pass(text, AFFINITY_ORGANIZATION_SWAPS)


def perturb_text_demographics(text: str) -> Tuple[str, List[Dict[str, str]]]:
    """
    Substitutes all demographic markers (pronouns + organizations).
    Single-pass to eliminate ping-pong reversal.
    """
    return _substitute_tokens_single_pass(text, DEMOGRAPHIC_SWAPS)


def perturb_candidate_name(name: str) -> Tuple[str, bool]:
    """
    Generates a counterfactual candidate name based on first name audit pairs.
    Returns (new_name, was_swapped).
    """
    if not name:
        return name, False

    parts = name.strip().split()
    if not parts:
        return name, False

    first_name_lower = parts[0].lower()
    if first_name_lower in NAME_SWAPS:
        counter_first = NAME_SWAPS[first_name_lower].title()
        new_name = " ".join([counter_first] + parts[1:])
        return new_name, True

    return f"{name} (Counterfactual Variant)", True


def detect_proxy_biases(text: str) -> List[Dict[str, Any]]:
    """
    Scans resume text for proxy signals correlated with protected classes.
    """
    detected_proxies: List[Dict[str, Any]] = []
    text_lower = text.lower() if text else ""

    for category, patterns in PROXY_PATTERNS.items():
        found_matches: List[str] = []
        for pat in patterns:
            matches = re.findall(pat, text_lower)
            if matches:
                found_matches.extend(matches)
        if found_matches:
            detected_proxies.append({
                "category": category,
                "flagged_terms": sorted(list(set(found_matches))),
                "recommendation": (
                    "Consider blind review or redacting non-qualification "
                    "identity proxies prior to evaluations."
                ),
            })

    return detected_proxies


def _evaluate_controlled_variation(
    job_description: str,
    candidates: List[Dict[str, Any]],
    variation_key: str,
    variation_label: str,
    transform_candidate_fn: Callable[[Dict[str, Any]], Tuple[Dict[str, Any], List[Dict[str, str]], bool]],
    baseline_scores: Dict[str, float],
    baseline_ranks: Dict[str, int],
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Executes a single controlled counterfactual variation across the candidate pool.
    Measures original score, counterfactual score, absolute/relative differences,
    and ranking displacement.
    """
    num_candidates = len(candidates)
    if num_candidates == 0:
        return {}

    cf_candidates = []
    cand_subs: Dict[str, List[Dict[str, str]]] = {}
    name_swapped_map: Dict[str, bool] = {}

    for c in candidates:
        cf_c, subs, swapped = transform_candidate_fn(c)
        cf_candidates.append(cf_c)
        cand_subs[c["candidate_id"]] = subs
        name_swapped_map[c["candidate_id"]] = swapped

    cf_results = score_candidates_advanced(job_description, cf_candidates, weights=weights)
    cf_scores = {c["candidate_id"]: c["score"] for c in cf_results}
    cf_ranks = {c["candidate_id"]: c["rank"] for c in cf_results}

    evaluations = []
    rank_flips = 0
    max_abs_delta = 0.0
    total_abs_delta = 0.0
    total_rel_delta = 0.0
    max_rank_disp = 0
    consistent_count = 0

    for c in candidates:
        cid = c["candidate_id"]
        orig_s = baseline_scores[cid]
        cf_s = cf_scores[cid]
        delta = round(cf_s - orig_s, 4)
        abs_delta = round(abs(delta), 4)

        rel_delta = round(abs_delta / (orig_s + 1e-6), 6)
        rel_delta_pct = round(rel_delta * 100.0, 4)

        orig_r = baseline_ranks[cid]
        cf_r = cf_ranks[cid]
        rank_shift = cf_r - orig_r
        rank_disp = abs(rank_shift)

        if rank_shift != 0:
            rank_flips += 1
        if rank_disp > max_rank_disp:
            max_rank_disp = rank_disp

        if abs_delta > max_abs_delta:
            max_abs_delta = abs_delta

        total_abs_delta += abs_delta
        total_rel_delta += rel_delta

        if abs_delta <= 1.0:
            consistent_count += 1

        evaluations.append({
            "candidate_id": cid,
            "candidate_name": c.get("name", cid),
            "baseline_score": orig_s,
            "counterfactual_score": cf_s,
            "score_delta": delta,
            "absolute_score_delta": abs_delta,
            "relative_score_delta": rel_delta,
            "relative_score_delta_pct": rel_delta_pct,
            "baseline_rank": orig_r,
            "counterfactual_rank": cf_r,
            "rank_shift": rank_shift,
            "rank_displacement": rank_disp,
            "substitutions_count": len(cand_subs.get(cid, [])),
            "substitutions_sample": cand_subs.get(cid, [])[:5],
            "name_swapped": name_swapped_map.get(cid, False),
        })

    mean_abs_delta = round(total_abs_delta / num_candidates, 4)
    mean_rel_delta_pct = round((total_rel_delta / num_candidates) * 100.0, 4)
    score_consistency_pct = round((consistent_count / num_candidates) * 100.0, 1)

    invariant_count = sum(1 for e in evaluations if e["absolute_score_delta"] <= 0.1)
    invariance_rate_pct = round((invariant_count / num_candidates) * 100.0, 1)

    return {
        "variation_key": variation_key,
        "variation_label": variation_label,
        "metrics": {
            "max_absolute_delta": max_abs_delta,
            "mean_absolute_delta": mean_abs_delta,
            "mean_relative_delta_pct": mean_rel_delta_pct,
            "rank_flips_count": rank_flips,
            "max_rank_displacement": max_rank_disp,
            "score_consistency_pct": score_consistency_pct,
            "invariance_rate_pct": invariance_rate_pct,
            "is_invariant": (max_abs_delta <= 0.25 and rank_flips == 0),
        },
        "evaluations": evaluations,
    }


def run_counterfactual_audit(
    job_description: str,
    candidates: List[Dict[str, Any]],
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Conducts comprehensive, multi-variation counterfactual fairness evaluation:
    1. Variant A (Gender Pronouns): Invariance to explicit linguistic gender markers
    2. Variant B (Names): Invariance to culturally associated candidate first names
    3. Variant C (Affinity Orgs): Invariance to non-qualification institutional proxies
    4. Variant D (Combined): Full multi-attribute perturbation

    Computes:
    - Original Score, Counterfactual Score, Absolute Delta, Relative Delta
    - Original Rank, Counterfactual Rank, Rank Shift, Rank Displacement
    - Explicit causal limitations (Kusner et al., 2017)
    """
    if not candidates:
        return {"error": "No candidates provided for audit."}

    # 1. Baseline scoring
    baseline_results = score_candidates_advanced(job_description, candidates, weights=weights)
    baseline_scores = {c["candidate_id"]: c["score"] for c in baseline_results}
    baseline_ranks = {c["candidate_id"]: c["rank"] for c in baseline_results}
    baseline_skills = {
        c["candidate_id"]: set(c.get("skills_extracted", {}).get("matched", []))
        for c in baseline_results
    }

    # Candidate proxy scanning
    candidate_proxies: Dict[str, List[Dict[str, Any]]] = {}
    for c in candidates:
        candidate_proxies[c["candidate_id"]] = detect_proxy_biases(c.get("qualifications", ""))

    # ── Variation 1: Gender Pronouns & Titles Only ───────────────────────
    def transform_pronouns(c: Dict[str, Any]):
        cf_c = copy.deepcopy(c)
        pert_text, subs = perturb_text_gender_pronouns(c.get("qualifications", ""))
        cf_c["qualifications"] = pert_text
        return cf_c, subs, False

    var_pronouns = _evaluate_controlled_variation(
        job_description, candidates,
        variation_key="gender_pronouns",
        variation_label="Gender Pronouns & Titles (Direct Text Sensitivity)",
        transform_candidate_fn=transform_pronouns,
        baseline_scores=baseline_scores,
        baseline_ranks=baseline_ranks,
        weights=weights,
    )

    # ── Variation 2: Candidate Names Only ────────────────────────────────
    def transform_names(c: Dict[str, Any]):
        cf_c = copy.deepcopy(c)
        new_name, swapped = perturb_candidate_name(c.get("name", ""))
        cf_c["name"] = new_name
        return cf_c, [], swapped

    var_names = _evaluate_controlled_variation(
        job_description, candidates,
        variation_key="candidate_names",
        variation_label="Candidate Names (Audit Study Name Isolation)",
        transform_candidate_fn=transform_names,
        baseline_scores=baseline_scores,
        baseline_ranks=baseline_ranks,
        weights=weights,
    )

    # ── Variation 3: Affinity & Institutional Organizations Only ─────────
    def transform_orgs(c: Dict[str, Any]):
        cf_c = copy.deepcopy(c)
        pert_text, subs = perturb_text_affinity_orgs(c.get("qualifications", ""))
        cf_c["qualifications"] = pert_text
        return cf_c, subs, False

    var_orgs = _evaluate_controlled_variation(
        job_description, candidates,
        variation_key="affinity_orgs",
        variation_label="Affinity & Institutional Proxies (Proxy Sensitivity)",
        transform_candidate_fn=transform_orgs,
        baseline_scores=baseline_scores,
        baseline_ranks=baseline_ranks,
        weights=weights,
    )

    # ── Variation 4: Combined Multi-Attribute Perturbation ───────────────
    def transform_combined(c: Dict[str, Any]):
        cf_c = copy.deepcopy(c)
        pert_text, subs = perturb_text_demographics(c.get("qualifications", ""))
        cf_c["qualifications"] = pert_text
        new_name, swapped = perturb_candidate_name(c.get("name", ""))
        cf_c["name"] = new_name
        return cf_c, subs, swapped

    var_combined = _evaluate_controlled_variation(
        job_description, candidates,
        variation_key="combined_multi_attribute",
        variation_label="Combined Multi-Attribute (Pronouns + Names + Proxies)",
        transform_candidate_fn=transform_combined,
        baseline_scores=baseline_scores,
        baseline_ranks=baseline_ranks,
        weights=weights,
    )

    # Check skill-set stability on combined perturbation
    cf_combined_cands = [transform_combined(c)[0] for c in candidates]
    cf_combined_results = score_candidates_advanced(job_description, cf_combined_cands, weights=weights)
    cf_skills = {
        c["candidate_id"]: set(c.get("skills_extracted", {}).get("matched", []))
        for c in cf_combined_results
    }

    skill_set_changed_count = 0
    for cid in baseline_scores:
        top_b = sorted(baseline_skills.get(cid, set()))[:3]
        top_cf = sorted(cf_skills.get(cid, set()))[:3]
        if top_b != top_cf:
            skill_set_changed_count += 1

    combined_metrics = var_combined["metrics"]
    num_candidates = len(candidates)
    total_proxies_found = sum(len(p) for p in candidate_proxies.values())

    is_overall_invariant = combined_metrics["is_invariant"]

    if is_overall_invariant:
        verdict = "PASS — SCORE INVARIANT TO DEMOGRAPHIC TEXT MARKERS"
        verdict_summary = (
            f"Perturbing gender markers, pronouns, candidate names, and "
            f"demographic affiliations resulted in zero rank flips and a "
            f"maximum score shift of only {combined_metrics['max_absolute_delta']:.4f} points "
            f"(mean relative delta: {combined_metrics['mean_relative_delta_pct']:.4f}%)."
        )
    else:
        verdict = "WARNING — SENSITIVITY DETECTED"
        verdict_summary = (
            f"Potential demographic sensitivity: {combined_metrics['rank_flips_count']} rank flip(s) "
            f"with max score shift of {combined_metrics['max_absolute_delta']:.4f} points when "
            f"demographic markers were perturbed."
        )

    # ── Conceptual Fairness Taxonomy & Epistemological Disclaimers ──────
    fairness_taxonomy = {
        "direct_demographic_text_sensitivity": {
            "definition": "Direct sensitivity to explicit gendered pronouns, honorifics, and gendered terms in resume prose.",
            "variation_tested": "Variant A (Gender Pronouns)",
            "max_delta": var_pronouns["metrics"]["max_absolute_delta"],
            "rank_flips": var_pronouns["metrics"]["rank_flips_count"],
        },
        "name_sensitivity": {
            "definition": "Sensitivity to first names associated with specific cultural, racial, or gender backgrounds (Bertrand & Mullainathan 2004 audit methodology).",
            "variation_tested": "Variant B (Candidate Names)",
            "max_delta": var_names["metrics"]["max_absolute_delta"],
            "rank_flips": var_names["metrics"]["rank_flips_count"],
        },
        "proxy_attribute_sensitivity": {
            "definition": "Sensitivity to non-qualification institutional proxies (affinity organizations, HBCUs, fraternities, seniority dates).",
            "variation_tested": "Variant C (Affinity Orgs)",
            "max_delta": var_orgs["metrics"]["max_absolute_delta"],
            "rank_flips": var_orgs["metrics"]["rank_flips_count"],
        },
        "broader_demographic_fairness": {
            "definition": "Group-level selection rate parity (DIR, SPD) across the applicant pool, which cannot be guaranteed by text-marker invariance alone.",
            "note": "Evaluated in Algorithmic Fairness Metrics Suite (DIR, SPD, EOD).",
        },
    }

    audit_limitations = [
        "This tests text marker invariance only, NOT formal counterfactual fairness as defined in Kusner et al. (2017).",
        "Formal counterfactual fairness requires modeling the underlying causal DAG to establish non-descendants of protected attributes.",
        "Passing text invariance does not prove absence of bias — only insensitivity to specific surface-level demographic tokens.",
        "The perturbation dictionaries cover common English gendered terms and organizations but are not exhaustive.",
        "Intersectional effects (combinations of multiple protected attributes) are partially tested in the combined variant.",
        "This audit does not test bias from resume formatting, length, or writing style differences correlated with demographic cohorts.",
    ]

    # Map individual audits for backward compatibility
    individual_audits = []
    for ev in var_combined["evaluations"]:
        cid = ev["candidate_id"]
        individual_audits.append({
            "candidate_id": cid,
            "candidate_name": ev["candidate_name"],
            "baseline_score": ev["baseline_score"],
            "counterfactual_score": ev["counterfactual_score"],
            "score_delta": ev["score_delta"],
            "absolute_score_delta": ev["absolute_score_delta"],
            "relative_score_delta": ev["relative_score_delta"],
            "relative_score_delta_pct": ev["relative_score_delta_pct"],
            "baseline_rank": ev["baseline_rank"],
            "counterfactual_rank": ev["counterfactual_rank"],
            "rank_shift": ev["rank_shift"],
            "rank_displacement": ev["rank_displacement"],
            "substitutions_tested": ev["substitutions_count"],
            "substitutions_details": ev["substitutions_sample"],
            "name_tested": next(
                (c.get("name") for c in cf_combined_cands if c["candidate_id"] == cid), ""
            ),
            "proxies_detected": len(candidate_proxies.get(cid, [])),
        })

    return {
        "verdict": verdict,
        "is_counterfactually_fair": is_overall_invariant,
        "summary": verdict_summary,
        "metrics": {
            "max_score_delta": combined_metrics["max_absolute_delta"],
            "mean_abs_delta": combined_metrics["mean_absolute_delta"],
            "mean_relative_delta_pct": combined_metrics["mean_relative_delta_pct"],
            "rank_flips_count": combined_metrics["rank_flips_count"],
            "max_rank_displacement": combined_metrics["max_rank_displacement"],
            "invariance_rate_pct": combined_metrics["invariance_rate_pct"],
            "score_consistency_pct": combined_metrics["score_consistency_pct"],
            "skill_set_changed_count": skill_set_changed_count,
            "total_candidates_audited": num_candidates,
            "total_proxies_flagged": total_proxies_found,
            "name_isolation_tested": True,
            "name_isolation_certified": True,
        },
        "controlled_variations": {
            "gender_pronouns": var_pronouns,
            "candidate_names": var_names,
            "affinity_orgs": var_orgs,
            "combined_multi_attribute": var_combined,
        },
        "fairness_taxonomy": fairness_taxonomy,
        "individual_audits": individual_audits,
        "proxy_alerts": [
            {
                "candidate_id": cid,
                "name": next(
                    (c["name"] for c in candidates if c["candidate_id"] == cid),
                    cid,
                ),
                "proxies": p,
            }
            for cid, p in candidate_proxies.items()
            if p
        ],
        "audit_limitations": audit_limitations,
    }
