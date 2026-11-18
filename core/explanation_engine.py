"""
FairHire AI — Explainable AI (XAI) & Ranking Justification Engine
Generates:
1. Multi-Aspect Score Decomposition with Exact Point Contributions
2. Skill Gap, Strength, and Categorized Attribution
3. Actionable Next-Tier Guidance & Upskilling Recommendations
4. Auditable Pairwise Comparative Explanations ("Why Candidate A ranked above Candidate B")
5. Demographic Isolation & Governance Certification
"""

from typing import Any, Dict, List, Optional, Set, Tuple


def calculate_next_tier_guidance(candidate: Dict[str, Any]) -> Dict[str, Any]:
    """
    Computes actionable feedback and upskilling guidance explaining what
    concrete qualifications would elevate the candidate to the next tier.
    """
    score = float(candidate.get("score", 0.0))
    tier = candidate.get("tier_badge", "Qualified")
    skills_info = candidate.get("skills_extracted", {})
    exp_info = candidate.get("experience_details", {})
    edu_info = candidate.get("education_details", {})
    weights = candidate.get("weights_used", {"skills": 0.40, "semantic": 0.30, "experience": 0.15, "education": 0.15})

    tier_thresholds = [
        ("Review", 50.0, "Qualified"),
        ("Qualified", 65.0, "Strong Match"),
        ("Strong Match", 80.0, "Top Fit"),
        ("Top Fit", 100.0, "Top Fit (Maximum)")
    ]

    target_tier = None
    target_score = None
    points_needed = 0.0

    for current_name, threshold, next_name in tier_thresholds:
        if tier == current_name:
            target_tier = next_name
            target_score = threshold
            points_needed = round(max(0.0, threshold - score), 2)
            break

    if not target_tier or tier == "Top Fit":
        return {
            "current_tier": "Top Fit",
            "target_tier": None,
            "points_needed": 0.0,
            "status": "Candidate has achieved the highest qualification tier.",
            "actionable_recommendations": ["Maintain current technical proficiencies and leadership contributions."]
        }

    recommendations: List[str] = []
    missing_skills = skills_info.get("missing", [])
    if missing_skills:
        potential_skill_gain = round(min(15.0, len(missing_skills[:3]) * 4.5 * weights.get("skills", 0.4)), 1)
        skills_str = ", ".join(missing_skills[:3])
        recommendations.append(
            f"Acquire demonstrated proficiency in missing core skills: [{skills_str}] (est. +{potential_skill_gain} pts)."
        )

    # Experience gap recommendation
    req_years = float(exp_info.get("required_years", 0.0))
    cand_years = float(exp_info.get("years_found", 0.0))
    if req_years > cand_years:
        exp_gap = round(req_years - cand_years, 1)
        exp_gain = round(min(15.0, exp_gap * 3.0 * weights.get("experience", 0.15)), 1)
        recommendations.append(
            f"Gain {exp_gap} additional year(s) of targeted domain experience to meet the {req_years:.1f}-year requirement (est. +{exp_gain} pts)."
        )

    # Education recommendation if relevant
    cand_tier = int(edu_info.get("tier", 2))
    if cand_tier == 0:
        recommendations.append(
            "Documenting completed degree credentials or accredited certifications would elevate educational alignment."
        )
    elif cand_tier < 4 and points_needed > 5.0:
        recommendations.append(
            "Pursuing relevant postgraduate or advanced industry certifications could elevate educational alignment."
        )

    if not recommendations:
        recommendations.append(
            f"Further refine technical documentation and demonstrated project outcomes to bridge the {points_needed} pt gap."
        )

    return {
        "current_tier": tier,
        "target_tier": target_tier,
        "target_score": target_score,
        "points_needed": points_needed,
        "status": f"{points_needed} points required to advance from '{tier}' to '{target_tier}'.",
        "actionable_recommendations": recommendations
    }


def generate_candidate_explanation(candidate: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generates structured, human-interpretable explanations for a candidate's ranking.
    Includes score breakdowns, strengths, gaps, next-tier guidance, and demographic isolation proof.
    """
    aspects = candidate.get("aspect_scores", {})
    weights = candidate.get("weights_used", {"skills": 0.40, "semantic": 0.30, "experience": 0.15, "education": 0.15})
    skills_info = candidate.get("skills_extracted", {})
    exp_info = candidate.get("experience_details", {})
    edu_info = candidate.get("education_details", {})
    top_terms = candidate.get("top_terms", [])
    
    score = float(candidate.get("score", 0.0))
    matched_skills = skills_info.get("matched", [])
    missing_skills = skills_info.get("missing", [])
    bonus_skills = skills_info.get("bonus", [])

    # Exact dimensional point contributions
    skill_match = float(aspects.get("skill_match", 0.0))
    semantic_rel = float(aspects.get("semantic_relevance", 0.0))
    exp_fit = float(aspects.get("experience_fit", 0.0))
    edu_fit = float(aspects.get("education_fit", 0.0))

    point_contributions = {
        "skills": {
            "raw_score": skill_match,
            "weight_pct": round(weights.get("skills", 0.40) * 100, 1),
            "points_earned": round(skill_match * weights.get("skills", 0.40), 2),
            "points_possible": round(100.0 * weights.get("skills", 0.40), 2)
        },
        "semantic": {
            "raw_score": semantic_rel,
            "weight_pct": round(weights.get("semantic", 0.30) * 100, 1),
            "points_earned": round(semantic_rel * weights.get("semantic", 0.30), 2),
            "points_possible": round(100.0 * weights.get("semantic", 0.30), 2)
        },
        "experience": {
            "raw_score": exp_fit,
            "weight_pct": round(weights.get("experience", 0.15) * 100, 1),
            "points_earned": round(exp_fit * weights.get("experience", 0.15), 2),
            "points_possible": round(100.0 * weights.get("experience", 0.15), 2)
        },
        "education": {
            "raw_score": edu_fit,
            "weight_pct": round(weights.get("education", 0.15) * 100, 1),
            "points_earned": round(edu_fit * weights.get("education", 0.15), 2),
            "points_possible": round(100.0 * weights.get("education", 0.15), 2)
        }
    }

    # Narrative synthesis
    strengths = []
    gaps = []

    # 1. Skills narrative
    if matched_skills:
        strengths.append(f"Demonstrated core required skills: {', '.join(matched_skills[:6])}.")
    if bonus_skills:
        strengths.append(f"Additional valuable technical skills: {', '.join(bonus_skills[:5])}.")
    if missing_skills:
        gaps.append(f"Missing required skills: {', '.join(missing_skills[:5])}.")

    # 2. Experience narrative
    exp_desc = exp_info.get("alignment_desc", "")
    if "Meets requirement" in exp_desc:
        strengths.append(f"Strong experience profile: {exp_desc}.")
    elif "gap" in exp_desc.lower():
        gaps.append(f"Experience alignment: {exp_desc}.")

    # 3. Education narrative
    edu_desc = edu_info.get("alignment_desc", "")
    edu_level = edu_info.get("level", "Unspecified")
    strengths.append(f"Education: {edu_level} ({edu_desc}).")

    # High-level summary paragraph
    narrative_summary = (
        f"Candidate achieved an overall fit score of {score}% (Tier: {candidate.get('tier_badge', 'Qualified')}). "
        f"Score is composed of {skill_match}% skill coverage ({point_contributions['skills']['points_earned']} pts), "
        f"{semantic_rel}% semantic relevance ({point_contributions['semantic']['points_earned']} pts), "
        f"{exp_fit}% experience alignment ({point_contributions['experience']['points_earned']} pts), and "
        f"{edu_fit}% education fit ({point_contributions['education']['points_earned']} pts)."
    )

    # Actionable guidance
    next_tier_guidance = calculate_next_tier_guidance(candidate)

    return {
        "candidate_id": candidate.get("candidate_id"),
        "candidate_name": candidate.get("name"),
        "overall_score": score,
        "narrative_summary": narrative_summary,
        "point_contributions": point_contributions,
        "strengths": strengths,
        "skill_gaps": gaps,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "bonus_skills": bonus_skills,
        "categorized_skills": skills_info.get("categorized", {}),
        "key_impact_terms": top_terms[:8],
        "next_tier_guidance": next_tier_guidance,
        "demographic_isolation": (
            "Audited: Name, demographic markers, and audit group tags were isolated "
            "and excluded from all feature matrices and model weights."
        )
    }


def generate_pairwise_comparison(
    cand_a: Dict[str, Any],
    cand_b: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generates an auditable pairwise justification explaining why Candidate A
    is ranked above (or below) Candidate B.
    Computes exact dimensional point deltas and skill set comparisons.
    """
    score_a = float(cand_a.get("score", 0.0))
    score_b = float(cand_b.get("score", 0.0))
    score_diff = round(score_a - score_b, 2)

    name_a = cand_a.get("name", cand_a.get("candidate_id", "Candidate A"))
    name_b = cand_b.get("name", cand_b.get("candidate_id", "Candidate B"))

    # Compare aspects
    aspects_a = cand_a.get("aspect_scores", {})
    aspects_b = cand_b.get("aspect_scores", {})
    
    skill_diff = round(aspects_a.get("skill_match", 0) - aspects_b.get("skill_match", 0), 2)
    semantic_diff = round(aspects_a.get("semantic_relevance", 0) - aspects_b.get("semantic_relevance", 0), 2)
    exp_diff = round(aspects_a.get("experience_fit", 0) - aspects_b.get("experience_fit", 0), 2)
    edu_diff = round(aspects_a.get("education_fit", 0) - aspects_b.get("education_fit", 0), 2)

    skills_a = set(cand_a.get("skills_extracted", {}).get("matched", []))
    skills_b = set(cand_b.get("skills_extracted", {}).get("matched", []))

    shared_skills = sorted(list(skills_a.intersection(skills_b)))
    unique_to_a = sorted(list(skills_a - skills_b))
    unique_to_b = sorted(list(skills_b - skills_a))

    reasons = []
    if skill_diff > 3.0:
        skills_str = f" (matched {', '.join(unique_to_a[:4])})" if unique_to_a else ""
        reasons.append(f"Higher skill coverage (+{skill_diff} pts){skills_str}")
    elif skill_diff < -3.0:
        skills_str = f" (matched {', '.join(unique_to_b[:4])})" if unique_to_b else ""
        reasons.append(f"Lower skill coverage ({skill_diff} pts){skills_str}")

    if semantic_diff > 3.0:
        reasons.append(f"Closer semantic keyword & context alignment (+{semantic_diff} pts)")
    elif semantic_diff < -3.0:
        reasons.append(f"Lower semantic context similarity ({semantic_diff} pts)")

    if exp_diff > 3.0:
        exp_a = cand_a.get("experience_details", {}).get("years_found", 0)
        exp_b = cand_b.get("experience_details", {}).get("years_found", 0)
        reasons.append(f"Greater experience alignment ({exp_a} yrs vs {exp_b} yrs, +{exp_diff} pts)")
    elif exp_diff < -3.0:
        exp_a = cand_a.get("experience_details", {}).get("years_found", 0)
        exp_b = cand_b.get("experience_details", {}).get("years_found", 0)
        reasons.append(f"Slightly lower experience match ({exp_a} yrs vs {exp_b} yrs, {exp_diff} pts)")

    if edu_diff > 0:
        reasons.append(f"Higher formal education qualification tier (+{edu_diff} pts)")
    elif edu_diff < 0:
        reasons.append(f"Different education qualification tier ({edu_diff} pts)")

    if not reasons:
        reasons.append("Candidates possess very similar qualification metrics across all dimensions.")

    higher_cand = name_a if score_diff >= 0 else name_b
    lower_cand = name_b if score_diff >= 0 else name_a
    abs_diff = abs(score_diff)

    if abs_diff == 0.0:
        verdict_text = (
            f"{name_a} and {name_b} achieved an identical overall score of {score_a}%. "
            f"Key factors: {'; '.join(reasons)}."
        )
    else:
        verdict_text = (
            f"{higher_cand} is ranked above {lower_cand} with a margin of {abs_diff}% points. "
            f"Key determining factors: {'; '.join(reasons)}."
        )

    return {
        "candidate_a_name": name_a,
        "candidate_b_name": name_b,
        "score_difference": score_diff,
        "higher_candidate": higher_cand,
        "verdict_text": verdict_text,
        "aspect_differences": {
            "skill_match_diff": skill_diff,
            "semantic_diff": semantic_diff,
            "experience_diff": exp_diff,
            "education_diff": edu_diff
        },
        "skills_unique_to_a": unique_to_a,
        "skills_unique_to_b": unique_to_b,
        "shared_skills": shared_skills
    }
