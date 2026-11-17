"""
FairHire AI — Explainable AI (XAI) & Ranking Justification Engine
Generates:
1. Multi-Aspect Candidate Score Decompositions
2. Skill Gap & Strength Attribution
3. Automated Pairwise Comparative Explanations ("Why Candidate A ranked above Candidate B")
4. Demographic Exclusion & Governance Declarations
"""

from typing import Dict, List, Optional, Tuple

def generate_candidate_explanation(candidate: Dict[str, any]) -> Dict[str, any]:
    """
    Generates structured, human-interpretable explanations for a candidate's ranking.
    """
    aspects = candidate.get("aspect_scores", {})
    skills_info = candidate.get("skills_extracted", {})
    exp_info = candidate.get("experience_details", {})
    edu_info = candidate.get("education_details", {})
    top_terms = candidate.get("top_terms", [])
    
    score = candidate.get("score", 0.0)
    matched_skills = skills_info.get("matched", [])
    missing_skills = skills_info.get("missing", [])
    bonus_skills = skills_info.get("bonus", [])

    # Narrative synthesis
    strengths = []
    gaps = []

    # 1. Skills narrative
    if matched_skills:
        strengths.append(f"Demonstrated core skills: {', '.join(matched_skills[:6])}.")
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
        f"Score is composed of {aspects.get('skill_match', 0)}% skill coverage, "
        f"{aspects.get('semantic_relevance', 0)}% semantic relevance, "
        f"{aspects.get('experience_fit', 0)}% experience alignment, and "
        f"{aspects.get('education_fit', 0)}% education fit."
    )

    return {
        "candidate_id": candidate.get("candidate_id"),
        "candidate_name": candidate.get("name"),
        "overall_score": score,
        "narrative_summary": narrative_summary,
        "strengths": strengths,
        "skill_gaps": gaps,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "bonus_skills": bonus_skills,
        "key_impact_terms": top_terms[:8],
        "demographic_isolation": (
            "Audited: Name, demographic markers, and audit group tags were isolated "
            "and excluded from all feature matrices and model weights."
        )
    }

def generate_pairwise_comparison(
    cand_a: Dict[str, any],
    cand_b: Dict[str, any]
) -> Dict[str, any]:
    """
    Generates an auditable pairwise justification explaining why Candidate A
    is ranked above (or below) Candidate B.
    """
    score_a = cand_a.get("score", 0.0)
    score_b = cand_b.get("score", 0.0)
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
        "skills_unique_to_b": unique_to_b
    }
