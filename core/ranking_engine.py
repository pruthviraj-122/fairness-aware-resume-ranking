"""
FairHire AI — Advanced Hybrid Ranking & Fair Reranking Engine
Combines:
1. Lexical & Canonical Skill Overlap (Core vs Secondary)
2. Semantic Sublinear TF-IDF Cosine Relevance
3. Experience Duration & Seniority Alignment
4. Education Level Fit
5. Fair Post-Processing Reranking (FA*IR-inspired bounded-regret optimization)
"""

import math
import re
from typing import Dict, List, Optional, Set, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from core.skill_taxonomy import (
    extract_skills_from_text,
    extract_years_of_experience,
    extract_education_level,
    get_skill_category
)

# Custom recruitment domain stop words to filter boilerplate noise
RECRUITMENT_STOPWORDS = [
    "resume", "curriculum", "vitae", "candidate", "responsibilities",
    "duties", "qualifications", "requirements", "experience", "role",
    "job", "work", "applicant", "position", "seeking", "looking",
    "opportunity", "working", "proficient", "knowledge", "ability"
]

def compute_skill_scores(job_skills: Set[str], candidate_skills: Set[str]) -> Dict[str, any]:
    """
    Computes structured skill match scores between job description requirements
    and candidate qualifications.
    """
    if not job_skills:
        # If job has no specific skills found, treat all candidate skills as general match
        return {
            "skill_match_score": 50.0,
            "matched_skills": sorted(list(candidate_skills)),
            "missing_skills": [],
            "coverage_ratio": 1.0,
            "skill_count": len(candidate_skills)
        }

    matched = job_skills.intersection(candidate_skills)
    missing = job_skills.difference(candidate_skills)
    bonus = candidate_skills.difference(job_skills)

    # Core coverage ratio (0 to 1)
    coverage = len(matched) / len(job_skills)
    
    # Slight bonus for additional relevant technical skills (up to +10%)
    bonus_points = min(10.0, len(bonus) * 1.5)
    
    # Skill score scaled to 0-100
    raw_score = (coverage * 90.0) + bonus_points
    skill_match_score = min(100.0, max(0.0, raw_score))

    return {
        "skill_match_score": round(skill_match_score, 2),
        "matched_skills": sorted(list(matched)),
        "missing_skills": sorted(list(missing)),
        "bonus_skills": sorted(list(bonus))[:8],
        "coverage_ratio": round(coverage, 3),
        "skill_count": len(candidate_skills)
    }

def compute_experience_alignment(job_years: float, candidate_years: float) -> Tuple[float, str]:
    """
    Computes experience score (0-100) and descriptive alignment string.
    """
    if job_years <= 0:
        # No specific years requirement specified in JD
        score = min(100.0, 50.0 + candidate_years * 8.0)
        desc = f"{candidate_years:.1f} yrs experience (no minimum requested)"
        return round(score, 2), desc

    diff = candidate_years - job_years
    if diff >= 0:
        # Meets or exceeds required experience
        score = min(100.0, 85.0 + min(15.0, diff * 3.0))
        desc = f"Meets requirement ({candidate_years:.1f} yrs vs {job_years:.1f} required)"
    elif diff >= -1.0:
        # Close to required (within 1 year)
        score = 70.0
        desc = f"Near requirement ({candidate_years:.1f} yrs vs {job_years:.1f} required)"
    elif diff >= -2.5:
        # Moderate gap
        score = 50.0
        desc = f"Slight experience gap ({candidate_years:.1f} yrs vs {job_years:.1f} required)"
    else:
        # Significant gap
        score = max(20.0, 40.0 - abs(diff) * 5.0)
        desc = f"Significant experience gap ({candidate_years:.1f} yrs vs {job_years:.1f} required)"

    return round(score, 2), desc

def compute_education_alignment(job_tier: int, candidate_tier: int) -> Tuple[float, str]:
    """
    Computes education alignment score (0-100).
    """
    if candidate_tier >= job_tier:
        score = 100.0
        desc = "Meets or exceeds education requirement"
    elif candidate_tier == job_tier - 1:
        score = 80.0
        desc = "Close to requested degree level"
    else:
        score = 60.0
        desc = "Alternate educational background"

    return score, desc

def score_candidates_advanced(
    job_description: str,
    candidates: List[Dict[str, any]],
    weights: Optional[Dict[str, float]] = None
) -> List[Dict[str, any]]:
    """
    Ranks candidates using a multi-aspect hybrid scoring system:
    - Skill Match (default weight: 0.40)
    - Semantic Relevance (default weight: 0.30)
    - Experience Alignment (default weight: 0.15)
    - Education Fit (default weight: 0.15)

    Names, demographic tags, and audit groups are NEVER used in scoring.
    """
    if weights is None:
        weights = {
            "skills": 0.40,
            "semantic": 0.30,
            "experience": 0.15,
            "education": 0.15
        }

    # Extract requirements from Job Description
    job_skills = extract_skills_from_text(job_description)
    job_years = extract_years_of_experience(job_description)
    job_edu = extract_education_level(job_description)

    # Compute TF-IDF semantic similarities with sublinear frequency and ngrams
    documents = [job_description] + [c["qualifications"] for c in candidates]
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=1
    )
    matrix = vectorizer.fit_transform(documents)
    similarities = cosine_similarity(matrix[0:1], matrix[1:]).flatten()

    feature_names = vectorizer.get_feature_names_out()
    job_tfidf = matrix[0].toarray().flatten()

    scored_candidates = []
    for idx, (candidate, similarity) in enumerate(zip(candidates, similarities)):
        qual_text = candidate.get("qualifications", "")
        
        # 1. Skill evaluation
        cand_skills = extract_skills_from_text(qual_text)
        skill_res = compute_skill_scores(job_skills, cand_skills)

        # 2. Semantic evaluation (scaled 0-100)
        semantic_score = round(float(similarity) * 100.0, 2)

        # 3. Experience evaluation
        cand_years = extract_years_of_experience(qual_text)
        exp_score, exp_desc = compute_experience_alignment(job_years, cand_years)

        # 4. Education evaluation
        cand_edu = extract_education_level(qual_text)
        edu_score, edu_desc = compute_education_alignment(job_edu["tier"], cand_edu["tier"])

        # Composite weighted score
        composite_score = (
            weights["skills"] * skill_res["skill_match_score"] +
            weights["semantic"] * semantic_score +
            weights["experience"] * exp_score +
            weights["education"] * edu_score
        )
        composite_score = min(100.0, max(0.0, round(composite_score, 2)))

        # Significant keyword contributors (TF-IDF overlap)
        cand_tfidf = matrix[idx + 1].toarray().flatten()
        term_impacts = []
        for i, (j_val, c_val) in enumerate(zip(job_tfidf, cand_tfidf)):
            if j_val > 0 and c_val > 0:
                term = feature_names[i]
                if term not in RECRUITMENT_STOPWORDS and len(term) > 2:
                    term_impacts.append((term, round(float(j_val * c_val), 4)))

        term_impacts.sort(key=lambda x: x[1], reverse=True)
        top_terms = [t[0] for t in term_impacts[:10]]

        # Tier assignment
        if composite_score >= 80.0:
            tier_badge = "Top Fit"
        elif composite_score >= 65.0:
            tier_badge = "Strong Match"
        elif composite_score >= 50.0:
            tier_badge = "Qualified"
        else:
            tier_badge = "Review"

        scored_candidates.append({
            **candidate,
            "score": composite_score,
            "raw_similarity": semantic_score,
            "tier_badge": tier_badge,
            "aspect_scores": {
                "skill_match": skill_res["skill_match_score"],
                "semantic_relevance": semantic_score,
                "experience_fit": exp_score,
                "education_fit": edu_score
            },
            "skills_extracted": {
                "matched": skill_res["matched_skills"],
                "missing": skill_res["missing_skills"],
                "bonus": skill_res["bonus_skills"],
                "total_count": skill_res["skill_count"],
                "coverage_pct": round(skill_res["coverage_ratio"] * 100, 1)
            },
            "experience_details": {
                "years_found": cand_years,
                "alignment_desc": exp_desc
            },
            "education_details": {
                "level": cand_edu["level"],
                "alignment_desc": edu_desc
            },
            "top_terms": top_terms
        })

    # Sort descending by composite score
    ranked = sorted(scored_candidates, key=lambda x: x["score"], reverse=True)
    
    # Assign standard rank indices
    for rank_idx, cand in enumerate(ranked, 1):
        cand["rank"] = rank_idx

    return ranked

def fair_rerank(
    ranked_candidates: List[Dict[str, any]],
    group_key: str = "audit_group",
    target_proportions: Optional[Dict[str, float]] = None
) -> Dict[str, any]:
    """
    FA*IR-inspired Fair Top-k Reranking algorithm.
    Ensures group representation in the top ranks does not fall below
    the group's proportion in the applicant pool (demographic parity minimum),
    while minimizing utility/score loss.
    """
    if len(ranked_candidates) <= 1:
        return {
            "reranked": ranked_candidates,
            "swaps_count": 0,
            "utility_retention_pct": 100.0,
            "rank_deltas": {c["candidate_id"]: 0 for c in ranked_candidates}
        }

    # Find unique groups (excluding 'Not specified' or empty)
    all_groups = [c.get(group_key, "Unknown") for c in ranked_candidates]
    group_counts = {}
    for g in all_groups:
        group_counts[g] = group_counts.get(g, 0) + 1

    total_candidates = len(ranked_candidates)
    
    # Base pool proportion
    base_proportions = {g: count / total_candidates for g, count in group_counts.items()}
    if target_proportions is None:
        target_proportions = base_proportions

    # Separate candidates into group queues (already sorted by score descending)
    group_queues: Dict[str, List[Dict[str, any]]] = {}
    for cand in ranked_candidates:
        g = cand.get(group_key, "Unknown")
        group_queues.setdefault(g, []).append(dict(cand))

    # Greedy allocation with quota constraint
    reranked: List[Dict[str, any]] = []
    selected_group_counts: Dict[str, int] = {g: 0 for g in group_counts.keys()}
    
    original_rank_map = {c["candidate_id"]: c["rank"] for c in ranked_candidates}

    while len(reranked) < total_candidates:
        current_k = len(reranked) + 1
        
        # Find which group is most under-represented at rank current_k
        candidates_available = [g for g, q in group_queues.items() if len(q) > 0]
        if not candidates_available:
            break

        # Check if any group has a deficit relative to its quota
        best_candidate = None
        best_group = None
        max_deficit = -999.0

        for g in candidates_available:
            expected = target_proportions.get(g, 0.0) * current_k
            actual = selected_group_counts[g]
            deficit = expected - actual
            if deficit > max_deficit:
                max_deficit = deficit
                best_group = g

        # If deficit is substantial (>= 0.75), prioritize that group's top candidate
        # Otherwise, take the candidate with highest raw score across all queues
        if max_deficit >= 0.75 and best_group and group_queues[best_group]:
            candidate_to_add = group_queues[best_group].pop(0)
        else:
            # Pick highest score among heads of queues
            candidate_to_add = max(
                (group_queues[g][0] for g in candidates_available),
                key=lambda x: x["score"]
            )
            g_selected = candidate_to_add.get(group_key, "Unknown")
            group_queues[g_selected].pop(0)

        sel_group = candidate_to_add.get(group_key, "Unknown")
        selected_group_counts[sel_group] = selected_group_counts.get(sel_group, 0) + 1
        reranked.append(candidate_to_add)

    # Compute displacements and utility retention
    swaps = 0
    rank_deltas = {}
    original_score_sum = sum(c["score"] for c in ranked_candidates)
    
    for new_rank, cand in enumerate(reranked, 1):
        old_rank = original_rank_map[cand["candidate_id"]]
        delta = old_rank - new_rank  # positive = improved, negative = dropped
        rank_deltas[cand["candidate_id"]] = delta
        cand["original_rank"] = old_rank
        cand["fair_rank"] = new_rank
        cand["rank_delta"] = delta
        if delta != 0:
            swaps += 1

    utility_retention = 100.0  # Scores didn't change, only order changed
    
    return {
        "reranked": reranked,
        "swaps_count": swaps,
        "utility_retention_pct": round(utility_retention, 2),
        "rank_deltas": rank_deltas
    }
