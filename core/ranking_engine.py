"""
FairHire AI 3.0 — Hybrid Ranking & Fair Reranking Engine

Scoring components:
1. Lexical & Canonical Skill Overlap (Categorized)
2. Semantic TF-IDF Cosine Relevance (sublinear, domain-stopword filtered)
3. Optional Sentence-Transformer semantic similarity (graceful degradation)
4. Experience Duration & Seniority Alignment
5. Educational Credential Tiering (Unknown = 0 points, not auto-awarded)
6. Deterministic Tie-Breaking
7. FA*IR-inspired Fair Post-Processing Reranking with DCG Utility Retention

Changes from v2.0:
- Sentence Transformers integration (optional, cached model)
- SEMANTIC_AVAILABLE flag for feature detection
- Education tier 0 (Unknown) → 0 points (no auto-award)
- compute_semantic_similarity() helper
- NDCG@K and Precision@K computation
- Unknown groups excluded from fairness quotas by default
"""

import math
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from core.skill_taxonomy import (
    extract_skills_from_text,
    extract_years_of_experience,
    extract_education_level,
    get_skill_category,
)

# ── Optional Sentence Transformers ──────────────────────────────────────
SEMANTIC_AVAILABLE = False
_SEMANTIC_MODEL = None

try:
    from sentence_transformers import SentenceTransformer
    SEMANTIC_AVAILABLE = True
except ImportError:
    pass


def get_semantic_model():
    """Lazily load and cache the sentence-transformer model."""
    global _SEMANTIC_MODEL
    if not SEMANTIC_AVAILABLE:
        return None
    if _SEMANTIC_MODEL is None:
        _SEMANTIC_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _SEMANTIC_MODEL


def compute_semantic_similarity(text_a: str, text_b: str) -> float:
    """
    Compute cosine similarity between two texts using sentence-transformers.
    Returns 0.0 if sentence_transformers is unavailable.
    """
    model = get_semantic_model()
    if model is None or not text_a.strip() or not text_b.strip():
        return 0.0
    try:
        embeddings = model.encode([text_a, text_b])
        from numpy import dot
        from numpy.linalg import norm
        a, b = embeddings[0], embeddings[1]
        sim = float(dot(a, b) / (norm(a) * norm(b) + 1e-9))
        return max(0.0, min(1.0, sim))
    except Exception:
        return 0.0


# ── Custom recruitment stop words ───────────────────────────────────────
RECRUITMENT_STOPWORDS = [
    "resume", "curriculum", "vitae", "candidate", "responsibilities",
    "duties", "qualifications", "requirements", "experience", "role",
    "job", "work", "applicant", "position", "seeking", "looking",
    "opportunity", "working", "proficient", "knowledge", "ability",
]


def normalize_weights(weights: Optional[Dict[str, float]]) -> Dict[str, float]:
    """
    Normalizes aspect scoring weights to ensure they sum to exactly 1.0.
    Falls back to default balanced weights if missing or invalid.
    """
    defaults = {
        "skills": 0.40,
        "semantic": 0.30,
        "experience": 0.15,
        "education": 0.15,
    }
    if not weights:
        return defaults

    w_skills = max(0.0, float(weights.get("skills", defaults["skills"])))
    w_sem = max(0.0, float(weights.get("semantic", defaults["semantic"])))
    w_exp = max(0.0, float(weights.get("experience", defaults["experience"])))
    w_edu = max(0.0, float(weights.get("education", defaults["education"])))

    total = w_skills + w_sem + w_exp + w_edu
    if total <= 1e-6:
        return defaults

    return {
        "skills": round(w_skills / total, 4),
        "semantic": round(w_sem / total, 4),
        "experience": round(w_exp / total, 4),
        "education": round(w_edu / total, 4),
    }


def compute_skill_scores(
    job_skills: Set[str], candidate_skills: Set[str]
) -> Dict[str, Any]:
    """
    Computes structured skill match scores between job requirements
    and candidate qualifications, categorized by domain.
    """
    if not job_skills:
        categorized_matched: Dict[str, List[str]] = {}
        for s in candidate_skills:
            cat = get_skill_category(s)
            categorized_matched.setdefault(cat, []).append(s)

        return {
            "skill_match_score": 50.0,
            "matched_skills": sorted(list(candidate_skills)),
            "missing_skills": [],
            "bonus_skills": [],
            "coverage_ratio": 1.0,
            "skill_count": len(candidate_skills),
            "categorized_matched": categorized_matched,
        }

    matched = job_skills.intersection(candidate_skills)
    missing = job_skills.difference(candidate_skills)
    bonus = candidate_skills.difference(job_skills)

    coverage = len(matched) / len(job_skills)
    bonus_points = min(10.0, len(bonus) * 1.5)

    raw_score = (coverage * 90.0) + bonus_points
    skill_match_score = min(100.0, max(0.0, raw_score))

    categorized_matched = {}
    for s in matched:
        cat = get_skill_category(s)
        categorized_matched.setdefault(cat, []).append(s)

    return {
        "skill_match_score": round(skill_match_score, 2),
        "matched_skills": sorted(list(matched)),
        "missing_skills": sorted(list(missing)),
        "bonus_skills": sorted(list(bonus))[:8],
        "coverage_ratio": round(coverage, 3),
        "skill_count": len(candidate_skills),
        "categorized_matched": categorized_matched,
    }


def compute_experience_alignment(
    job_years: float, candidate_years: float
) -> Tuple[float, str]:
    """
    Computes experience score (0-100) and descriptive alignment string.
    """
    if job_years <= 0:
        score = min(100.0, 50.0 + candidate_years * 8.0)
        desc = f"{candidate_years:.1f} yrs experience (no minimum requested)"
        return round(score, 2), desc

    diff = candidate_years - job_years
    if diff >= 0:
        score = min(100.0, 85.0 + min(15.0, diff * 3.0))
        desc = f"Meets requirement ({candidate_years:.1f} yrs vs {job_years:.1f} required)"
    elif diff >= -1.0:
        score = 70.0
        desc = f"Near requirement ({candidate_years:.1f} yrs vs {job_years:.1f} required)"
    elif diff >= -2.5:
        score = 50.0
        desc = f"Slight experience gap ({candidate_years:.1f} yrs vs {job_years:.1f} required)"
    else:
        score = max(20.0, 40.0 - abs(diff) * 5.0)
        desc = f"Significant experience gap ({candidate_years:.1f} yrs vs {job_years:.1f} required)"

    return round(score, 2), desc


def compute_education_alignment(
    job_tier: int, candidate_tier: int
) -> Tuple[float, str]:
    """
    Computes education alignment score (0-100).
    Tier 0 (Unknown) is NOT auto-awarded points.
    """
    if candidate_tier == 0:
        return 0.0, "Education not specified — no automatic points awarded"

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
    candidates: List[Dict[str, Any]],
    weights: Optional[Dict[str, float]] = None,
    use_semantic: bool = True,
) -> List[Dict[str, Any]]:
    """
    Ranks candidates using a multi-aspect hybrid scoring system:
    - Skill Match
    - Semantic Relevance (TF-IDF, optionally blended with sentence-transformer)
    - Experience Alignment
    - Education Fit

    Names, demographic tags, and audit groups are NEVER used in scoring.
    """
    normalized_weights = normalize_weights(weights)

    # Extract requirements from job description
    job_skills = extract_skills_from_text(job_description)
    job_years = extract_years_of_experience(job_description)
    job_edu = extract_education_level(job_description)

    # TF-IDF semantic similarities
    documents = [job_description] + [c.get("qualifications", "") for c in candidates]
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=1,
    )
    matrix = vectorizer.fit_transform(documents)
    similarities = cosine_similarity(matrix[0:1], matrix[1:]).flatten()

    feature_names = vectorizer.get_feature_names_out()
    job_tfidf = matrix[0].toarray().flatten()

    # Optional: sentence-transformer similarity
    st_similarities = []
    if use_semantic and SEMANTIC_AVAILABLE:
        for c in candidates:
            qual = c.get("qualifications", "")
            st_sim = compute_semantic_similarity(job_description, qual)
            st_similarities.append(st_sim)
    else:
        st_similarities = [0.0] * len(candidates)

    scored_candidates = []
    for idx, (candidate, tfidf_sim) in enumerate(zip(candidates, similarities)):
        qual_text = candidate.get("qualifications", "")

        # 1. Skill evaluation
        cand_skills = extract_skills_from_text(qual_text)
        skill_res = compute_skill_scores(job_skills, cand_skills)

        # 2. Semantic evaluation (TF-IDF baseline, optionally blended)
        tfidf_score = float(tfidf_sim) * 100.0
        if use_semantic and SEMANTIC_AVAILABLE and st_similarities[idx] > 0:
            # Blend: 60% TF-IDF, 40% sentence-transformer
            st_score = st_similarities[idx] * 100.0
            semantic_score = round(0.6 * tfidf_score + 0.4 * st_score, 2)
        else:
            semantic_score = round(tfidf_score, 2)

        # 3. Experience evaluation
        cand_years = extract_years_of_experience(qual_text)
        exp_score, exp_desc = compute_experience_alignment(job_years, cand_years)

        # 4. Education evaluation
        cand_edu = extract_education_level(qual_text)
        edu_score, edu_desc = compute_education_alignment(
            job_edu["tier"], cand_edu["tier"]
        )

        # Composite weighted score
        composite_score = (
            normalized_weights["skills"] * skill_res["skill_match_score"]
            + normalized_weights["semantic"] * semantic_score
            + normalized_weights["experience"] * exp_score
            + normalized_weights["education"] * edu_score
        )
        composite_score = min(100.0, max(0.0, round(composite_score, 2)))

        # Significant keyword contributors
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
            "raw_similarity": round(tfidf_score, 2),
            "semantic_model_score": round(st_similarities[idx] * 100, 2) if SEMANTIC_AVAILABLE else None,
            "tier_badge": tier_badge,
            "aspect_scores": {
                "skill_match": skill_res["skill_match_score"],
                "semantic_relevance": semantic_score,
                "experience_fit": exp_score,
                "education_fit": edu_score,
            },
            "weights_used": normalized_weights,
            "skills_extracted": {
                "matched": skill_res["matched_skills"],
                "missing": skill_res["missing_skills"],
                "bonus": skill_res["bonus_skills"],
                "total_count": skill_res["skill_count"],
                "coverage_pct": round(skill_res["coverage_ratio"] * 100, 1),
                "categorized": skill_res.get("categorized_matched", {}),
            },
            "experience_details": {
                "years_found": cand_years,
                "required_years": job_years,
                "alignment_desc": exp_desc,
            },
            "education_details": {
                "level": cand_edu["level"],
                "tier": cand_edu["tier"],
                "alignment_desc": edu_desc,
            },
            "top_terms": top_terms,
        })

    # Sort descending with deterministic tie-breaking
    ranked = sorted(
        scored_candidates,
        key=lambda x: (
            x["score"],
            x["aspect_scores"]["skill_match"],
            x["aspect_scores"]["experience_fit"],
            x.get("candidate_id", ""),
        ),
        reverse=True,
    )

    for rank_idx, cand in enumerate(ranked, 1):
        cand["rank"] = rank_idx

    return ranked


def calculate_dcg(ranked_candidates: List[Dict[str, Any]]) -> float:
    """
    Computes Discounted Cumulative Gain (DCG):
    DCG = sum_{r=1}^n score(r) / log2(1 + r)
    """
    dcg = 0.0
    for r, c in enumerate(ranked_candidates, 1):
        score = float(c.get("score", 0.0))
        discount = 1.0 / math.log2(1.0 + r)
        dcg += score * discount
    return dcg


def calculate_ndcg(
    standard_ranked: List[Dict[str, Any]],
    comparison_ranked: List[Dict[str, Any]],
    k: Optional[int] = None,
) -> Dict[str, float]:
    """
    Computes NDCG@K comparing standard (ideal) vs comparison ranking.
    Uses standard ranking as the ideal ordering.
    """
    if not standard_ranked:
        return {"ndcg_standard": 0.0, "ndcg_comparison": 0.0, "ndcg_at_k": 1.0}

    if k is None or k <= 0:
        k = len(standard_ranked)
    k = min(k, len(standard_ranked))

    ideal_dcg = calculate_dcg(standard_ranked[:k])
    comp_dcg = calculate_dcg(comparison_ranked[:k])

    ndcg_val = (comp_dcg / ideal_dcg) if ideal_dcg > 0 else 1.0

    return {
        "ndcg_standard": round(ideal_dcg, 4),
        "ndcg_comparison": round(comp_dcg, 4),
        "ndcg_at_k": round(min(1.0, ndcg_val), 4),
        "k": k,
    }


def calculate_precision_at_k(
    ranked: List[Dict[str, Any]],
    k: int,
    threshold: float = 60.0,
) -> float:
    """
    Computes Precision@K — fraction of top-k candidates above score threshold.
    Note: This uses score-based relevance, not independent labels.
    """
    if k <= 0 or not ranked:
        return 0.0
    k = min(k, len(ranked))
    relevant_in_topk = sum(
        1 for c in ranked[:k] if float(c.get("score", 0.0)) >= threshold
    )
    return round(relevant_in_topk / k, 4)


def fair_rerank(
    ranked_candidates: List[Dict[str, Any]],
    group_key: str = "audit_group",
    target_proportions: Optional[Dict[str, float]] = None,
    exclude_unknown: bool = True,
) -> Dict[str, Any]:
    """
    FA*IR-inspired Fair Top-k Reranking algorithm.
    Ensures group representation in the top ranks does not fall below
    the group's proportion in the applicant pool (demographic parity minimum),
    while minimizing utility/score loss.

    exclude_unknown: If True, Unknown/Not-specified groups are not included
                     in fairness quota calculations (they fill remaining slots).

    Assumptions documented:
    - Group proportions default to pool proportions
    - Greedy allocation with 0.75 deficit threshold
    - Unknown groups get no quota boost by default
    """
    if len(ranked_candidates) <= 1:
        return {
            "reranked": ranked_candidates,
            "swaps_count": 0,
            "utility_retention_pct": 100.0,
            "rank_deltas": {c["candidate_id"]: 0 for c in ranked_candidates},
            "avg_rank_shift": 0.0,
            "max_rank_shift": 0,
            "assumptions": [
                "Single candidate — no reranking needed.",
            ],
        }

    # Find unique groups
    all_groups = [c.get(group_key, "Unknown") for c in ranked_candidates]
    group_counts: Dict[str, int] = {}
    for g in all_groups:
        group_counts[g] = group_counts.get(g, 0) + 1

    total_candidates = len(ranked_candidates)

    # Determine which groups participate in quota
    excluded_groups = {"Unknown", "Not specified"} if exclude_unknown else set()
    quota_groups = {g for g in group_counts if g not in excluded_groups}

    # Base pool proportion
    base_proportions = {
        g: count / total_candidates for g, count in group_counts.items()
    }
    if target_proportions is None:
        target_proportions = base_proportions

    # Separate candidates into group queues
    group_queues: Dict[str, List[Dict[str, Any]]] = {}
    for cand in ranked_candidates:
        g = cand.get(group_key, "Unknown")
        group_queues.setdefault(g, []).append(dict(cand))

    # Greedy allocation with quota constraint
    reranked: List[Dict[str, Any]] = []
    selected_group_counts: Dict[str, int] = {g: 0 for g in group_counts.keys()}
    original_rank_map = {c["candidate_id"]: c["rank"] for c in ranked_candidates}

    while len(reranked) < total_candidates:
        current_k = len(reranked) + 1

        candidates_available = [g for g, q in group_queues.items() if len(q) > 0]
        if not candidates_available:
            break

        # Check quota groups for deficit
        best_group = None
        max_deficit = -999.0

        for g in candidates_available:
            if g not in quota_groups:
                continue
            expected = target_proportions.get(g, 0.0) * current_k
            actual = selected_group_counts[g]
            deficit = expected - actual
            if deficit > max_deficit:
                max_deficit = deficit
                best_group = g

        if max_deficit >= 0.75 and best_group and group_queues.get(best_group):
            candidate_to_add = group_queues[best_group].pop(0)
        else:
            candidate_to_add = max(
                (group_queues[g][0] for g in candidates_available),
                key=lambda x: x["score"],
            )
            g_selected = candidate_to_add.get(group_key, "Unknown")
            group_queues[g_selected].pop(0)

        sel_group = candidate_to_add.get(group_key, "Unknown")
        selected_group_counts[sel_group] = (
            selected_group_counts.get(sel_group, 0) + 1
        )
        reranked.append(candidate_to_add)

    # Compute displacement metrics
    swaps = 0
    rank_deltas: Dict[str, int] = {}
    total_abs_shift = 0
    max_shift = 0

    for new_rank, cand in enumerate(reranked, 1):
        old_rank = original_rank_map[cand["candidate_id"]]
        delta = old_rank - new_rank
        rank_deltas[cand["candidate_id"]] = delta
        cand["original_rank"] = old_rank
        cand["fair_rank"] = new_rank
        cand["rank_delta"] = delta
        if delta != 0:
            swaps += 1
            abs_delta = abs(delta)
            total_abs_shift += abs_delta
            if abs_delta > max_shift:
                max_shift = abs_delta

    dcg_standard = calculate_dcg(ranked_candidates)
    dcg_fair = calculate_dcg(reranked)
    dcg_utility_retention = (
        (dcg_fair / dcg_standard * 100.0) if dcg_standard > 0 else 100.0
    )
    dcg_utility_retention = round(min(100.0, dcg_utility_retention), 2)

    avg_rank_shift = (
        round(total_abs_shift / total_candidates, 2)
        if total_candidates > 0
        else 0.0
    )

    assumptions = [
        "Group proportions default to pool demographic proportions.",
        "Greedy allocation with 0.75 deficit threshold triggers quota boost.",
        f"{'Unknown/Not-specified groups excluded from quota.' if exclude_unknown else 'All groups participate in quota.'}",
        "Utility measured via DCG (Discounted Cumulative Gain).",
        "This reranking is a heuristic post-processor, not an optimal solution.",
    ]

    return {
        "reranked": reranked,
        "swaps_count": swaps,
        "utility_retention_pct": dcg_utility_retention,
        "rank_deltas": rank_deltas,
        "avg_rank_shift": avg_rank_shift,
        "max_rank_shift": max_shift,
        "assumptions": assumptions,
    }
