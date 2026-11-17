"""
FairHire AI — Comprehensive Algorithmic Fairness Metrics Suite
Implements:
1. Disparate Impact Ratio (DIR / EEOC 4/5ths Rule)
2. Statistical / Demographic Parity Difference (SPD)
3. Equal Opportunity Difference (EOD)
4. Rank-Biased Positional Exposure Disparity
5. Group-level Score Distribution & Cohen's d Effect Size
6. Top-k Skew & Representation Analysis
"""

import math
from typing import Dict, List, Optional, Tuple, Union

def calculate_group_distributions(
    candidates: List[Dict[str, any]],
    group_key: str = "audit_group"
) -> Dict[str, Dict[str, any]]:
    """
    Computes statistical summaries for each group: count, mean score, median score, std dev.
    """
    groups: Dict[str, List[float]] = {}
    for c in candidates:
        g = c.get(group_key, "Not specified")
        score = float(c.get("score", 0.0))
        groups.setdefault(g, []).append(score)

    stats: Dict[str, Dict[str, any]] = {}
    for g, scores in groups.items():
        n = len(scores)
        if n == 0:
            continue
        mean_score = sum(scores) / n
        sorted_scores = sorted(scores)
        median_score = (
            sorted_scores[n // 2]
            if n % 2 != 0
            else (sorted_scores[n // 2 - 1] + sorted_scores[n // 2]) / 2.0
        )
        variance = sum((s - mean_score) ** 2 for s in scores) / n if n > 1 else 0.0
        std_dev = math.sqrt(variance)

        stats[g] = {
            "count": n,
            "mean_score": round(mean_score, 2),
            "median_score": round(median_score, 2),
            "std_dev": round(std_dev, 2),
            "min_score": round(min(scores), 2),
            "max_score": round(max(scores), 2)
        }

    return stats

def calculate_cohens_d(group_a_scores: List[float], group_b_scores: List[float]) -> float:
    """Computes Cohen's d effect size between two score distributions."""
    n1, n2 = len(group_a_scores), len(group_b_scores)
    if n1 < 2 or n2 < 2:
        return 0.0

    m1 = sum(group_a_scores) / n1
    m2 = sum(group_b_scores) / n2
    v1 = sum((s - m1) ** 2 for s in group_a_scores) / (n1 - 1)
    v2 = sum((s - m2) ** 2 for s in group_b_scores) / (n2 - 1)

    pooled_std = math.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    if pooled_std < 1e-9:
        return 0.0

    return round((m1 - m2) / pooled_std, 3)

def calculate_fairness_metrics(
    ranked_candidates: List[Dict[str, any]],
    group_key: str = "audit_group",
    top_k: Optional[int] = None,
    qualification_threshold: float = 60.0
) -> Dict[str, any]:
    """
    Computes all standard algorithmic fairness metrics on the ranked candidate pool.
    """
    total_candidates = len(ranked_candidates)
    if total_candidates == 0:
        return {"error": "Candidate pool is empty."}

    # Selection threshold (default to top 50% or min 2)
    if top_k is None or top_k <= 0 or top_k > total_candidates:
        top_k = max(1, math.ceil(total_candidates / 2))

    # Identify candidate groups
    group_stats = calculate_group_distributions(ranked_candidates, group_key)
    active_groups = [g for g in group_stats.keys() if g not in ["Not specified", "Unknown"]]

    # If fewer than 2 distinct audit groups exist, evaluate against whole pool
    if len(active_groups) < 2:
        return {
            "top_k_used": top_k,
            "group_distributions": group_stats,
            "disparate_impact_ratio": None,
            "four_fifths_compliant": True,
            "demographic_parity_diff": 0.0,
            "equal_opportunity_diff": 0.0,
            "exposure_disparity_ratio": 1.0,
            "status_message": "Single group or unspecified audit groups. Multi-group audit requires at least two identified groups.",
            "group_comparisons": []
        }

    # Top-k selection indicator
    selected_cids = {c["candidate_id"] for c in ranked_candidates[:top_k]}

    # Primary pairwise comparison: between first two active groups (e.g. Group A vs Group B)
    g_a, g_b = active_groups[0], active_groups[1]
    
    cands_a = [c for c in ranked_candidates if c.get(group_key) == g_a]
    cands_b = [c for c in ranked_candidates if c.get(group_key) == g_b]

    n_a, n_b = len(cands_a), len(cands_b)
    sel_a = sum(1 for c in cands_a if c["candidate_id"] in selected_cids)
    sel_b = sum(1 for c in cands_b if c["candidate_id"] in selected_cids)

    rate_a = (sel_a / n_a) if n_a > 0 else 0.0
    rate_b = (sel_b / n_b) if n_b > 0 else 0.0

    # 1. Disparate Impact Ratio (DIR)
    # DIR = Selection Rate (unprivileged) / Selection Rate (privileged)
    if max(rate_a, rate_b) > 0:
        dir_val = min(rate_a, rate_b) / max(rate_a, rate_b)
    else:
        dir_val = 1.0
    dir_val = round(dir_val, 3)

    # EEOC 4/5ths Rule: DIR >= 0.80
    four_fifths_compliant = dir_val >= 0.80

    # 2. Demographic Parity Difference (SPD)
    spd_val = round(abs(rate_a - rate_b), 3)

    # 3. Equal Opportunity Difference (EOD)
    # Evaluates True Positive Rate among qualified applicants (score >= threshold)
    qual_a = [c for c in cands_a if c.get("score", 0.0) >= qualification_threshold]
    qual_b = [c for c in cands_b if c.get(group_key) == g_b and c.get("score", 0.0) >= qualification_threshold]

    tpr_a = (sum(1 for c in qual_a if c["candidate_id"] in selected_cids) / len(qual_a)) if qual_a else 0.0
    tpr_b = (sum(1 for c in qual_b if c["candidate_id"] in selected_cids) / len(qual_b)) if qual_b else 0.0
    eod_val = round(abs(tpr_a - tpr_b), 3)

    # 4. Rank-Biased Exposure Disparity
    # Positional attention discount: 1 / log2(1 + rank)
    def calculate_group_exposure(candidates_subset: List[Dict[str, any]]) -> float:
        if not candidates_subset:
            return 0.0
        total_exp = 0.0
        for cand in candidates_subset:
            r = cand.get("rank", 1)
            discount = 1.0 / math.log2(1.0 + r)
            total_exp += discount
        return total_exp / len(candidates_subset)

    exp_a = calculate_group_exposure(cands_a)
    exp_b = calculate_group_exposure(cands_b)

    if max(exp_a, exp_b) > 0:
        exposure_ratio = round(min(exp_a, exp_b) / max(exp_a, exp_b), 3)
    else:
        exposure_ratio = 1.0

    # Cohen's d effect size on scores
    scores_a = [float(c.get("score", 0.0)) for c in cands_a]
    scores_b = [float(c.get("score", 0.0)) for c in cands_b]
    cohens_d = calculate_cohens_d(scores_a, scores_b)

    # Compliance status assessment
    if dir_val >= 0.80 and spd_val <= 0.20:
        status = "COMPLIANT"
        status_msg = f"Candidate ranking satisfies the EEOC 80% Rule (DIR = {dir_val:.2f}) and demographic parity."
    elif dir_val >= 0.65:
        status = "BORDERLINE"
        status_msg = f"Disparate impact ratio ({dir_val:.2f}) is below the recommended 0.80 threshold. Review recommended."
    else:
        status = "ADVERSE_IMPACT_ALERT"
        status_msg = f"Significant disparate impact detected (DIR = {dir_val:.2f} < 0.80). Selection rates differ markedly."

    return {
        "status": status,
        "status_message": status_msg,
        "top_k_used": top_k,
        "group_distributions": group_stats,
        "primary_comparison": {
            "group_a": g_a,
            "group_b": g_b,
            "group_a_selection_rate": round(rate_a * 100, 1),
            "group_b_selection_rate": round(rate_b * 100, 1),
            "group_a_count": n_a,
            "group_b_count": n_b,
            "group_a_selected": sel_a,
            "group_b_selected": sel_b,
            "cohens_d_score_difference": cohens_d
        },
        "disparate_impact_ratio": dir_val,
        "four_fifths_compliant": four_fifths_compliant,
        "demographic_parity_diff": spd_val,
        "equal_opportunity_diff": eod_val,
        "exposure_disparity": {
            "group_a_exposure": round(exp_a, 4),
            "group_b_exposure": round(exp_b, 4),
            "exposure_ratio": exposure_ratio,
            "exposure_difference": round(abs(exp_a - exp_b), 4)
        }
    }
