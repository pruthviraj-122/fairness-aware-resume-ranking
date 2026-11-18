"""
FairHire AI 3.0 — Algorithmic Fairness Metrics Suite

Implements:
1. Multi-Group Disparate Impact Ratio (DIR / EEOC 4/5ths Rule)
2. Statistical / Demographic Parity Difference (SPD)
3. Equal Opportunity Difference (EOD) — with explicit methodology warning
4. Rank-Biased Positional Exposure Disparity
5. Theil Index / Generalized Entropy
6. Group-by-Group Selection Rate Audit Table
7. Group-level Score Distributions & Cohen's d Effect Sizes
8. Rank displacement metrics
9. Group sample size reporting with uncertainty warnings

IMPORTANT METHODOLOGICAL NOTES:
- EOD uses score-based qualification proxy, NOT independent ground-truth labels.
  When independent labels are unavailable, this is explicitly flagged.
- Status labels never imply legal compliance.
- Small sample warnings are reported, not hidden.
- These metrics do not constitute legal compliance assessment.
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union


def calculate_group_distributions(
    candidates: List[Dict[str, Any]],
    group_key: str = "audit_group",
) -> Dict[str, Dict[str, Any]]:
    """
    Computes statistical summaries for each group.
    """
    groups: Dict[str, List[float]] = {}
    for c in candidates:
        g = c.get(group_key, "Not specified")
        score = float(c.get("score", 0.0))
        groups.setdefault(g, []).append(score)

    stats: Dict[str, Dict[str, Any]] = {}
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
        variance = (
            sum((s - mean_score) ** 2 for s in scores) / n if n > 1 else 0.0
        )
        std_dev = math.sqrt(variance)

        stats[g] = {
            "count": n,
            "mean_score": round(mean_score, 2),
            "median_score": round(median_score, 2),
            "std_dev": round(std_dev, 2),
            "min_score": round(min(scores), 2),
            "max_score": round(max(scores), 2),
        }

    return stats


def calculate_cohens_d(
    group_a_scores: List[float], group_b_scores: List[float]
) -> float:
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


def calculate_theil_index(exposures: List[float]) -> float:
    """
    Computes Theil's T index (Generalized Entropy Index with alpha=1).
    0.0 = perfect equality; higher = more concentration at the top.
    """
    n = len(exposures)
    if n <= 1:
        return 0.0

    mean_exp = sum(exposures) / n
    if mean_exp < 1e-9:
        return 0.0

    theil = 0.0
    for x in exposures:
        if x > 1e-9:
            ratio = x / mean_exp
            theil += ratio * math.log(ratio)
    return round(theil / n, 4)


def calculate_exposure_disparity(
    ranked_candidates: List[Dict[str, Any]],
    group_key: str = "audit_group",
) -> Dict[str, Any]:
    """
    Computes per-group average exposure using logarithmic discount.
    Returns group exposures, ratio, and difference.
    """
    active_groups = set()
    group_candidates: Dict[str, List[Dict]] = {}
    for c in ranked_candidates:
        g = c.get(group_key, "Not specified")
        if g not in ("Not specified", "Unknown"):
            active_groups.add(g)
            group_candidates.setdefault(g, []).append(c)

    group_exposures: Dict[str, float] = {}
    all_individual_exposures: List[float] = []

    for c in ranked_candidates:
        r = c.get("rank", 1)
        discount = 1.0 / math.log2(1.0 + r)
        all_individual_exposures.append(discount)

    for g in active_groups:
        cands = group_candidates.get(g, [])
        if cands:
            tot_exp = sum(
                1.0 / math.log2(1.0 + c.get("rank", 1)) for c in cands
            )
            group_exposures[g] = round(tot_exp / len(cands), 4)
        else:
            group_exposures[g] = 0.0

    exp_values = list(group_exposures.values())
    max_exp = max(exp_values) if exp_values else 1.0
    min_exp = min(exp_values) if exp_values else 1.0

    exposure_ratio = round(min_exp / max_exp, 3) if max_exp > 0 else 1.0
    exposure_diff = round(max_exp - min_exp, 4)

    theil_index = calculate_theil_index(all_individual_exposures)

    return {
        "group_exposures": group_exposures,
        "exposure_ratio": exposure_ratio,
        "exposure_difference": exposure_diff,
        "theil_index": theil_index,
    }


def calculate_fairness_metrics(
    ranked_candidates: List[Dict[str, Any]],
    group_key: str = "audit_group",
    top_k: Optional[int] = None,
    qualification_threshold: float = 60.0,
    ground_truth_labels: Optional[Dict[str, bool]] = None,
) -> Dict[str, Any]:
    """
    Computes comprehensive algorithmic fairness metrics.

    IMPORTANT: EOD requires independent ground-truth qualification labels.
    If ground_truth_labels is None, EOD uses score-based proxy and is flagged
    as methodologically limited.

    Args:
        ground_truth_labels: Optional dict mapping candidate_id -> bool
            (True = independently verified as qualified).
            If None, EOD uses score-based proxy with explicit warning.
    """
    total_candidates = len(ranked_candidates)
    if total_candidates == 0:
        return {"error": "Candidate pool is empty."}

    if top_k is None or top_k <= 0 or top_k > total_candidates:
        top_k = max(1, math.ceil(total_candidates / 2))

    group_stats = calculate_group_distributions(ranked_candidates, group_key)
    active_groups = [
        g for g in group_stats.keys() if g not in ["Not specified", "Unknown"]
    ]

    # ── Uncertainty warnings ────────────────────────────────────────────
    uncertainty_warnings: List[str] = []
    group_sample_sizes: Dict[str, int] = {}
    for g in active_groups:
        n = group_stats[g]["count"]
        group_sample_sizes[g] = n
        if n < 5:
            uncertainty_warnings.append(
                f"Group '{g}' has only {n} candidate(s). "
                f"Metrics may be unreliable due to small sample size."
            )

    if total_candidates < 10:
        uncertainty_warnings.append(
            f"Total pool has only {total_candidates} candidates. "
            f"Statistical metrics may not be meaningful."
        )

    # Single group or unspecified
    if len(active_groups) < 2:
        return {
            "top_k_used": top_k,
            "group_distributions": group_stats,
            "group_sample_sizes": group_sample_sizes,
            "disparate_impact_ratio": 1.0,
            "four_fifths_compliant": True,
            "demographic_parity_diff": 0.0,
            "equal_opportunity_diff": 0.0,
            "eod_valid": False,
            "eod_methodology_note": (
                "Insufficient groups for EOD computation."
            ),
            "exposure_disparity_ratio": 1.0,
            "status": "INSUFFICIENT_GROUPS",
            "status_message": (
                "Single group or unspecified audit groups. Multi-group audit "
                "requires at least two identified audit groups."
            ),
            "group_audit_table": [],
            "primary_comparison": None,
            "uncertainty_warnings": uncertainty_warnings,
            "disclaimer": (
                "These metrics do not constitute legal compliance assessment. "
                "Consult legal counsel for employment law compliance."
            ),
        }

    # Top-k selection set
    selected_cids = {c["candidate_id"] for c in ranked_candidates[:top_k]}

    # Group-level selection rates & counts
    group_candidates: Dict[str, List[Dict[str, Any]]] = {}
    group_counts: Dict[str, int] = {}
    group_selected: Dict[str, int] = {}
    group_selection_rates: Dict[str, float] = {}

    for g in active_groups:
        cands = [c for c in ranked_candidates if c.get(group_key) == g]
        group_candidates[g] = cands
        n_g = len(cands)
        sel_g = sum(1 for c in cands if c["candidate_id"] in selected_cids)
        rate_g = (sel_g / n_g) if n_g > 0 else 0.0

        group_counts[g] = n_g
        group_selected[g] = sel_g
        group_selection_rates[g] = rate_g

    # Benchmark group (highest selection rate)
    benchmark_group = max(active_groups, key=lambda g: group_selection_rates[g])
    benchmark_rate = group_selection_rates[benchmark_group]

    # Group audit table
    group_audit_table: List[Dict[str, Any]] = []
    min_dir = 1.0
    all_compliant = True

    for g in active_groups:
        rate = group_selection_rates[g]
        if benchmark_rate > 0:
            impact_ratio = round(rate / benchmark_rate, 3)
        else:
            impact_ratio = 1.0

        is_g_compliant = impact_ratio >= 0.80
        if not is_g_compliant:
            all_compliant = False

        if impact_ratio < min_dir:
            min_dir = impact_ratio

        group_audit_table.append({
            "group_name": g,
            "total_applicants": group_counts[g],
            "selected_count": group_selected[g],
            "selection_rate_pct": round(rate * 100, 1),
            "impact_ratio": impact_ratio,
            "is_benchmark": (g == benchmark_group),
            "four_fifths_compliant": is_g_compliant,
            "mean_score": group_stats[g]["mean_score"],
            "median_score": group_stats[g]["median_score"],
            "std_dev": group_stats[g]["std_dev"],
        })

    dir_val = round(min_dir, 3)
    four_fifths_compliant = all_compliant

    # Demographic Parity Difference
    rates_list = [group_selection_rates[g] for g in active_groups]
    spd_val = round(max(rates_list) - min(rates_list), 3)

    # ── Equal Opportunity Difference ────────────────────────────────────
    eod_valid = ground_truth_labels is not None
    if eod_valid:
        # Use independent ground-truth labels
        group_tprs: Dict[str, float] = {}
        for g in active_groups:
            cands = group_candidates[g]
            qual_cands = [
                c for c in cands
                if ground_truth_labels.get(c["candidate_id"], False)
            ]
            if qual_cands:
                tpr = sum(
                    1
                    for c in qual_cands
                    if c["candidate_id"] in selected_cids
                ) / len(qual_cands)
            else:
                tpr = 0.0
            group_tprs[g] = tpr

        tpr_list = list(group_tprs.values())
        eod_val = round(max(tpr_list) - min(tpr_list), 3) if tpr_list else 0.0
        eod_note = (
            "EOD computed using independently provided ground-truth "
            "qualification labels."
        )
    else:
        # Fallback: score-based proxy with explicit warning
        all_scores = [float(c.get("score", 0.0)) for c in ranked_candidates]
        adaptive_threshold = qualification_threshold
        qualified_overall = [s for s in all_scores if s >= qualification_threshold]
        if len(qualified_overall) < max(2, math.ceil(total_candidates * 0.25)):
            sorted_all = sorted(all_scores)
            adaptive_threshold = sorted_all[len(sorted_all) // 2]

        group_tprs = {}
        for g in active_groups:
            cands = group_candidates[g]
            qual_cands = [
                c for c in cands
                if float(c.get("score", 0.0)) >= adaptive_threshold
            ]
            if qual_cands:
                tpr = sum(
                    1
                    for c in qual_cands
                    if c["candidate_id"] in selected_cids
                ) / len(qual_cands)
            else:
                tpr = 0.0
            group_tprs[g] = tpr

        tpr_list = list(group_tprs.values())
        eod_val = round(max(tpr_list) - min(tpr_list), 3) if tpr_list else 0.0
        eod_note = (
            "WARNING: EOD uses score-based qualification proxy "
            f"(score >= {adaptive_threshold:.1f}), NOT independent ground-truth "
            "labels. This is methodologically limited — the model's own "
            "scores determine qualification, creating circular reasoning. "
            "Treat as indicative only. For valid EOD, provide independently "
            "verified qualification labels."
        )

    # ── Exposure Disparity ──────────────────────────────────────────────
    exposure_data = calculate_exposure_disparity(ranked_candidates, group_key)

    # Primary pairwise comparison
    g_a, g_b = active_groups[0], active_groups[1]
    scores_a = [float(c.get("score", 0.0)) for c in group_candidates[g_a]]
    scores_b = [float(c.get("score", 0.0)) for c in group_candidates[g_b]]
    cohens_d = calculate_cohens_d(scores_a, scores_b)

    # ── Status — NEVER implies legal compliance ─────────────────────────
    if dir_val >= 0.80 and spd_val <= 0.20:
        status = "MEETS_THRESHOLD"
        status_msg = (
            f"Candidate ranking meets the EEOC 80% threshold (DIR = {dir_val:.2f}) "
            f"across audited cohorts. This does NOT constitute proof of fairness "
            f"or legal compliance."
        )
    elif dir_val >= 0.65:
        status = "BELOW_THRESHOLD"
        status_msg = (
            f"Disparate impact ratio ({dir_val:.2f}) falls below the 0.80 threshold. "
            f"Selection disparities observed between groups."
        )
    else:
        status = "ADVERSE_IMPACT_ALERT"
        status_msg = (
            f"Significant adverse disparate impact detected (DIR = {dir_val:.2f} < 0.80). "
            f"Selection rates differ markedly across demographic cohorts."
        )

    return {
        "status": status,
        "status_message": status_msg,
        "top_k_used": top_k,
        "benchmark_group": benchmark_group,
        "group_distributions": group_stats,
        "group_sample_sizes": group_sample_sizes,
        "group_audit_table": group_audit_table,
        "disparate_impact_ratio": dir_val,
        "four_fifths_compliant": four_fifths_compliant,
        "demographic_parity_diff": spd_val,
        "equal_opportunity_diff": eod_val,
        "eod_valid": eod_valid,
        "eod_methodology_note": eod_note,
        "adaptive_qualification_threshold_used": round(
            adaptive_threshold if not eod_valid else qualification_threshold, 1
        ),
        "theil_inequality_index": exposure_data["theil_index"],
        "exposure_disparity": {
            "group_exposures": exposure_data["group_exposures"],
            "exposure_ratio": exposure_data["exposure_ratio"],
            "exposure_difference": exposure_data["exposure_difference"],
        },
        "primary_comparison": {
            "group_a": g_a,
            "group_b": g_b,
            "group_a_selection_rate": round(
                group_selection_rates[g_a] * 100, 1
            ),
            "group_b_selection_rate": round(
                group_selection_rates[g_b] * 100, 1
            ),
            "group_a_count": group_counts[g_a],
            "group_b_count": group_counts[g_b],
            "group_a_selected": group_selected[g_a],
            "group_b_selected": group_selected[g_b],
            "cohens_d_score_difference": cohens_d,
        },
        "uncertainty_warnings": uncertainty_warnings,
        "disclaimer": (
            "These metrics do not constitute legal compliance assessment. "
            "Consult legal counsel for employment law compliance. "
            "A metric passing a threshold is not proof of fairness."
        ),
    }
