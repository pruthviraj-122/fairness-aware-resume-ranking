"""
FairHire AI 3.0 — Academic Evaluation & Experimental Research Framework

Compares ranking methodologies across benchmark datasets:
1. TF-IDF lexical baseline (weights: semantic 1.0, skills 0.0, exp 0.0, edu 0.0)
2. Hybrid multi-aspect ranking (skills 0.40, semantic 0.30, exp 0.15, edu 0.15)
3. Hybrid with FA*IR post-processed reranking

Evaluates:
- Ranking quality: NDCG@K, Precision@K
- Fairness: Demographic Parity Difference (SPD), Disparate Impact Ratio (DIR),
  Equal Opportunity Difference (EOD), Exposure Disparity, Theil Index
- Stability: Kendall's Tau rank correlation, Rank displacement, Score variance
"""

import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

from core.ranking_engine import (
    score_candidates_advanced,
    fair_rerank,
    calculate_dcg,
    calculate_ndcg,
    calculate_precision_at_k,
    SEMANTIC_AVAILABLE,
)
from core.fairness_metrics import calculate_fairness_metrics


def kendall_tau(ranks_a: List[int], ranks_b: List[int]) -> float:
    """
    Computes Kendall's tau rank correlation coefficient between two rankings.
    Does not require scipy.
    Returns value in [-1.0, 1.0], where 1.0 = identical rankings, -1.0 = reversed.
    """
    n = len(ranks_a)
    if n <= 1:
        return 1.0
    if len(ranks_b) != n:
        raise ValueError("Both rank lists must have identical length.")

    concordant = 0
    discordant = 0

    for i in range(n):
        for j in range(i + 1, n):
            sign_a = (ranks_a[i] > ranks_a[j]) - (ranks_a[i] < ranks_a[j])
            sign_b = (ranks_b[i] > ranks_b[j]) - (ranks_b[i] < ranks_b[j])
            product = sign_a * sign_b
            if product > 0:
                concordant += 1
            elif product < 0:
                discordant += 1

    total_pairs = (n * (n - 1)) / 2.0
    if total_pairs == 0:
        return 1.0
    return round((concordant - discordant) / total_pairs, 4)


class RankingEvaluator:
    """
    Evaluation suite for evaluating ranking models and fairness trade-offs.
    """

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed

    @staticmethod
    def load_sample_data(path: Optional[str] = None) -> Dict[str, Any]:
        """Loads benchmark datasets from JSON."""
        if path is None:
            path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "sample_data",
                "sample_job_and_resumes.json",
            )
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f).get("datasets", {})

    def run_tfidf_baseline(
        self,
        job_description: str,
        candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Runs pure TF-IDF semantic relevance baseline (skills=0, semantic=1.0, exp=0, edu=0).
        """
        weights = {"skills": 0.0, "semantic": 1.0, "experience": 0.0, "education": 0.0}
        return score_candidates_advanced(
            job_description, candidates, weights=weights, use_semantic=False
        )

    def run_hybrid_model(
        self,
        job_description: str,
        candidates: List[Dict[str, Any]],
        weights: Optional[Dict[str, float]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Runs the full multi-aspect hybrid model.
        """
        return score_candidates_advanced(
            job_description, candidates, weights=weights, use_semantic=True
        )

    def compute_stability_metrics(
        self,
        ranked_a: List[Dict[str, Any]],
        ranked_b: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """
        Computes ranking stability between two candidate orderings:
        - Kendall's Tau rank correlation
        - Mean absolute rank displacement
        - Max rank displacement
        """
        cid_to_rank_a = {c["candidate_id"]: c.get("fair_rank", c.get("rank", i + 1)) for i, c in enumerate(ranked_a)}
        cid_to_rank_b = {c["candidate_id"]: c.get("fair_rank", c.get("rank", i + 1)) for i, c in enumerate(ranked_b)}

        common_cids = sorted(list(set(cid_to_rank_a.keys()).intersection(cid_to_rank_b.keys())))
        if not common_cids:
            return {"kendall_tau": 0.0, "avg_rank_displacement": 0.0, "max_rank_displacement": 0.0}

        list_a = [cid_to_rank_a[cid] for cid in common_cids]
        list_b = [cid_to_rank_b[cid] for cid in common_cids]

        tau = kendall_tau(list_a, list_b)
        displacements = [abs(a - b) for a, b in zip(list_a, list_b)]
        avg_disp = round(sum(displacements) / len(displacements), 2)
        max_disp = max(displacements) if displacements else 0

        return {
            "kendall_tau": tau,
            "avg_rank_displacement": avg_disp,
            "max_rank_displacement": max_disp,
        }

    def evaluate_cohort(
        self,
        job_description: str,
        candidates: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Runs end-to-end multi-model comparison on a single candidate cohort.
        """
        n = len(candidates)
        if top_k is None or top_k <= 0 or top_k > n:
            top_k = max(1, math.ceil(n / 2))

        # 1. TF-IDF Baseline
        tfidf_ranked = self.run_tfidf_baseline(job_description, candidates)
        tfidf_metrics = calculate_fairness_metrics(tfidf_ranked, top_k=top_k)

        # 2. Hybrid Model
        hybrid_ranked = self.run_hybrid_model(job_description, candidates, weights=weights)
        hybrid_metrics = calculate_fairness_metrics(hybrid_ranked, top_k=top_k)

        # 3. Fair Reranked Hybrid
        fair_obj = fair_rerank(hybrid_ranked)
        fair_ranked = fair_obj["reranked"]
        fair_metrics = calculate_fairness_metrics(fair_ranked, top_k=top_k)

        # 4. NDCG and Precision
        ndcg_fair = calculate_ndcg(hybrid_ranked, fair_ranked, k=top_k)
        prec_tfidf = calculate_precision_at_k(tfidf_ranked, k=top_k)
        prec_hybrid = calculate_precision_at_k(hybrid_ranked, k=top_k)
        prec_fair = calculate_precision_at_k(fair_ranked, k=top_k)

        # 5. Stability comparisons
        stab_tfidf_vs_hybrid = self.compute_stability_metrics(tfidf_ranked, hybrid_ranked)
        stab_hybrid_vs_fair = self.compute_stability_metrics(hybrid_ranked, fair_ranked)

        return {
            "candidate_count": n,
            "top_k": top_k,
            "models": {
                "tfidf_baseline": {
                    "disparate_impact_ratio": tfidf_metrics.get("disparate_impact_ratio"),
                    "demographic_parity_diff": tfidf_metrics.get("demographic_parity_diff"),
                    "equal_opportunity_diff": tfidf_metrics.get("equal_opportunity_diff"),
                    "precision_at_k": prec_tfidf,
                    "four_fifths_compliant": tfidf_metrics.get("four_fifths_compliant"),
                    "theil_inequality_index": tfidf_metrics.get("theil_inequality_index"),
                },
                "hybrid_ranking": {
                    "disparate_impact_ratio": hybrid_metrics.get("disparate_impact_ratio"),
                    "demographic_parity_diff": hybrid_metrics.get("demographic_parity_diff"),
                    "equal_opportunity_diff": hybrid_metrics.get("equal_opportunity_diff"),
                    "precision_at_k": prec_hybrid,
                    "four_fifths_compliant": hybrid_metrics.get("four_fifths_compliant"),
                    "theil_inequality_index": hybrid_metrics.get("theil_inequality_index"),
                },
                "fair_reranked_hybrid": {
                    "disparate_impact_ratio": fair_metrics.get("disparate_impact_ratio"),
                    "demographic_parity_diff": fair_metrics.get("demographic_parity_diff"),
                    "equal_opportunity_diff": fair_metrics.get("equal_opportunity_diff"),
                    "precision_at_k": prec_fair,
                    "ndcg_at_k": ndcg_fair.get("ndcg_at_k"),
                    "utility_retention_pct": fair_obj.get("utility_retention_pct"),
                    "swaps_count": fair_obj.get("swaps_count"),
                    "four_fifths_compliant": fair_metrics.get("four_fifths_compliant"),
                    "theil_inequality_index": fair_metrics.get("theil_inequality_index"),
                },
            },
            "stability": {
                "tfidf_vs_hybrid": stab_tfidf_vs_hybrid,
                "hybrid_vs_fair_rerank": stab_hybrid_vs_fair,
            },
            "methodology_notes": [
                "All metrics computed on standardized benchmark candidate cohorts.",
                "EOD uses score-based qualification proxies when independent labels are not available.",
                "Fairness thresholds are indicators and do not certify legal compliance.",
            ],
        }

    def run_full_suite(self) -> Dict[str, Any]:
        """Runs evaluation over all benchmark datasets."""
        datasets = self.load_sample_data()
        results = {}
        for key, data in datasets.items():
            results[key] = {
                "title": data.get("title", key),
                "evaluation": self.evaluate_cohort(
                    data["job_description"], data["candidates"]
                ),
            }
        return {
            "experiment_version": "3.0",
            "semantic_transformers_enabled": SEMANTIC_AVAILABLE,
            "cohort_evaluations": results,
        }


def run_reproducible_experiment(
    output_path: str = "evaluation/results.json",
    random_seed: int = 42,
) -> Dict[str, Any]:
    """
    Executes the reproducible experimental benchmark and writes results to JSON.
    """
    evaluator = RankingEvaluator(random_seed=random_seed)
    results = evaluator.run_full_suite()

    # Save to disk
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("=" * 65)
    print(" FAIRHIRE AI 3.0 — REPRODUCIBLE EXPERIMENTAL BENCHMARK")
    print("=" * 65)
    print(f"Output File: {output_path}")
    print(f"Sentence-Transformers Active: {SEMANTIC_AVAILABLE}")
    print("-" * 65)
    for cohort_id, cohort_res in results.get("cohort_evaluations", {}).items():
        title = cohort_res["title"]
        ev = cohort_res["evaluation"]["models"]
        print(f"\nCohort: {title} ({cohort_id})")
        print(f"  TF-IDF Baseline:        DIR={ev['tfidf_baseline']['disparate_impact_ratio']} | SPD={ev['tfidf_baseline']['demographic_parity_diff']}")
        print(f"  Hybrid Ranking:         DIR={ev['hybrid_ranking']['disparate_impact_ratio']} | SPD={ev['hybrid_ranking']['demographic_parity_diff']}")
        print(f"  Fair Reranked Hybrid:   DIR={ev['fair_reranked_hybrid']['disparate_impact_ratio']} | Ret={ev['fair_reranked_hybrid']['utility_retention_pct']}% | NDCG={ev['fair_reranked_hybrid']['ndcg_at_k']}")
    print("=" * 65)

    return results


if __name__ == "__main__":
    run_reproducible_experiment()
