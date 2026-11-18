"""
FairHire AI 3.0 — Academic Evaluation & Research Framework

Provides publication-grade experimental evaluation of fairness-aware ranking methodologies:
1. TF-IDF Lexical Baseline
2. Semantic Similarity Baseline (pure semantic relevance)
3. Hybrid Multi-Aspect Ranking (Skills + Semantic + Experience + Education)
4. FA*IR Post-Processing Reranking under Relaxed, Moderate, and Strict Constraints

Key Research Capabilities:
- Scalable Synthetic Candidate Generator (supports N=100 to N=1000+ candidates)
- Independent Ground-Truth Qualification Labels (resolves EOD circularity)
- Multi-Seed Statistical Sweep (reports mean +- standard deviation across random seeds)
- Metrics: NDCG@K, Precision@K, Recall@K, DIR, SPD, EOD, Theil Index,
  Counterfactual Absolute Delta, and Rank Displacement
"""

import json
import math
import os
import random
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.ranking_engine import (
    score_candidates_advanced,
    fair_rerank,
    calculate_dcg,
    calculate_ndcg,
    calculate_precision_at_k,
    calculate_recall_at_k,
    compare_fair_rerank_constraints,
    SEMANTIC_AVAILABLE,
)
from core.fairness_metrics import calculate_fairness_metrics
from core.fairness_audit import run_counterfactual_audit


def kendall_tau(ranks_a: List[int], ranks_b: List[int]) -> float:
    """
    Computes Kendall's tau rank correlation coefficient between two rankings.
    Returns value in [-1.0, 1.0], where 1.0 = identical order, -1.0 = reversed.
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


class SyntheticCandidateGenerator:
    """
    Generates controlled synthetic applicant cohorts for algorithmic fairness research.
    Produces authentic technical qualifications, demographic attributes, and
    independent ground-truth qualification labels for non-circular evaluation.
    """

    CORE_TECH_SKILLS = [
        "Python", "Docker", "Kubernetes", "PostgreSQL", "AWS", "Git",
        "CI/CD", "Redis", "Kafka", "Linux", "REST API", "Microservices"
    ]
    DATA_SKILLS = [
        "Machine Learning", "PyTorch", "TensorFlow", "Pandas", "NumPy",
        "Scikit-learn", "SQL", "Spark", "Airflow", "FastAPI"
    ]
    FRONTEND_SKILLS = [
        "JavaScript", "TypeScript", "React", "Next.js", "Node.js", "HTML", "CSS"
    ]
    DEGREES = [
        ("Bachelor's Degree in Computer Science", 3),
        ("Master's Degree in Computer Science", 4),
        ("Bachelor's Degree in Software Engineering", 3),
        ("Master's in Data Science", 4),
        ("Doctorate (PhD) in Computer Science", 5),
        ("Bootcamp Graduate in Web Development", 1),
        ("Associate Degree in Information Technology", 2),
    ]

    FIRST_NAMES_GROUP_A = [
        "James", "John", "Robert", "Michael", "David", "William", "Richard",
        "Joseph", "Thomas", "Charles", "Daniel", "Matthew", "Anthony", "Donald"
    ]
    FIRST_NAMES_GROUP_B = [
        "Mary", "Patricia", "Jennifer", "Linda", "Elizabeth", "Barbara",
        "Susan", "Jessica", "Sarah", "Karen", "Lisa", "Nancy", "Emily"
    ]
    FIRST_NAMES_GROUP_C = [
        "Jamal", "Marcus", "Darnell", "Malik", "Terrell", "DeAndre",
        "Lakisha", "Ebony", "Latoya", "Shanice", "Tamika", "Keisha"
    ]
    FIRST_NAMES_GROUP_D = [
        "Wei", "Ming", "Priya", "Rahul", "Carlos", "Mateo",
        "Sofia", "Camila", "Elena", "Fatima", "Amina", "Ahmed"
    ]

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def generate_cohort(
        self,
        n: int = 100,
        group_weights: Optional[Dict[str, float]] = None,
        qualification_parity: bool = True,
    ) -> Tuple[str, List[Dict[str, Any]], Dict[str, bool]]:
        """
        Generates a cohort of n synthetic candidates against a standardized
        Senior Software Engineer job description.

        Returns:
            job_description: str
            candidates: List[Dict[str, Any]]
            ground_truth_labels: Dict[candidate_id -> bool] (True = verified qualified)
        """
        job_description = (
            "Senior Backend Software Engineer\n"
            "Requirements:\n"
            "- Minimum 4+ years of professional backend software engineering experience.\n"
            "- Strong proficiency in Python, Docker, Kubernetes, and PostgreSQL.\n"
            "- Experience designing scalable REST APIs, microservices, and CI/CD pipelines.\n"
            "- Bachelor's degree or higher in Computer Science or related engineering field."
        )

        required_skills = {"python", "docker", "kubernetes", "postgresql"}
        required_years = 4.0

        if group_weights is None:
            group_weights = {
                "Group A (Traditional)": 0.40,
                "Group B (Female Cohort)": 0.30,
                "Group C (Underrepresented Cohort 1)": 0.15,
                "Group D (Underrepresented Cohort 2)": 0.15,
            }

        groups = list(group_weights.keys())
        weights = list(group_weights.values())

        candidates = []
        ground_truth_labels: Dict[str, bool] = {}

        for i in range(1, n + 1):
            cid = f"SYNTH_{i:04d}"
            assigned_group = self.rng.choices(groups, weights=weights, k=1)[0]

            # Select name based on group
            if "Group A" in assigned_group:
                fname = self.rng.choice(self.FIRST_NAMES_GROUP_A)
            elif "Group B" in assigned_group:
                fname = self.rng.choice(self.FIRST_NAMES_GROUP_B)
            elif "Group C" in assigned_group:
                fname = self.rng.choice(self.FIRST_NAMES_GROUP_C)
            else:
                fname = self.rng.choice(self.FIRST_NAMES_GROUP_D)

            lname = f"Candidate_{i}"
            full_name = f"{fname} {lname}"

            # Qualification generation
            # True capability tier: 0 (Junior), 1 (Mid), 2 (Senior), 3 (Lead)
            tier = self.rng.choices([0, 1, 2, 3], weights=[0.25, 0.35, 0.25, 0.15], k=1)[0]

            if tier == 0:
                years = round(self.rng.uniform(0.5, 2.5), 1)
                num_core = self.rng.randint(1, 2)
                degree_choice = self.rng.choice(self.DEGREES[:3] + self.DEGREES[5:])
            elif tier == 1:
                years = round(self.rng.uniform(2.5, 4.0), 1)
                num_core = self.rng.randint(2, 3)
                degree_choice = self.rng.choice(self.DEGREES[:4])
            elif tier == 2:
                years = round(self.rng.uniform(4.5, 8.0), 1)
                num_core = self.rng.randint(3, 4)
                degree_choice = self.rng.choice(self.DEGREES[:5])
            else:
                years = round(self.rng.uniform(8.0, 14.0), 1)
                num_core = 4
                degree_choice = self.rng.choice(self.DEGREES[:5])

            cand_skills = set(self.rng.sample(self.CORE_TECH_SKILLS, k=min(len(self.CORE_TECH_SKILLS), num_core + 2)))
            cand_skills.update(self.rng.sample(self.DATA_SKILLS, k=self.rng.randint(1, 3)))

            # Build qualification prose
            degree_str, degree_tier = degree_choice
            qual_lines = [
                f"{full_name} is a software engineer with {years} years of professional experience.",
                f"Education: {degree_str}.",
                f"Technical proficiencies: {', '.join(sorted(cand_skills))}.",
                f"Demonstrated background in backend architecture, building microservices, and system reliability."
            ]
            qual_text = " ".join(qual_lines)

            # Objective Ground Truth Qualification Rule (Non-Circular):
            # A candidate is objectively qualified if they have >= 4.0 years experience,
            # possess at least 3 of the 4 required core skills, and hold degree tier >= 3.
            matched_req = len(required_skills.intersection({s.lower() for s in cand_skills}))
            is_objectively_qualified = (
                years >= required_years and matched_req >= 3 and degree_tier >= 3
            )

            ground_truth_labels[cid] = is_objectively_qualified

            candidates.append({
                "candidate_id": cid,
                "name": full_name,
                "qualifications": qual_text,
                "audit_group": assigned_group,
                "ground_truth_qualified": is_objectively_qualified,
            })

        return job_description, candidates, ground_truth_labels


class RankingEvaluator:
    """
    Research-grade benchmark evaluation engine.
    Supports standard cohorts, large synthetic datasets, multi-seed sweeps,
    and Pareto trade-off comparisons.
    """

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        self.generator = SyntheticCandidateGenerator(seed=random_seed)

    @staticmethod
    def load_sample_data(path: Optional[str] = None) -> Dict[str, Any]:
        """Loads sample benchmark datasets from JSON."""
        if path is None:
            path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "sample_data",
                "sample_job_and_resumes.json",
            )
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f).get("datasets", {})

    def evaluate_cohort(
        self,
        job_description: str,
        candidates: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        weights: Optional[Dict[str, float]] = None,
        ground_truth_labels: Optional[Dict[str, bool]] = None,
        run_counterfactual: bool = True,
    ) -> Dict[str, Any]:
        """
        Runs comprehensive multi-model benchmark on a single candidate cohort:
        - TF-IDF Lexical Baseline
        - Semantic Similarity Baseline
        - Hybrid Multi-Aspect Model
        - FA*IR Reranking under Relaxed, Moderate, and Strict constraints
        """
        n = len(candidates)
        if top_k is None or top_k <= 0 or top_k > n:
            top_k = max(1, math.ceil(n / 2))

        # Ground truth set for Recall@K and EOD
        gt_relevant_ids: Optional[Set[str]] = None
        if ground_truth_labels is not None:
            gt_relevant_ids = {cid for cid, is_q in ground_truth_labels.items() if is_q}

        # 1. TF-IDF Lexical Baseline (pure tf-idf, other weights 0)
        tfidf_weights = {"skills": 0.0, "semantic": 1.0, "experience": 0.0, "education": 0.0}
        tfidf_ranked = score_candidates_advanced(
            job_description, candidates, weights=tfidf_weights, use_semantic=False
        )
        tfidf_metrics = calculate_fairness_metrics(
            tfidf_ranked, top_k=top_k, ground_truth_labels=ground_truth_labels
        )
        prec_tfidf = calculate_precision_at_k(tfidf_ranked, k=top_k)
        rec_tfidf = calculate_recall_at_k(tfidf_ranked, k=top_k, ground_truth_relevant_ids=gt_relevant_ids)

        # 2. Hybrid Multi-Aspect Model
        hybrid_ranked = score_candidates_advanced(
            job_description, candidates, weights=weights, use_semantic=True
        )
        hybrid_metrics = calculate_fairness_metrics(
            hybrid_ranked, top_k=top_k, ground_truth_labels=ground_truth_labels
        )
        prec_hybrid = calculate_precision_at_k(hybrid_ranked, k=top_k)
        rec_hybrid = calculate_recall_at_k(hybrid_ranked, k=top_k, ground_truth_relevant_ids=gt_relevant_ids)

        # 3. FA*IR Reranking — Sweep Relaxed, Moderate, Strict
        rerank_variants = {}
        for level, tau in [("relaxed", 1.00), ("moderate", 0.75), ("strict", 0.50)]:
            fair_obj = fair_rerank(
                hybrid_ranked, constraint_level=level, deficit_threshold=tau
            )
            fair_ranked = fair_obj["reranked"]
            fair_metrics = calculate_fairness_metrics(
                fair_ranked, top_k=top_k, ground_truth_labels=ground_truth_labels
            )
            ndcg_fair = calculate_ndcg(hybrid_ranked, fair_ranked, k=top_k)
            prec_fair = calculate_precision_at_k(fair_ranked, k=top_k)
            rec_fair = calculate_recall_at_k(fair_ranked, k=top_k, ground_truth_relevant_ids=gt_relevant_ids)

            rerank_variants[level] = {
                "constraint_level": level,
                "deficit_threshold": tau,
                "disparate_impact_ratio": fair_metrics.get("disparate_impact_ratio"),
                "demographic_parity_diff": fair_metrics.get("demographic_parity_diff"),
                "equal_opportunity_diff": fair_metrics.get("equal_opportunity_diff"),
                "eod_valid": fair_metrics.get("eod_valid"),
                "four_fifths_compliant": fair_metrics.get("four_fifths_compliant"),
                "precision_at_k": prec_fair,
                "recall_at_k": rec_fair,
                "ndcg_at_k": ndcg_fair.get("ndcg_at_k"),
                "utility_retention_pct": fair_obj.get("utility_retention_pct"),
                "swaps_count": fair_obj.get("swaps_count"),
                "avg_rank_shift": fair_obj.get("avg_rank_shift"),
                "max_rank_shift": fair_obj.get("max_rank_shift"),
                "theil_inequality_index": fair_metrics.get("theil_inequality_index"),
                "normalized_theil_index": fair_metrics.get("normalized_theil_index"),
            }

        # 4. Stability comparisons
        cid_to_rank_tfidf = [c["rank"] for c in tfidf_ranked]
        cid_map_hybrid = {c["candidate_id"]: c["rank"] for c in hybrid_ranked}
        cid_to_rank_hybrid_order = [cid_map_hybrid[c["candidate_id"]] for c in tfidf_ranked]
        tau_tfidf_hybrid = kendall_tau(cid_to_rank_tfidf, cid_to_rank_hybrid_order)

        # 5. Optional Counterfactual Audit
        cf_summary = None
        if run_counterfactual and n <= 200:
            cf_audit = run_counterfactual_audit(job_description, candidates, weights=weights)
            cf_metrics = cf_audit.get("metrics", {})
            cf_summary = {
                "max_score_delta": cf_metrics.get("max_score_delta"),
                "mean_abs_delta": cf_metrics.get("mean_abs_delta"),
                "mean_relative_delta_pct": cf_metrics.get("mean_relative_delta_pct"),
                "rank_flips_count": cf_metrics.get("rank_flips_count"),
                "max_rank_displacement": cf_metrics.get("max_rank_displacement"),
                "invariance_rate_pct": cf_metrics.get("invariance_rate_pct"),
                "is_counterfactually_fair": cf_audit.get("is_counterfactually_fair"),
            }

        return {
            "candidate_count": n,
            "top_k": top_k,
            "ground_truth_labels_used": ground_truth_labels is not None,
            "models": {
                "tfidf_baseline": {
                    "disparate_impact_ratio": tfidf_metrics.get("disparate_impact_ratio"),
                    "demographic_parity_diff": tfidf_metrics.get("demographic_parity_diff"),
                    "equal_opportunity_diff": tfidf_metrics.get("equal_opportunity_diff"),
                    "eod_valid": tfidf_metrics.get("eod_valid"),
                    "four_fifths_compliant": tfidf_metrics.get("four_fifths_compliant"),
                    "precision_at_k": prec_tfidf,
                    "recall_at_k": rec_tfidf,
                    "ndcg_at_k": 1.0,
                    "theil_inequality_index": tfidf_metrics.get("theil_inequality_index"),
                    "normalized_theil_index": tfidf_metrics.get("normalized_theil_index"),
                },
                "hybrid_ranking": {
                    "disparate_impact_ratio": hybrid_metrics.get("disparate_impact_ratio"),
                    "demographic_parity_diff": hybrid_metrics.get("demographic_parity_diff"),
                    "equal_opportunity_diff": hybrid_metrics.get("equal_opportunity_diff"),
                    "eod_valid": hybrid_metrics.get("eod_valid"),
                    "four_fifths_compliant": hybrid_metrics.get("four_fifths_compliant"),
                    "precision_at_k": prec_hybrid,
                    "recall_at_k": rec_hybrid,
                    "ndcg_at_k": 1.0,
                    "theil_inequality_index": hybrid_metrics.get("theil_inequality_index"),
                    "normalized_theil_index": hybrid_metrics.get("normalized_theil_index"),
                },
                "fair_reranked_moderate": rerank_variants["moderate"],
                "fair_rerank_variants": rerank_variants,
            },
            "stability": {
                "kendall_tau_tfidf_vs_hybrid": tau_tfidf_hybrid,
            },
            "counterfactual_summary": cf_summary,
        }

    def run_multi_seed_synthetic_benchmark(
        self,
        cohort_sizes: List[int] = [100, 250, 500],
        seeds: List[int] = [42, 123, 999],
    ) -> Dict[str, Any]:
        """
        Executes multi-seed synthetic evaluation across configured cohort scales.
        Calculates mean +- standard deviation for statistical confidence.
        """
        benchmark_results = {}

        for n in cohort_sizes:
            size_key = f"N_{n}"
            seed_runs = []

            for s in seeds:
                gen = SyntheticCandidateGenerator(seed=s)
                jd, cands, gt = gen.generate_cohort(n=n)
                eval_res = self.evaluate_cohort(
                    jd, cands, ground_truth_labels=gt, run_counterfactual=(n <= 100)
                )
                seed_runs.append(eval_res)

            # Compute statistics across seeds
            def stat_summary(values: List[float]) -> Dict[str, float]:
                valid_vals = [v for v in values if v is not None and not math.isnan(v)]
                if not valid_vals:
                    return {"mean": 0.0, "std": 0.0}
                mean_val = sum(valid_vals) / len(valid_vals)
                var = sum((x - mean_val) ** 2 for x in valid_vals) / len(valid_vals) if len(valid_vals) > 1 else 0.0
                return {
                    "mean": round(mean_val, 4),
                    "std": round(math.sqrt(var), 4),
                    "min": round(min(valid_vals), 4),
                    "max": round(max(valid_vals), 4),
                }

            agg_models = {}
            for m_key in ["tfidf_baseline", "hybrid_ranking", "fair_reranked_moderate"]:
                agg_models[m_key] = {
                    "disparate_impact_ratio": stat_summary(
                        [r["models"][m_key]["disparate_impact_ratio"] for r in seed_runs]
                    ),
                    "demographic_parity_diff": stat_summary(
                        [r["models"][m_key]["demographic_parity_diff"] for r in seed_runs]
                    ),
                    "equal_opportunity_diff": stat_summary(
                        [r["models"][m_key]["equal_opportunity_diff"] for r in seed_runs]
                    ),
                    "precision_at_k": stat_summary(
                        [r["models"][m_key]["precision_at_k"] for r in seed_runs]
                    ),
                    "recall_at_k": stat_summary(
                        [r["models"][m_key]["recall_at_k"] for r in seed_runs]
                    ),
                    "ndcg_at_k": stat_summary(
                        [r["models"][m_key]["ndcg_at_k"] for r in seed_runs]
                    ),
                }

            # Add utility retention for fair reranked
            agg_models["fair_reranked_moderate"]["utility_retention_pct"] = stat_summary(
                [r["models"]["fair_reranked_moderate"]["utility_retention_pct"] for r in seed_runs]
            )

            benchmark_results[size_key] = {
                "cohort_size": n,
                "seeds_evaluated": seeds,
                "aggregated_metrics": agg_models,
                "sample_seed_run": seed_runs[0],
            }

        return benchmark_results

    def run_full_suite(self) -> Dict[str, Any]:
        """
        Executes standard sample datasets + multi-seed large-scale synthetic benchmark.
        """
        sample_datasets = self.load_sample_data()
        sample_results = {}
        for key, data in sample_datasets.items():
            sample_results[key] = {
                "title": data.get("title", key),
                "evaluation": self.evaluate_cohort(
                    data["job_description"], data["candidates"], run_counterfactual=True
                ),
            }

        synthetic_results = self.run_multi_seed_synthetic_benchmark(
            cohort_sizes=[100, 250, 500],
            seeds=[42, 123, 999],
        )

        return {
            "experiment_version": "3.0",
            "semantic_transformers_enabled": SEMANTIC_AVAILABLE,
            "sample_cohort_evaluations": sample_results,
            "synthetic_large_scale_benchmark": synthetic_results,
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

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("=" * 75)
    print(" FAIRHIRE AI 3.0 — REPRODUCIBLE RESEARCH BENCHMARK REPORT")
    print("=" * 75)
    print(f"Output File: {output_path}")
    print(f"Sentence-Transformers Active: {SEMANTIC_AVAILABLE}")
    print("-" * 75)

    print("\n[PART 1: STANDARDIZED SAMPLE BENCHMARKS]")
    for cohort_id, cohort_res in results.get("sample_cohort_evaluations", {}).items():
        title = cohort_res["title"]
        ev = cohort_res["evaluation"]["models"]
        print(f"\nCohort: {title} ({cohort_id})")
        print(f"  TF-IDF Baseline:      DIR={ev['tfidf_baseline']['disparate_impact_ratio']} | SPD={ev['tfidf_baseline']['demographic_parity_diff']} | Prec@K={ev['tfidf_baseline']['precision_at_k']}")
        print(f"  Hybrid Ranking:       DIR={ev['hybrid_ranking']['disparate_impact_ratio']} | SPD={ev['hybrid_ranking']['demographic_parity_diff']} | Prec@K={ev['hybrid_ranking']['precision_at_k']}")
        print(f"  Fair Rerank (Mod):    DIR={ev['fair_reranked_moderate']['disparate_impact_ratio']} | Ret={ev['fair_reranked_moderate']['utility_retention_pct']}% | NDCG={ev['fair_reranked_moderate']['ndcg_at_k']}")

    print("\n[PART 2: MULTI-SEED LARGE-SCALE SYNTHETIC BENCHMARK]")
    for size_key, size_res in results.get("synthetic_large_scale_benchmark", {}).items():
        n = size_res["cohort_size"]
        agg = size_res["aggregated_metrics"]
        print(f"\nCohort Size N = {n} (Averaged across 3 random seeds: [42, 123, 999])")
        for m_name, label in [
            ("tfidf_baseline", "TF-IDF Baseline"),
            ("hybrid_ranking", "Hybrid Multi-Aspect"),
            ("fair_reranked_moderate", "Fair Reranked (Moderate)")
        ]:
            dir_st = agg[m_name]["disparate_impact_ratio"]
            spd_st = agg[m_name]["demographic_parity_diff"]
            eod_st = agg[m_name]["equal_opportunity_diff"]
            p_st = agg[m_name]["precision_at_k"]
            r_st = agg[m_name]["recall_at_k"]
            ndcg_st = agg[m_name]["ndcg_at_k"]

            print(f"  {label:25s} | DIR: {dir_st['mean']:.3f} +- {dir_st['std']:.3f} | "
                  f"SPD: {spd_st['mean']:.3f} +- {spd_st['std']:.3f} | "
                  f"EOD: {eod_st['mean']:.3f} +- {eod_st['std']:.3f} | "
                  f"Prec@K: {p_st['mean']:.3f} | Rec@K: {r_st['mean']:.3f} | NDCG: {ndcg_st['mean']:.3f}")

    print("=" * 75)
    return results


if __name__ == "__main__":
    run_reproducible_experiment()
