# FairHire AI 3.0 — Fairness-Aware Resume Ranking & Algorithmic Audit System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-76%20Passed-success.svg)]()
[![Compliance](https://img.shields.io/badge/Audit%20Framework-EEOC%20%7C%20NYC%20LL144%20%7C%20EU%20AI%20Act-teal.svg)]()
[![Code%20Audit](https://img.shields.io/badge/Audit%20Report-PROJECT__AUDIT.md-blue.svg)](PROJECT_AUDIT.md)

**FairHire AI 3.0** is an explainable, audit-ready applicant ranking and algorithmic fairness platform designed to mitigate demographic bias in recruitment intelligence. Moving far beyond trivial group-tag swaps, FairHire AI integrates a multi-aspect hybrid scoring engine, multi-variation counterfactual perturbation testing (pronouns, names, affinity orgs), EEOC 4/5ths Rule compliance analytics, rank-biased positional exposure metrics, configurable FA*IR post-processed reranking with Pareto trade-off sensitivity, and a research-grade synthetic evaluation framework.

---

## Table of Contents
- [Architecture & System Overview](#architecture--system-overview)
- [Key Features in FairHire AI 3.0](#key-features-in-fairhire-ai-30)
- [Installation & Quickstart](#installation--quickstart)
- [Running Automated Tests](#running-automated-tests)
- [Experimental Evaluation Framework](#experimental-evaluation-framework)
- [REST API Reference](#rest-api-reference)
- [Security & Privacy Controls](#security--privacy-controls)
- [What This System Does NOT Do](#what-this-system-does-not-do)
- [Academic & Ethical Methodology](#academic--ethical-methodology)

---

## Architecture & System Overview

```
                      ┌──────────────────────────────────────────────┐
                      │    Input: Job Description & Resumes/Files    │
                      │         (PDF, DOCX, TXT, Manual Entry)       │
                      └──────────────────────┬───────────────────────┘
                                             │
                        ┌────────────────────▼────────────────────┐
                        │   Secure Document Parser & Extractor    │
                        │ (Magic Bytes, Page Limit, PII Redaction)│
                        └────────────────────┬────────────────────┘
                                             │
                        ┌────────────────────▼────────────────────┐
                        │       Canonical Skill Taxonomy &        │
                        │        Experience / Degree Normalizer   │
                        └────────────────────┬────────────────────┘
                                             │
             ┌───────────────────────────────┼───────────────────────────────┐
             │                               │                               │
┌────────────▼────────────┐     ┌────────────▼────────────┐     ┌────────────▼────────────┐
│   Multi-Aspect Hybrid   │     │  Counterfactual Invariance│     │   Algorithmic Fairness  │
│      Scoring Engine     │     │      Sensitivity Audit  │     │      Metrics Suite      │
│ - Skills Match (40%)    │     │ - Single-Pass Swaps     │     │ - Disparate Impact (DIR)│
│ - TF-IDF Semantic (30%) │     │ - Name Perturbation     │     │ - Demographic Parity    │
│ - Experience Fit (15%)  │     │ - Score Shift & Flips   │     │ - EOD (Proxy Flagged)   │
│ - Education Fit (15%)   │     │ - Proxy Bias Detection  │     │ - Positional Exposure   │
│ - Dense Embeddings (Opt)│     │ - Causal Limitations    │     │ - Theil Entropy Index   │
└────────────┬────────────┘     └────────────┬────────────┘     └────────────┬────────────┘
             │                               │                               │
             └───────────────────────────────┼───────────────────────────────┘
                                             │
                        ┌────────────────────▼────────────────────┐
                        │       FA*IR Bounded-Regret Fair        │
                        │         Post-Processed Reranking        │
                        │   (DCG Utility Retention >95%)          │
                        └────────────────────┬────────────────────┘
                                             │
                        ┌────────────────────▼────────────────────┐
                        │  Interactive Dark UI Dashboard (7 Tabs) │
                        │   CSV & JSON Downloadable Reports       │
                        └─────────────────────────────────────────┘
```

---

## Key Features in FairHire AI 3.0

### 1. Multi-Aspect Hybrid Candidate Ranking
- **Lexical & Canonical Skill Overlap**: 100+ categorized skills with multi-word canonical normalization (e.g. `k8s` &rarr; `kubernetes`, `golang` &rarr; `go`, `rag` &rarr; `retrieval augmented generation`).
- **Semantic Text Relevance**: Sublinear TF-IDF cosine similarity filtered with domain-specific recruitment stop words, optionally blended with dense Sentence-Transformers embeddings when available.
- **Experience Duration & Alignment**: Smooth non-linear scoring based on required vs demonstrated tenure, guarding against graduation-year confusion.
- **Educational Credential Tiering & Missing-Information Strategy**: Tiering from Doctorate (5) down to Bootcamp/Certificate (1). Supports **Proportional Weight Rebalancing** so candidates without explicit degree metadata are evaluated purely on demonstrated skills and experience without an automatic punitive zero-score penalty.

### 2. Methodologically Grounded Fairness Metrics
- **Multi-Group Disparate Impact Ratio (DIR)**: Evaluates selection rates relative to the benchmark group using the EEOC 4/5ths Rule threshold ($DIR \ge 0.80$).
- **Statistical / Demographic Parity Difference (SPD)**: Maximum selection rate disparity across any two audited demographic cohorts.
- **Equal Opportunity Difference (EOD)**: Supports independent ground-truth qualification labels to avoid circular evaluation. When proxy thresholds are used, methodological limitations are explicitly flagged.
- **Rank-Biased Positional Exposure**: Eye-tracking logarithmic visibility discounting ($v(r) = 1/\log_2(1+r)$) to measure screen real-estate allocation disparities.
- **Theil Inequality & Normalized Theil Index**: Generalized entropy index ($\alpha=1$) measuring inequality in attention distribution across candidate queue positions.
- **Uncertainty & Sample Size Reporting**: Flags small group sizes ($n < 5$) with prominent warning banners.

### 3. Multi-Variation Counterfactual & Proxy Bias Auditing
- **Controlled Variations Evaluated**:
  1. *Variant A*: Direct Gender Pronoun & Honorific Invariance (`he/him` &harr; `she/her`, `Mr` &harr; `Ms`).
  2. *Variant B*: Candidate Name Substitution (Audit study methodology: Bertrand & Mullainathan 2004).
  3. *Variant C*: Affinity & Institutional Proxies (`SWE`, `Women Who Code`, `HBCUs` &harr; neutral counterparts).
  4. *Variant D*: Combined Multi-Attribute Perturbation.
- **Granular Metrics**: Computes Original Score, Counterfactual Score, Absolute Delta ($|\Delta s|$), Relative Delta ($|\Delta s| / s_{\text{orig}}$), Original Rank, Counterfactual Rank, Rank Shift, and Rank Displacement ($|r_{\text{orig}} - r_{\text{cf}}|$) for every applicant.
- **Transparent Causal Disclaimer**: Documents Kusner et al. (2017) causal limitations: text-marker invariance tests observational surface sensitivity and does not prove absence of latent structural bias.

### 4. Configurable Fair Reranking (FA*IR) & Pareto Sensitivity
- **Configurable Deficit Thresholds**:
  * *Strict* ($\tau = 0.50$): High representation guarantee; maximizes DIR.
  * *Moderate* ($\tau = 0.75$, default): Balanced representation and ranking fidelity.
  * *Relaxed* ($\tau = 1.00$): High ranking utility retention ($>99.9\%$ DCG retention).
- **Pareto Trade-off Table**: Displays empirical trade-offs between demographic representation (DIR, SPD) and ranking utility (NDCG@K, DCG retention, rank displacement).

### 5. Research-Grade Evaluation Framework
- **Scalable Synthetic Generator**: Simulates cohorts of $N=100, 250, 500, 1000$ candidates with independent ground-truth qualification labels.
- **Multi-Seed Statistical Sweep**: Runs benchmarks across random seeds `[42, 123, 999]`, reporting $\mu \pm \sigma$ for Precision@K, Recall@K, NDCG@K, DIR, SPD, and EOD.

### 6. Explainable AI (XAI) & Interactive Pairwise Comparison
- Multi-dimensional score decompositions with exact point contributions.
- Actionable next-tier upskilling guidance explaining points required to advance.
- Interactive pairwise candidate comparison detailing dimensional deltas and unique skill attributions.

### 7. Modern Dark Navy & Emerald Green Dashboard
- 7 comprehensive interactive tabs:
  1. **Rankings & Explanations**
  2. **Fairness Metrics Suite**
  3. **Counterfactual Invariance Audit (with Multi-Variation Breakdown)**
  4. **Fair Reranking Trade-offs (with Pareto Sensitivity Analysis)**
  5. **Interactive Pairwise XAI**
  6. **Evaluation Reports (CSV & JSON Download)**
  7. **Model Comparison & Academic Methodology**

---

## Installation & Quickstart

### Prerequisites
- Python 3.10+ (Tested on Python 3.14)
- `pip` package manager

### Setup
```bash
# 1. Clone or navigate to the project directory
cd /path/to/fairness_aware_resume_ranking

# 2. Install required dependencies
python3 -m pip install -r requirements.txt

# 3. Optional: Install dense semantic transformers (requires ~500MB disk space)
# python3 -m pip install sentence-transformers
```

### Running the Application
```bash
# Production mode (default)
python3 app.py

# Development mode
FLASK_DEBUG=true SECRET_KEY=my-secret-key python3 app.py
```
Open your browser to `http://127.0.0.1:5000`.

---

## Running Automated Tests

FairHire AI 3.0 includes an automated test suite with **68 tests** covering all modules:

```bash
# Run complete test suite
python3 -m pytest -v

# Run specific module test suites
python3 -m pytest tests/test_ranking_engine.py -v
python3 -m pytest tests/test_fairness_metrics.py -v
python3 -m pytest tests/test_fairness_audit.py -v
python3 -m pytest tests/test_document_parser.py -v
python3 -m pytest tests/test_app.py -v
python3 -m pytest tests/test_explanation_engine.py -v
python3 -m pytest tests/test_skill_taxonomy.py -v
```

---

## Experimental Evaluation Framework

The project includes a standalone, reproducible benchmark suite in `evaluation/eval_framework.py`:

```bash
# Run the evaluation benchmark
python3 -m evaluation.eval_framework
```

This runs evaluations across all three benchmark cohorts (Software Engineer, ML Scientist, Full Stack Cloud Cohort) and outputs results to `evaluation/results.json`.

---

## REST API Reference

### 1. `GET /api/health`
Health and capability check.
```json
{
  "status": "ok",
  "service": "FairHire AI",
  "version": "3.0",
  "semantic_model_available": false
}
```

### 2. `POST /api/rank`
Ranks candidate pool, executes fairness metrics, runs counterfactual audit, and fair reranks.
- **Request Body**:
  ```json
  {
    "job_description": "Senior Python engineer with Docker and PostgreSQL.",
    "candidates": [
      {
        "candidate_id": "C1",
        "name": "Candidate A",
        "qualifications": "6 years Python, Docker, PostgreSQL.",
        "audit_group": "Group A"
      },
      {
        "candidate_id": "C2",
        "name": "Candidate B",
        "qualifications": "Junior Python developer with 1 year experience.",
        "audit_group": "Group B"
      }
    ],
    "top_k": 1,
    "weights": {"skills": 0.40, "semantic": 0.30, "experience": 0.15, "education": 0.15}
  }
  ```

### 3. `POST /api/parse-resume`
Extracts structured metadata from uploaded PDF, DOCX, or TXT file. Protects PII by returning text length and sanitized preview rather than raw body.

### 4. `POST /api/pairwise-compare`
Generates comparative justification between Candidate A and Candidate B.

### 5. `GET /api/export-audit`
Exports current session's evaluation and compliance metrics as formatted JSON.

### 6. `GET /api/export-ranking-csv`
Downloads candidate rankings as a CSV spreadsheet.

---

## Security & Privacy Controls

FairHire AI 3.0 has undergone a rigorous security hardening process:
1. **Isolated Session Storage**: Global report storage has been replaced with per-session memory stores (`AUDIT_STORE`) to eliminate cross-user data leakage.
2. **File Size & Content Validation**: Enforces a strict 5MB upload ceiling, file extension whitelist, and binary magic-byte verification (`%PDF`, `PK`).
3. **PDF Security Limits**: Enforces a 10-page limit and blocks password-encrypted or malformed documents.
4. **PII Protection**: File parsing APIs omit raw document bodies from network responses.
5. **HTTP Defense-in-Depth**: Automatically applies `Content-Security-Policy`, `X-Content-Type-Options: nosniff`, and `X-Frame-Options: SAMEORIGIN` headers.
6. **No Debug in Production**: Debug mode defaults to `False` unless explicitly enabled via `FLASK_DEBUG=true`.

---

## What This System Does NOT Do

To ensure scientific honesty and ethical deployment, we explicitly document operational boundaries:

1. **Does NOT certify legal compliance**: An algorithm achieving a Disparate Impact Ratio $\ge 0.80$ satisfies a mathematical guideline but does not constitute legal certification under Title VII, NYC Local Law 144, or the EU AI Act.
2. **Does NOT prove complete absence of bias**: Passing counterfactual text-invariance tests confirms insensitivity to specific surface tokens (e.g. pronouns), not absence of systemic or historical bias in qualifications.
3. **Does NOT make autonomous hiring decisions**: This software is designed exclusively as an explainable decision-support tool. Autonomous decisions without human review are outside its intended scope.
4. **Does NOT assume skill mentions equal proficiency**: Skill extraction measures demonstrated keyword overlap and contextual evidence, not verified competency.
5. **Does NOT perform optical character recognition (OCR)**: Scanned image-only PDFs without embedded text streams are rejected with clear error guidance.

---

## Academic & Ethical Methodology

For in-depth mathematical formulations, causal graph diagrams, algorithmic proofs, and empirical benchmark findings, please consult [RESEARCH.md](file:///home/pruthvirajalagur/Downloads/project/fairness_aware_resume_ranking/RESEARCH.md).
