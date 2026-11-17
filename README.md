# FairHire AI — Fairness-Aware Resume Ranking & Algorithmic Audit System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-32%20Passed-success.svg)]()
[![Compliance](https://img.shields.io/badge/Compliance-EEOC%20%7C%20NYC%20LL144%20%7C%20EU%20AI%20Act-teal.svg)]()

**FairHire AI** is an explainable, audit-ready applicant ranking and algorithmic fairness platform designed to mitigate demographic bias in recruitment intelligence. Moving far beyond trivial group-tag swaps, FairHire AI integrates a multi-aspect hybrid scoring engine, non-trivial counterfactual token perturbation testing, EEOC 4/5ths Rule compliance analytics, rank-biased positional exposure metrics, and post-processed fair reranking.

---

## Table of Contents
- [Architecture & System Overview](#architecture--system-overview)
- [Key Features Implemented](#key-features-implemented)
  - [1. Non-Trivial Counterfactual & Sensitivity Audit](#1-non-trivial-counterfactual--sensitivity-audit)
  - [2. Comprehensive Algorithmic Fairness Metrics](#2-comprehensive-algorithmic-fairness-metrics)
  - [3. Multi-Aspect Hybrid Ranking & Fair Reranking](#3-multi-aspect-hybrid-ranking--fair-reranking)
  - [4. Explainable AI (XAI) & Pairwise Justifications](#4-explainable-ai-xai--pairwise-justifications)
  - [5. Native PDF/DOCX/TXT Multi-Format Parsing](#5-native-pdfdocxtxt-multi-format-parsing)
  - [6. Interactive Compliance & Audit Dashboard](#6-interactive-compliance--audit-dashboard)
  - [7. Automated Pytest Test Suite](#7-automated-pytest-test-suite)
- [Installation & Quickstart](#installation--quickstart)
- [Running Automated Tests](#running-automated-tests)
- [REST API Reference](#rest-api-reference)
- [Regulatory Compliance Mapping](#regulatory-compliance-mapping)
- [Research Documentation](#research-documentation)

---

## Architecture & System Overview

```
                      ┌──────────────────────────────────────────────┐
                      │    Input: Job Description & Resumes/Files    │
                      │         (PDF, DOCX, TXT, Manual Entry)       │
                      └──────────────────────┬───────────────────────┘
                                             │
                        ┌────────────────────▼────────────────────┐
                        │   Document Parser & Section Extractor   │
                        │    (Sanitization, Contact Info, Layout) │
                        └────────────────────┬────────────────────┘
                                             │
                        ┌────────────────────▼────────────────────┐
                        │       Canonical Skill Taxonomy &        │
                        │        Experience / Degree Normalizer   │
                        └────────────────────┬────────────────────┘
                                             │
            ┌────────────────────────────────┼────────────────────────────────┐
            │                                │                                │
┌───────────▼───────────┐        ┌───────────▼───────────┐        ┌───────────▼───────────┐
│  Multi-Aspect Hybrid  │        │ Rigorous Non-Trivial  │        │ Algorithmic Fairness  │
│    Scoring Engine     │        │ Counterfactual Audit  │        │     Metrics Suite     │
│  - Skills (40%)       │        │ - Pronoun Swaps       │        │ - EEOC 4/5ths Rule    │
│  - Semantic (30%)     │        │ - Title/Affiliations  │        │ - Demographic Parity  │
│  - Experience (15%)   │        │ - Score Shift & Flips │        │ - Equal Opportunity   │
│  - Education (15%)    │        │ - Proxy Bias Flags    │        │ - Positional Exposure │
└───────────┬───────────┘        └───────────┬───────────┘        └───────────┬───────────┘
            │                                │                                │
            └────────────────────────────────┼────────────────────────────────┘
                                             │
                        ┌────────────────────▼────────────────────┐
                        │       FA*IR Bounded-Regret Fair        │
                        │         Post-Processed Reranking        │
                        └────────────────────┬────────────────────┘
                                             │
                        ┌────────────────────▼────────────────────┐
                        │      Interactive Audit Dashboard &      │
                        │      JSON Compliance Export Report      │
                        └─────────────────────────────────────────┘
```

---

## Key Features Implemented

### 1. Non-Trivial Counterfactual & Sensitivity Audit
Replaced trivial group-tag switches (which tested only unread metadata) with genuine linguistic perturbation:
- **Demographic Token Perturbation**: Systematically swaps real gendered markers within resume text (e.g. `he/she`, `his/her`, `Mr./Ms.`, `father/mother`, `Society of Women Engineers` vs honors societies) and recalculates candidate rankings.
- **Invariance & Stability Verification**: Quantifies individual fairness through Maximum Score Delta ($\max |\Delta S_i|$), Mean Absolute Shift, and Rank-Flip Counters.
- **Proxy Bias Detection**: Identifies covert demographic proxies in resume text (early graduation dates, seniority markers, gendered affiliations, parental leave mentions) with actionable auditing alerts.

### 2. Comprehensive Algorithmic Fairness Metrics
Implements standard academic and legal statistical criteria:
- **Disparate Impact Ratio (DIR / 80% Rule)**: Verifies that selection rates for protected/audited groups satisfy the EEOC threshold ($\text{DIR} \ge 0.80$).
- **Statistical / Demographic Parity Difference (SPD)**: Measures absolute disparity between group selection probabilities ($|P(\hat{Y}=1|G_A) - P(\hat{Y}=1|G_B)|$).
- **Equal Opportunity Difference (EOD)**: Compares True Positive Rates across groups among qualified candidates meeting baseline score thresholds.
- **Rank-Biased Positional Exposure**: Implements logarithmic eye-tracking attention discounting ($v(r) = 1/\log_2(1+r)$) to measure recruiter visibility disparity between demographic groups.
- **Score Distribution Analytics**: Reports group-level mean, median, standard deviation, and Cohen's $d$ effect sizes.

### 3. Multi-Aspect Hybrid Ranking & Fair Reranking
- **Domain-Specific Skill Taxonomy**: Over 300+ technical, cloud, AI, and soft skills with canonical mapping and synonym resolution (e.g., `k8s` &rarr; `kubernetes`, `postgres` &rarr; `postgresql`, `py` &rarr; `python`).
- **Balanced Multi-Aspect Scoring**:
  - **Skill Match (40%)**: Core required skill coverage vs secondary/bonus skills.
  - **Semantic Relevance (30%)**: Sublinear TF-IDF cosine similarity with domain stop words.
  - **Experience Alignment (15%)**: Automated detection of years of experience vs job requirements.
  - **Education Fit (15%)**: Recognition of Doctorate, Master's, Bachelor's, Associate degrees.
- **FA*IR Post-Processing Reranking**: Provides a bounded-regret fair reranking algorithm that dynamically enforces demographic parity constraints while preserving applicant merit and score utility.

### 4. Explainable AI (XAI) & Pairwise Justifications
- **Multi-Dimensional Decompositions**: Explicit score breakdowns for every candidate across all four evaluation aspects.
- **Skill Gap & Strength Attribution**: Identifies matched core skills, missing required skills, and candidate bonuses.
- **Automated Pairwise Comparative Rationale**: Generates transparent justifications comparing Candidate A and Candidate B (e.g., higher core skill coverage + matched required database experience, despite similar education).
- **Demographic Isolation Proof**: Audits and explicitly confirms that candidate names, gender pronouns, and demographic markers were isolated from model feature matrices.

### 5. Native PDF/DOCX/TXT Multi-Format Parsing
- **Direct Multi-File Ingestion**: Upload PDF resumes via `pypdf`, DOCX resumes via `python-docx`, and plain text with automatic multi-encoding fallback (`utf-8`, `latin-1`, `cp1252`).
- **Metadata Extraction**: Automatically extracts candidate names, emails, phone numbers, GitHub/LinkedIn links, and segments text into functional sections (Summary, Skills, Experience, Education).

### 6. Interactive Compliance & Audit Dashboard
- **Modern UI & Visualization**: Clean design system built with Plus Jakarta Sans and high-contrast color coding.
- **Tabbed Audit Center**:
  - *Rankings & Explanations*: Leaderboard with tier badges (Top Fit, Strong Match, Qualified, Review), score bars, and expandable breakdown cards.
  - *Fairness Metrics Suite*: EEOC 4/5ths Rule status banner, DIR gauge, metric cards, and demographic distribution tables.
  - *Counterfactual & Proxy Audit*: Invariance rate meter, individual perturbation diff table, and proxy alerts.
  - *Fair Reranking Comparison*: Side-by-side comparison of Unconstrained Ranking vs Fair Top-$k$ Reranking.
- **One-Click Benchmark Datasets**: Instantly load realistic Software Engineer and ML Scientist cohorts to run demonstrations.
- **Audit Report Export**: One-click download of auditable JSON reports for HR recordkeeping and legal compliance.

### 7. Automated Pytest Test Suite
32 unit and integration tests covering parser accuracy, skill extraction, ranking monotonicity, fair reranking, fairness metrics, counterfactual invariance, and Flask REST endpoints.

---

## Installation & Quickstart

### Prerequisites
- Python 3.10+
- pip

### 1. Clone Repository & Setup Virtual Environment

**On Linux / macOS:**
```bash
cd fairness_aware_resume_ranking
python3 -m venv .venv
source .venv/bin/activate
```

**On Windows:**
```powershell
cd fairness_aware_resume_ranking
python -m venv .venv
.venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch Application
```bash
python3 app.py
```
Open **`http://127.0.0.1:5000`** in your browser.

---

## Running Automated Tests

Run the full automated test suite using `pytest`:
```bash
pytest -v
```

Expected output:
```
tests/test_app.py::test_get_index PASSED
tests/test_app.py::test_post_sample_dataset_load PASSED
tests/test_app.py::test_post_candidate_ranking_and_audit PASSED
tests/test_app.py::test_api_sample_datasets PASSED
tests/test_app.py::test_api_rank_endpoint PASSED
tests/test_app.py::test_api_parse_resume_txt PASSED
tests/test_app.py::test_api_export_audit PASSED
tests/test_document_parser.py::test_sanitize_text PASSED
tests/test_document_parser.py::test_parse_docx_in_memory PASSED
tests/test_document_parser.py::test_parse_pdf_in_memory PASSED
tests/test_document_parser.py::test_parse_text_multi_encoding PASSED
tests/test_document_parser.py::test_extract_candidate_name PASSED
tests/test_document_parser.py::test_extract_contact_info PASSED
tests/test_document_parser.py::test_extract_resume_sections PASSED
tests/test_explanation_engine.py::test_generate_candidate_explanation PASSED
tests/test_explanation_engine.py::test_generate_pairwise_comparison PASSED
tests/test_fairness_audit.py::test_perturb_text_demographics_swaps PASSED
tests/test_fairness_audit.py::test_detect_proxy_biases PASSED
tests/test_fairness_audit.py::test_run_counterfactual_audit_invariance PASSED
tests/test_fairness_metrics.py::test_calculate_group_distributions PASSED
tests/test_fairness_metrics.py::test_calculate_cohens_d PASSED
tests/test_fairness_metrics.py::test_fairness_metrics_four_fifths_compliant PASSED
tests/test_fairness_metrics.py::test_fairness_metrics_adverse_impact_violation PASSED
tests/test_ranking_engine.py::test_compute_skill_scores PASSED
tests/test_ranking_engine.py::test_compute_experience_alignment PASSED
tests/test_ranking_engine.py::test_score_candidates_advanced_bounds_and_ranking PASSED
tests/test_ranking_engine.py::test_fair_rerank_preserves_candidates_and_corrects_quota PASSED
tests/test_skill_taxonomy.py::test_extract_skills_canonical_and_synonyms PASSED
tests/test_skill_taxonomy.py::test_extract_skills_word_boundary_isolation PASSED
tests/test_skill_taxonomy.py::test_extract_years_of_experience PASSED
tests/test_skill_taxonomy.py::test_extract_education_level PASSED
tests/test_skill_taxonomy.py::test_get_skill_category PASSED

============================== 32 passed in ~1.0s ==============================
```

---

## REST API Reference

### 1. `POST /api/rank`
Ranks candidate pool and computes full fairness audit.
- **Request Body (JSON)**:
  ```json
  {
    "job_description": "Senior Python developer with Docker and PostgreSQL.",
    "candidates": [
      {
        "candidate_id": "C1",
        "name": "Candidate 1",
        "qualifications": "5 years Python, Docker, PostgreSQL. BS in Computer Science.",
        "audit_group": "Group A"
      },
      {
        "candidate_id": "C2",
        "name": "Candidate 2",
        "qualifications": "4 years Python, AWS, PostgreSQL. MS in CS.",
        "audit_group": "Group B"
      }
    ],
    "ranking_mode": "standard",
    "top_k": 1
  }
  ```
- **Response**: Returns standard ranking, fair reranking, DIR, SPD, EOD, exposure disparity, and counterfactual audit metrics.

### 2. `POST /api/parse-resume`
Uploads and parses a PDF, DOCX, or TXT resume file.
- **Request**: Multipart form data with `file`.
- **Response**: JSON object with candidate name, email, phone, extracted skills, sections, and sanitized text.

### 3. `POST /api/pairwise-compare`
Generates comparative justification between two candidates.
- **Request Body (JSON)**: `{ "candidate_a": {...}, "candidate_b": {...} }`
- **Response**: Comparative breakdown, score delta, and natural language justification.

### 4. `GET /api/sample-datasets`
Returns pre-configured benchmark evaluation cohorts (e.g. Software Engineer, Data Scientist).

### 5. `GET /api/export-audit`
Downloads full audit certificate in JSON format for the latest evaluation.

---

## Regulatory Compliance Mapping

| Legal / Regulatory Standard | Specific Requirement | FairHire AI Implementation |
|:----------------------------|:---------------------|:---------------------------|
| **EEOC Uniform Guidelines (UGESP)** | 4/5ths Rule (Adverse Impact Ratio $\ge 0.80$) | Computes Disparate Impact Ratio across demographic cohorts; flags alerts. |
| **NYC Local Law 144 (AEDT)** | Annual independent bias audit of automated employment decision tools | Publishes selection rates, impact ratios, and auditable JSON export. |
| **EU AI Act (Annex III High-Risk AI)** | Mandatory risk mitigation & human oversight for employment AI | Explainable feature attribution, proxy bias detection, and fair reranking. |

---

## Research Documentation
For mathematical derivations, algorithmic details, and in-depth academic literature review, see **[RESEARCH.md](file:///home/pruthvirajalagur/Downloads/project/fairness_aware_resume_ranking/RESEARCH.md)**.
