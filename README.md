# FairHire AI — Fairness-Aware Resume Ranking

A small academic prototype that compares candidate qualifications with a job description, ranks candidates using TF-IDF cosine similarity, provides keyword-based explanations, and runs a controlled counterfactual invariance check.

## Requirements
- Python 3.10+
- pip

## Run on Fedora / Linux

```bash
cd fairness_aware_resume_ranking
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000` in your browser.

## How it works
1. Enter a job description.
2. Add at least two candidate profiles with skills, education, projects and experience.
3. The app vectorizes the job description and qualifications using TF-IDF.
4. Cosine similarity is used as a relevance score.
5. Matching terms are displayed as a basic explanation.
6. The counterfactual audit switches the audit-only group label while holding qualifications fixed, then checks whether scores change.

## Important limitations
- This is a prototype, not a validated hiring tool.
- TF-IDF similarity is not a measure of candidate quality or job performance.
- The displayed matching terms are simple keyword evidence, not SHAP explanations.
- The current counterfactual test proves only that the audit label is not used by this implementation. It does not establish fairness across real demographic groups or detect proxy bias in resume wording.
- Do not use this prototype to make real hiring decisions. Human review, representative evaluation data, privacy safeguards and legal review are necessary.

## Suggested next research milestones
1. Add PDF/DOCX resume parsing.
2. Build a reviewed skills taxonomy and structured feature extraction.
3. Add a supervised ranking model only if you have a suitable, consented and representative dataset.
4. Add SHAP explanations for a trained model.
5. Evaluate group-level selection rates, equal opportunity difference and ranking disparities on appropriately governed test data.
6. Design stronger counterfactual pairs that change one protected attribute while holding qualifications constant, and report score/rank changes.
7. Add tests, model cards, privacy controls and human-review workflow.
