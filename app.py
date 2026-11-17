from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

app = Flask(__name__)

# Ranking uses qualifications only. Names and protected attributes are never model inputs.
def tokenize_skills(text):
    return sorted(set(re.findall(r"[a-zA-Z][a-zA-Z+#. -]{1,30}", text.lower())))

def score_candidates(job_description, candidates):
    documents = [job_description] + [c["qualifications"] for c in candidates]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(documents)
    similarities = cosine_similarity(matrix[0:1], matrix[1:]).flatten()

    job_terms = set(vectorizer.get_feature_names_out())
    results = []
    for candidate, similarity in zip(candidates, similarities):
        candidate_terms = set(re.findall(r"[a-zA-Z][a-zA-Z+#. -]{1,30}", candidate["qualifications"].lower()))
        matched = sorted(term for term in candidate_terms if term in job_terms and len(term) > 2)
        results.append({
            **candidate,
            "score": round(float(similarity) * 100, 2),
            "matched_terms": matched[:12],
            "explanation": (
                f"Relevance score is based on text similarity between the job description "
                f"and this candidate's skills, education and experience. "
                f"Matching terms: {', '.join(matched[:8]) if matched else 'No clear keyword overlap'}."
            )
        })
    return sorted(results, key=lambda x: x["score"], reverse=True)

def counterfactual_audit(job_description, candidates):
    """
    Counterfactual test: hold qualifications fixed and change audit-only group label.
    The ranking function must return the same score because group labels are not inputs.
    This tests implementation-level invariance, not real-world fairness by itself.
    """
    baseline = score_candidates(job_description, candidates)
    changed = []
    for c in candidates:
        altered = dict(c)
        altered["audit_group"] = (
            "Group B" if c.get("audit_group") == "Group A" else "Group A"
        )
        changed.append(altered)
    counterfactual = score_candidates(job_description, changed)

    baseline_scores = {x["candidate_id"]: x["score"] for x in baseline}
    cf_scores = {x["candidate_id"]: x["score"] for x in counterfactual}
    deltas = {
        cid: round(cf_scores[cid] - score, 6)
        for cid, score in baseline_scores.items()
    }
    invariant = all(abs(delta) < 1e-9 for delta in deltas.values())
    return {
        "invariant": invariant,
        "deltas": deltas,
        "message": (
            "Scores remained unchanged when audit-only group labels were switched."
            if invariant else
            "Warning: scores changed after audit labels were switched."
        )
    }

@app.route("/", methods=["GET", "POST"])
def index():
    results = None
    audit = None
    error = None
    job_description = ""
    if request.method == "POST":
        job_description = request.form.get("job_description", "").strip()
        names = request.form.getlist("candidate_name[]")
        qualifications = request.form.getlist("qualifications[]")
        groups = request.form.getlist("audit_group[]")

        candidates = []
        for i, (name, qualification) in enumerate(zip(names, qualifications)):
            if qualification.strip():
                candidates.append({
                    "candidate_id": f"C{i+1}",
                    "name": name.strip() or f"Candidate {i+1}",
                    "qualifications": qualification.strip(),
                    "audit_group": groups[i] if i < len(groups) else "Not specified"
                })

        if not job_description:
            error = "Please enter a job description."
        elif len(candidates) < 2:
            error = "Please enter at least two candidates with qualifications."
        else:
            try:
                results = score_candidates(job_description, candidates)
                audit = counterfactual_audit(job_description, candidates)
            except ValueError:
                error = "Please provide more descriptive job and candidate text."

    return render_template("index.html", results=results, audit=audit,
                           error=error, job_description=job_description)

if __name__ == "__main__":
    app.run(debug=True)
