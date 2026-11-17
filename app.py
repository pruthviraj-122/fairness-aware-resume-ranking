"""
FairHire AI — Fairness-Aware Resume Ranking & Compliance Audit System
Flask Application & API Service
"""

import json
import os
from typing import Dict, List, Optional
from flask import Flask, jsonify, render_template, request, send_file, Response

from core.document_parser import (
    parse_document,
    extract_candidate_name,
    extract_contact_info,
    extract_resume_sections
)
from core.skill_taxonomy import extract_skills_from_text
from core.ranking_engine import score_candidates_advanced, fair_rerank
from core.fairness_metrics import calculate_fairness_metrics
from core.fairness_audit import run_counterfactual_audit
from core.explanation_engine import (
    generate_candidate_explanation,
    generate_pairwise_comparison
)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max upload

# Load sample datasets
SAMPLE_DATA_PATH = os.path.join(os.path.dirname(__file__), "sample_data", "sample_job_and_resumes.json")
SAMPLE_DATA = {}
if os.path.exists(SAMPLE_DATA_PATH):
    try:
        with open(SAMPLE_DATA_PATH, "r", encoding="utf-8") as f:
            SAMPLE_DATA = json.load(f).get("datasets", {})
    except Exception as e:
        app.logger.warning(f"Could not load sample datasets: {e}")

# In-memory storage for latest audit export
LATEST_AUDIT_REPORT = {}

@app.route("/", methods=["GET", "POST"])
def index():
    global LATEST_AUDIT_REPORT
    results = None
    audit = None
    fairness_metrics = None
    fair_results = None
    error = None
    job_description = ""
    ranking_mode = "standard"
    selected_sample = ""
    candidate_input_list = []

    if request.method == "POST":
        action = request.form.get("action", "rank")
        selected_sample = request.form.get("sample_preset", "")
        ranking_mode = request.form.get("ranking_mode", "standard")

        # Handle 1-click sample load
        if action == "load_sample" and selected_sample in SAMPLE_DATA:
            sample = SAMPLE_DATA[selected_sample]
            job_description = sample["job_description"]
            candidate_input_list = sample["candidates"]
        else:
            job_description = request.form.get("job_description", "").strip()
            names = request.form.getlist("candidate_name[]")
            qualifications = request.form.getlist("qualifications[]")
            groups = request.form.getlist("audit_group[]")

            for i in range(max(len(qualifications), len(names))):
                q = qualifications[i].strip() if i < len(qualifications) else ""
                n = names[i].strip() if i < len(names) else ""
                g = groups[i].strip() if i < len(groups) else "Not specified"
                if q or n:
                    candidate_input_list.append({
                        "candidate_id": f"C{len(candidate_input_list) + 1}",
                        "name": n or f"Candidate {len(candidate_input_list) + 1}",
                        "qualifications": q,
                        "audit_group": g
                    })

            # Handle file uploads (PDF, DOCX, TXT)
            uploaded_files = request.files.getlist("resume_files")
            file_groups = request.form.getlist("file_audit_group[]")
            
            for file_idx, file_obj in enumerate(uploaded_files):
                if file_obj and file_obj.filename:
                    try:
                        extracted_text = parse_document(file_obj.stream, filename=file_obj.filename)
                        if extracted_text.strip():
                            inferred_name = extract_candidate_name(extracted_text, file_obj.filename)
                            file_group = file_groups[file_idx] if file_idx < len(file_groups) else "Not specified"
                            candidate_input_list.append({
                                "candidate_id": f"C{len(candidate_input_list) + 1}",
                                "name": inferred_name,
                                "qualifications": extracted_text,
                                "audit_group": file_group,
                                "source_file": file_obj.filename
                            })
                    except Exception as e:
                        error = f"Error processing file '{file_obj.filename}': {str(e)}"

        # Validate minimum inputs
        valid_candidates = [c for c in candidate_input_list if c["qualifications"].strip()]

        if not job_description and not error:
            error = "Please enter or select a job description."
        elif len(valid_candidates) < 2 and not error:
            error = "Please provide at least two candidates with qualifications (via text or file upload)."
        elif not error:
            try:
                # 1. Advanced Ranking
                ranked = score_candidates_advanced(job_description, valid_candidates)

                # 2. Enrich with structured explanations
                for c in ranked:
                    c["explanation_details"] = generate_candidate_explanation(c)

                # 3. Comprehensive Fairness Metrics
                fairness_metrics = calculate_fairness_metrics(ranked, group_key="audit_group")

                # 4. Rigorous Counterfactual Audit
                audit = run_counterfactual_audit(job_description, valid_candidates)

                # 5. Fair Reranking (if enabled or for side-by-side comparison)
                fair_result_obj = fair_rerank(ranked, group_key="audit_group")
                fair_results = fair_result_obj["reranked"]
                for fc in fair_results:
                    fc["explanation_details"] = generate_candidate_explanation(fc)

                results = fair_results if ranking_mode == "fair" else ranked

                # Store audit report for export
                LATEST_AUDIT_REPORT = {
                    "job_description": job_description,
                    "ranking_mode": ranking_mode,
                    "total_candidates": len(valid_candidates),
                    "ranking_results": results,
                    "fairness_metrics": fairness_metrics,
                    "counterfactual_audit": audit,
                    "fair_reranking_comparison": {
                        "swaps_count": fair_result_obj["swaps_count"],
                        "utility_retention_pct": fair_result_obj["utility_retention_pct"],
                        "rank_deltas": fair_result_obj["rank_deltas"]
                    }
                }
            except Exception as e:
                error = f"Evaluation error: {str(e)}"

    return render_template(
        "index.html",
        results=results,
        fair_results=fair_results,
        audit=audit,
        fairness_metrics=fairness_metrics,
        error=error,
        job_description=job_description,
        ranking_mode=ranking_mode,
        selected_sample=selected_sample,
        candidates=candidate_input_list,
        sample_keys=list(SAMPLE_DATA.keys())
    )

@app.route("/api/sample-datasets", methods=["GET"])
def get_sample_datasets():
    """Returns available realistic benchmark datasets."""
    return jsonify({"status": "success", "datasets": SAMPLE_DATA})

@app.route("/api/rank", methods=["POST"])
def api_rank_candidates():
    """Programmatic API endpoint for candidate ranking, fairness audit, and explanations."""
    data = request.get_json(silent=True) or {}
    job_description = data.get("job_description", "").strip()
    candidates = data.get("candidates", [])
    group_key = data.get("group_key", "audit_group")
    ranking_mode = data.get("ranking_mode", "standard")
    top_k = data.get("top_k")

    if not job_description:
        return jsonify({"status": "error", "message": "job_description is required."}), 400
    if len(candidates) < 2:
        return jsonify({"status": "error", "message": "At least 2 candidates are required."}), 400

    try:
        ranked = score_candidates_advanced(job_description, candidates)
        for c in ranked:
            c["explanation_details"] = generate_candidate_explanation(c)

        metrics = calculate_fairness_metrics(ranked, group_key=group_key, top_k=top_k)
        audit = run_counterfactual_audit(job_description, candidates)
        fair_obj = fair_rerank(ranked, group_key=group_key)

        return jsonify({
            "status": "success",
            "ranking_mode": ranking_mode,
            "standard_ranking": ranked,
            "fair_reranked": fair_obj["reranked"],
            "rerank_stats": {
                "swaps_count": fair_obj["swaps_count"],
                "utility_retention_pct": fair_obj["utility_retention_pct"]
            },
            "fairness_metrics": metrics,
            "counterfactual_audit": audit
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/parse-resume", methods=["POST"])
def api_parse_resume():
    """Parses an uploaded resume file and returns text and extracted metadata."""
    if "file" not in request.files:
        return jsonify({"status": "error", "message": "No file uploaded."}), 400
    file = request.files["file"]
    if not file.filename:
        return jsonify({"status": "error", "message": "Empty filename."}), 400

    try:
        text = parse_document(file.stream, filename=file.filename)
        name = extract_candidate_name(text, file.filename)
        contact = extract_contact_info(text)
        sections = extract_resume_sections(text)
        skills = list(extract_skills_from_text(text))

        return jsonify({
            "status": "success",
            "filename": file.filename,
            "candidate_name": name,
            "contact_info": contact,
            "skills": skills,
            "sections": sections,
            "full_text": text
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/pairwise-compare", methods=["POST"])
def api_pairwise_compare():
    """Compares two candidates and returns detailed comparative justification."""
    data = request.get_json(silent=True) or {}
    cand_a = data.get("candidate_a")
    cand_b = data.get("candidate_b")

    if not cand_a or not cand_b:
        return jsonify({"status": "error", "message": "Both candidate_a and candidate_b objects required."}), 400

    comparison = generate_pairwise_comparison(cand_a, cand_b)
    return jsonify({"status": "success", "comparison": comparison})

@app.route("/api/export-audit", methods=["GET"])
def export_audit_report():
    """Exports the latest audit and ranking report as formatted JSON."""
    global LATEST_AUDIT_REPORT
    if not LATEST_AUDIT_REPORT:
        return jsonify({"status": "error", "message": "No evaluation has been run yet."}), 404

    report_json = json.dumps(LATEST_AUDIT_REPORT, indent=2)
    return Response(
        report_json,
        mimetype="application/json",
        headers={"Content-Disposition": "attachment;filename=fairhire_audit_report.json"}
    )

if __name__ == "__main__":
    app.run(debug=True, port=5000)
