"""
FairHire AI 3.0 — Fairness-Aware Resume Ranking & Compliance Audit System
Flask Application & Secure API Service

Security & Architectural Improvements:
- Isolated per-session storage for evaluation reports (prevents cross-user data leakage)
- Configurable environment variable for SECRET_KEY and FLASK_DEBUG
- Strict upload size limit (5MB) and extension whitelist
- HTTP security headers (CSP, nosniff, frame-ancestors)
- Sanitized API outputs without raw PII/resume body exposure
- Export endpoints for JSON audit reports and CSV ranking reports
- Health check API endpoint (/api/health)
"""

import collections
import csv
import io
import json
import os
import secrets
from typing import Any, Dict, List, Optional
from flask import (
    Flask,
    jsonify,
    render_template,
    request,
    Response,
    session,
)

from core.document_parser import (
    parse_document,
    extract_candidate_name,
    extract_contact_info,
    extract_resume_sections,
    DocumentParseError,
)
from core.skill_taxonomy import extract_skills_from_text
from core.ranking_engine import (
    score_candidates_advanced,
    fair_rerank,
    calculate_ndcg,
    calculate_precision_at_k,
    calculate_recall_at_k,
    compare_fair_rerank_constraints,
    SEMANTIC_AVAILABLE,
)
from core.fairness_metrics import calculate_fairness_metrics
from core.fairness_audit import run_counterfactual_audit
from core.explanation_engine import (
    generate_candidate_explanation,
    generate_pairwise_comparison,
)

app = Flask(__name__)

# Security: Configurable secret key & debug mode from environment
app.secret_key = os.environ.get("SECRET_KEY", secrets.token_hex(32))
debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() == "true"

# Security: Strict 5MB maximum upload limit
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

# Allowed resume file extensions
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}

# Isolated session-based report store with bounded LRU/FIFO eviction
AUDIT_STORE: collections.OrderedDict[str, Dict[str, Any]] = collections.OrderedDict()
MAX_STORE_SESSIONS = 200

# Load sample benchmark datasets
SAMPLE_DATA_PATH = os.path.join(
    os.path.dirname(__file__), "sample_data", "sample_job_and_resumes.json"
)
SAMPLE_DATA = {}
if os.path.exists(SAMPLE_DATA_PATH):
    try:
        with open(SAMPLE_DATA_PATH, "r", encoding="utf-8") as f:
            SAMPLE_DATA = json.load(f).get("datasets", {})
    except Exception as e:
        app.logger.warning(f"Could not load sample datasets: {e}")


def get_or_create_session_id() -> str:
    """Returns or generates a secure session identifier for per-user isolation."""
    if "session_id" not in session:
        session["session_id"] = secrets.token_hex(16)
    return session["session_id"]


def set_session_report(sid: str, report: Dict[str, Any]) -> None:
    """Stores an evaluation report isolated to the specific user session."""
    if len(AUDIT_STORE) >= MAX_STORE_SESSIONS:
        AUDIT_STORE.popitem(last=False)
    AUDIT_STORE[sid] = report


def get_session_report(sid: str) -> Optional[Dict[str, Any]]:
    """Retrieves the latest evaluation report for the current session."""
    return AUDIT_STORE.get(sid)


def validate_file_extension(filename: str) -> bool:
    """Checks whether a filename matches the allowed extension whitelist."""
    if not filename or "." not in filename:
        return False
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXTENSIONS


def parse_weights_from_request(req_data: Any) -> Optional[Dict[str, float]]:
    """Helper to parse scoring weights from form or JSON."""
    try:
        w_skills = float(req_data.get("weight_skills", 40))
        w_sem = float(req_data.get("weight_semantic", 30))
        w_exp = float(req_data.get("weight_experience", 15))
        w_edu = float(req_data.get("weight_education", 15))
        total = w_skills + w_sem + w_exp + w_edu
        if total > 0:
            return {
                "skills": round(w_skills / total, 4),
                "semantic": round(w_sem / total, 4),
                "experience": round(w_exp / total, 4),
                "education": round(w_edu / total, 4),
            }
    except (TypeError, ValueError):
        pass
    return None


@app.after_request
def apply_security_headers(response: Response) -> Response:
    """Applies defense-in-depth security headers to all responses."""
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src 'self' https://fonts.gstatic.com; "
        "img-src 'self' data:;"
    )
    return response


@app.errorhandler(413)
def handle_file_too_large(e):
    """Graceful JSON or HTML handler for payload size limit violations."""
    if request.path.startswith("/api/"):
        return jsonify({
            "status": "error",
            "message": "File exceeds maximum upload size limit (5MB)."
        }), 413
    return render_template(
        "index.html",
        error="Uploaded file exceeds the maximum 5MB size limit. Please upload a smaller document.",
        sample_keys=list(SAMPLE_DATA.keys()),
        ui_weights={"skills": 40, "semantic": 30, "experience": 15, "education": 15}
    ), 413


@app.route("/api/health", methods=["GET"])
def health_check():
    """Service health check endpoint."""
    return jsonify({
        "status": "ok",
        "service": "FairHire AI",
        "version": "3.0",
        "semantic_model_available": SEMANTIC_AVAILABLE,
    })


@app.route("/", methods=["GET", "POST"])
def index():
    sid = get_or_create_session_id()
    results = None
    audit = None
    fairness_metrics = None
    fair_results = None
    error = None
    job_description = ""
    ranking_mode = "standard"
    selected_sample = ""
    candidate_input_list = []
    top_k = None
    weights = None
    fair_result_obj = None
    pareto_comparison = None
    missing_info_strategy = "rebalance"
    constraint_level = "moderate"

    # Form defaults for UI sliders
    ui_weights = {"skills": 40, "semantic": 30, "experience": 15, "education": 15}

    if request.method == "POST":
        action = request.form.get("action", "rank")
        selected_sample = request.form.get("sample_preset", "")
        ranking_mode = request.form.get("ranking_mode", "standard")
        missing_info_strategy = request.form.get("missing_info_strategy", "rebalance")
        constraint_level = request.form.get("constraint_level", "moderate")

        # Parse custom weights
        custom_weights = parse_weights_from_request(request.form)
        if custom_weights:
            weights = custom_weights
            ui_weights = {
                "skills": int(round(custom_weights["skills"] * 100)),
                "semantic": int(round(custom_weights["semantic"] * 100)),
                "experience": int(round(custom_weights["experience"] * 100)),
                "education": int(round(custom_weights["education"] * 100)),
            }

        # Parse top_k
        top_k_str = request.form.get("top_k", "").strip()
        if top_k_str.isdigit() and int(top_k_str) > 0:
            top_k = int(top_k_str)

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
                        "audit_group": g,
                    })

            # Handle file uploads (PDF, DOCX, TXT)
            uploaded_files = request.files.getlist("resume_files")
            file_groups = request.form.getlist("file_audit_group[]")

            for file_idx, file_obj in enumerate(uploaded_files):
                if file_obj and file_obj.filename:
                    if not validate_file_extension(file_obj.filename):
                        error = f"Unsupported file format for '{file_obj.filename}'. Only PDF, DOCX, and TXT are supported."
                        break
                    try:
                        extracted_text = parse_document(
                            file_obj.stream, filename=file_obj.filename
                        )
                        if extracted_text.strip():
                            inferred_name = extract_candidate_name(
                                extracted_text, file_obj.filename
                            )
                            file_group = (
                                file_groups[file_idx]
                                if file_idx < len(file_groups)
                                else "Not specified"
                            )
                            candidate_input_list.append({
                                "candidate_id": f"C{len(candidate_input_list) + 1}",
                                "name": inferred_name,
                                "qualifications": extracted_text,
                                "audit_group": file_group,
                                "source_file": file_obj.filename,
                            })
                    except (DocumentParseError, ValueError) as parse_err:
                        error = f"Error processing file '{file_obj.filename}': {str(parse_err)}"
                        break
                    except Exception as e:
                        error = f"Unexpected error reading '{file_obj.filename}'."
                        break

        # Validate minimum inputs
        valid_candidates = [
            c for c in candidate_input_list if c["qualifications"].strip()
        ]

        if not job_description and not error:
            error = "Please enter or select a job description."
        elif len(valid_candidates) < 2 and not error:
            error = "Please provide at least two candidates with qualifications (via text or file upload)."
        elif not error:
            try:
                # 1. Advanced Multi-Aspect Hybrid Ranking with missing-information handling
                ranked = score_candidates_advanced(
                    job_description,
                    valid_candidates,
                    weights=weights,
                    missing_info_strategy=missing_info_strategy,
                )

                # 2. Enrich with structured explanations & next-tier guidance
                for c in ranked:
                    c["explanation_details"] = generate_candidate_explanation(c)

                # 3. Comprehensive Fairness Metrics Suite (with top_k)
                fairness_metrics = calculate_fairness_metrics(
                    ranked, group_key="audit_group", top_k=top_k
                )

                # 4. Rigorous Multi-Variation Counterfactual Audit
                audit = run_counterfactual_audit(
                    job_description, valid_candidates, weights=weights
                )

                # 5. Fair Reranking (FA*IR-inspired) with configurable constraint level
                fair_result_obj = fair_rerank(
                    ranked, group_key="audit_group", constraint_level=constraint_level
                )
                fair_results = fair_result_obj["reranked"]
                for fc in fair_results:
                    fc["explanation_details"] = generate_candidate_explanation(fc)

                # 6. Pareto Frontier Multi-Constraint Comparison
                pareto_comparison = compare_fair_rerank_constraints(
                    ranked, group_key="audit_group", top_k=top_k
                )

                # Compute NDCG@K
                ndcg_metrics = calculate_ndcg(ranked, fair_results, k=top_k)

                results = fair_results if ranking_mode == "fair" else ranked

                # Store audit report isolated to this user session
                report_data = {
                    "job_description": job_description,
                    "ranking_mode": ranking_mode,
                    "missing_info_strategy": missing_info_strategy,
                    "constraint_level": constraint_level,
                    "total_candidates": len(valid_candidates),
                    "weights_used": weights
                    or {
                        "skills": 0.40,
                        "semantic": 0.30,
                        "experience": 0.15,
                        "education": 0.15,
                    },
                    "top_k_used": fairness_metrics.get("top_k_used"),
                    "ranking_results": results,
                    "standard_ranking": ranked,
                    "fair_reranked_results": fair_results,
                    "fairness_metrics": fairness_metrics,
                    "counterfactual_audit": audit,
                    "pareto_comparison": pareto_comparison,
                    "ndcg_metrics": ndcg_metrics,
                    "fair_reranking_comparison": {
                        "swaps_count": fair_result_obj["swaps_count"],
                        "utility_retention_pct": fair_result_obj[
                            "utility_retention_pct"
                        ],
                        "avg_rank_shift": fair_result_obj.get(
                            "avg_rank_shift", 0.0
                        ),
                        "max_rank_shift": fair_result_obj.get(
                            "max_rank_shift", 0
                        ),
                        "rank_deltas": fair_result_obj["rank_deltas"],
                        "constraint_level_used": fair_result_obj.get(
                            "constraint_level_used", "moderate"
                        ),
                        "deficit_threshold_used": fair_result_obj.get(
                            "deficit_threshold_used", 0.75
                        ),
                        "assumptions": fair_result_obj.get("assumptions", []),
                    },
                }
                set_session_report(sid, report_data)

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
        sample_keys=list(SAMPLE_DATA.keys()),
        ui_weights=ui_weights,
        top_k=top_k,
        fair_rerank_stats=fair_result_obj,
        pareto_comparison=pareto_comparison,
        missing_info_strategy=missing_info_strategy,
        constraint_level=constraint_level,
    )


@app.route("/api/sample-datasets", methods=["GET"])
def get_sample_datasets():
    """Returns available benchmark datasets."""
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
    weights = data.get("weights")
    missing_info_strategy = data.get("missing_info_strategy", "rebalance")
    constraint_level = data.get("constraint_level", "moderate")

    if missing_info_strategy not in ["rebalance", "neutral", "zero"]:
        missing_info_strategy = "rebalance"
    if constraint_level not in ["strict", "moderate", "relaxed"]:
        constraint_level = "moderate"

    if not job_description:
        return jsonify({"status": "error", "message": "job_description is required."}), 400
    if len(candidates) < 2:
        return jsonify({"status": "error", "message": "At least 2 candidates are required."}), 400

    try:
        ranked = score_candidates_advanced(
            job_description,
            candidates,
            weights=weights,
            missing_info_strategy=missing_info_strategy,
        )
        for c in ranked:
            c["explanation_details"] = generate_candidate_explanation(c)

        metrics = calculate_fairness_metrics(ranked, group_key=group_key, top_k=top_k)
        audit = run_counterfactual_audit(job_description, candidates, weights=weights)
        fair_obj = fair_rerank(
            ranked, group_key=group_key, constraint_level=constraint_level
        )
        for fc in fair_obj["reranked"]:
            fc["explanation_details"] = generate_candidate_explanation(fc)

        pareto_comparison = compare_fair_rerank_constraints(
            ranked, group_key=group_key, top_k=top_k
        )
        ndcg_metrics = calculate_ndcg(ranked, fair_obj["reranked"], k=top_k)

        # Store in session for export
        sid = get_or_create_session_id()
        set_session_report(sid, {
            "job_description": job_description,
            "ranking_mode": ranking_mode,
            "missing_info_strategy": missing_info_strategy,
            "constraint_level": constraint_level,
            "total_candidates": len(candidates),
            "weights_used": ranked[0].get("weights_used", {}),
            "top_k_used": metrics.get("top_k_used"),
            "ranking_results": fair_obj["reranked"] if ranking_mode == "fair" else ranked,
            "standard_ranking": ranked,
            "fair_reranked_results": fair_obj["reranked"],
            "fairness_metrics": metrics,
            "counterfactual_audit": audit,
            "pareto_comparison": pareto_comparison,
            "ndcg_metrics": ndcg_metrics,
            "fair_reranking_comparison": {
                "swaps_count": fair_obj["swaps_count"],
                "utility_retention_pct": fair_obj["utility_retention_pct"],
                "avg_rank_shift": fair_obj.get("avg_rank_shift", 0.0),
                "max_rank_shift": fair_obj.get("max_rank_shift", 0),
                "rank_deltas": fair_obj["rank_deltas"],
                "constraint_level_used": fair_obj.get("constraint_level_used", "moderate"),
                "deficit_threshold_used": fair_obj.get("deficit_threshold_used", 0.75),
            },
        })

        return jsonify({
            "status": "success",
            "ranking_mode": ranking_mode,
            "missing_info_strategy": missing_info_strategy,
            "constraint_level": constraint_level,
            "weights_used": ranked[0].get("weights_used", {}),
            "standard_ranking": ranked,
            "fair_reranked": fair_obj["reranked"],
            "rerank_stats": {
                "swaps_count": fair_obj["swaps_count"],
                "utility_retention_pct": fair_obj["utility_retention_pct"],
                "avg_rank_shift": fair_obj.get("avg_rank_shift", 0.0),
                "max_rank_shift": fair_obj.get("max_rank_shift", 0),
                "constraint_level_used": fair_obj.get("constraint_level_used", "moderate"),
                "deficit_threshold_used": fair_obj.get("deficit_threshold_used", 0.75),
            },
            "pareto_comparison": pareto_comparison,
            "ndcg_metrics": ndcg_metrics,
            "fairness_metrics": metrics,
            "counterfactual_audit": audit,
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/parse-resume", methods=["POST"])
def api_parse_resume():
    """
    Parses an uploaded resume file with validation.
    Protects candidate privacy: does NOT return raw full resume text in response.
    """
    if "file" not in request.files:
        return jsonify({"status": "error", "message": "No file uploaded."}), 400
    file = request.files["file"]
    if not file.filename:
        return jsonify({"status": "error", "message": "Empty filename."}), 400

    if not validate_file_extension(file.filename):
        return jsonify({
            "status": "error",
            "message": f"Unsupported format. Allowed extensions: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        }), 400

    try:
        text = parse_document(file.stream, filename=file.filename)
        name = extract_candidate_name(text, file.filename)
        contact = extract_contact_info(text)
        sections = extract_resume_sections(text)
        skills = sorted(list(extract_skills_from_text(text)))

        # Privacy protection: redact/truncate raw text
        preview = text[:200] + ("..." if len(text) > 200 else "")

        return jsonify({
            "status": "success",
            "filename": file.filename,
            "candidate_name": name,
            "contact_info": {
                "has_email": bool(contact.get("email")),
                "has_phone": bool(contact.get("phone")),
                "github": contact.get("github"),
                "linkedin": contact.get("linkedin"),
                "portfolio": contact.get("portfolio"),
            },
            "skills": skills,
            "sections_found": [k for k, v in sections.items() if v and k != "other"],
            "text_length": len(text),
            "text_preview": preview,
        })
    except (DocumentParseError, ValueError) as pe:
        return jsonify({"status": "error", "message": str(pe)}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": "Failed to parse document safely."}), 500


@app.route("/api/pairwise-compare", methods=["POST"])
def api_pairwise_compare():
    """Compares two candidates and returns detailed comparative justification."""
    sid = get_or_create_session_id()
    report = get_session_report(sid) or {}

    data = request.get_json(silent=True) or {}
    cand_a = data.get("candidate_a")
    cand_b = data.get("candidate_b")

    # If IDs provided instead of full objects, look them up in session report
    active_results = report.get("ranking_results", [])
    if not cand_a and data.get("candidate_a_id") and active_results:
        cid_a = data.get("candidate_a_id")
        cand_a = next((c for c in active_results if c.get("candidate_id") == cid_a), None)

    if not cand_b and data.get("candidate_b_id") and active_results:
        cid_b = data.get("candidate_b_id")
        cand_b = next((c for c in active_results if c.get("candidate_id") == cid_b), None)

    if not cand_a or not cand_b:
        return jsonify({
            "status": "error",
            "message": "Both candidate_a and candidate_b objects or valid IDs from active evaluation required."
        }), 400

    comparison = generate_pairwise_comparison(cand_a, cand_b)
    return jsonify({"status": "success", "comparison": comparison})


@app.route("/api/export-audit", methods=["GET"])
def export_audit_report():
    """Exports the current session's audit and ranking report as formatted JSON."""
    sid = get_or_create_session_id()
    report = get_session_report(sid)
    if not report:
        return jsonify({"status": "error", "message": "No evaluation has been run in this session yet."}), 404

    report_json = json.dumps(report, indent=2)
    return Response(
        report_json,
        mimetype="application/json",
        headers={"Content-Disposition": "attachment;filename=fairhire_audit_report.json"}
    )


@app.route("/api/export-ranking-csv", methods=["GET"])
def export_ranking_csv():
    """Exports the current session's candidate ranking table as downloadable CSV."""
    sid = get_or_create_session_id()
    report = get_session_report(sid)
    if not report or not report.get("ranking_results"):
        return jsonify({"status": "error", "message": "No evaluation results available to export."}), 404

    ranked_candidates = report.get("ranking_results", [])
    output = io.StringIO()
    writer = csv.writer(output)

    # Header
    writer.writerow([
        "Rank", "Candidate ID", "Candidate Name", "Composite Score",
        "Tier Badge", "Audit Group", "Skill Match Score",
        "Semantic Relevance", "Experience Fit", "Education Fit",
        "Matched Skills Count", "Matched Skills", "Missing Skills"
    ])

    for c in ranked_candidates:
        aspects = c.get("aspect_scores", {})
        skills_info = c.get("skills_extracted", {})
        matched = "; ".join(skills_info.get("matched", []))
        missing = "; ".join(skills_info.get("missing", []))

        writer.writerow([
            c.get("fair_rank", c.get("rank", "")),
            c.get("candidate_id", ""),
            c.get("name", ""),
            c.get("score", 0.0),
            c.get("tier_badge", ""),
            c.get("audit_group", "Unknown"),
            aspects.get("skill_match", 0.0),
            aspects.get("semantic_relevance", 0.0),
            aspects.get("experience_fit", 0.0),
            aspects.get("education_fit", 0.0),
            skills_info.get("total_count", 0),
            matched,
            missing,
        ])

    csv_data = output.getvalue()
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=fairhire_candidate_rankings.csv"}
    )


if __name__ == "__main__":
    app.run(debug=debug_mode, host="127.0.0.1", port=5000)
