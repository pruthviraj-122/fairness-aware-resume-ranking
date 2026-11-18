"""
FairHire AI 3.0 — Skill Taxonomy & Canonical Dictionary
Provides structured categorisation of technical, analytical, and domain skills,
including synonym normalization and word-boundary–safe regex matching.

Improvements in v3.0:
- 20+ new skills (GraphQL, gRPC, Prometheus, Grafana, etc.)
- extract_skills_detailed() with evidence snippets & mandatory/optional distinction
- extract_skills_from_text() backward-compatible (returns Set[str])
- Stricter word-boundary matching to prevent false positives
- Education extraction handles missing info as Unknown
- Experience extraction guards against graduation-year confusion
"""

import re
from typing import Any, Dict, List, Optional, Set, Tuple

# ── Skill Synonyms ──────────────────────────────────────────────────────
SKILL_SYNONYMS: Dict[str, List[str]] = {
    # Languages
    "python": ["python", "python3", "py"],
    "javascript": ["javascript", "js", "ecmascript"],
    "typescript": ["typescript", "ts"],
    "java": ["java", "java8", "java11", "java17"],
    "c++": ["c++", "cpp", "cplusplus"],
    "c#": ["c#", "csharp", "c sharp", ".net"],
    "go": ["go", "golang"],
    "rust": ["rust", "rustlang"],
    "sql": ["sql", "structured query language", "tsql", "plsql", "pl/sql"],
    "r": ["r language", "r programming"],
    "ruby": ["ruby", "ruby on rails"],
    "php": ["php"],
    "scala": ["scala"],
    "swift": ["swift"],
    "kotlin": ["kotlin"],
    "bash": ["bash", "shell scripting", "shell", "sh"],

    # Frameworks & Libraries
    "react": ["react", "react.js", "reactjs"],
    "angular": ["angular", "angularjs", "angular 2+"],
    "vue": ["vue", "vue.js", "vuejs"],
    "next.js": ["next.js", "nextjs"],
    "node.js": ["node.js", "nodejs"],
    "express": ["express", "express.js", "expressjs"],
    "django": ["django", "django rest framework", "drf"],
    "flask": ["flask"],
    "fastapi": ["fastapi", "fast api"],
    "spring": ["spring", "spring boot", "springboot"],
    "pytorch": ["pytorch", "torch"],
    "tensorflow": ["tensorflow", "tf"],
    "scikit-learn": ["scikit-learn", "sklearn", "scikit learn"],
    "pandas": ["pandas"],
    "numpy": ["numpy"],
    "keras": ["keras"],

    # Cloud & DevOps
    "aws": ["aws", "amazon web services", "ec2", "s3", "lambda"],
    "azure": ["azure", "microsoft azure"],
    "gcp": ["gcp", "google cloud", "google cloud platform"],
    "docker": ["docker", "containerization", "containers"],
    "kubernetes": ["kubernetes", "k8s"],
    "terraform": ["terraform"],
    "ansible": ["ansible"],
    "ci/cd": ["ci/cd", "cicd", "continuous integration", "continuous deployment",
              "jenkins", "github actions", "gitlab ci"],
    "linux": ["linux", "unix", "ubuntu", "debian", "redhat", "centos"],
    "git": ["git", "github", "gitlab", "version control"],
    "helm": ["helm", "helm chart", "helm charts"],
    "istio": ["istio", "service mesh"],
    "vault": ["vault", "hashicorp vault"],

    # Databases & Big Data
    "postgresql": ["postgresql", "postgres"],
    "mysql": ["mysql"],
    "mongodb": ["mongodb", "mongo"],
    "redis": ["redis"],
    "elasticsearch": ["elasticsearch", "elastic search"],
    "kafka": ["kafka", "apache kafka"],
    "spark": ["spark", "apache spark", "pyspark"],
    "snowflake": ["snowflake"],
    "bigquery": ["bigquery", "big query"],
    "airflow": ["airflow", "apache airflow"],
    "celery": ["celery"],
    "rabbitmq": ["rabbitmq", "rabbit mq"],

    # Monitoring & Observability
    "prometheus": ["prometheus"],
    "grafana": ["grafana"],

    # API & Protocols
    "graphql": ["graphql", "graph ql"],
    "grpc": ["grpc", "g-rpc"],

    # Auth & Security
    "oauth": ["oauth", "oauth2", "oauth 2.0"],
    "jwt": ["jwt", "json web token", "json web tokens"],

    # AI & Core Concepts
    "machine learning": ["machine learning", "ml"],
    "deep learning": ["deep learning", "dl", "neural networks"],
    "nlp": ["nlp", "natural language processing", "text mining"],
    "computer vision": ["computer vision", "cv", "image processing", "opencv"],
    "llm": ["llm", "llms", "large language models", "prompt engineering", "langchain"],
    "rag": ["rag", "retrieval augmented generation"],
    "vector database": ["vector database", "vector db", "pinecone", "weaviate",
                        "chromadb", "milvus", "faiss"],
    "openai api": ["openai api", "openai", "chatgpt api"],
    "hugging face": ["hugging face", "huggingface", "transformers library"],

    # Data Tools
    "dbt": ["dbt", "data build tool"],
    "looker": ["looker"],
    "tableau": ["tableau"],

    # Engineering & Practices
    "microservices": ["microservices", "microservice architecture", "distributed systems"],
    "system design": ["system design", "architecture", "scalability"],
    "rest api": ["rest api", "restful", "rest apis", "restful api", "api design"],
    "agile": ["agile", "scrum", "kanban"],
    "unit testing": ["unit testing", "tdd", "test driven development",
                     "automated testing", "pytest", "junit"],
}

# ── Skill Category Groupings ────────────────────────────────────────────
SKILL_CATEGORIES: Dict[str, List[str]] = {
    "Languages": ["python", "javascript", "typescript", "java", "c++", "c#",
                  "go", "rust", "sql", "r", "ruby", "php", "scala", "swift",
                  "kotlin", "bash"],
    "Frameworks & Web": ["react", "angular", "vue", "next.js", "node.js",
                         "express", "django", "flask", "fastapi", "spring"],
    "AI & Data Science": ["machine learning", "deep learning", "nlp",
                          "computer vision", "llm", "rag", "vector database",
                          "openai api", "hugging face", "pytorch", "tensorflow",
                          "scikit-learn", "pandas", "numpy", "keras"],
    "Cloud & DevOps": ["aws", "azure", "gcp", "docker", "kubernetes",
                       "terraform", "ansible", "ci/cd", "linux", "git",
                       "helm", "istio", "vault"],
    "Databases & Big Data": ["postgresql", "mysql", "mongodb", "redis",
                             "elasticsearch", "kafka", "spark", "snowflake",
                             "bigquery", "airflow", "celery", "rabbitmq"],
    "Monitoring": ["prometheus", "grafana"],
    "APIs & Protocols": ["graphql", "grpc", "rest api"],
    "Auth & Security": ["oauth", "jwt"],
    "Data Tools": ["dbt", "looker", "tableau"],
    "Engineering & Practices": ["microservices", "system design", "agile",
                                "unit testing"],
}

# ── Reverse Lookup ───────────────────────────────────────────────────────
SYNONYM_TO_CANONICAL: Dict[str, str] = {}
for canonical, syn_list in SKILL_SYNONYMS.items():
    SYNONYM_TO_CANONICAL[canonical.lower()] = canonical
    for syn in syn_list:
        SYNONYM_TO_CANONICAL[syn.lower()] = canonical


def get_skill_category(skill: str) -> str:
    """Returns the high-level category of a canonical skill."""
    skill_lower = skill.lower()
    for category, skills in SKILL_CATEGORIES.items():
        if skill_lower in [s.lower() for s in skills]:
            return category
    return "Other Technical"


def extract_skills_from_text(text: str) -> Set[str]:
    """
    Scans text for all known skill mentions (canonical names and synonyms)
    and returns a set of canonical skill names.
    Uses word boundary regexes to prevent substring false positives.
    Backward-compatible: always returns Set[str].
    """
    if not text:
        return set()

    found_skills: Set[str] = set()
    text_lower = f" {text.lower()} "

    # Sort synonyms by length descending so longer phrases match first
    sorted_synonyms = sorted(SYNONYM_TO_CANONICAL.keys(),
                             key=lambda s: len(s), reverse=True)

    for syn in sorted_synonyms:
        escaped_syn = re.escape(syn)
        pattern = rf"(?<![a-zA-Z0-9_]){escaped_syn}(?![a-zA-Z0-9_+#])"
        if re.search(pattern, text_lower):
            canonical = SYNONYM_TO_CANONICAL[syn]
            found_skills.add(canonical)

    return found_skills


def extract_skills_detailed(
    text: str,
    mandatory_skills: Optional[Set[str]] = None
) -> Dict[str, Any]:
    """
    Enhanced skill extraction with evidence and mandatory/optional distinction.

    Returns:
        {
            'found': Set[str],           # all canonical skills found
            'mandatory_matched': Set[str],
            'mandatory_missing': Set[str],
            'evidence': Dict[str, str],  # skill -> text snippet
        }
    """
    if not text:
        return {
            "found": set(),
            "mandatory_matched": set(),
            "mandatory_missing": mandatory_skills or set(),
            "evidence": {},
        }

    found_skills: Set[str] = set()
    evidence: Dict[str, str] = {}
    text_lower = f" {text.lower()} "

    sorted_synonyms = sorted(SYNONYM_TO_CANONICAL.keys(),
                             key=lambda s: len(s), reverse=True)

    for syn in sorted_synonyms:
        escaped_syn = re.escape(syn)
        pattern = rf"(?<![a-zA-Z0-9_]){escaped_syn}(?![a-zA-Z0-9_+#])"
        match = re.search(pattern, text_lower)
        if match:
            canonical = SYNONYM_TO_CANONICAL[syn]
            if canonical not in found_skills:
                found_skills.add(canonical)
                # Extract context snippet (up to 80 chars around the match)
                start = max(0, match.start() - 40)
                end = min(len(text_lower), match.end() + 40)
                snippet = text_lower[start:end].strip()
                evidence[canonical] = f"...{snippet}..."

    mandatory = mandatory_skills or set()
    mandatory_lower = {s.lower() for s in mandatory}
    found_lower = {s.lower() for s in found_skills}

    mandatory_matched = {s for s in mandatory if s.lower() in found_lower}
    mandatory_missing = {s for s in mandatory if s.lower() not in found_lower}

    return {
        "found": found_skills,
        "mandatory_matched": mandatory_matched,
        "mandatory_missing": mandatory_missing,
        "evidence": evidence,
    }


def extract_years_of_experience(text: str) -> float:
    """
    Extracts explicit years of experience from resume/job text.
    Handles: '5+ years of experience', '3-5 years', 'over 7 years'.
    Guards against graduation year confusion (rejects values > 40).
    """
    if not text:
        return 0.0

    patterns = [
        r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years|yrs)(?:\s+of)?(?:\s+experience)?",
        r"(?:experience(?:\s+of)?\s*)(\d+(?:\.\d+)?)\s*\+?\s*(?:years|yrs)",
        r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*(?:years|yrs)",
    ]

    max_years = 0.0
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            groups = match.groups()
            if len(groups) == 2 and groups[1] is not None:
                val = float(groups[1])
            else:
                val = float(groups[0])
            if val < 40.0:  # Sanity cap
                max_years = max(max_years, val)

    return max_years


def extract_education_level(text: str) -> Dict[str, Any]:
    """
    Extracts education qualifications: PhD, Master's, Bachelor's, Associate, Bootcamp.
    Returns level name and a normalized numeric tier (0 to 5).
    Tier 0 = Unknown/unspecified (no auto-award of points).
    """
    text_lower = text.lower() if text else ""

    if re.search(r"\b(ph\.?d|doctorate|doctor of philosophy)\b", text_lower):
        return {"level": "Doctorate (PhD)", "tier": 5}
    if re.search(r"\b(master(?:'?s)?|m\.?s\.?|m\.?sc|m\.?tech|mba)\b", text_lower):
        return {"level": "Master's Degree", "tier": 4}
    if re.search(r"\b(bachelor(?:'?s)?|b\.?s\.?|b\.?sc|b\.?tech|b\.?e\.?|undergraduate)\b",
                 text_lower):
        return {"level": "Bachelor's Degree", "tier": 3}
    if re.search(r"\b(associate(?:'?s)?|a\.?s\.?|diploma)\b", text_lower):
        return {"level": "Associate / Diploma", "tier": 2}
    if re.search(r"\b(bootcamp|certification|self-taught)\b", text_lower):
        return {"level": "Bootcamp / Certificate", "tier": 1}

    return {"level": "Unknown", "tier": 0}


def extract_education_details(text: str) -> Dict[str, Any]:
    """
    Extracts structured education info: degree, institution, field.
    Returns Unknown for missing components.
    """
    edu = extract_education_level(text)
    text_lower = text.lower() if text else ""

    # Try to extract institution
    institution = "Unknown"
    inst_match = re.search(
        r"\b([A-Za-z\s]{2,30}\s+(?:university|college|institute|polytechnic|school))\b|"
        r"\b((?:university|college|institute|polytechnic|school)\s+(?:of\s+)?[A-Za-z\s]{2,30})\b|"
        r"\b(mit|caltech|stanford|harvard|oxford|cambridge)\b",
        text_lower
    )
    if inst_match:
        matched_str = next((m for m in inst_match.groups() if m), None)
        if matched_str:
            matched_str = re.sub(r"^(?:from|at|in)\s+", "", matched_str.strip())
            institution = matched_str.strip().title()

    # Try to extract field
    field = "Unknown"
    field_match = re.search(
        r"(?:in|of)\s+(computer\s+science|software\s+engineering|data\s+science|"
        r"mathematics|statistics|information\s+technology|electrical\s+engineering|"
        r"computer\s+engineering|digital\s+media|information\s+systems)",
        text_lower
    )
    if field_match:
        field = field_match.group(1).strip().title()

    return {
        "level": edu["level"],
        "tier": edu["tier"],
        "institution": institution,
        "field": field,
    }


def extract_employment_periods(text: str) -> List[Dict[str, Any]]:
    """
    Extracts employment periods from resume text.
    Attempts to identify year ranges (e.g., '2019 - 2022', '2020 - Present').
    Returns list of dicts with start_year, end_year, duration_years.
    """
    if not text:
        return []

    periods = []
    # Pattern: (year) - (year|Present|Current)
    pattern = r"(20\d{2}|19\d{2})\s*[-–—to]+\s*(20\d{2}|19\d{2}|present|current|now)"
    for match in re.finditer(pattern, text, re.IGNORECASE):
        start_year = int(match.group(1))
        end_str = match.group(2).lower()
        end_year = 2024 if end_str in ("present", "current", "now") else int(end_str)

        if 1950 < start_year <= end_year <= 2030:
            duration = end_year - start_year
            periods.append({
                "start_year": start_year,
                "end_year": end_year if end_str not in ("present", "current", "now") else "Present",
                "duration_years": duration,
            })

    return periods
