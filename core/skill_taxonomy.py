"""
FairHire AI — Skill Taxonomy & Canonical Dictionary
Provides structured categorisation of technical, analytical, and domain skills,
including synonym normalization and fuzzy/regex keyword matching.
"""

import re
from typing import Dict, List, Set, Tuple

# Mapping of canonical skill names to recognized synonyms and alternate spellings
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
    "next.js": ["next.js", "nextjs", "next"],
    "node.js": ["node.js", "nodejs", "node"],
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
    "ci/cd": ["ci/cd", "cicd", "continuous integration", "continuous deployment", "jenkins", "github actions", "gitlab ci"],
    "linux": ["linux", "unix", "ubuntu", "debian", "redhat", "centos"],
    "git": ["git", "github", "gitlab", "version control"],

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

    # AI & Core Concepts
    "machine learning": ["machine learning", "ml"],
    "deep learning": ["deep learning", "dl", "neural networks"],
    "nlp": ["nlp", "natural language processing", "text mining"],
    "computer vision": ["computer vision", "cv", "image processing", "opencv"],
    "llm": ["llm", "llms", "large language models", "prompt engineering", "langchain"],
    "microservices": ["microservices", "microservice architecture", "distributed systems"],
    "system design": ["system design", "architecture", "scalability"],
    "rest api": ["rest api", "restful", "rest apis", "restful api", "api design", "graphql"],
    "agile": ["agile", "scrum", "kanban"],
    "unit testing": ["unit testing", "tdd", "test driven development", "automated testing", "pytest", "junit"],
}

# Skill category groupings
SKILL_CATEGORIES: Dict[str, List[str]] = {
    "Languages": ["python", "javascript", "typescript", "java", "c++", "c#", "go", "rust", "sql", "r", "ruby", "php", "scala", "swift", "kotlin", "bash"],
    "Frameworks & Web": ["react", "angular", "vue", "next.js", "node.js", "express", "django", "flask", "fastapi", "spring"],
    "AI & Data Science": ["machine learning", "deep learning", "nlp", "computer vision", "llm", "pytorch", "tensorflow", "scikit-learn", "pandas", "numpy", "keras"],
    "Cloud & DevOps": ["aws", "azure", "gcp", "docker", "kubernetes", "terraform", "ansible", "ci/cd", "linux", "git"],
    "Databases & Big Data": ["postgresql", "mysql", "mongodb", "redis", "elasticsearch", "kafka", "spark", "snowflake", "bigquery", "airflow"],
    "Engineering & Practices": ["microservices", "system design", "rest api", "agile", "unit testing"]
}

# Reverse lookup: synonym -> canonical name
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
    Uses word boundary regexes to prevent substring false positives (e.g., 'go' inside 'algorithm').
    """
    if not text:
        return set()

    found_skills: Set[str] = set()
    text_lower = f" {text.lower()} "

    # Sort synonyms by length descending so longer phrases match first
    sorted_synonyms = sorted(SYNONYM_TO_CANONICAL.keys(), key=lambda s: len(s), reverse=True)

    for syn in sorted_synonyms:
        escaped_syn = re.escape(syn)
        pattern = rf"(?<![a-zA-Z0-9_]){escaped_syn}(?![a-zA-Z0-9_+#])"
        if re.search(pattern, text_lower):
            canonical = SYNONYM_TO_CANONICAL[syn]
            found_skills.add(canonical)

    return found_skills

def extract_years_of_experience(text: str) -> float:
    """
    Extracts explicit or inferred years of experience from resume/job text.
    Handles patterns like '5+ years of experience', '3-5 years', 'over 7 years', etc.
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
                # Range like 3-5 years -> take upper bound or average
                val = float(groups[1])
            else:
                val = float(groups[0])
            if val < 45.0:  # Sanity cap to avoid graduation year mismatches like '2018 years'
                max_years = max(max_years, val)

    return max_years

def extract_education_level(text: str) -> Dict[str, any]:
    """
    Extracts education qualifications: PhD, Master's, Bachelor's, Associate, or Bootcamp.
    Returns level name and a normalized numeric tier (1 to 5).
    """
    text_lower = text.lower() if text else ""
    
    if re.search(r"\b(ph\.?d|doctorate|doctor of philosophy)\b", text_lower):
        return {"level": "Doctorate (PhD)", "tier": 5}
    if re.search(r"\b(master(?:'?s)?|m\.?s\.?|m\.?sc|m\.?tech|mba)\b", text_lower):
        return {"level": "Master's Degree", "tier": 4}
    if re.search(r"\b(bachelor(?:'?s)?|b\.?s\.?|b\.?sc|b\.?tech|b\.?e\.?|undergraduate)\b", text_lower):
        return {"level": "Bachelor's Degree", "tier": 3}
    if re.search(r"\b(associate(?:'?s)?|a\.?s\.?|diploma)\b", text_lower):
        return {"level": "Associate / Diploma", "tier": 2}
    if re.search(r"\b(bootcamp|certification|self-taught)\b", text_lower):
        return {"level": "Bootcamp / Certificate", "tier": 1}

    return {"level": "Unspecified / Practical Experience", "tier": 2}
