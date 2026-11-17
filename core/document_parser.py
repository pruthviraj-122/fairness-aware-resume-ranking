"""
FairHire AI — Document Parser & Resume Text Extractor
Supports PDF (pypdf), DOCX (python-docx), and plain text formats.
Extracts clean text, contact metadata, and identifies structured sections.
"""

import io
import re
from typing import Dict, List, Optional, Tuple, Union
import pypdf
import docx

class DocumentParseError(Exception):
    """Raised when a document cannot be parsed."""
    pass

def sanitize_text(text: str) -> str:
    """
    Cleans raw text while preserving meaningful technical symbols (C++, C#, .NET).
    Normalizes typographic quotes, dashes, and redundant whitespace.
    """
    if not text:
        return ""
    
    # Replace non-standard dashes and quotes
    text = re.sub(r"[\u2018\u2019]", "'", text)
    text = re.sub(r"[\u201C\u201D]", '"', text)
    text = re.sub(r"[\u2013\u2014]", "-", text)
    text = re.sub(r"\u2022", "*", text)
    
    # Normalize control characters except newlines/tabs
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)
    
    # Normalize multiple blank lines to a single newline
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    
    # Normalize horizontal whitespace
    text = re.sub(r"[ \t]+", " ", text)
    
    return text.strip()

def parse_pdf(file_input: Union[str, bytes, io.BytesIO]) -> str:
    """
    Extracts text from a PDF document using pypdf.
    Accepts filepath string or in-memory bytes/BytesIO.
    """
    try:
        if isinstance(file_input, (bytes, bytearray)):
            stream = io.BytesIO(file_input)
        elif isinstance(file_input, io.BytesIO):
            stream = file_input
        else:
            stream = open(file_input, "rb")

        reader = pypdf.PdfReader(stream)
        extracted_pages = []
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            if page_text.strip():
                extracted_pages.append(page_text.strip())

        full_text = "\n\n".join(extracted_pages)
        return sanitize_text(full_text)
    except Exception as e:
        raise DocumentParseError(f"Failed to parse PDF document: {str(e)}") from e

def parse_docx(file_input: Union[str, bytes, io.BytesIO]) -> str:
    """
    Extracts text from a DOCX document using python-docx.
    Includes paragraphs and table cells.
    """
    try:
        if isinstance(file_input, (bytes, bytearray)):
            stream = io.BytesIO(file_input)
        elif isinstance(file_input, io.BytesIO):
            stream = file_input
        else:
            stream = file_input

        doc = docx.Document(stream)
        text_parts = []
        
        # Paragraphs
        for p in doc.paragraphs:
            if p.text.strip():
                text_parts.append(p.text.strip())

        # Tables
        for table in doc.tables:
            for row in table.rows:
                row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                if row_text:
                    text_parts.append(row_text)

        full_text = "\n".join(text_parts)
        return sanitize_text(full_text)
    except Exception as e:
        raise DocumentParseError(f"Failed to parse DOCX document: {str(e)}") from e

def parse_text(file_input: Union[str, bytes, io.BytesIO]) -> str:
    """
    Extracts text from plain text or Markdown with multiple encoding fallbacks.
    """
    if isinstance(file_input, str):
        return sanitize_text(file_input)

    raw_bytes = file_input.read() if hasattr(file_input, "read") else file_input
    
    encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252", "iso-8859-1"]
    for enc in encodings:
        try:
            decoded = raw_bytes.decode(enc)
            return sanitize_text(decoded)
        except (UnicodeDecodeError, AttributeError):
            continue

    return sanitize_text(raw_bytes.decode("utf-8", errors="replace"))

def parse_document(file_input: Union[str, bytes, io.BytesIO], filename: str = "") -> str:
    """
    Automatically dispatches parsing based on file extension or filename.
    """
    ext = filename.lower().split(".")[-1] if "." in filename else ""
    
    if ext == "pdf":
        return parse_pdf(file_input)
    elif ext in ["docx", "doc"]:
        return parse_docx(file_input)
    elif ext in ["txt", "md", "csv", "json"]:
        return parse_text(file_input)
    else:
        # Try PDF, then DOCX, then TXT
        try:
            return parse_pdf(file_input)
        except Exception:
            try:
                return parse_docx(file_input)
            except Exception:
                return parse_text(file_input)

def extract_candidate_name(text: str, fallback_filename: str = "") -> str:
    """
    Heuristically extracts a candidate's name from resume text or filename.
    Looks at the first few lines of the resume before email/phone/address.
    """
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if lines:
        for line in lines[:4]:
            # If line is short, doesn't contain @, phone numbers, or section headers
            if 2 <= len(line.split()) <= 4 and not re.search(r"[@\d|:]|resume|curriculum|vitae|summary|skills", line, re.I):
                # Clean up punctuation
                cleaned = re.sub(r"[^\w\s-]", "", line).strip()
                if cleaned:
                    return cleaned

    # Fallback to filename
    if fallback_filename:
        clean_name = re.sub(r"(\.pdf|\.docx|\.txt|_resume|_cv|resume|cv)", "", fallback_filename, flags=re.I)
        clean_name = re.sub(r"[_-]", " ", clean_name).strip().title()
        if clean_name:
            return clean_name

    return "Candidate"

def extract_contact_info(text: str) -> Dict[str, Optional[str]]:
    """Extracts email, phone, and professional links from resume text."""
    email_match = re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b", text)
    email = email_match.group(0) if email_match else None

    phone_match = re.search(r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}", text)
    phone = phone_match.group(0) if phone_match else None

    github_match = re.search(r"(?:https?://)?(?:www\.)?github\.com/[A-Za-z0-9_.-]+", text, re.I)
    github = github_match.group(0) if github_match else None

    linkedin_match = re.search(r"(?:https?://)?(?:www\.)?linkedin\.com/in/[A-Za-z0-9_.-]+", text, re.I)
    linkedin = linkedin_match.group(0) if linkedin_match else None

    return {
        "email": email,
        "phone": phone,
        "github": github,
        "linkedin": linkedin
    }

def extract_resume_sections(text: str) -> Dict[str, str]:
    """
    Segments resume text into common functional sections:
    'summary', 'skills', 'experience', 'education', 'projects', 'other'.
    """
    section_patterns = {
        "summary": r"(?:professional\s+summary|summary|profile|about\s+me|objective)",
        "skills": r"(?:technical\s+skills|skills|technologies|competencies|core\s+skills)",
        "experience": r"(?:work\s+experience|professional\s+experience|experience|employment\s+history)",
        "education": r"(?:education|academic\s+background|degrees|certifications\s*&\s*education)",
        "projects": r"(?:projects|technical\s+projects|open\s+source|portfolio)",
    }

    # Combined header regex
    headers_regex = r"(?im)^(?:\s*#*\s*)(?P<header>" + "|".join(section_patterns.values()) + r")\s*[:\n]"
    matches = list(re.finditer(headers_regex, text))

    sections: Dict[str, str] = {k: "" for k in section_patterns.keys()}
    sections["other"] = ""

    if not matches:
        sections["other"] = text
        return sections

    # Content before first header
    if matches[0].start() > 0:
        sections["summary"] = text[:matches[0].start()].strip()

    for i, match in enumerate(matches):
        raw_header = match.group("header").strip().lower()
        start_idx = match.end()
        end_idx = matches[i + 1].start() if (i + 1 < len(matches)) else len(text)
        content = text[start_idx:end_idx].strip()

        # Categorize header
        assigned = False
        for sec_name, pattern in section_patterns.items():
            if re.search(rf"\b{pattern}\b", raw_header, re.I):
                sections[sec_name] = (sections[sec_name] + "\n" + content).strip()
                assigned = True
                break
        if not assigned:
            sections["other"] = (sections["other"] + "\n" + content).strip()

    return sections
