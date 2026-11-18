"""
FairHire AI 3.0 — Secure Document Parser & Resume Text Extractor
Supports PDF (pypdf), DOCX (python-docx), and plain text formats.
Extracts clean text, contact metadata, and identifies structured sections.

Security & Robustness Enhancements:
- File signature / magic-byte verification (prevents extension spoofing)
- Maximum PDF page limit (MAX_PDF_PAGES = 10)
- Encrypted PDF detection with clear user feedback
- Empty/unreadable document detection (min 20 characters)
- Safe handling of corrupted/malformed DOCX and PDF files
- Plaintext length capping (max 50,000 characters)
- Explicit OCR availability indicator (OCR_AVAILABLE = False; non-scanned only)
"""

import io
import re
import zipfile
from typing import Any, Dict, List, Optional, Tuple, Union

import pypdf
import docx

# Explicit OCR status declaration
OCR_AVAILABLE: bool = False
MAX_PDF_PAGES: int = 10
MAX_TEXT_LENGTH: int = 50000
MIN_READABLE_CHARS: int = 20


class DocumentParseError(Exception):
    """Raised when a document cannot be parsed or fails validation."""
    pass


def validate_file_content(file_input: Union[bytes, io.BytesIO], filename: str = "") -> bool:
    """
    Validates file magic bytes against the declared filename extension.
    Returns True if valid or if extension is text-based.
    """
    ext = filename.lower().split(".")[-1] if "." in filename else ""
    
    if isinstance(file_input, io.BytesIO):
        pos = file_input.tell()
        file_input.seek(0)
        header = file_input.read(8)
        file_input.seek(pos)
    elif isinstance(file_input, (bytes, bytearray)):
        header = bytes(file_input[:8])
    else:
        return True

    if ext == "pdf":
        return header.startswith(b"%PDF")
    elif ext in ("docx", "doc"):
        # DOCX is an OpenXML ZIP container starting with PK
        return header.startswith(b"PK")
    elif ext in ("txt", "md", "csv", "json"):
        return True
    return True


def sanitize_text(text: str) -> str:
    """
    Cleans raw text while preserving meaningful technical symbols (C++, C#, .NET).
    Normalizes typographic quotes, dashes, bullet points, and redundant whitespace.
    """
    if not text:
        return ""
    
    # Replace typographic quotes, dashes, and bullets
    text = re.sub(r"[\u2018\u2019]", "'", text)
    text = re.sub(r"[\u201C\u201D]", '"', text)
    text = re.sub(r"[\u2013\u2014]", "-", text)
    text = re.sub(r"[\u2022\u25cf\u25aa\u25b6\u2219]", "*", text)
    
    # Normalize control characters except newlines/tabs
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)
    
    # Normalize multiple blank lines to a single double newline
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    
    # Normalize horizontal whitespace
    text = re.sub(r"[ \t]+", " ", text)
    
    return text.strip()


def parse_pdf(file_input: Union[str, bytes, io.BytesIO]) -> str:
    """
    Extracts text from a PDF document using pypdf with security limits.
    Enforces page limits and encrypted document detection.
    """
    stream = None
    close_stream = False
    try:
        if isinstance(file_input, (bytes, bytearray)):
            stream = io.BytesIO(file_input)
        elif isinstance(file_input, io.BytesIO):
            stream = file_input
        elif isinstance(file_input, str):
            stream = open(file_input, "rb")
            close_stream = True
        else:
            stream = file_input

        # Magic byte check if stream supports seek
        if hasattr(stream, "seek") and hasattr(stream, "read"):
            pos = stream.tell()
            stream.seek(0)
            magic = stream.read(4)
            stream.seek(pos)
            if magic and not magic.startswith(b"%PDF"):
                raise DocumentParseError("Invalid PDF header signature. File is not a valid PDF.")

        reader = pypdf.PdfReader(stream)
        
        # Encrypted PDF check
        if getattr(reader, "is_encrypted", False):
            raise DocumentParseError("PDF is encrypted or password-protected. Please provide an unencrypted version.")

        total_pages = len(reader.pages)
        if total_pages == 0:
            raise DocumentParseError("PDF contains no pages.")

        pages_to_read = min(total_pages, MAX_PDF_PAGES)
        extracted_pages = []
        
        for i in range(pages_to_read):
            try:
                page = reader.pages[i]
                page_text = page.extract_text() or ""
                if page_text.strip():
                    extracted_pages.append(page_text.strip())
            except Exception as page_err:
                # Log page extraction error but continue with other pages if possible
                continue

        full_text = "\n\n".join(extracted_pages)
        sanitized = sanitize_text(full_text)
        
        if len(sanitized) < MIN_READABLE_CHARS:
            raise ValueError("Document appears to be empty or contains no readable text (scanned image PDFs require OCR).")
            
        return sanitized
    except (DocumentParseError, ValueError):
        raise
    except Exception as e:
        raise DocumentParseError(f"Failed to parse PDF document: {str(e)}") from e
    finally:
        if close_stream and stream:
            stream.close()


def parse_docx(file_input: Union[str, bytes, io.BytesIO]) -> str:
    """
    Extracts text from a DOCX document using python-docx.
    Includes paragraphs and table cells with error handling for malformed archives.
    """
    try:
        if isinstance(file_input, (bytes, bytearray)):
            stream = io.BytesIO(file_input)
        elif isinstance(file_input, io.BytesIO):
            stream = file_input
        else:
            stream = file_input

        # Magic byte check
        if hasattr(stream, "seek") and hasattr(stream, "read"):
            pos = stream.tell()
            stream.seek(0)
            magic = stream.read(4)
            stream.seek(pos)
            if magic and not magic.startswith(b"PK"):
                raise DocumentParseError("Invalid DOCX header signature. File is not a valid OpenXML document.")

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
        sanitized = sanitize_text(full_text)
        
        if len(sanitized) < MIN_READABLE_CHARS:
            raise ValueError("Document appears to be empty or contains no readable text.")
            
        return sanitized
    except zipfile.BadZipFile as bzf:
        raise DocumentParseError("Malformed or corrupted DOCX archive.") from bzf
    except PermissionError as pe:
        raise DocumentParseError("Permission denied while accessing document.") from pe
    except (DocumentParseError, ValueError):
        raise
    except Exception as e:
        raise DocumentParseError(f"Failed to parse DOCX document: {str(e)}") from e


def parse_text(file_input: Union[str, bytes, io.BytesIO]) -> str:
    """
    Extracts text from plain text or Markdown with multiple encoding fallbacks.
    Enforces maximum text length (MAX_TEXT_LENGTH).
    """
    if isinstance(file_input, str):
        content = file_input
    else:
        raw_bytes = file_input.read() if hasattr(file_input, "read") else file_input
        if isinstance(raw_bytes, str):
            content = raw_bytes
        else:
            encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252", "iso-8859-1"]
            content = None
            for enc in encodings:
                try:
                    content = raw_bytes.decode(enc)
                    break
                except (UnicodeDecodeError, AttributeError):
                    continue
            if content is None:
                content = raw_bytes.decode("utf-8", errors="replace")

    # Enforce text character limit
    if len(content) > MAX_TEXT_LENGTH:
        content = content[:MAX_TEXT_LENGTH]

    sanitized = sanitize_text(content)
    if len(sanitized) < MIN_READABLE_CHARS:
        raise ValueError("Document appears to be empty or contains no readable text.")

    return sanitized


def parse_document(file_input: Union[str, bytes, io.BytesIO], filename: str = "") -> str:
    """
    Automatically dispatches parsing based on validated file extension or magic bytes.
    """
    ext = filename.lower().split(".")[-1] if "." in filename else ""
    
    # If bytes or BytesIO, validate magic bytes if extension present
    if hasattr(file_input, "seek") or isinstance(file_input, (bytes, bytearray)):
        if not validate_file_content(file_input, filename):
            raise DocumentParseError(f"File content does not match expected format for .{ext}")

    if ext == "pdf":
        return parse_pdf(file_input)
    elif ext in ["docx", "doc"]:
        return parse_docx(file_input)
    elif ext in ["txt", "md", "csv", "json"]:
        return parse_text(file_input)
    else:
        # Fallback cascade: try PDF, then DOCX, then TXT
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
        for line in lines[:5]:
            clean_line = re.sub(r"^[\*\-\•\#\d\.\s]+", "", line).strip()
            words = clean_line.split()
            if 2 <= len(words) <= 4 and not re.search(
                r"[@\d|:]|resume|curriculum|vitae|summary|skills|experience|education|developer|engineer|manager",
                clean_line,
                re.IGNORECASE
            ):
                cleaned = re.sub(r"[^\w\s-]", "", clean_line).strip()
                if cleaned:
                    return cleaned

    # Fallback to filename
    if fallback_filename:
        clean_name = re.sub(r"(\.pdf|\.docx|\.txt|_resume|_cv|resume|cv)", "", fallback_filename, flags=re.IGNORECASE)
        clean_name = re.sub(r"[_-]", " ", clean_name).strip().title()
        if clean_name:
            return clean_name

    return "Candidate"


def extract_contact_info(text: str) -> Dict[str, Optional[str]]:
    """
    Extracts email, phone, and professional links from resume text.
    Supports international and US domestic phone formats.
    """
    email_match = re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,7}\b", text)
    email = email_match.group(0) if email_match else None

    phone_match = re.search(
        r"\+\d{1,3}(?:[-.\s]+\d+){2,5}|(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",
        text
    )
    phone = phone_match.group(0).strip() if phone_match else None

    github_match = re.search(r"(?:https?://)?(?:www\.)?github\.com/[A-Za-z0-9_.-]+", text, re.IGNORECASE)
    github = github_match.group(0) if github_match else None

    linkedin_match = re.search(r"(?:https?://)?(?:www\.)?linkedin\.com/in/[A-Za-z0-9_.-]+", text, re.IGNORECASE)
    linkedin = linkedin_match.group(0) if linkedin_match else None

    portfolio_match = re.search(r"(?:https?://)(?:www\.)?[A-Za-z0-9_.-]+\.(?:io|dev|me|tech|org|net)", text, re.IGNORECASE)
    portfolio = portfolio_match.group(0) if portfolio_match else None

    return {
        "email": email,
        "phone": phone,
        "github": github,
        "linkedin": linkedin,
        "portfolio": portfolio
    }


def extract_resume_sections(text: str) -> Dict[str, str]:
    """
    Segments resume text into common functional sections.
    """
    section_patterns = {
        "summary": r"(?:professional\s+summary|summary|profile|about\s+me|objective)",
        "skills": r"(?:technical\s+skills|skills|technologies|competencies|core\s+skills)",
        "experience": r"(?:work\s+experience|professional\s+experience|experience|employment\s+history)",
        "education": r"(?:education|academic\s+background|degrees|academic\s+credentials)",
        "projects": r"(?:projects|technical\s+projects|open\s+source|portfolio)",
        "certifications": r"(?:certifications|licenses|professional\s+development)"
    }

    headers_regex = r"(?im)^(?:\s*#*\s*)(?P<header>" + "|".join(section_patterns.values()) + r")\s*[:\n]"
    matches = list(re.finditer(headers_regex, text))

    sections: Dict[str, str] = {k: "" for k in section_patterns.keys()}
    sections["other"] = ""

    if not matches:
        sections["other"] = text
        return sections

    if matches[0].start() > 0:
        sections["summary"] = text[:matches[0].start()].strip()

    for i, match in enumerate(matches):
        raw_header = match.group("header").strip().lower()
        start_idx = match.end()
        end_idx = matches[i + 1].start() if (i + 1 < len(matches)) else len(text)
        content = text[start_idx:end_idx].strip()

        assigned = False
        for sec_name, pattern in section_patterns.items():
            if re.search(rf"\b{pattern}\b", raw_header, re.IGNORECASE):
                sections[sec_name] = (sections[sec_name] + "\n" + content).strip()
                assigned = True
                break
        if not assigned:
            sections["other"] = (sections["other"] + "\n" + content).strip()

    return sections
