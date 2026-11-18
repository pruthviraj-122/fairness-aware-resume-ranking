import io
import pytest
import pypdf
import docx
from core.document_parser import (
    sanitize_text,
    parse_pdf,
    parse_docx,
    parse_text,
    parse_document,
    validate_file_content,
    extract_candidate_name,
    extract_contact_info,
    extract_resume_sections,
    DocumentParseError,
    OCR_AVAILABLE,
    MAX_PDF_PAGES,
)

def test_sanitize_text():
    raw = "Senior  Engineer \u2013 Python\u2019s ecosystem \u2022 AWS\x00"
    cleaned = sanitize_text(raw)
    assert cleaned == "Senior Engineer - Python's ecosystem * AWS"

def test_parse_docx_in_memory():
    doc = docx.Document()
    doc.add_heading("Jane Doe - Resume", level=1)
    doc.add_paragraph("Experienced Cloud Architect with 8 years in AWS and Kubernetes.")
    
    table = doc.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "Skill: Python"
    table.rows[0].cells[1].text = "Level: Expert"
    
    stream = io.BytesIO()
    doc.save(stream)
    stream.seek(0)

    extracted = parse_docx(stream)
    assert "Jane Doe - Resume" in extracted
    assert "Cloud Architect" in extracted
    assert "Skill: Python" in extracted

def test_parse_empty_document_raises_value_error():
    # Blank PDF page has no readable characters -> should raise ValueError
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=200, height=200)
    stream = io.BytesIO()
    writer.write(stream)
    stream.seek(0)
    
    with pytest.raises(ValueError):
        parse_pdf(stream)

def test_parse_text_multi_encoding():
    utf8_bytes = "Candidate résumé with UTF-8 symbols: €100k, naïve Bayes. Minimum twenty characters satisfied.".encode("utf-8")
    extracted_utf8 = parse_text(io.BytesIO(utf8_bytes))
    assert "résumé" in extracted_utf8

    latin1_bytes = "Candidate résumé Latin-1 with extra characters to satisfy minimum threshold.".encode("latin-1")
    extracted_latin1 = parse_text(io.BytesIO(latin1_bytes))
    assert "résumé" in extracted_latin1

def test_parse_document_dispatch():
    txt_content = "Python developer skills and experience exceeding twenty characters requirement."
    res_txt = parse_document(txt_content, filename="resume.txt")
    assert "Python developer skills" in res_txt

def test_validate_file_content():
    # Valid PDF magic bytes
    pdf_bytes = io.BytesIO(b"%PDF-1.4 header content")
    assert validate_file_content(pdf_bytes, "doc.pdf") is True

    # Invalid PDF magic bytes
    fake_pdf = io.BytesIO(b"NOT_A_PDF_CONTENT")
    assert validate_file_content(fake_pdf, "doc.pdf") is False

    # Valid DOCX ZIP magic bytes
    docx_bytes = io.BytesIO(b"PK\x03\x04ZIP_CONTENT")
    assert validate_file_content(docx_bytes, "doc.docx") is True

def test_parse_corrupt_pdf_raises_error():
    corrupt_bytes = b"NOT_A_VALID_PDF_HEADER_123456"
    with pytest.raises(DocumentParseError):
        parse_pdf(corrupt_bytes)

def test_extract_candidate_name():
    resume_text = "Johnathan Reynolds\njohn.reynolds@example.com | (555) 234-5678\nSummary: Software Engineer"
    name = extract_candidate_name(resume_text, "resume.pdf")
    assert name == "Johnathan Reynolds"

    # Fallback to filename
    no_name_text = "john@example.com\nSummary: Software Engineer"
    name_from_file = extract_candidate_name(no_name_text, "Sarah_Connor_Resume.pdf")
    assert name_from_file == "Sarah Connor"

def test_extract_contact_info():
    resume_text = (
        "Alice Smith\nEmail: alice.smith@techcorp.io\n"
        "Phone: +1 (555) 019-2834\n"
        "GitHub: github.com/alicesmith\n"
        "LinkedIn: linkedin.com/in/alice-smith\n"
        "Portfolio: https://alicesmith.dev"
    )
    contact = extract_contact_info(resume_text)
    assert contact["email"] == "alice.smith@techcorp.io"
    assert "555" in contact["phone"]
    assert "alicesmith" in contact["github"]
    assert "alice-smith" in contact["linkedin"]
    assert "alicesmith.dev" in contact["portfolio"]

def test_extract_contact_info_international_phone():
    resume_uk = "Dr. Arthur Pendelton\narthur@oxford.ac.uk\n+44 20 7946 0958"
    contact_uk = extract_contact_info(resume_uk)
    assert "+44" in contact_uk["phone"]

def test_extract_resume_sections():
    resume_text = (
        "Alice Smith\n\n"
        "Professional Summary:\n"
        "Senior developer with 6 years experience.\n\n"
        "Technical Skills:\n"
        "Python, Docker, Kubernetes, AWS\n\n"
        "Work Experience:\n"
        "Software Engineer at Acme Corp (2020-Present)\n\n"
        "Education:\n"
        "B.S. in Computer Science"
    )
    sections = extract_resume_sections(resume_text)
    assert "Python, Docker" in sections["skills"]
    assert "Acme Corp" in sections["experience"]
    assert "Computer Science" in sections["education"]

def test_ocr_and_limits_constants():
    assert OCR_AVAILABLE is False
    assert MAX_PDF_PAGES == 10
