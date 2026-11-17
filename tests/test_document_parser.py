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
    extract_candidate_name,
    extract_contact_info,
    extract_resume_sections
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

def test_parse_pdf_in_memory():
    # Create a minimal valid PDF using pypdf
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=200, height=200)
    
    stream = io.BytesIO()
    writer.write(stream)
    stream.seek(0)
    
    # Empty page should not crash
    extracted = parse_pdf(stream)
    assert isinstance(extracted, str)

def test_parse_text_multi_encoding():
    utf8_bytes = "Candidate résumé with UTF-8 symbols: €100k, naïve Bayes".encode("utf-8")
    extracted_utf8 = parse_text(io.BytesIO(utf8_bytes))
    assert "résumé" in extracted_utf8

    latin1_bytes = "Candidate résumé Latin-1".encode("latin-1")
    extracted_latin1 = parse_text(io.BytesIO(latin1_bytes))
    assert "résumé" in extracted_latin1

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
        "LinkedIn: linkedin.com/in/alice-smith"
    )
    contact = extract_contact_info(resume_text)
    assert contact["email"] == "alice.smith@techcorp.io"
    assert "555" in contact["phone"]
    assert "alicesmith" in contact["github"]
    assert "alice-smith" in contact["linkedin"]

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
