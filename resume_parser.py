"""Extracts text and structured candidate info from PDF resumes."""
import re
import pdfplumber
import spacy

nlp = spacy.load("en_core_web_sm")

SKILL_DB = [
    "python", "java", "c++", "c#", "javascript", "typescript", "sql", "r",
    "html", "css", "php", "go", "rust", "matlab", "scala",
    "machine learning", "deep learning", "nlp", "natural language processing",
    "computer vision", "reinforcement learning", "tensorflow", "pytorch",
    "keras", "scikit-learn", "sklearn", "pandas", "numpy", "opencv",
    "huggingface", "transformers", "langchain", "llamaindex", "spacy",
    "nltk", "yolo", "easyocr", "gradient boosting", "xgboost", "lightgbm",
    "django", "flask", "fastapi", "streamlit", "react", "node.js", "next.js",
    "angular", "vue", "spring boot", ".net",
    "mysql", "postgresql", "mongodb", "firebase", "supabase", "sqlite",
    "redis", "chromadb", "faiss", "pinecone",
    "aws", "azure", "gcp", "docker", "kubernetes", "git", "github", "linux",
    "ci/cd", "jenkins",
    "power bi", "tableau", "excel", "data analysis", "data visualization",
    "statistics", "a/b testing",
    "communication", "leadership", "teamwork", "problem solving",
    "project management", "agile", "scrum",
]

SECTION_HEADERS = {
    "experience": ["experience", "work experience", "employment history",
                   "professional experience"],
    "education": ["education", "academic background", "qualifications"],
    "skills": ["skills", "technical skills", "core competencies"],
    "projects": ["projects", "personal projects", "academic projects"],
    "certifications": ["certifications", "certificates", "licenses"],
}

EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
PHONE_RE = re.compile(r"(\+?\d[\d\-\s()]{8,14}\d)")
DATE_RANGE_RE = re.compile(
    r"(\d{4})\s*(?:-|to|–)\s*(present|current|\d{4})", re.IGNORECASE
)


def extract_text_from_pdf(file) -> str:
    """file: a file-like object (e.g. from Streamlit uploader)."""
    text_parts = []
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    return "\n".join(text_parts)


def _split_into_sections(text: str) -> dict:
    lines = text.split("\n")
    sections = {}
    current = "header"
    sections[current] = []

    for line in lines:
        stripped = line.strip().lower()
        matched_section = None
        for key, headers in SECTION_HEADERS.items():
            if any(stripped == h or stripped.startswith(h) for h in headers):
                matched_section = key
                break
        if matched_section:
            current = matched_section
            sections[current] = []
        else:
            sections.setdefault(current, []).append(line)
    return sections


def extract_name(text: str) -> str:
    # Heuristic: first few non-empty lines, look for a PERSON entity
    first_lines = "\n".join([l for l in text.split("\n") if l.strip()][:5])
    doc = nlp(first_lines)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text.strip()
    # Fallback: first non-empty line if short and looks like a name
    for line in text.split("\n"):
        line = line.strip()
        if 2 <= len(line.split()) <= 4 and not EMAIL_RE.search(line):
            return line
    return "Unknown"


def extract_email(text: str) -> str:
    match = EMAIL_RE.search(text)
    return match.group(0) if match else ""


def extract_phone(text: str) -> str:
    match = PHONE_RE.search(text)
    return match.group(0).strip() if match else ""


def extract_skills(text: str) -> list:
    text_lower = text.lower()
    found = set()
    for skill in SKILL_DB:
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, text_lower):
            found.add(skill)
    return sorted(found)


def estimate_years_experience(text: str) -> float:
    ranges = DATE_RANGE_RE.findall(text)
    total_years = 0
    for start, end in ranges:
        start_year = int(start)
        end_year = 2026 if end.lower() in ("present", "current") else int(end)
        if end_year >= start_year:
            total_years += (end_year - start_year)
    return float(total_years) if total_years > 0 else 0.0


def _clean_section(lines: list) -> list:
    return [l.strip("•- \t") for l in lines if l.strip()]


def parse_resume(file, filename: str = "") -> dict:
    text = extract_text_from_pdf(file)
    sections = _split_into_sections(text)

    return {
        "filename": filename,
        "name": extract_name(text),
        "email": extract_email(text),
        "phone": extract_phone(text),
        "skills": extract_skills(text),
        "education": _clean_section(sections.get("education", [])),
        "certifications": _clean_section(sections.get("certifications", [])),
        "experience": _clean_section(sections.get("experience", [])),
        "projects": _clean_section(sections.get("projects", [])),
        "years_experience": estimate_years_experience(text),
        "raw_text": text,
    }