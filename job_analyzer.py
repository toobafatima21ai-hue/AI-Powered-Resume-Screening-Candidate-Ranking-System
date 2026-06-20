"""Parses free-text job descriptions into structured requirements."""
import re
from resume_parser import SKILL_DB

EXPERIENCE_RE = re.compile(
    r"(\d+)\+?\s*(?:-\s*\d+\s*)?years?", re.IGNORECASE
)

PREFERRED_MARKERS = [
    "preferred", "nice to have", "bonus", "good to have", "plus"
]


def extract_required_skills(jd_text: str) -> list:
    text_lower = jd_text.lower()
    found = set()
    for skill in SKILL_DB:
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, text_lower):
            found.add(skill)
    return sorted(found)


def extract_preferred_skills(jd_text: str) -> list:
    """Skills mentioned near 'preferred'/'nice to have' markers."""
    text_lower = jd_text.lower()
    preferred = set()
    for marker in PREFERRED_MARKERS:
        idx = text_lower.find(marker)
        if idx != -1:
            window = text_lower[idx: idx + 400]
            for skill in SKILL_DB:
                if re.search(r"\b" + re.escape(skill) + r"\b", window):
                    preferred.add(skill)
    return sorted(preferred)


def extract_min_experience(jd_text: str) -> float:
    matches = EXPERIENCE_RE.findall(jd_text)
    years = [int(m) for m in matches]
    return float(max(years)) if years else 0.0


def parse_job_description(title: str, jd_text: str) -> dict:
    required = extract_required_skills(jd_text)
    preferred = extract_preferred_skills(jd_text)
    # required skills shouldn't double-count as preferred
    required = [s for s in required if s not in preferred]

    return {
        "title": title,
        "raw_text": jd_text,
        "required_skills": required,
        "preferred_skills": preferred,
        "min_experience": extract_min_experience(jd_text),
    }