import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

# Get API key safely
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

_configured = False


def _ensure_configured():
    """Configure Gemini only once and safely."""
    global _configured

    if not GEMINI_API_KEY:
        return False

    if not _configured:
        genai.configure(api_key=GEMINI_API_KEY)
        _configured = True

    return True


def is_available() -> bool:
    """Check if Gemini is usable."""
    return bool(GEMINI_API_KEY)


def _generate(prompt: str) -> str:
    """Core safe Gemini call."""
    if not _ensure_configured():
        return "AI features unavailable: GEMINI_API_KEY is not set."

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Gemini error: {str(e)}"


# ---------------------------
# FEATURES
# ---------------------------

def summarize_candidate(candidate: dict, match: dict) -> str:
    prompt = f"""
Summarize this candidate for a recruiter in 4-5 sentences.

Name: {candidate.get('name', '')}
Skills: {', '.join(candidate.get('skills', []))}
Years of experience: {candidate.get('years_experience', 0)}
Matched required skills: {', '.join(match.get('matched_skills', []))}
Missing skills: {', '.join(match.get('missing_skills', []))}
Education: {' | '.join(candidate.get('education', [])[:3])}

Write a concise, neutral, professional summary highlighting fit and gaps.
"""
    return _generate(prompt)


def skill_gap_analysis(candidate: dict, job: dict, match: dict) -> str:
    prompt = f"""
A candidate is missing these skills for a job: {', '.join(match.get('missing_skills', []))}.

Job title: {job.get('title', '')}
Candidate's existing skills: {', '.join(candidate.get('skills', []))}

Provide a brief skill gap analysis (3-4 bullet points) explaining impact and learning suggestions.
"""
    return _generate(prompt)


def generate_interview_questions(candidate: dict, job: dict, match: dict, n: int = 5) -> str:
    """
    FIXED: 'match' was missing in your original version.
    Now passed properly as argument.
    """

    prompt = f"""
Generate {n} targeted interview questions for this candidate applying to the role "{job.get('title', '')}".

Candidate skills: {', '.join(candidate.get('skills', []))}
Job required skills: {', '.join(job.get('required_skills', []))}
Missing skills: {', '.join(match.get('missing_skills', []))}

Mix technical questions and gap-based questions.
Return as numbered list only.
"""
    return _generate(prompt)


def resume_improvement_suggestions(candidate: dict, job: dict) -> str:
    prompt = f"""
Give 3-4 concise resume improvement suggestions.

Job: {job.get('title', '')}
Candidate skills: {', '.join(candidate.get('skills', []))}
Job required skills: {', '.join(job.get('required_skills', []))}

Focus on keywords, structure, and missing skills.
Return bullet points only.
"""
    return _generate(prompt)