"""Optional LLM-powered features via Gemini. Degrades gracefully if no API key
or if the configured model is unavailable."""

import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables from .env
load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
_configured = False
_working_model_name = None  # cached once we find a model that actually works

# Tried in order. Update this list if Google renames/retires models again.
CANDIDATE_MODELS = [
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-flash-latest",
    "gemini-1.5-pro",
    "gemini-pro",
]


def _ensure_configured() -> bool:
    global _configured
    if GEMINI_API_KEY and not _configured:
        genai.configure(api_key=GEMINI_API_KEY)
        _configured = True
    return bool(GEMINI_API_KEY)


def is_available() -> bool:
    return bool(GEMINI_API_KEY)


def _discover_working_model() -> str:
    """
    Find a usable model name. First tries the candidate list directly.
    If all fail, queries the API for models that support generateContent.
    """
    global _working_model_name

    if _working_model_name:
        return _working_model_name

    last_error = None

    # Fast path
    for name in CANDIDATE_MODELS:
        try:
            model = genai.GenerativeModel(name)
            model.generate_content("ping")
            _working_model_name = name
            return name
        except Exception as e:
            last_error = e
            continue

    # Fallback: discover available models dynamically
    try:
        available = [
            m.name
            for m in genai.list_models()
            if "generateContent"
            in getattr(m, "supported_generation_methods", [])
        ]

        if available:
            chosen = available[0]
            _working_model_name = chosen
            return chosen

    except Exception as e:
        last_error = e

    raise RuntimeError(
        "No usable Gemini model found for this API key. "
        f"Last error: {last_error}. "
        "Check your GEMINI_API_KEY or update CANDIDATE_MODELS."
    )


def _generate(prompt: str) -> str:
    """Generate text from Gemini with full error handling."""

    if not _ensure_configured():
        return (
            "⚠️ AI features unavailable: "
            "GEMINI_API_KEY environment variable is not set."
        )

    try:
        model_name = _discover_working_model()
    except RuntimeError as e:
        return f"⚠️ {e}"

    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)

        if not response.candidates:
            return (
                "⚠️ Gemini returned no response "
                "(possibly blocked by safety filters)."
            )

        if not response.text:
            return "⚠️ Gemini returned an empty response."

        return response.text

    except Exception as e:
        error_str = str(e)

        # Model retired / unavailable
        if "404" in error_str or "not found" in error_str.lower():
            global _working_model_name
            _working_model_name = None

            try:
                fallback_name = _discover_working_model()
                model = genai.GenerativeModel(fallback_name)
                response = model.generate_content(prompt)

                if response.text:
                    return response.text

                return "⚠️ Gemini returned an empty response."

            except Exception as e2:
                return (
                    "⚠️ Gemini model unavailable. "
                    f"Fallback failed: {e2}"
                )

        # Quota / rate limits
        if (
            "429" in error_str
            or "quota" in error_str.lower()
            or "rate limit" in error_str.lower()
        ):
            return (
                "⚠️ Gemini API quota exceeded or rate limit reached. "
                "Please try again later."
            )

        # Invalid API key
        if (
            "401" in error_str
            or "403" in error_str
            or "api key" in error_str.lower()
        ):
            return (
                "⚠️ Invalid Gemini API key. "
                "Check your GEMINI_API_KEY in the .env file."
            )

        return f"⚠️ Gemini error: {error_str}"


def summarize_candidate(candidate: dict, match: dict) -> str:
    prompt = f"""
Summarize this candidate for a recruiter in 4-5 sentences.

Name: {candidate['name']}
Skills: {', '.join(candidate['skills'])}
Years of experience: {candidate['years_experience']}
Matched required skills: {', '.join(match['matched_skills'])}
Missing skills: {', '.join(match['missing_skills'])}
Education: {' | '.join(candidate['education'][:3])}

Write a concise, neutral, professional summary highlighting fit and gaps.
"""
    return _generate(prompt)


def skill_gap_analysis(candidate: dict, job: dict, match: dict) -> str:
    prompt = f"""
A candidate is missing these skills:
{', '.join(match['missing_skills'])}

Job title: {job['title']}
Candidate skills: {', '.join(candidate['skills'])}

Provide a brief skill-gap analysis (3-4 bullet points) explaining:
- Impact of each missing skill
- One practical suggestion to learn each skill

Keep it concise.
"""
    return _generate(prompt)


# FIXED: added match parameter
def generate_interview_questions(
    candidate: dict,
    job: dict,
    match: dict,
    n: int = 5
) -> str:

    prompt = f"""
Generate {n} targeted interview questions for this candidate.

Role: {job['title']}
Candidate skills: {', '.join(candidate['skills'])}
Required skills: {', '.join(job['required_skills'])}
Missing skills: {', '.join(match['missing_skills'])}

Mix technical questions with 1-2 questions probing missing skills.

Return only a numbered list.
"""

    return _generate(prompt)


def resume_improvement_suggestions(candidate: dict, job: dict) -> str:
    prompt = f"""
Give 3-4 concise suggestions to improve this resume.

Target role: {job['title']}
Candidate skills: {', '.join(candidate['skills'])}
Required skills: {', '.join(job['required_skills'])}

Focus on:
- Missing keywords
- Resume structure
- Relevant achievements
- ATS optimization

Return bullet points only.
"""
    return _generate(prompt)
