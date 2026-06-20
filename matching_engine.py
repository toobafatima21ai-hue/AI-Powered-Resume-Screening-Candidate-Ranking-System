"""Scores candidates against a job: skill match + semantic similarity + experience."""
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

_model = None

WEIGHTS = {"skill": 0.5, "semantic": 0.3, "experience": 0.2}


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")  # free, local, ~80MB
    return _model


def skill_match_score(candidate_skills: list, required_skills: list,
                       preferred_skills: list) -> tuple:
    cand_set = set(s.lower() for s in candidate_skills)
    req_set = set(s.lower() for s in required_skills)
    pref_set = set(s.lower() for s in preferred_skills)

    matched_required = cand_set & req_set
    matched_preferred = cand_set & pref_set
    missing_required = req_set - cand_set

    if not req_set and not pref_set:
        score = 50.0  # neutral if JD had no detectable skills
    else:
        req_component = (len(matched_required) / len(req_set)) * 80 if req_set else 80
        pref_component = (len(matched_preferred) / len(pref_set)) * 20 if pref_set else 0
        score = req_component + pref_component

    return score, sorted(matched_required | matched_preferred), sorted(missing_required)


def semantic_similarity_score(resume_text: str, jd_text: str) -> float:
    model = get_model()
    embeddings = model.encode([resume_text[:5000], jd_text[:5000]])
    sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return float(max(0, min(100, sim * 100)))


def experience_score(candidate_years: float, min_required: float) -> float:
    if min_required <= 0:
        return 100.0
    if candidate_years >= min_required:
        return 100.0
    return max(0.0, (candidate_years / min_required) * 100)


def calculate_match(candidate: dict, job: dict) -> dict:
    skill_score, matched, missing = skill_match_score(
        candidate["skills"], job["required_skills"], job["preferred_skills"]
    )
    sem_score = semantic_similarity_score(candidate["raw_text"], job["raw_text"])
    exp_score = experience_score(candidate["years_experience"], job["min_experience"])

    final_score = (
        skill_score * WEIGHTS["skill"]
        + sem_score * WEIGHTS["semantic"]
        + exp_score * WEIGHTS["experience"]
    )

    return {
        "final_score": round(final_score, 1),
        "skill_score": round(skill_score, 1),
        "semantic_score": round(sem_score, 1),
        "experience_score": round(exp_score, 1),
        "matched_skills": matched,
        "missing_skills": missing,
    }


def rank_candidates(candidates: list, job: dict) -> list:
    results = []
    for c in candidates:
        m = calculate_match(c, job)
        m["candidate"] = c
        results.append(m)
    return sorted(results, key=lambda x: x["final_score"], reverse=True)