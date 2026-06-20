"""SQLite storage layer for candidates, jobs, and match results."""
import sqlite3
import json
from datetime import datetime

DB_PATH = "resume_screener.db"


def get_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS candidates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            email TEXT,
            phone TEXT,
            skills TEXT,
            education TEXT,
            certifications TEXT,
            experience TEXT,
            projects TEXT,
            raw_text TEXT,
            years_experience REAL,
            filename TEXT,
            uploaded_at TEXT,
            embedding BLOB
        )
    """)

    # Migration: add embedding column if upgrading an existing DB created
    # before this feature existed.
    try:
        cur.execute("ALTER TABLE candidates ADD COLUMN embedding BLOB")
        conn.commit()
    except sqlite3.OperationalError:
        pass  # column already exists

    cur.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            raw_text TEXT,
            required_skills TEXT,
            preferred_skills TEXT,
            min_experience REAL,
            created_at TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS matches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            candidate_id INTEGER,
            job_id INTEGER,
            final_score REAL,
            skill_score REAL,
            semantic_score REAL,
            experience_score REAL,
            matched_skills TEXT,
            missing_skills TEXT,
            created_at TEXT,
            FOREIGN KEY(candidate_id) REFERENCES candidates(id),
            FOREIGN KEY(job_id) REFERENCES jobs(id)
        )
    """)
    conn.commit()
    conn.close()


def save_candidate(data: dict) -> int:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO candidates (name, email, phone, skills, education,
            certifications, experience, projects, raw_text, years_experience,
            filename, uploaded_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        data.get("name"), data.get("email"), data.get("phone"),
        json.dumps(data.get("skills", [])),
        json.dumps(data.get("education", [])),
        json.dumps(data.get("certifications", [])),
        json.dumps(data.get("experience", [])),
        json.dumps(data.get("projects", [])),
        data.get("raw_text", ""),
        data.get("years_experience", 0),
        data.get("filename", ""),
        datetime.now().isoformat()
    ))
    conn.commit()
    cid = cur.lastrowid
    conn.close()
    return cid


def save_job(data: dict) -> int:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO jobs (title, raw_text, required_skills, preferred_skills,
            min_experience, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        data.get("title"), data.get("raw_text"),
        json.dumps(data.get("required_skills", [])),
        json.dumps(data.get("preferred_skills", [])),
        data.get("min_experience", 0),
        datetime.now().isoformat()
    ))
    conn.commit()
    jid = cur.lastrowid
    conn.close()
    return jid


def save_match(data: dict) -> int:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO matches (candidate_id, job_id, final_score, skill_score,
            semantic_score, experience_score, matched_skills, missing_skills,
            created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        data["candidate_id"], data["job_id"], data["final_score"],
        data["skill_score"], data["semantic_score"], data["experience_score"],
        json.dumps(data["matched_skills"]), json.dumps(data["missing_skills"]),
        datetime.now().isoformat()
    ))
    conn.commit()
    mid = cur.lastrowid
    conn.close()
    return mid


def get_all_candidates():
    conn = get_connection()
    rows = conn.execute("SELECT * FROM candidates").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_matches_for_job(job_id: int):
    conn = get_connection()
    rows = conn.execute("""
        SELECT m.*, c.name, c.email, c.filename
        FROM matches m JOIN candidates c ON m.candidate_id = c.id
        WHERE m.job_id = ?
        ORDER BY m.final_score DESC
    """, (job_id,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_jobs():
    conn = get_connection()
    rows = conn.execute("SELECT * FROM jobs ORDER BY id DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def clear_all():
    conn = get_connection()
    for t in ("candidates", "jobs", "matches"):
        conn.execute(f"DELETE FROM {t}")
    conn.commit()
    conn.close()