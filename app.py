"""Streamlit dashboard for AI-powered resume screening."""
import json
import streamlit as st
import pandas as pd

from database import (
    init_db, save_candidate, save_job, save_match,
    get_all_candidates, get_matches_for_job, get_jobs, clear_all
)
from resume_parser import parse_resume
from job_analyzer import parse_job_description
from matching_engine import rank_candidates
import ai_features
from vector_search import search_candidates, build_index
from clustering import cluster_candidates, describe_cluster

st.set_page_config(page_title="AI Resume Screener", layout="wide")
init_db()

st.title("🎯 AI-Powered Resume Screening & Candidate Ranking")

tabs = st.tabs([
    "📤 Upload Resumes", "📝 Job Description", "🏆 Rankings",
    "👤 Candidate Detail", "⚖️ Compare Candidates", "📥 Export",
    "🔍 Semantic Search", "🧩 Candidate Clustering"
])
# ---------------- Upload Resumes ----------------
with tabs[0]:
    st.subheader("Upload Resumes (PDF)")
    files = st.file_uploader("Select one or more PDF resumes", type=["pdf"],
                              accept_multiple_files=True)
    if st.button("Process Resumes") and files:
        progress = st.progress(0)
        for i, f in enumerate(files):
            try:
                parsed = parse_resume(f, filename=f.name)
                save_candidate(parsed)
                st.success(f"✅ Parsed: {parsed['name']} ({f.name})")
            except Exception as e:
                st.error(f"❌ Failed to parse {f.name}: {e}")
            progress.progress((i + 1) / len(files))
        st.success(f"Processed {len(files)} resume(s).")

    st.divider()
    candidates = get_all_candidates()
    st.write(f"**{len(candidates)} candidate(s) in database**")
    if candidates:
        df = pd.DataFrame([{
            "Name": c["name"], "Email": c["email"], "Phone": c["phone"],
            "Years Exp": c["years_experience"], "File": c["filename"]
        } for c in candidates])
        st.dataframe(df, use_container_width=True)

    if st.button("🗑️ Clear All Data"):
        clear_all()
        st.rerun()

# ---------------- Job Description ----------------
with tabs[1]:
    st.subheader("Add a Job Description")
    job_title = st.text_input("Job Title")
    jd_text = st.text_area("Paste the job description", height=250)

    if st.button("Analyze Job Description") and jd_text:
        parsed_job = parse_job_description(job_title or "Untitled Role", jd_text)
        job_id = save_job(parsed_job)
        st.session_state["last_job_id"] = job_id
        st.success(f"Job saved (ID {job_id})")
        col1, col2, col3 = st.columns(3)
        col1.metric("Required Skills Found", len(parsed_job["required_skills"]))
        col2.metric("Preferred Skills Found", len(parsed_job["preferred_skills"]))
        col3.metric("Min. Experience (yrs)", parsed_job["min_experience"])
        st.write("**Required:**", ", ".join(parsed_job["required_skills"]) or "—")
        st.write("**Preferred:**", ", ".join(parsed_job["preferred_skills"]) or "—")

    st.divider()
    jobs = get_jobs()
    if jobs:
        st.write("**Saved jobs:**")
        for j in jobs:
            st.text(f"#{j['id']} — {j['title']}  (min exp: {j['min_experience']} yrs)")

# ---------------- Rankings ----------------
with tabs[2]:
    st.subheader("Candidate Rankings")
    jobs = get_jobs()
    if not jobs:
        st.info("Add a job description first.")
    else:
        job_options = {f"#{j['id']} — {j['title']}": j for j in jobs}
        selected_label = st.selectbox("Select job", list(job_options.keys()))
        selected_job = job_options[selected_label]

        if st.button("Run Matching"):
            candidates = get_all_candidates()
            if not candidates:
                st.warning("No candidates uploaded yet.")
            else:
                import json
                cand_clean = []
                for c in candidates:
                    cand_clean.append({
                        **c,
                        "skills": json.loads(c["skills"]),
                        "education": json.loads(c["education"]),
                    })
                job_clean = {
                    **selected_job,
                    "required_skills": json.loads(selected_job["required_skills"]),
                    "preferred_skills": json.loads(selected_job["preferred_skills"]),
                }
                results = rank_candidates(cand_clean, job_clean)
                for r in results:
                    save_match({
                        "candidate_id": r["candidate"]["id"],
                        "job_id": selected_job["id"],
                        **{k: r[k] for k in (
                            "final_score", "skill_score", "semantic_score",
                            "experience_score", "matched_skills", "missing_skills")}
                    })
                st.session_state["rank_results"] = results
                st.success("Matching complete.")

        if "rank_results" in st.session_state:
            results = st.session_state["rank_results"]
            df = pd.DataFrame([{
                "Rank": i + 1,
                "Name": r["candidate"]["name"],
                "Final Score": r["final_score"],
                "Skill Score": r["skill_score"],
                "Semantic Score": r["semantic_score"],
                "Experience Score": r["experience_score"],
                "Matched Skills": ", ".join(r["matched_skills"][:6]),
                "Missing Skills": ", ".join(r["missing_skills"][:6]),
            } for i, r in enumerate(results)])
            st.dataframe(df, use_container_width=True)

# ---------------- Candidate Detail ----------------
with tabs[3]:
    st.subheader("Candidate Detail")
    candidates = get_all_candidates()
    if candidates:
        names = {f"{c['name']} ({c['filename']})": c for c in candidates}
        choice = st.selectbox("Select candidate", list(names.keys()))
        c = names[choice]
        import json
        st.write("**Email:**", c["email"])
        st.write("**Phone:**", c["phone"])
        st.write("**Years Experience (est.):**", c["years_experience"])
        st.write("**Skills:**", ", ".join(json.loads(c["skills"])))
        st.write("**Education:**")
        for e in json.loads(c["education"]):
            st.text(f"- {e}")
        st.write("**Certifications:**")
        for cert in json.loads(c["certifications"]):
            st.text(f"- {cert}")
        st.write("**Experience:**")
        for e in json.loads(c["experience"]):
            st.text(f"- {e}")
        st.write("**Projects:**")
        for p in json.loads(c["projects"]):
            st.text(f"- {p}")

        if ai_features.is_available() and "rank_results" in st.session_state:
            match = next((r for r in st.session_state["rank_results"]
                          if r["candidate"]["id"] == c["id"]), None)
            if match and st.button("Generate AI Summary"):
                with st.spinner("Generating..."):
                    summary = ai_features.summarize_candidate(
                        {**c, "skills": json.loads(c["skills"]),
                         "education": json.loads(c["education"])}, match)
                    st.info(summary)
    else:
        st.info("No candidates yet.")

# ---------------- Compare Candidates ----------------
with tabs[4]:
    st.subheader("Compare Candidates")
    if "rank_results" in st.session_state:
        results = st.session_state["rank_results"]
        names = [r["candidate"]["name"] for r in results]
        chosen = st.multiselect("Pick candidates to compare", names, max_selections=4)
        if chosen:
            selected = [r for r in results if r["candidate"]["name"] in chosen]
            df = pd.DataFrame([{
                "Name": r["candidate"]["name"],
                "Final Score": r["final_score"],
                "Skill Score": r["skill_score"],
                "Semantic Score": r["semantic_score"],
                "Experience Score": r["experience_score"],
                "Years Exp": r["candidate"]["years_experience"],
                "Matched Skills": ", ".join(r["matched_skills"]),
                "Missing Skills": ", ".join(r["missing_skills"]),
            } for r in selected])
            st.dataframe(df.set_index("Name").T, use_container_width=True)
    else:
        st.info("Run matching in the Rankings tab first.")

# ---------------- Export ----------------
with tabs[5]:
    st.subheader("Export Results")
    if "rank_results" in st.session_state:
        results = st.session_state["rank_results"]
        df = pd.DataFrame([{
            "Rank": i + 1, "Name": r["candidate"]["name"],
            "Email": r["candidate"]["email"], "Final Score": r["final_score"],
            "Skill Score": r["skill_score"], "Semantic Score": r["semantic_score"],
            "Experience Score": r["experience_score"],
            "Matched Skills": ", ".join(r["matched_skills"]),
            "Missing Skills": ", ".join(r["missing_skills"]),
        } for i, r in enumerate(results)])
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", csv, "candidate_rankings.csv", "text/csv")
        st.dataframe(df, use_container_width=True)
    else:
        st.info("Nothing to export yet — run matching first.")
# ---------------- Semantic Search (FAISS) ----------------
with tabs[6]:
    st.subheader("🔍 Semantic Candidate Search")
    st.caption(
        "Search candidates by meaning, not just keywords. Try a free-text "
        "query like a job description or 'Python developer with NLP and AWS experience'."
    )

    candidates = get_all_candidates()
    if not candidates:
        st.info("Upload resumes first before searching.")
    else:
        col_a, col_b = st.columns([3, 1])
        with col_a:
            query = st.text_area("Search query", height=100,
                                  placeholder="e.g. Machine learning engineer with PyTorch and computer vision experience")
        with col_b:
            top_k = st.number_input("Results to return", min_value=1, max_value=20, value=5)
            rebuild = st.checkbox("Rebuild index", value=False,
                                   help="Check this after uploading new resumes so they're included in search.")

        if st.button("Search"):
            if not query.strip():
                st.warning("Please enter a search query.")
            else:
                try:
                    if rebuild:
                        with st.spinner("Rebuilding FAISS index..."):
                            build_index()
                    with st.spinner("Searching..."):
                        results = search_candidates(query, k=int(top_k))

                    if not results:
                        st.info("No matching candidates found.")
                    else:
                        for r in results:
                            c = r["candidate"]
                            with st.container(border=True):
                                col1, col2 = st.columns([3, 1])
                                col1.markdown(f"**{c['name']}**  \n{c['email']} | {c['phone']}")
                                col2.metric("Similarity", f"{r['similarity']}%")
                                skills = json.loads(c["skills"]) if c.get("skills") else []
                                st.caption(f"Skills: {', '.join(skills[:10]) or '—'}")
                except ValueError as ve:
                    st.warning(str(ve))
                except RuntimeError as re_err:
                    st.error(f"Search error: {re_err}")
                except Exception as e:
                    st.error(f"Unexpected error during search: {e}")

# ---------------- Candidate Clustering ----------------
with tabs[7]:
    st.subheader("🧩 Candidate Clustering")
    st.caption(
        "Groups candidates into clusters of similar profiles based on resume "
        "content — useful for spotting talent pools (e.g. 'backend engineers', "
        "'data scientists') across a large applicant pool."
    )

    candidates = get_all_candidates()
    if len(candidates) < 2:
        st.info("Need at least 2 candidates in the database to run clustering.")
    else:
        auto_k = st.checkbox("Auto-select number of clusters", value=True)
        manual_k = None
        if not auto_k:
            max_k = max(2, len(candidates) - 1)
            manual_k = st.slider("Number of clusters", min_value=2,
                                  max_value=min(10, max_k), value=min(3, max_k))

        if st.button("Run Clustering"):
            try:
                with st.spinner("Clustering candidates..."):
                    result = cluster_candidates(candidates, n_clusters=None if auto_k else manual_k)

                st.success(
                    f"Found {result['n_clusters']} cluster(s)"
                    + (f" (silhouette score: {result['silhouette_score']})"
                       if result["silhouette_score"] is not None else "")
                )

                for cluster_id, members in sorted(result["clusters"].items()):
                    summary = describe_cluster(members)
                    with st.expander(
                        f"Cluster {cluster_id + 1} — {summary['size']} candidate(s), "
                        f"avg {summary['avg_experience']} yrs exp"
                    ):
                        st.write("**Common skills:**", ", ".join(summary["top_skills"]) or "—")
                        for m in members:
                            st.text(f"• {m['name']} ({m['email']})")
            except ValueError as ve:
                st.warning(str(ve))
            except RuntimeError as re_err:
                st.error(f"Clustering error: {re_err}")
            except Exception as e:
                st.error(f"Unexpected error during clustering: {e}")