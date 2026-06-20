"""
Retrieval-Augmented Generation (RAG) recruiter chat assistant.

Pipeline:
  1. CHUNK   - split each candidate's raw resume text into overlapping passages
  2. EMBED   - encode each chunk with the same MiniLM model used elsewhere
  3. INDEX   - store chunk vectors in a dedicated FAISS index (separate from
               the candidate-level index in vector_search.py)
  4. RETRIEVE- given a recruiter question, embed it and pull the top-k most
               relevant resume passages
  5. GENERATE- inject only those retrieved passages into a Gemini prompt and
               ask it to answer strictly from that grounding context, with
               citations back to which candidate each fact came from
"""
import os
import pickle
import json
import numpy as np

try:
    import faiss
except ImportError as e:
    raise ImportError(
        "faiss-cpu is not installed. Run: pip install faiss-cpu --break-system-packages"
    ) from e

from matching_engine import get_model
from database import get_connection, get_all_candidates
import ai_features

CHUNK_INDEX_PATH = "chunk_index.faiss"
CHUNK_META_PATH = "chunk_index_meta.pkl"

CHUNK_SIZE = 700        # characters per chunk
CHUNK_OVERLAP = 150     # overlap between consecutive chunks, preserves context across boundaries


# ---------------------------------------------------------------------------
# 1. CHUNKING
# ---------------------------------------------------------------------------
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> list:
    """Split text into overlapping character chunks. Returns list of strings."""
    if not text or not text.strip():
        return []
    text = text.strip()
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks


# ---------------------------------------------------------------------------
# 2-3. EMBED + INDEX
# ---------------------------------------------------------------------------
def build_chunk_index(candidates: list = None):
    """
    Build a FAISS index over resume chunks (not whole resumes).
    Each vector maps back to metadata: which candidate, which chunk text.
    """
    try:
        if candidates is None:
            candidates = get_all_candidates()
        if not candidates:
            raise ValueError("No candidates in the database. Upload resumes first.")

        model = get_model()
        all_vectors, metadata = [], []

        for c in candidates:
            text = c.get("raw_text", "")
            chunks = chunk_text(text)
            if not chunks:
                continue
            try:
                vectors = model.encode(chunks)
            except Exception as e:
                print(f"Embedding failed for candidate {c.get('id')}: {e}")
                continue

            for chunk_str, vec in zip(chunks, vectors):
                all_vectors.append(vec)
                metadata.append({
                    "candidate_id": c["id"],
                    "name": c.get("name", "Unknown"),
                    "email": c.get("email", ""),
                    "chunk_text": chunk_str,
                })

        if not all_vectors:
            raise ValueError("No resume text could be chunked/embedded.")

        matrix = np.vstack(all_vectors).astype(np.float32)
        faiss.normalize_L2(matrix)
        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)

        faiss.write_index(index, CHUNK_INDEX_PATH)
        with open(CHUNK_META_PATH, "wb") as f:
            pickle.dump(metadata, f)

        return index, metadata
    except Exception as e:
        raise RuntimeError(f"Failed to build chunk index: {e}")


def load_chunk_index():
    if not os.path.exists(CHUNK_INDEX_PATH) or not os.path.exists(CHUNK_META_PATH):
        return None, None
    try:
        index = faiss.read_index(CHUNK_INDEX_PATH)
        with open(CHUNK_META_PATH, "rb") as f:
            metadata = pickle.load(f)
        return index, metadata
    except Exception as e:
        print(f"Failed to load chunk index, will rebuild: {e}")
        return None, None


# ---------------------------------------------------------------------------
# 4. RETRIEVE
# ---------------------------------------------------------------------------
def retrieve_relevant_chunks(query: str, k: int = 6) -> list:
    """
    Embed the query and return the top-k most similar resume chunks,
    each with candidate metadata and a similarity score.
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")
    try:
        index, metadata = load_chunk_index()
        if index is None or index.ntotal == 0:
            index, metadata = build_chunk_index()

        model = get_model()
        query_vec = model.encode([query]).astype(np.float32)
        faiss.normalize_L2(query_vec)

        k = min(k, index.ntotal)
        scores, indices = index.search(query_vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            meta = metadata[idx]
            results.append({**meta, "similarity": round(float(score) * 100, 1)})
        return results
    except Exception as e:
        raise RuntimeError(f"Chunk retrieval failed: {e}")


# ---------------------------------------------------------------------------
# 5. GENERATE (grounded in retrieved chunks only)
# ---------------------------------------------------------------------------
def _build_grounded_prompt(question: str, chunks: list) -> str:
    context_blocks = []
    for i, ch in enumerate(chunks, 1):
        context_blocks.append(
            f"[Source {i} | Candidate: {ch['name']} ({ch['email']})]\n{ch['chunk_text']}"
        )
    context = "\n\n".join(context_blocks)

    return f"""You are a recruiting assistant. Answer the recruiter's question
using ONLY the resume excerpts provided below as context. Do not use any
outside knowledge or make assumptions beyond what's written.

If the excerpts don't contain enough information to answer confidently, say so
explicitly rather than guessing.

When you state a fact, cite which candidate it came from by name, e.g. "(Source 2 - Jane Doe)".

RESUME EXCERPTS:
{context}

RECRUITER QUESTION:
{question}

ANSWER (grounded strictly in the excerpts above, with citations):"""


def ask_recruiter_assistant(question: str, k: int = 6) -> dict:
    """
    Full RAG call: retrieve relevant resume chunks, then ask Gemini to answer
    using only that retrieved context. Returns the answer plus the sources
    used, so the UI can show provenance.
    """
    if not ai_features.is_available():
        return {
            "answer": "AI Recruiter Chat is unavailable: set the GEMINI_API_KEY "
                      "environment variable to enable this feature.",
            "sources": [],
        }
    try:
        chunks = retrieve_relevant_chunks(question, k=k)
        if not chunks:
            return {
                "answer": "No resume content found to answer this question. "
                          "Make sure resumes have been uploaded.",
                "sources": [],
            }

        prompt = _build_grounded_prompt(question, chunks)
        answer = ai_features._generate(prompt)

        return {"answer": answer, "sources": chunks}
    except ValueError as ve:
        return {"answer": f"⚠️ {ve}", "sources": []}
    except RuntimeError as re_err:
        return {"answer": f"⚠️ {re_err}", "sources": []}
    except Exception as e:
        return {"answer": f"⚠️ Unexpected error: {e}", "sources": []}