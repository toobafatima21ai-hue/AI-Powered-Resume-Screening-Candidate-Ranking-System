"""FAISS-based vector search and embedding cache for candidates."""
import os
import pickle
import numpy as np

try:
    import faiss
except ImportError as e:
    raise ImportError(
        "faiss-cpu is not installed. Run: pip install faiss-cpu --break-system-packages"
    ) from e

from matching_engine import get_model
from database import get_connection, get_all_candidates

INDEX_PATH = "candidate_index.faiss"
ID_MAP_PATH = "candidate_index_ids.pkl"


def _serialize_embedding(vec: np.ndarray) -> bytes:
    return pickle.dumps(vec.astype(np.float32))


def _deserialize_embedding(blob: bytes) -> np.ndarray:
    return pickle.loads(blob)


def compute_and_store_embedding(candidate_id: int, text: str) -> np.ndarray:
    """Compute embedding for a candidate's resume text and cache it in DB."""
    try:
        model = get_model()
        embedding = model.encode([text[:5000]])[0]
        conn = get_connection()
        conn.execute(
            "UPDATE candidates SET embedding = ? WHERE id = ?",
            (_serialize_embedding(embedding), candidate_id)
        )
        conn.commit()
        conn.close()
        return embedding
    except Exception as e:
        raise RuntimeError(f"Failed to compute/store embedding for candidate {candidate_id}: {e}")


def get_or_compute_embedding(candidate: dict) -> np.ndarray:
    """Return cached embedding if available in the candidate dict, else compute it."""
    blob = candidate.get("embedding")
    if blob:
        try:
            return _deserialize_embedding(blob)
        except Exception:
            pass  # corrupted cache, fall through and recompute
    return compute_and_store_embedding(candidate["id"], candidate["raw_text"])


def build_index(candidates: list = None):
    """Build (and persist to disk) a FAISS index over all candidates' embeddings."""
    try:
        if candidates is None:
            candidates = get_all_candidates()
        if not candidates:
            raise ValueError("No candidates in the database. Upload resumes first.")

        embeddings, id_map = [], []
        for c in candidates:
            try:
                emb = get_or_compute_embedding(c)
                embeddings.append(emb)
                id_map.append(c["id"])
            except Exception as e:
                print(f"Skipping candidate {c.get('id')}: {e}")

        if not embeddings:
            raise ValueError("No embeddings could be computed for any candidate.")

        matrix = np.vstack(embeddings).astype(np.float32)
        faiss.normalize_L2(matrix)  # enables cosine similarity via inner product
        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)

        faiss.write_index(index, INDEX_PATH)
        with open(ID_MAP_PATH, "wb") as f:
            pickle.dump(id_map, f)

        return index, id_map
    except Exception as e:
        raise RuntimeError(f"Failed to build FAISS index: {e}")


def load_index():
    """Load a persisted index from disk, if present and valid."""
    if not os.path.exists(INDEX_PATH) or not os.path.exists(ID_MAP_PATH):
        return None, None
    try:
        index = faiss.read_index(INDEX_PATH)
        with open(ID_MAP_PATH, "rb") as f:
            id_map = pickle.load(f)
        return index, id_map
    except Exception as e:
        print(f"Failed to load FAISS index, will rebuild: {e}")
        return None, None


def search_candidates(query_text: str, k: int = 5) -> list:
    """Semantic search: return top-k candidates most similar to query_text
    (e.g. a job description, or 'looking for a Python ML engineer with NLP experience')."""
    if not query_text or not query_text.strip():
        raise ValueError("Query text cannot be empty.")
    try:
        index, id_map = load_index()
        if index is None or index.ntotal == 0:
            index, id_map = build_index()

        model = get_model()
        query_emb = model.encode([query_text[:5000]]).astype(np.float32)
        faiss.normalize_L2(query_emb)

        k = min(k, index.ntotal)
        scores, indices = index.search(query_emb, k)

        results = []
        conn = get_connection()
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            candidate_id = id_map[idx]
            row = conn.execute(
                "SELECT * FROM candidates WHERE id = ?", (candidate_id,)
            ).fetchone()
            if row:
                results.append({
                    "candidate": dict(row),
                    "similarity": round(float(score) * 100, 1)
                })
        conn.close()
        return results
    except Exception as e:
        raise RuntimeError(f"Semantic search failed: {e}")