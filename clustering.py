"""KMeans clustering of candidates based on resume embeddings."""
import json
from collections import Counter

import numpy as np

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
except ImportError as e:
    raise ImportError("scikit-learn is required for clustering.") from e

from vector_search import get_or_compute_embedding


def cluster_candidates(candidates: list, n_clusters: int = None) -> dict:
    """
    Group candidates into clusters of similar resumes.
    If n_clusters is None, automatically picks the best k (2-6) via silhouette score.
    Returns: {"clusters": {cluster_id: [candidates]}, "n_clusters": int, "silhouette_score": float|None}
    """
    try:
        if len(candidates) < 2:
            raise ValueError("Need at least 2 candidates to perform clustering.")

        embeddings, valid_candidates = [], []
        for c in candidates:
            try:
                emb = get_or_compute_embedding(c)
                embeddings.append(emb)
                valid_candidates.append(c)
            except Exception as e:
                print(f"Skipping candidate {c.get('id')} in clustering: {e}")

        if len(valid_candidates) < 2:
            raise ValueError("Not enough valid embeddings to cluster.")

        X = np.vstack(embeddings)
        max_possible_k = len(valid_candidates) - 1

        if n_clusters is None:
            n_clusters = _auto_select_k(X, max_possible_k)
        else:
            n_clusters = max(2, min(n_clusters, max_possible_k))

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)

        clusters = {}
        for label, candidate in zip(labels, valid_candidates):
            clusters.setdefault(int(label), []).append(candidate)

        sil_score = None
        if len(set(labels)) > 1:
            try:
                sil_score = round(float(silhouette_score(X, labels)), 3)
            except Exception:
                sil_score = None

        return {"clusters": clusters, "n_clusters": n_clusters, "silhouette_score": sil_score}
    except Exception as e:
        raise RuntimeError(f"Clustering failed: {e}")


def _auto_select_k(X: np.ndarray, max_k: int) -> int:
    """Try k=2..min(6,max_k), return the k with the best silhouette score."""
    best_k, best_score = 2, -1
    upper = min(6, max_k)
    if upper < 2:
        return 2
    for k in range(2, upper + 1):
        try:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X)
            if len(set(labels)) < 2:
                continue
            score = silhouette_score(X, labels)
            if score > best_score:
                best_score, best_k = score, k
        except Exception:
            continue
    return best_k


def describe_cluster(candidates: list) -> dict:
    """Summarize a cluster: top skills, average years of experience."""
    all_skills, total_exp = [], 0
    for c in candidates:
        skills = c.get("skills")
        if isinstance(skills, str):
            try:
                skills = json.loads(skills)
            except Exception:
                skills = []
        all_skills.extend(skills or [])
        total_exp += c.get("years_experience", 0) or 0

    top_skills = [s for s, _ in Counter(all_skills).most_common(5)]
    avg_exp = round(total_exp / len(candidates), 1) if candidates else 0

    return {"size": len(candidates), "top_skills": top_skills, "avg_experience": avg_exp}