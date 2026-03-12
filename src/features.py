"""Publication-level features with disk caching for embeddings and TF-IDF."""
from __future__ import annotations

import hashlib
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

CACHE_DIR = Path(os.environ.get("CASE_COMP_CACHE", ".cache"))
DEFAULT_WEIGHTS = (0.6, 0.3, 0.1)


def _content_hash(clean_df: pd.DataFrame) -> str:
    """Stable hash of dataframe content for cache invalidation."""
    key = pd.util.hash_pandas_object(clean_df[["person_uuid", "article_uuid", "combined_text", "keyword_str"]]).values
    key = hashlib.sha256(np.ascontiguousarray(key).tobytes()).hexdigest()[:16]
    return key


def build_publication_features(clean_df: pd.DataFrame):
    """
    Build publication-level features (no cache).
    Returns: (abs_embeddings, keyword_matrix, sdg_matrix, tfidf_vectorizer, sbert_model)
    """
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    model = SentenceTransformer("all-mpnet-base-v2")
    abs_embeddings = model.encode(clean_df["combined_text"].tolist(), show_progress_bar=True)

    tfidf = TfidfVectorizer()
    keyword_sparse = tfidf.fit_transform(clean_df["keyword_str"])
    keyword_matrix = keyword_sparse.toarray()

    sdg_matrix = np.vstack(clean_df["sdg_vec"].to_numpy())
    expected_rows = len(clean_df)
    if abs_embeddings.shape[0] != expected_rows or keyword_matrix.shape[0] != expected_rows or sdg_matrix.shape[0] != expected_rows:
        raise ValueError("Feature matrix row count mismatch")

    return abs_embeddings, keyword_matrix, sdg_matrix, tfidf, model


def build_or_load_publication_features(clean_df: pd.DataFrame):
    """
    Build or load from cache: publication embeddings and TF-IDF.
    Returns: (abs_embeddings, keyword_matrix, sdg_matrix, tfidf_vectorizer, sbert_model)
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    content_hash = _content_hash(clean_df)
    cache_prefix = CACHE_DIR / f"pub_features_{content_hash}"

    abs_path = cache_prefix.with_suffix(".abs.npy")
    kw_path = cache_prefix.with_suffix(".kw.npy")
    sdg_path = cache_prefix.with_suffix(".sdg.npy")
    tfidf_path = cache_prefix.with_suffix(".tfidf.pkl")
    meta_path = cache_prefix.with_suffix(".meta.json")

    if abs_path.exists() and kw_path.exists() and sdg_path.exists() and tfidf_path.exists():
        try:
            abs_emb = np.load(abs_path)
            kw_mat = np.load(kw_path)
            sdg_mat = np.load(sdg_path)
            with open(tfidf_path, "rb") as f:
                tfidf = pickle.load(f)
            with open(meta_path, "r") as f:
                meta = json.load(f)
            if meta.get("n_rows") == len(clean_df):
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer("all-mpnet-base-v2")
                return abs_emb, kw_mat, sdg_mat, tfidf, model
        except Exception:
            pass

    abs_emb, kw_mat, sdg_mat, tfidf, model = build_publication_features(clean_df)
    np.save(abs_path, abs_emb)
    np.save(kw_path, kw_mat)
    np.save(sdg_path, sdg_mat)
    with open(tfidf_path, "wb") as f:
        pickle.dump(tfidf, f)
    with open(meta_path, "w") as f:
        json.dump({"n_rows": len(clean_df)}, f)
    return abs_emb, kw_mat, sdg_mat, tfidf, model
