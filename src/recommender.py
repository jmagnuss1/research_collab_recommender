"""Faculty and paper profile building, similarity matrices, and recommendation APIs."""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

DEFAULT_TOP_K = 25
DEFAULT_WEIGHTS = (0.6, 0.3, 0.1)


@dataclass
class RecommenderArtifacts:
    mode: str
    entity_ids: List[str]
    id_to_idx: Dict[str, int]
    metadata: pd.DataFrame
    combined_vectors: np.ndarray
    similarity_matrix: np.ndarray
    abs_sim: np.ndarray
    kw_sim: np.ndarray
    sdg_sim: np.ndarray
    abs_norm: np.ndarray
    kw_norm: np.ndarray
    sdg_norm: np.ndarray
    keyword_counters: Dict[str, Counter]
    sdg_tags: Dict[str, set]
    weights: Tuple[float, float, float] = DEFAULT_WEIGHTS


def _aggregate_profile_blocks(
    groups: Dict[str, np.ndarray],
    abs_embeddings: np.ndarray,
    keyword_matrix: np.ndarray,
    sdg_matrix: np.ndarray,
    source_df: pd.DataFrame,
) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray, Dict[str, Counter], Dict[str, set]]:
    entity_ids = list(groups.keys())
    abs_profiles, kw_profiles, sdg_profiles = [], [], []
    keyword_counters: Dict[str, Counter] = {}
    sdg_tags: Dict[str, set] = {}

    for entity_id in entity_ids:
        idx = np.array(groups[entity_id], dtype=int)
        abs_profiles.append(abs_embeddings[idx].mean(axis=0))
        kw_profiles.append(keyword_matrix[idx].mean(axis=0))
        sdg_profiles.append(sdg_matrix[idx].mean(axis=0))
        tokens = [kw for kws in source_df.iloc[idx]["keyword_list"].tolist() for kw in kws]
        keyword_counters[entity_id] = Counter(tokens)
        tag_set = set()
        for c in ["top 1", "top 2", "top 3"]:
            tag_set.update(int(v) for v in source_df.iloc[idx][c].tolist() if 1 <= int(v) <= 17)
        sdg_tags[entity_id] = tag_set

    return (
        entity_ids,
        np.vstack(abs_profiles),
        np.vstack(kw_profiles),
        np.vstack(sdg_profiles),
        keyword_counters,
        sdg_tags,
    )


def _weighted_similarity(
    abs_profiles: np.ndarray,
    kw_profiles: np.ndarray,
    sdg_profiles: np.ndarray,
    weights: Tuple[float, float, float] = DEFAULT_WEIGHTS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    w_abs, w_kw, w_sdg = weights
    abs_norm = normalize(abs_profiles)
    kw_norm = normalize(kw_profiles)
    sdg_norm = normalize(sdg_profiles)
    combined_vectors = np.hstack([w_abs * abs_norm, w_kw * kw_norm, w_sdg * sdg_norm])
    similarity_matrix = cosine_similarity(combined_vectors)
    abs_sim = cosine_similarity(abs_norm)
    kw_sim = cosine_similarity(kw_norm)
    sdg_sim = cosine_similarity(sdg_norm)
    return combined_vectors, similarity_matrix, abs_sim, kw_sim, sdg_sim, abs_norm, kw_norm, sdg_norm


def build_mode_artifacts(
    clean_df: pd.DataFrame,
    mode: str,
    abs_embeddings: np.ndarray,
    keyword_matrix: np.ndarray,
    sdg_matrix: np.ndarray,
    weights: Tuple[float, float, float] = DEFAULT_WEIGHTS,
) -> RecommenderArtifacts:
    """Create mode-specific profile vectors and similarity matrices."""
    if mode not in {"faculty", "paper"}:
        raise ValueError("mode must be one of: 'faculty', 'paper'")

    if mode == "faculty":
        groups = clean_df.groupby("person_uuid", sort=False).indices
        metadata = (
            clean_df.groupby("person_uuid", as_index=False)
            .agg(
                name=("name", "first"),
                email=("email", "first"),
                department=("department", "first"),
                publication_count=("article_uuid", "nunique"),
            )
            .rename(columns={"person_uuid": "entity_id"})
        )
    else:
        groups = clean_df.groupby("article_uuid", sort=False).indices
        metadata = (
            clean_df.groupby("article_uuid", as_index=False)
            .agg(
                title=("title", "first"),
                publication_year=("publication_year", "first"),
                author_count=("person_uuid", "nunique"),
                sample_department=("department", "first"),
            )
            .rename(columns={"article_uuid": "entity_id"})
        )

    (
        entity_ids,
        abs_profiles,
        kw_profiles,
        sdg_profiles,
        keyword_counters,
        sdg_tags,
    ) = _aggregate_profile_blocks(groups, abs_embeddings, keyword_matrix, sdg_matrix, clean_df)

    (
        combined_vectors,
        similarity_matrix,
        abs_sim,
        kw_sim,
        sdg_sim,
        abs_norm,
        kw_norm,
        sdg_norm,
    ) = _weighted_similarity(abs_profiles, kw_profiles, sdg_profiles, weights)

    metadata = metadata.set_index("entity_id").reindex(entity_ids).reset_index()

    return RecommenderArtifacts(
        mode=mode,
        entity_ids=entity_ids,
        id_to_idx={eid: i for i, eid in enumerate(entity_ids)},
        metadata=metadata,
        combined_vectors=combined_vectors,
        similarity_matrix=similarity_matrix,
        abs_sim=abs_sim,
        kw_sim=kw_sim,
        sdg_sim=sdg_sim,
        abs_norm=abs_norm,
        kw_norm=kw_norm,
        sdg_norm=sdg_norm,
        keyword_counters=keyword_counters,
        sdg_tags=sdg_tags,
        weights=weights,
    )


def _build_recommendation_df(artifacts: RecommenderArtifacts, source_id: str, top_k: int) -> pd.DataFrame:
    from src.explain import explain_recommendation_pair

    if source_id not in artifacts.id_to_idx:
        raise KeyError(f"Unknown {artifacts.mode} id: {source_id}")
    i = artifacts.id_to_idx[source_id]
    sims = artifacts.similarity_matrix[i].copy()
    sims[i] = -1.0
    top_idx = np.argsort(sims)[-top_k:][::-1]
    rows = []
    for j in top_idx:
        target_id = artifacts.entity_ids[j]
        explanation = explain_recommendation_pair(artifacts, source_id, target_id)
        rows.append({"source_id": source_id, "recommended_id": target_id, "similarity_score": float(sims[j]), **explanation})
    rec = pd.DataFrame(rows)
    if rec.empty:
        return rec
    meta_src = artifacts.metadata.rename(columns={"entity_id": "source_id"})
    src_cols = [c for c in meta_src.columns if c != "source_id"]
    meta_src = meta_src.rename(columns={c: f"source_{c}" for c in src_cols})
    meta_tgt = artifacts.metadata.rename(columns={"entity_id": "recommended_id"})
    tgt_cols = [c for c in meta_tgt.columns if c != "recommended_id"]
    meta_tgt = meta_tgt.rename(columns={c: f"recommended_{c}" for c in tgt_cols})
    rec = rec.merge(meta_src, on="source_id", how="left").merge(meta_tgt, on="recommended_id", how="left")
    return rec.sort_values("similarity_score", ascending=False).reset_index(drop=True)


def get_faculty_recommendations(
    person_uuid: str,
    artifacts: RecommenderArtifacts,
    top_k: int = DEFAULT_TOP_K,
) -> pd.DataFrame:
    """Top-k similar faculty with explainability."""
    rec = _build_recommendation_df(artifacts, source_id=person_uuid, top_k=top_k)
    if rec.empty:
        return rec
    return rec.rename(columns={
        "source_id": "faculty_uuid", "recommended_id": "recommended_uuid",
        "source_name": "name", "source_email": "email", "source_department": "department",
        "recommended_name": "recommended_name", "recommended_email": "recommended_email",
        "recommended_department": "recommended_department",
    })


def get_paper_recommendations(
    article_uuid: str,
    artifacts: RecommenderArtifacts,
    top_k: int = DEFAULT_TOP_K,
) -> pd.DataFrame:
    """Top-k similar papers with explainability."""
    rec = _build_recommendation_df(artifacts, source_id=article_uuid, top_k=top_k)
    if rec.empty:
        return rec
    return rec.rename(columns={
        "source_id": "article_uuid", "recommended_id": "recommended_article_uuid",
        "source_title": "title", "recommended_title": "recommended_title",
    })


def paper_to_faculty_recommendations(
    article_uuid: str,
    faculty_artifacts: RecommenderArtifacts,
    paper_artifacts: RecommenderArtifacts,
    top_k: int = DEFAULT_TOP_K,
) -> pd.DataFrame:
    """For one paper, return top-k faculty whose research is closest to that paper."""
    from src.explain import _top_shared_keywords, _contribution_summary

    if article_uuid not in paper_artifacts.id_to_idx:
        raise KeyError(f"Unknown article: {article_uuid}")
    paper_idx = paper_artifacts.id_to_idx[article_uuid]
    paper_vec = paper_artifacts.combined_vectors[paper_idx : paper_idx + 1]
    sims = cosine_similarity(paper_vec, faculty_artifacts.combined_vectors).ravel()
    top_idx = np.argsort(sims)[-top_k:][::-1]
    paper_kw = paper_artifacts.keyword_counters.get(article_uuid, Counter())
    paper_sdg = paper_artifacts.sdg_tags.get(article_uuid, set())
    weights = faculty_artifacts.weights
    rows = []
    for j in top_idx:
        fid = faculty_artifacts.entity_ids[j]
        meta = faculty_artifacts.metadata.iloc[j]
        abs_c = float(cosine_similarity(
            paper_artifacts.abs_norm[paper_idx : paper_idx + 1],
            faculty_artifacts.abs_norm[j : j + 1],
        )[0, 0])
        kw_c = float(cosine_similarity(
            paper_artifacts.kw_norm[paper_idx : paper_idx + 1],
            faculty_artifacts.kw_norm[j : j + 1],
        )[0, 0])
        sdg_c = float(cosine_similarity(
            paper_artifacts.sdg_norm[paper_idx : paper_idx + 1],
            faculty_artifacts.sdg_norm[j : j + 1],
        )[0, 0])
        shared_kw = _top_shared_keywords(paper_kw, faculty_artifacts.keyword_counters.get(fid, Counter()), top_n=5)
        shared_sdg = sorted(paper_sdg & faculty_artifacts.sdg_tags.get(fid, set()))
        total = float(sims[j])
        summary = _contribution_summary(total, abs_c, kw_c, sdg_c, shared_kw, shared_sdg, weights)
        rows.append({
            "recommended_uuid": fid,
            "recommended_name": meta.get("name"),
            "recommended_email": meta.get("email"),
            "recommended_department": meta.get("department"),
            "similarity_score": total,
            "abstract_component": abs_c,
            "keyword_component": kw_c,
            "sdg_component": sdg_c,
            "shared_keywords": shared_kw,
            "shared_sdg_tags": shared_sdg,
            "contribution_summary": summary,
        })
    return pd.DataFrame(rows)


def build_full_recommendation_table(artifacts: RecommenderArtifacts, top_k: int = DEFAULT_TOP_K) -> pd.DataFrame:
    """Full recommendation table for all entities in this mode."""
    frames = [_build_recommendation_df(artifacts, source_id=eid, top_k=top_k) for eid in artifacts.entity_ids]
    out = pd.concat(frames, ignore_index=True)
    if artifacts.mode == "faculty":
        out = out.rename(columns={
            "source_id": "faculty_uuid", "recommended_id": "recommended_uuid",
            "source_name": "name", "source_email": "email", "source_department": "department",
            "recommended_name": "recommended_name", "recommended_email": "recommended_email",
            "recommended_department": "recommended_department",
        })
    else:
        out = out.rename(columns={
            "source_id": "article_uuid", "recommended_id": "recommended_article_uuid",
            "source_title": "title", "recommended_title": "recommended_title",
        })
    return out


def parse_sdg_query(sdg_query: Union[int, str, Sequence[int]]) -> List[int]:
    """Parse SDG/theme query into SDG IDs (1-17)."""
    if isinstance(sdg_query, int):
        return [sdg_query] if 1 <= sdg_query <= 17 else []
    if isinstance(sdg_query, (list, tuple, set)):
        return sorted(set(int(x) for x in sdg_query if 1 <= int(x) <= 17))
    if isinstance(sdg_query, str):
        text = sdg_query.lower()
        ids = [int(x) for x in re.findall(r"\b([1-9]|1[0-7])\b", text)]
        theme_map = {
            "poverty": 1, "hunger": 2, "health": 3, "education": 4, "gender": 5,
            "water": 6, "energy": 7, "work": 8, "industry": 9, "inequality": 10,
            "city": 11, "consumption": 12, "climate": 13, "ocean": 14, "land": 15,
            "peace": 16, "partnership": 17,
        }
        for keyword, sdg_id in theme_map.items():
            if keyword in text:
                ids.append(sdg_id)
        return sorted(set(i for i in ids if 1 <= i <= 17))
    return []


def topic_to_faculty_lookup(
    topic: str,
    faculty_artifacts: RecommenderArtifacts,
    sbert_model,
    tfidf_vectorizer,
    weights: Tuple[float, float, float] = DEFAULT_WEIGHTS,
    top_k: int = 10,
) -> pd.DataFrame:
    """Student persona: free-text topic -> recommended faculty."""
    if not isinstance(topic, str) or not topic.strip():
        raise ValueError("topic must be a non-empty string")
    query_text = topic.strip()
    w_abs, w_kw, w_sdg = weights
    query_abs = sbert_model.encode([query_text], show_progress_bar=False)
    query_kw = tfidf_vectorizer.transform([query_text]).toarray()
    query_sdg = np.zeros((1, 17), dtype=float)
    query_abs_norm = normalize(query_abs)
    query_kw_norm = normalize(query_kw)
    query_sdg_norm = normalize(query_sdg)
    query_vec = np.hstack([w_abs * query_abs_norm, w_kw * query_kw_norm, w_sdg * query_sdg_norm])
    sims = cosine_similarity(query_vec, faculty_artifacts.combined_vectors).ravel()
    top_idx = np.argsort(sims)[-top_k:][::-1]
    query_terms = {t.lower() for t in re.findall(r"[A-Za-z]+", topic)}
    rows = []
    for idx in top_idx:
        fid = faculty_artifacts.entity_ids[idx]
        meta = faculty_artifacts.metadata.iloc[idx]
        # Prefer keywords that overlap with the query terms; if none, fall back to this faculty member's top keywords.
        counter = faculty_artifacts.keyword_counters[fid]
        kw_matches = [kw for kw in counter if kw.lower() in query_terms][:5]
        if not kw_matches:
            kw_matches = [kw for kw, _ in counter.most_common(3)]
        rows.append({
            "faculty_uuid": fid,
            "name": meta.get("name"),
            "department": meta.get("department"),
            "email": meta.get("email"),
            "topic_similarity": float(sims[idx]),
            "matched_keywords": kw_matches,
        })
    return pd.DataFrame(rows)


def sdg_to_faculty_lookup(
    sdg_query: Union[int, str, Sequence[int]],
    faculty_artifacts: RecommenderArtifacts,
    top_k: int = 10,
) -> pd.DataFrame:
    """Donor persona: SDG/theme -> recommended faculty."""
    sdg_ids = parse_sdg_query(sdg_query)
    if not sdg_ids:
        raise ValueError("Could not parse SDG query into tags 1-17")
    q = np.zeros((1, 17), dtype=float)
    for sdg_id in sdg_ids:
        q[0, sdg_id - 1] = 1.0
    q_norm = normalize(q)
    sims = cosine_similarity(q_norm, faculty_artifacts.sdg_norm).ravel()
    top_idx = np.argsort(sims)[-top_k:][::-1]
    rows = []
    for idx in top_idx:
        fid = faculty_artifacts.entity_ids[idx]
        meta = faculty_artifacts.metadata.iloc[idx]
        tags = faculty_artifacts.sdg_tags[fid]
        overlap = sorted(tags.intersection(set(sdg_ids)))
        # If there is no strict overlap (e.g. data is sparse), fall back to showing this faculty member's SDG tags.
        matched = overlap if overlap else sorted(tags)
        rows.append({
            "faculty_uuid": fid,
            "name": meta.get("name"),
            "department": meta.get("department"),
            "email": meta.get("email"),
            "sdg_similarity": float(sims[idx]),
            "matched_sdg_tags": matched,
        })
    return pd.DataFrame(rows)


def recommendations_alternate_mode(
    source_id: str,
    artifacts: RecommenderArtifacts,
    mode: str,
    top_k: int = DEFAULT_TOP_K,
) -> pd.DataFrame:
    """
    Get recommendations using only one signal (for Alternatives tab).
    mode: "bert_only" | "tfidf_only" | "sdg_only"
    """
    from src.explain import explain_recommendation_pair

    i = artifacts.id_to_idx[source_id]
    if mode == "bert_only":
        sims = artifacts.abs_sim[i]
    elif mode == "tfidf_only":
        sims = artifacts.kw_sim[i]
    elif mode == "sdg_only":
        sims = artifacts.sdg_sim[i]
    else:
        raise ValueError("mode must be bert_only, tfidf_only, or sdg_only")
    sims = sims.copy()
    sims[i] = -1.0
    top_idx = np.argsort(sims)[-top_k:][::-1]
    rows = []
    for j in top_idx:
        target_id = artifacts.entity_ids[j]
        expl = explain_recommendation_pair(artifacts, source_id, target_id)
        meta = artifacts.metadata.iloc[j]
        rows.append({
            "recommended_id": target_id,
            "recommended_name": meta.get("name"),
            "recommended_email": meta.get("email"),
            "recommended_department": meta.get("department"),
            "similarity_score": float(sims[j]),
            **expl,
        })
    rec = pd.DataFrame(rows)
    if artifacts.mode == "faculty":
        rec = rec.rename(columns={"recommended_id": "recommended_uuid"})
    return rec
