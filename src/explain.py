"""Explainability helpers and data quality reporting."""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Sequence

import pandas as pd

from src.recommender import RecommenderArtifacts

DEFAULT_WEIGHTS = (0.6, 0.3, 0.1)


def _top_shared_keywords(counter_a: Counter, counter_b: Counter, top_n: int = 5) -> List[str]:
    overlap = []
    for token in set(counter_a) & set(counter_b):
        overlap.append((token, min(counter_a[token], counter_b[token])))
    overlap.sort(key=lambda x: (-x[1], x[0].lower()))
    return [token for token, _ in overlap[:top_n]]


def _contribution_summary(
    total_score: float,
    a: float,
    k: float,
    s: float,
    shared_kw: Sequence[str],
    shared_sdg: Sequence[int],
    weights: tuple = DEFAULT_WEIGHTS,
) -> str:
    """Plain-language summary of why this recommendation makes sense."""
    w_abs, w_kw, w_sdg = weights
    weighted_parts = {
        "publication summaries": w_abs * a,
        "shared keywords": w_kw * k,
        "shared sustainability goals": w_sdg * s,
    }
    strongest = max(weighted_parts, key=weighted_parts.get)

    if total_score >= 0.75:
        level = "strong"
    elif total_score >= 0.45:
        level = "good"
    else:
        level = "emerging"

    if shared_kw:
        kw_phrase = f"shared topics such as {', '.join(shared_kw)}"
    else:
        kw_phrase = "very limited direct topic overlap"

    if shared_sdg:
        sdg_phrase = f"shared sustainability goals (SDG {', '.join(map(str, shared_sdg))})"
    else:
        sdg_phrase = "only light overlap on sustainability goals"

    return (
        f"This looks like a {level} research fit. "
        f"Most of the match comes from {strongest}. "
        f"Evidence: {kw_phrase}, plus {sdg_phrase}."
    )


def explain_recommendation_pair(
    artifacts: RecommenderArtifacts,
    source_id: str,
    target_id: str,
) -> Dict[str, object]:
    """Explainability signals for one recommendation pair."""
    i = artifacts.id_to_idx[source_id]
    j = artifacts.id_to_idx[target_id]
    abs_component = float(artifacts.abs_sim[i, j])
    keyword_component = float(artifacts.kw_sim[i, j])
    sdg_component = float(artifacts.sdg_sim[i, j])
    total_score = float(artifacts.similarity_matrix[i, j])
    shared_keywords = _top_shared_keywords(
        artifacts.keyword_counters[source_id],
        artifacts.keyword_counters[target_id],
        top_n=5,
    )
    shared_sdg = sorted(artifacts.sdg_tags[source_id] & artifacts.sdg_tags[target_id])
    # If there are no exact shared SDGs but either side has tags, fall back to the union so users still see SDG focus.
    if not shared_sdg:
        shared_sdg = sorted(artifacts.sdg_tags[source_id] | artifacts.sdg_tags[target_id])
    weights = getattr(artifacts, "weights", DEFAULT_WEIGHTS)
    summary = _contribution_summary(
        total_score, abs_component, keyword_component, sdg_component,
        shared_keywords, shared_sdg, weights,
    )
    return {
        "abstract_component": abs_component,
        "keyword_component": keyword_component,
        "sdg_component": sdg_component,
        "shared_keywords": shared_keywords,
        "shared_sdg_tags": shared_sdg,
        "contribution_summary": summary,
    }


def build_data_quality_report(clean_df: pd.DataFrame) -> pd.DataFrame:
    """Per-faculty data quality and confidence for Evidence panel."""
    work = clean_df.copy()
    work["missing_abstract"] = work["clean_abstract"].eq("") | work["clean_abstract"].str.strip().eq("")
    work["missing_keywords"] = work["keyword_list"].apply(lambda x: len(x) == 0)
    work["has_sdg"] = work[["top 1", "top 2", "top 3"]].gt(0).any(axis=1)
    report = (
        work.groupby(["person_uuid", "name", "department"], as_index=False)
        .agg(
            publication_count=("article_uuid", "nunique"),
            missing_abstract_count=("missing_abstract", "sum"),
            missing_keyword_count=("missing_keywords", "sum"),
            sdg_tag_coverage=("has_sdg", "mean"),
        )
        .rename(columns={"person_uuid": "faculty_uuid"})
    )
    report["missing_abstract_rate"] = report["missing_abstract_count"] / report["publication_count"].clip(lower=1)
    report["missing_keyword_rate"] = report["missing_keyword_count"] / report["publication_count"].clip(lower=1)

    def confidence_label(row: pd.Series) -> str:
        if (
            row["publication_count"] >= 5
            and row["missing_abstract_rate"] <= 0.20
            and row["missing_keyword_rate"] <= 0.20
            and row["sdg_tag_coverage"] >= 0.60
        ):
            return "High"
        if (
            row["publication_count"] >= 3
            and row["missing_abstract_rate"] <= 0.50
            and row["missing_keyword_rate"] <= 0.50
        ):
            return "Medium"
        return "Low"

    report["confidence_score"] = report.apply(confidence_label, axis=1)
    report["data_sources"] = "Faculty publications: abstract, keywords, SDG top tags"
    report["sdg_assignment_note"] = "SDG tags use top 1 / top 2 / top 3 fields per publication"
    return report.sort_values(["confidence_score", "publication_count"], ascending=[True, False]).reset_index(drop=True)
