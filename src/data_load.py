"""Data loading, cleaning, and faculty metadata for the recommender."""
from __future__ import annotations

import os
import re
from typing import List, Union

import pandas as pd

ALLOWED_DEPARTMENTS = {"Finance", "Accountancy", "Business Administration"}
DEFAULT_DATA_PATH = os.environ.get("CASE_COMP_DATA", "case_competition.csv")


def clean_html(text: str) -> str:
    """Remove HTML tags from abstract text; return empty string for missing values."""
    if pd.isna(text):
        return ""
    return re.sub(r"<.*?>", "", str(text)).strip()


def parse_keywords(raw: str) -> List[str]:
    """Parse semicolon-delimited keywords and keep non-empty alphabetical tokens."""
    if pd.isna(raw):
        return []
    tokens = [t.strip() for t in str(raw).split(";") if t and t.strip()]
    return [t for t in tokens if any(c.isalpha() for c in t)]


def normalize_sdg_value(value: Union[int, float, str]) -> int:
    """Convert SDG entries to integer tags in [0, 17], using 0 for missing/invalid."""
    if pd.isna(value):
        return 0
    try:
        ivalue = int(float(value))
        return ivalue if 0 <= ivalue <= 17 else 0
    except Exception:
        return 0


def sdg_vector_from_row(row: pd.Series):
    """Build a 17-dim SDG frequency vector from top-1/top-2/top-3 SDG labels."""
    import numpy as np

    vec = np.zeros(17, dtype=float)
    for col in ["top 1", "top 2", "top 3"]:
        tag = row[col]
        if 1 <= tag <= 17:
            vec[int(tag) - 1] += 1.0
    return vec


def validate_required_columns(df: pd.DataFrame) -> None:
    required = {
        "person_uuid", "name", "email", "department",
        "article_uuid", "title", "abstract", "keywords",
        "top 1", "top 2", "top 3", "is_sustain", "publication_year",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")


def enforce_department_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Restrict dataset to requested departments."""
    filtered = df[df["department"].isin(ALLOWED_DEPARTMENTS)].copy().reset_index(drop=True)
    if filtered.empty:
        raise ValueError("Department filter removed all rows. Check input data.")
    return filtered


def prepare_clean_dataframe(csv_path: str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load and enrich publication-level records used by the recommender."""
    df = pd.read_csv(csv_path)
    validate_required_columns(df)
    df = enforce_department_filter(df)

    for col in ["top 1", "top 2", "top 3"]:
        df[col] = df[col].apply(normalize_sdg_value)

    df["clean_abstract"] = df["abstract"].apply(clean_html)
    df["keyword_list"] = df["keywords"].apply(parse_keywords)
    df["keyword_str"] = df["keyword_list"].apply(lambda lst: " ".join(lst))
    df["sdg_vec"] = df.apply(sdg_vector_from_row, axis=1)

    df["combined_text"] = (
        df["clean_abstract"].fillna("").astype(str).str.strip()
        + " "
        + df["keyword_str"].fillna("")
    ).str.strip()
    df.loc[df["combined_text"] == "", "combined_text"] = "no abstract available"

    clean_df = df[
        [
            "person_uuid", "name", "email", "department",
            "article_uuid", "title", "clean_abstract", "keyword_list", "keyword_str",
            "combined_text", "sdg_vec",
            "top 1", "top 2", "top 3", "is_sustain", "publication_year",
        ]
    ].copy()

    if clean_df[["person_uuid", "article_uuid"]].isna().any().any():
        raise ValueError("person_uuid/article_uuid contain nulls after cleaning.")

    return clean_df


def build_faculty_meta(clean_df: pd.DataFrame) -> pd.DataFrame:
    """One row per faculty: faculty_uuid, name, email, department, publication_count."""
    meta = (
        clean_df.groupby("person_uuid", as_index=False)
        .agg(
            name=("name", "first"),
            email=("email", "first"),
            department=("department", "first"),
            publication_count=("article_uuid", "nunique"),
        )
        .rename(columns={"person_uuid": "faculty_uuid"})
    )
    return meta
