"""
Research Collaboration Recommender — Streamlit UI.
Run: streamlit run app.py
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import streamlit as st

from src.data_load import DEFAULT_DATA_PATH, build_faculty_meta, prepare_clean_dataframe
from src.explain import build_data_quality_report
from src.features import build_or_load_publication_features, build_publication_features
from src.recommender import (
    DEFAULT_TOP_K,
    DEFAULT_WEIGHTS,
    build_full_recommendation_table,
    build_mode_artifacts,
    get_faculty_recommendations,
    paper_to_faculty_recommendations,
    recommendations_alternate_mode,
    sdg_to_faculty_lookup,
    topic_to_faculty_lookup,
)

st.set_page_config(
    page_title="Research Collaboration Recommender",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ----- UI constants (Illinois brand) -----
ILLINI_BLUE = "#13294B"
ILLINI_ORANGE = "#FF5F05"
CARD_BG = "#E8F0FE"
SPACER = "<div style='height: 1.25rem;'></div>"


def _card(html: str) -> str:
    return f"""
    <div style="background:{CARD_BG}; border:1px solid #D0E0F0; border-radius:10px; padding:1rem 1.25rem; margin:0.5rem 0;">
      {html}
    </div>
    """


# ----- Demo data (no file, in-memory only) -----
@st.cache_data(show_spinner="Building demo data…")
def make_demo_clean_df() -> pd.DataFrame:
    """Minimal synthetic dataset so the app can run without a real CSV."""
    rows = [
        {
            "person_uuid": "demo-person-1",
            "name": "Alex Chen",
            "email": "achen@illinois.edu",
            "department": "Finance",
            "article_uuid": "demo-paper-1",
            "title": "Sustainable investing and climate risk",
            "clean_abstract": "We study how climate risk is priced in sustainable investment portfolios.",
            "keyword_list": ["sustainability", "climate", "finance"],
            "keyword_str": "sustainability climate finance",
            "combined_text": "We study how climate risk is priced in sustainable investment portfolios. sustainability climate finance",
            "top 1": 13,
            "top 2": 7,
            "top 3": 12,
            "is_sustain": 1,
            "publication_year": 2023,
        },
        {
            "person_uuid": "demo-person-1",
            "name": "Alex Chen",
            "email": "achen@illinois.edu",
            "department": "Finance",
            "article_uuid": "demo-paper-2",
            "title": "ESG disclosure and firm value",
            "clean_abstract": "This paper examines the link between ESG disclosure and firm value.",
            "keyword_list": ["ESG", "disclosure", "firm value"],
            "keyword_str": "ESG disclosure firm value",
            "combined_text": "This paper examines the link between ESG disclosure and firm value. ESG disclosure firm value",
            "top 1": 12,
            "top 2": 9,
            "top 3": 13,
            "is_sustain": 1,
            "publication_year": 2022,
        },
        {
            "person_uuid": "demo-person-2",
            "name": "Jordan Smith",
            "email": "jsmith@illinois.edu",
            "department": "Accountancy",
            "article_uuid": "demo-paper-3",
            "title": "Carbon accounting and reporting standards",
            "clean_abstract": "We analyze carbon accounting standards and their impact on reporting quality.",
            "keyword_list": ["carbon", "accounting", "reporting"],
            "keyword_str": "carbon accounting reporting",
            "combined_text": "We analyze carbon accounting standards and their impact on reporting quality. carbon accounting reporting",
            "top 1": 13,
            "top 2": 12,
            "top 3": 9,
            "is_sustain": 1,
            "publication_year": 2023,
        },
    ]
    clean_df = pd.DataFrame(rows)
    vec = np.zeros(17, dtype=float)
    clean_df["sdg_vec"] = clean_df.apply(
        lambda r: np.array([1.0 if i + 1 in (r["top 1"], r["top 2"], r["top 3"]) else 0.0 for i in range(17)]),
        axis=1,
    )
    return clean_df


@st.cache_data(show_spinner="Loading demo…")
def load_demo_pipeline():
    """Run the full pipeline on demo data (no file, not cached to disk)."""
    clean_df = make_demo_clean_df()
    abs_emb, kw_mat, sdg_mat, tfidf, model = build_publication_features(clean_df)
    faculty_artifacts = build_mode_artifacts(clean_df, "faculty", abs_emb, kw_mat, sdg_mat, DEFAULT_WEIGHTS)
    paper_artifacts = build_mode_artifacts(clean_df, "paper", abs_emb, kw_mat, sdg_mat, DEFAULT_WEIGHTS)
    rec_df = build_full_recommendation_table(faculty_artifacts, top_k=DEFAULT_TOP_K)
    faculty_meta = build_faculty_meta(clean_df)
    quality_report = build_data_quality_report(clean_df)
    return {
        "clean_df": clean_df,
        "faculty_artifacts": faculty_artifacts,
        "paper_artifacts": paper_artifacts,
        "rec_df": rec_df,
        "faculty_meta": faculty_meta,
        "quality_report": quality_report,
        "tfidf": tfidf,
        "sbert_model": model,
    }


# ----- Cached pipeline (real data) -----
@st.cache_data(show_spinner="Loading data and building profiles…")
def load_pipeline(csv_path: str = DEFAULT_DATA_PATH):
    """Load clean data, build or load cached features, build faculty and paper artifacts."""
    clean_df = prepare_clean_dataframe(csv_path)
    abs_emb, kw_mat, sdg_mat, tfidf, model = build_or_load_publication_features(clean_df)
    faculty_artifacts = build_mode_artifacts(clean_df, "faculty", abs_emb, kw_mat, sdg_mat, DEFAULT_WEIGHTS)
    paper_artifacts = build_mode_artifacts(clean_df, "paper", abs_emb, kw_mat, sdg_mat, DEFAULT_WEIGHTS)
    rec_df = build_full_recommendation_table(faculty_artifacts, top_k=DEFAULT_TOP_K)
    faculty_meta = build_faculty_meta(clean_df)
    quality_report = build_data_quality_report(clean_df)
    return {
        "clean_df": clean_df,
        "faculty_artifacts": faculty_artifacts,
        "paper_artifacts": paper_artifacts,
        "rec_df": rec_df,
        "faculty_meta": faculty_meta,
        "quality_report": quality_report,
        "tfidf": tfidf,
        "sbert_model": model,
    }


def faculty_options(faculty_meta: pd.DataFrame) -> list[tuple[str, str]]:
    out = []
    meta = faculty_meta.drop_duplicates(subset=["faculty_uuid"]).reset_index(drop=True)
    name_counts = meta["name"].value_counts()
    for _, r in meta.iterrows():
        display = r["name"] if name_counts.get(r["name"], 0) <= 1 else f"{r['name']} ({r['department']})"
        out.append((display, r["faculty_uuid"]))
    return out


def paper_options(paper_artifacts) -> list[tuple[str, str]]:
    meta = paper_artifacts.metadata
    out = []
    for _, row in meta.iterrows():
        title = str(row.get("title", ""))[:70]
        if len(str(row.get("title", ""))) > 70:
            title += "…"
        year = row.get("publication_year", "")
        display = f"{title} ({year})" if year else title
        out.append((display, row["entity_id"]))
    return out


def ensure_flag_file(path: str = "flagged_issues.csv") -> None:
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write("timestamp,entity_id,entity_type,issue_type,notes\n")


def flag_issue(entity_id: str, entity_type: str, issue_type: str, notes: str = "") -> None:
    ensure_flag_file()
    ts = pd.Timestamp.utcnow().isoformat()
    line = f'"{ts}","{entity_id}","{entity_type}","{issue_type}","{notes.replace(chr(34), chr(39))}"\n'
    with open("flagged_issues.csv", "a", encoding="utf-8") as f:
        f.write(line)


# ----- Load pipeline or show no-data -----
pipe = None
use_demo = st.session_state.get("use_demo", False)

if use_demo:
    pipe = load_demo_pipeline()
else:
    try:
        pipe = load_pipeline()
    except Exception:
        pipe = None

if pipe is None and not use_demo:
    st.markdown(f"""
    <div style="background:{ILLINI_BLUE}; color:white; padding:20px; border-radius:10px; margin-bottom:20px;">
      <h2 style="margin:0 0 8px 0;">Research Collaboration Recommender · Illinois</h2>
      <p style="margin:0; font-size:15px;">Find faculty collaborators by research fit.</p>
    </div>
    """, unsafe_allow_html=True)
    no_data_html = (
        "<h3 style='margin:0 0 12px 0; color:#13294B;'>No dataset found</h3>"
        "<p style='margin:0 0 8px 0;'>To run the recommender with your own data, add a CSV file and start the app again.</p>"
        "<ul style='margin:8px 0 0 0; padding-left:1.25rem;'><li>Place your file at <code>case_competition.csv</code> in this project folder, or</li>"
        "<li>Set the environment variable <code>CASE_COMP_DATA</code> to your file path.</li></ul>"
        "<p style='margin:12px 0 0 0;'>See the README for required columns and format.</p>"
    )
    st.markdown(_card(no_data_html), unsafe_allow_html=True)
    st.markdown(SPACER, unsafe_allow_html=True)
    if st.button("Try demo", type="primary", help="Run the app with sample data so you can explore the interface"):
        st.session_state["use_demo"] = True
        st.rerun()
    st.stop()

clean_df = pipe["clean_df"]
faculty_artifacts = pipe["faculty_artifacts"]
paper_artifacts = pipe["paper_artifacts"]
rec_df = pipe["rec_df"]
faculty_meta = pipe["faculty_meta"]
quality_report = pipe["quality_report"]
tfidf = pipe["tfidf"]
sbert_model = pipe["sbert_model"]
is_demo = use_demo

faculty_choices = faculty_options(faculty_meta)
paper_choices = paper_options(paper_artifacts)
departments = sorted(faculty_meta["department"].dropna().unique().tolist())

# ----- Header -----
demo_badge = " — <span style='opacity:0.9;'>Demo mode: sample data only</span>" if is_demo else ""
st.markdown(f"""
<div style="background:{ILLINI_BLUE}; color:white; padding:18px 20px; border-radius:10px; margin-bottom:20px;">
  <h2 style="margin:0 0 6px 0;">Research Collaboration Recommender · Illinois</h2>
  <p style="margin:0; font-size:14px;">Find faculty collaborators by research fit — for students, donors, and leaders.{demo_badge}</p>
</div>
""", unsafe_allow_html=True)

# ----- Tabs -----
tab_landing, tab_explore, tab_paper, tab_alternatives, tab_leadership, tab_how = st.tabs([
    "Home",
    "Find collaborators",
    "Match by paper",
    "Compare ranking methods",
    "Leadership",
    "How it works",
])

# ----- Landing -----
with tab_landing:
    st.subheader("Choose how you’d like to explore")
    st.markdown(SPACER, unsafe_allow_html=True)
    landing_intro = (
        "<p style='margin:0 0 12px 0; color:#13294B; font-weight:600;'>Choose a starting point. We'll show faculty whose research best matches your interest.</p>"
        "<div style='display:flex; gap:1rem; flex-wrap:wrap; margin-top:12px;'>"
        "<span style='flex:1; min-width:140px;'><strong>Students</strong> — start with a topic or question.</span>"
        "<span style='flex:1; min-width:140px;'><strong>Donors & partners</strong> — start with a sustainability goal.</span>"
        "<span style='flex:1; min-width:140px;'><strong>Leaders</strong> — browse by department and review the leadership view.</span>"
        "</div>"
    )
    st.markdown(_card(landing_intro), unsafe_allow_html=True)
    st.markdown(SPACER, unsafe_allow_html=True)

    # Student: topic-based search
    st.markdown("---")
    st.subheader("Student: explore faculty by topic")
    st.caption("Type a topic or question and we’ll suggest faculty whose work is closest.")
    default_topic = st.session_state.get("home_topic", "sustainability")
    topic = st.text_input(
        "Example: climate risk in finance",
        value=default_topic,
        key="home_topic",
    )
    run_topic = st.button("Find matching faculty", key="home_topic_btn")
    if run_topic:
        st.session_state["home_topic_last"] = topic.strip()

    topic_last = st.session_state.get("home_topic_last", "").strip()
    if topic_last:
        with st.spinner("Finding faculty who match your topic…"):
            try:
                df = topic_to_faculty_lookup(topic_last, faculty_artifacts, sbert_model, tfidf, DEFAULT_WEIGHTS, top_k=10)
                st.subheader(f"Faculty whose research matches “{topic_last}”")
                if not df.empty:
                    st.caption("Here are your top matches.")
                    # Single high-level explanation instead of repeating per recommendation
                    st.markdown(_card(
                        "<p style='margin:0 0 4px 0; color:#13294B;'>We highlight faculty whose publication summaries and keywords are closest to your topic. Use <em>Find collaborators</em> for a deeper dive into any person.</p>"
                    ), unsafe_allow_html=True)
                    show = df[["name", "department", "topic_similarity", "matched_keywords"]].copy()
                    show.columns = ["Name", "Department", "Research fit", "Matched keywords"]
                    st.dataframe(
                        show.style.format({"Research fit": "{:.3f}"}),
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Name": st.column_config.TextColumn("Name", help="Researcher's full name"),
                            "Department": st.column_config.TextColumn("Department", help="Academic department"),
                            "Research fit": st.column_config.NumberColumn("Research fit", help="How closely this researcher's work matches your topic (0–1)."),
                            "Matched keywords": st.column_config.TextColumn("Matched keywords", help="Keywords from their work that overlap with your search."),
                        },
                    )
                else:
                    st.info("No matches yet — try a different topic.")
            except Exception as e:
                st.error(f"We couldn't run that search. {e}")

    # Donor: SDG-based search
    st.markdown(SPACER, unsafe_allow_html=True)
    st.markdown("---")
    st.subheader("Donor: explore by sustainability goal")
    st.caption("Choose a UN Sustainable Development Goal to see aligned faculty.")
    sdg_labels = {
        1: "No poverty", 2: "Zero hunger", 3: "Good health", 4: "Quality education",
        5: "Gender equality", 6: "Clean water", 7: "Affordable energy", 8: "Decent work",
        9: "Industry & innovation", 10: "Reduced inequality", 11: "Sustainable cities",
        12: "Responsible consumption", 13: "Climate action", 14: "Life below water",
        15: "Life on land", 16: "Peace & justice", 17: "Partnerships",
    }
    cols = st.columns(3)
    for i, (sdg_id, label) in enumerate(sdg_labels.items()):
        with cols[i % 3]:
            if st.button(
                f"Goal {sdg_id}: {label}",
                key=f"home_sdg_{sdg_id}",
                use_container_width=True,
                help=f"UN Sustainable Development Goal {sdg_id}: {label}",
            ):
                st.session_state["home_sdg_last"] = sdg_id

    selected_sdg = st.session_state.get("home_sdg_last")
    if selected_sdg:
        with st.spinner("Finding faculty by sustainability goal…"):
            try:
                df = sdg_to_faculty_lookup(selected_sdg, faculty_artifacts, top_k=10)
                st.subheader(f"Faculty aligned with Goal {selected_sdg}: {sdg_labels.get(selected_sdg, '')}")
                if not df.empty:
                    st.caption("We’ll show researchers whose work aligns with this goal.")
                    st.markdown(_card(
                        "<p style='margin:0 0 4px 0; color:#13294B;'>We highlight faculty whose SDG tags and topics are most closely aligned with this goal. Use <em>Find collaborators</em> for more detail on any person.</p>"
                    ), unsafe_allow_html=True)
                    show = df[["name", "department", "sdg_similarity", "matched_sdg_tags"]].copy()
                    show.columns = ["Name", "Department", "Goal alignment", "Matched SDGs"]
                    st.dataframe(
                        show.style.format({"Goal alignment": "{:.3f}"}),
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Name": st.column_config.TextColumn("Name", help="Researcher's full name"),
                            "Department": st.column_config.TextColumn("Department", help="Academic department"),
                            "Goal alignment": st.column_config.NumberColumn("Goal alignment", help="How closely this researcher's work aligns with the selected goal (0–1)."),
                            "Matched SDGs": st.column_config.TextColumn("Matched SDGs", help="SDG numbers that appear in both the researcher's work and your selection."),
                        },
                    )
                else:
                    st.info("No matches yet — try another goal.")
            except Exception as e:
                st.error(f"We couldn't run that search. {e}")

    st.markdown(SPACER, unsafe_allow_html=True)
    home_evidence = "<p style='margin:0; color:#13294B;'><strong>Evidence & data</strong> — We use publication abstracts, keywords, and sustainability (SDG) tags. You can see data sources and confidence for each profile under <strong>Find collaborators</strong> and in the <strong>Leadership</strong> view.</p>"
    st.markdown(_card(home_evidence), unsafe_allow_html=True)

# ----- Explore (Find collaborators) -----
with tab_explore:
    st.subheader("Find collaborators")
    st.caption("Choose a faculty member to see who researches similar topics — and why we recommend them.")

    search_mode = st.radio("Search by", ["Faculty name", "Department"], horizontal=True, label_visibility="collapsed")
    if search_mode == "Faculty name":
        choice = st.selectbox(
            "Choose a faculty member",
            options=[d for d, _ in faculty_choices],
            key="explore_faculty",
            help="We'll show their top similar colleagues and why they match",
        )
        if choice:
            _, person_uuid = next((d, u) for d, u in faculty_choices if d == choice)
            with st.spinner("Loading recommendations…"):
                rec = get_faculty_recommendations(person_uuid, faculty_artifacts, top_k=DEFAULT_TOP_K)
            if rec.empty:
                st.warning("We don't have recommendations for this person yet.")
            else:
                row_meta = faculty_meta[faculty_meta["faculty_uuid"] == person_uuid].iloc[0]
                st.markdown(f"**{row_meta['name']}** — {row_meta['department']}")
                st.markdown(SPACER, unsafe_allow_html=True)
                st.caption("Here are your top matches. Expand any row to see why we recommend them.")
                top = rec.head(10)
                for _, r in top.iterrows():
                    exp_title = f"{r['recommended_name']} — {r['recommended_department']} · {r['similarity_score']:.2f} match"
                    with st.expander(exp_title):
                        st.markdown("**Why this match stands out**")
                        st.write(r.get("contribution_summary", ""))
                        st.write("**Shared keywords:**", ", ".join(r.get("shared_keywords", []) or ["—"]))
                        st.write("**Shared SDGs:**", ", ".join(map(str, r.get("shared_sdg_tags", []) or [])) or "—")
                        a = float(r.get("abstract_component", 0.0))
                        k = float(r.get("keyword_component", 0.0))
                        s = float(r.get("sdg_component", 0.0))
                        total = a + k + s or 1e-6
                        a_pct = a / total * 100.0
                        k_pct = k / total * 100.0
                        s_pct = s / total * 100.0
                        st.write(
                            "**What drove this recommendation:** "
                            f"Publication summaries {a_pct:.0f}% · Keywords {k_pct:.0f}% · Sustainability goals {s_pct:.0f}%"
                        )

                st.markdown(SPACER, unsafe_allow_html=True)
                st.markdown("---")
                st.subheader("Evidence and confidence")
                evidence_card = "<p style='margin:0 0 8px 0; color:#13294B;'>How we rate this profile and where the data comes from.</p>"
                st.markdown(_card(evidence_card), unsafe_allow_html=True)
                qr = quality_report[quality_report["faculty_uuid"] == person_uuid]
                if not qr.empty:
                    q = qr.iloc[0]
                    miss_abs_pct = q["missing_abstract_rate"] * 100.0
                    miss_kw_pct = q["missing_keyword_rate"] * 100.0
                    pub_count = int(q["publication_count"])
                    conf = q["confidence_score"]
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Publications used", pub_count)
                    c2.metric("Missing abstracts", f"{miss_abs_pct:.1f}%")
                    c3.metric("Missing keywords", f"{miss_kw_pct:.1f}%")
                    st.write("**Data we use:** Abstract text, keywords, and SDG tags from publications.")
                    st.write("**How SDG tags are set:** From the dataset fields top 1 / top 2 / top 3 per publication.")
                    if conf == "High":
                        interp = "High confidence: this profile is based on several publications with very little missing detail."
                    elif conf == "Medium":
                        interp = "Medium confidence: there is enough evidence to recommend, but some publication details are missing."
                    else:
                        interp = "Low confidence: results rely on a smaller set of publications or have more missing detail."
                    st.write(f"**Confidence:** {conf}. {interp}")
                    st.markdown("**If something looks wrong:** Use the form below to flag this record (e.g. incorrect profile, missing data, wrong SDG). Your report is saved for follow-up.")
                with st.expander("Flag an issue with this profile"):
                    with st.form("flag_form"):
                        issue_type = st.selectbox("Issue type", ["Incorrect profile", "Missing data", "Wrong SDG", "Other"], key="flag_type")
                        notes = st.text_area("Notes (optional)", key="flag_notes")
                        if st.form_submit_button("Flag this record"):
                            flag_issue(person_uuid, "faculty", issue_type, notes)
                            st.success("Thanks — we've recorded your feedback.")
                st.download_button("Export this list to CSV", top[["recommended_name", "recommended_department", "similarity_score", "contribution_summary"]].to_csv(index=False).encode("utf-8"), file_name="recommendations.csv", mime="text/csv")
    else:
        dept = st.selectbox("Department", departments, key="explore_dept")
        dept_faculty = faculty_meta[faculty_meta["department"] == dept].drop_duplicates("faculty_uuid")
        if dept_faculty.empty:
            st.warning("No faculty in this department.")
        else:
            st.caption("Top recommendations for each person in this department.")
            for _, row in dept_faculty.head(10).iterrows():
                pid = row["faculty_uuid"]
                rec = get_faculty_recommendations(pid, faculty_artifacts, top_k=5)
                st.markdown(f"**{row['name']}**")
                if not rec.empty:
                    st.dataframe(rec[["recommended_name", "recommended_department", "similarity_score"]].rename(columns={"recommended_name": "Recommended", "recommended_department": "Dept", "similarity_score": "Score"}), use_container_width=True, hide_index=True)
                st.markdown("")

# ----- By paper -----
with tab_paper:
    st.subheader("Match by paper")
    st.caption("Pick a publication — we'll show which faculty are closest to that paper's topics.")
    if not paper_choices:
        st.warning("No papers in the dataset.")
    else:
        title_choice = st.selectbox("Choose a paper (by title)", [t for t, _ in paper_choices], key="paper_select")
        if title_choice:
            _, article_uuid = next((t, u) for t, u in paper_choices if t == title_choice)
            with st.spinner("Finding faculty closest to this paper…"):
                paper_rec = paper_to_faculty_recommendations(article_uuid, faculty_artifacts, paper_artifacts, top_k=10)
            if paper_rec.empty:
                st.warning("We don't have faculty recommendations for this paper yet.")
            else:
                st.subheader("Faculty closest to this paper")
                st.caption("Here are your top matches. Expand a row to see why.")
                for _, r in paper_rec.head(10).iterrows():
                    exp_title = f"{r['recommended_name']} — {r['recommended_department']} · {r['similarity_score']:.2f} match"
                    with st.expander(exp_title):
                        st.write(r.get("contribution_summary", ""))
                        st.write("**Shared keywords:**", ", ".join(r.get("shared_keywords", []) or ["—"]))
                        st.write("**Shared SDGs:**", ", ".join(map(str, r.get("shared_sdg_tags", []) or [])) or "—")
                        a = float(r.get("abstract_component", 0.0))
                        k = float(r.get("keyword_component", 0.0))
                        s = float(r.get("sdg_component", 0.0))
                        total = a + k + s or 1e-6
                        a_pct = a / total * 100.0
                        k_pct = k / total * 100.0
                        s_pct = s / total * 100.0
                        st.write(
                            "**What drove this recommendation:** "
                            f"Publication summaries {a_pct:.0f}% · Keywords {k_pct:.0f}% · Sustainability goals {s_pct:.0f}%"
                        )
                st.markdown(SPACER, unsafe_allow_html=True)
                st.markdown("---")
                st.subheader("About this paper")
                paper_card = "<p style='margin:0; color:#13294B;'>Details we use to match this paper to faculty.</p>"
                st.markdown(_card(paper_card), unsafe_allow_html=True)
                paper_meta = paper_artifacts.metadata[paper_artifacts.metadata["entity_id"] == article_uuid].iloc[0]
                st.write("**Title:**", paper_meta.get("title", ""))
                st.write("**Year:**", paper_meta.get("publication_year", ""))
                sub = clean_df[clean_df["article_uuid"] == article_uuid]
                if not sub.empty:
                    row = sub.iloc[0]
                    abst = (row.get("clean_abstract") or "")[:300]
                    if len(str(row.get("clean_abstract", ""))) > 300:
                        abst += "…"
                    st.write("**Abstract (excerpt):**", abst or "—")
                    st.write("**Keywords:**", ", ".join(row.get("keyword_list", []) or []))
                    st.write("**SDG tags:**", row.get("top 1", ""), row.get("top 2", ""), row.get("top 3", ""))

# ----- Alternatives -----
with tab_alternatives:
    st.subheader("Compare ranking methods")
    st.caption("See how recommendations change when we use only one signal (e.g. abstract meaning only) instead of all three.")
    alt_faculty = st.selectbox("Choose a faculty member to compare", [d for d, _ in faculty_choices], key="alt_faculty")
    if alt_faculty:
        _, person_uuid = next((d, u) for d, u in faculty_choices if d == alt_faculty)
        mode = st.radio("Ranking mode", ["Default (all signals)", "Publication summaries only", "Keywords only (TF-IDF)", "SDG only"], key="alt_mode")
        if mode == "Default (all signals)":
            rec = get_faculty_recommendations(person_uuid, faculty_artifacts, top_k=10)
            st.caption("Top 10 using our combined score (abstract + keywords + SDG).")
        elif mode == "Publication summaries only":
            rec = recommendations_alternate_mode(person_uuid, faculty_artifacts, "bert_only", top_k=10)
            st.caption("Top 10 by abstract similarity only.")
        elif mode == "Keywords only (TF-IDF)":
            rec = recommendations_alternate_mode(person_uuid, faculty_artifacts, "tfidf_only", top_k=10)
            st.caption("Top 10 by keyword overlap only.")
        else:
            rec = recommendations_alternate_mode(person_uuid, faculty_artifacts, "sdg_only", top_k=10)
            st.caption("Top 10 by SDG alignment only.")
        if not rec.empty:
            cols = [c for c in ["recommended_name", "recommended_department", "similarity_score"] if c in rec.columns]
            st.dataframe(rec[cols].head(10), use_container_width=True, hide_index=True)
    st.markdown(SPACER, unsafe_allow_html=True)
    st.markdown("---")
    how_card = (
        "<p style='margin:0 0 8px 0; color:#13294B;'><strong>Weights:</strong> By default we lean most on publication summaries, then keywords, then SDG tags (roughly 60/30/10). You can see how rankings change when each piece is used on its own.</p>"
        "<p style='margin:0; color:#13294B;'><strong>How similarity works:</strong> We compare how similar two research profiles are across publication summaries, keywords, and SDG tags. Higher scores mean a closer research fit.</p>"
    )
    st.markdown(_card(how_card), unsafe_allow_html=True)

# ----- Leadership -----
with tab_leadership:
    st.subheader("Leadership summary")
    st.caption("High-level view for resource allocation and storytelling.")
    st.markdown(SPACER, unsafe_allow_html=True)
    # Most connected faculty (ranked table)
    freq = rec_df.groupby("recommended_uuid").size().sort_values(ascending=False)
    top_freq = freq.head(10)
    rows = []
    for uid, count in top_freq.items():
        m = faculty_meta[faculty_meta["faculty_uuid"] == uid]
        if not m.empty:
            name = m.iloc[0]["name"]
            dept = m.iloc[0]["department"]
        else:
            name = str(uid)
            dept = ""
        rows.append({"Name": name, "Department": dept, "Times recommended": int(count)})
    if rows:
        st.markdown("**Most connected faculty**")
        st.caption("Faculty who appear most often in others' top recommendation lists.")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.markdown(SPACER, unsafe_allow_html=True)
    dept_matrix = rec_df.pivot_table(index="department", columns="recommended_department", values="similarity_score", aggfunc="mean").fillna(0)
    st.markdown("**Cross-department similarity (average)**")
    st.caption("Higher values mean faculty in these departments tend to publish on more similar topics.")
    st.dataframe(dept_matrix.style.format("{:.3f}"), use_container_width=True, hide_index=True)
    st.markdown(SPACER, unsafe_allow_html=True)
    from collections import Counter
    sdg_counts = Counter()
    for col in ["top 1", "top 2", "top 3"]:
        for v in clean_df[col].dropna():
            try:
                if 1 <= int(v) <= 17:
                    sdg_counts[int(v)] += 1
            except (ValueError, TypeError):
                pass
    st.markdown("**Top SDGs in this dataset**")
    sdg_rows = [{"Goal": f"SDG {k}", "Publications tagged": v} for k, v in sdg_counts.most_common(10)]
    if sdg_rows:
        st.dataframe(pd.DataFrame(sdg_rows), use_container_width=True, hide_index=True)

    # Simple collaboration clusters based on similarity threshold
    sim = faculty_artifacts.similarity_matrix
    ids = faculty_artifacts.entity_ids
    threshold = 0.72
    visited = set()
    cluster_sizes = []
    for i in range(len(ids)):
        if i in visited:
            continue
        stack = [i]
        comp = set()
        while stack:
            node = stack.pop()
            if node in comp:
                continue
            comp.add(node)
            visited.add(node)
            neighbors = np.where(sim[node] >= threshold)[0].tolist()
            for n in neighbors:
                if n != node and n not in comp:
                    stack.append(n)
        cluster_sizes.append(len(comp))
    if cluster_sizes:
        cluster_sizes_sorted = sorted(cluster_sizes, reverse=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Collaboration clusters", len(cluster_sizes_sorted))
        c2.metric("Largest cluster size", cluster_sizes_sorted[0])
        avg_size = sum(cluster_sizes_sorted) / len(cluster_sizes_sorted)
        c3.metric("Average cluster size", f"{avg_size:.1f}")

    summary_data = [
        {"Metric": "Most connected faculty (names)", "Value": ", ".join([r["Name"] for r in rows]) if rows else ""},
        {"Metric": "Top SDGs (counts)", "Value": ", ".join([f"SDG{k}({v})" for k, v in sdg_counts.most_common(10)])},
    ]
    st.markdown(SPACER, unsafe_allow_html=True)
    st.download_button("Save leadership summary as CSV", pd.DataFrame(summary_data).to_csv(index=False).encode("utf-8"), file_name="leadership_summary.csv", mime="text/csv", key="lead_export")

# ----- How it works -----
with tab_how:
    st.subheader("How recommendations work")
    st.markdown("""
    We combine **three pieces of evidence** to match researchers:

    - **Publication summaries** — Short descriptions of each paper, so we can understand topics in plain language.
    - **Keywords** — Author-provided keywords that highlight specific methods or themes.
    - **Sustainability goals (SDGs)** — Up to three SDG tags per publication that describe impact areas.

    We compare how similar two research profiles are across these three pieces. Profiles that line up on summaries, keywords, and goals rise to the top of the list.
    """)
    st.markdown(SPACER, unsafe_allow_html=True)
    st.subheader("Data and confidence")
    st.markdown("Data comes from your own publication records (abstracts, keywords, SDG tags). Confidence is **High** when we have many publications and few missing abstracts/keywords; **Low** when data is sparse. If something looks wrong, use **Flag this record** in the Find collaborators tab. Your data is read locally and is not shared with us.")
    st.markdown(SPACER, unsafe_allow_html=True)
    st.subheader("Alternatives we tried")
    st.markdown("We tried using only one source of evidence at a time (publication summaries only, keywords only, SDG tags only) and different weight splits. You can compare these in the **Compare ranking methods** tab.")
