# Research Collaboration Recommender

A **public-facing app** that recommends faculty collaborators using a hybrid similarity model: Sentence-BERT (abstracts), TF-IDF (keywords), and SDG vectors (sustainability). Built for **students, donors, administrators, and peer institutions** — clean UI, no jargon, with explainability and evidence.

---

## Judge quick start

**Local run (one command from this folder):**

```bash
pip install -r requirements.txt && streamlit run app.py
```

Then open the URL shown (e.g. http://localhost:8501).

**Note on deployment (privacy):** This app is straightforward to deploy as a public web link, but we intentionally did not publish an open-access instance because the dataset includes school employee / faculty information.

---

## What the app does

- **Find collaborators by faculty name or department** — See top similar researchers with “Why this match” (shared keywords, SDGs, signal breakdown).
- **Student flow:** “Find a sustainability mentor” — Enter a topic (e.g. sustainability, renewable energy) and get faculty whose work matches.
- **Donor flow:** “Find projects to fund” — Browse by UN Sustainable Development Goal (SDG) and see aligned researchers.
- **By paper:** Pick a publication by title and see which faculty are closest to that paper’s topics.
- **Evidence & confidence:** For each profile, see data sources, missingness (abstracts/keywords/SDGs), publication count, and a High/Medium/Low confidence label. **Flag/correct** flow writes to `flagged_issues.csv`.
- **Alternatives considered:** Compare default ranking with BERT-only, TF-IDF-only, and SDG-only rankings.
- **Leadership:** Most connected faculty, cross-department similarity, top SDGs, with CSV export.

**Departments included:** Finance, Accountancy, Business Administration.

---

## Setup

1. **Clone or download** this repo and open a terminal in the project folder.

2. **Create a virtual environment** (recommended):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   First run may download the Sentence-BERT model (requires internet).

4. **Data:** Place `case_competition.csv` in the project root, or set:
   ```bash
   export CASE_COMP_DATA=/path/to/your/data.csv
   ```

---

## How to run

From the **project root** (the folder that contains `app.py`):

```bash
streamlit run app.py
```

Or run the entrypoint script:

```bash
python run_app.py
```

You can also double-click `run_app.py` (or run `python run_app.py` from this folder in a terminal) to start the app. Then open the URL shown (e.g. http://localhost:8501).

**No data file?** If you don't have a CSV yet, the app will show a short message and a **Try demo** button so you can explore the interface with sample data.

**Caching:** Embeddings are cached under `.cache/`. Delete `.cache/` to force a full rebuild after data changes.

---

## Data use and sharing

- Faculty and publication data may be subject to **institutional or campus policy**. Do not redistribute real datasets without checking with your institution. See [DATA_USE.md](DATA_USE.md) for more.
- **This repo does not include real faculty data.** You supply your own CSV.

**Sharing this app:** To share with others, zip this folder (or share the repo) **without** including `case_competition.csv` or any real data. Recipients install dependencies and add their own data file.

---

## How we address judge feedback

| Feedback | Where it’s addressed |
|--------|------------------------|
| **Personas & journey maps** | **Home** tab: “Find a sustainability mentor” (Student) and “Find projects to fund” (Donor) run real topic/SDG searches and show results. **Find collaborators** supports both flows. |
| **Explainability** | Each recommendation has expandable “Why this match”: shared keywords, shared SDG tags, abstract/keyword/SDG signal breakdown, and a plain-English summary. **How it works** tab explains the model in simple language. |
| **Evidence & data quality** | **Evidence & confidence** panel (in Find collaborators): data sources, how SDG tags are assigned, missingness %, publication count, confidence level. **“What to do if something looks wrong”** → Flag form that appends to `flagged_issues.csv`. |
| **Alternatives considered** | **Alternatives** tab: BERT-only, TF-IDF-only, SDG-only rankings plus short explanation of weights and cosine similarity. |
| **Leadership layer** | **Leadership** tab: most connected faculty, cross-department similarity table, top SDGs. **Export** button for leadership summary CSV. |
| **User-friendly** | No UUIDs in the UI; faculty and papers are chosen by **name** (and department or year for disambiguation). Plain-language labels, tooltips, and sensible defaults. |

---

---

## Project layout

- `app.py` — Streamlit UI (landing, explore, by paper, alternatives, leadership, how it works).
- `run_app.py` — Entrypoint: `python run_app.py` runs the app.
- `src/data_load.py` — Load CSV, clean, filter departments, build faculty meta.
- `src/features.py` — Publication-level embeddings (SBERT + TF-IDF + SDG) with disk cache.
- `src/recommender.py` — Faculty/paper profiles, similarity matrices, recommendations, topic/SDG search, paper→faculty, alternate modes.
- `src/explain.py` — “Why this match” explanations and data quality report.
- `case_competition.csv` — Your input data (optional; place in project root or set `CASE_COMP_DATA`). Not included in the repo.
- `flagged_issues.csv` — Created when users flag a record (timestamp, entity_id, type, notes). See `.gitignore`.

---

## Notes

- **No paid services** — Sentence-BERT and TF-IDF run locally; first run downloads the model once.
- **Jupyter:** The original ipywidgets dashboard is in `CaseCompExpanded1.py`; it expects a `src/` package and will work once you run from the same environment. The **recommended way to run** is Streamlit: `streamlit run app.py`.
