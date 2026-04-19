# Research Collaboration Recommender

A web app that helps students, donors, and university administrators find faculty researchers by topic, sustainability goal, or publication — with plain-English explanations of every match.

## Quick Start

You'll need Python 3 installed. If you don't have it, download it from python.org before continuing.

1. Download this project as a ZIP file and unzip it somewhere easy to find (your Desktop works well).

2. Open Terminal. On Mac, press Cmd+Space, type "Terminal", and press Enter. On Windows, open PowerShell.

3. Navigate to the unzipped project folder. The easiest way: open Terminal, type `cd ` (with a space after), drag the project folder into the Terminal window, and press Enter.

4. Run these commands, one at a time:

```bash
python3 -m venv .venv
```
```bash
source .venv/bin/activate
```
```bash
pip install -r requirements.txt
```
```bash
streamlit run app.py
```

5. Open your browser to http://localhost:8501

The first run downloads a language model (about 90MB) and caches it — this only happens once.

## What You'll See

The app is built around four distinct user flows.

Finding a collaborator. Search by faculty name and the app returns their closest research neighbors from across Finance, Accountancy, and Business Administration. Each result comes with a "Why this match" panel showing shared keywords, SDG overlap, and an abstract similarity score — so the ranking is never a black box.

Student mentor search. Type a topic like "renewable energy" or "behavioral economics" and get a ranked list of faculty whose published work is most relevant. You don't need to know any names or departments going in.

Donor SDG explorer. Browse by UN Sustainable Development Goal to find researchers whose work aligns with your funding priorities. Each result shows the faculty member's top SDGs and the specific publications that drove the match.

Paper-to-faculty matching. Pick any publication by title and find other faculty whose work is closest to that paper's themes — useful for spotting collaboration opportunities that wouldn't show up on an org chart.

Leadership view. A summary layer with the most-connected researchers across departments, a cross-department similarity table, and top SDGs across the college. Everything exports to CSV.

## How It Works

The recommender blends three signals. Sentence-BERT encodes publication abstracts into dense semantic vectors, capturing meaning beyond keyword overlap. TF-IDF weighs the terms that are most distinctive to each researcher's body of work. SDG vectors map research to the 17 UN Sustainable Development Goals using a curated keyword taxonomy. The final similarity score is a weighted combination of all three, and an Alternatives tab lets you compare the default ranking against any single signal so you can see what each approach contributes.

Embeddings are cached after the first run under `.cache/`. Delete that folder if you replace or update the data file.

## Data and Privacy

The repo includes no faculty data — you supply your own `case_competition.csv` in the project root. Faculty records may be subject to institutional policy; do not redistribute real datasets without checking with your institution first.

If something in the app looks wrong, every faculty profile has a Flag form that appends a timestamped note to `flagged_issues.csv` for later review.
