# Deploy the app (Streamlit Community Cloud)

Get a **public URL** so judges can open the app in a browser with no install.

---

## Option A: Deploy from a repo that has only this app (recommended)

1. **Create a new GitHub repo** (e.g. `research-collab-recommender`) and push **only the contents of the `submission` folder** as the repo root:
   - At the top level you should have: `app.py`, `run_app.py`, `requirements.txt`, `README.md`, `src/`, `.streamlit/`, etc.
   - Do **not** put the submission folder itself inside the repo; its contents should be the root.

2. **Go to [share.streamlit.io](https://share.streamlit.io)** and sign in with GitHub.

3. **New app**
   - **Repository:** `your-username/research-collab-recommender`
   - **Branch:** `main` (or your default)
   - **Main file path:** `app.py`
   - **App URL:** choose a name (e.g. `research-collab-recommender`).

4. **Deploy.** The first run may take a few minutes (Sentence-BERT is downloaded once).

5. **No data on Cloud:** The deployed app will show “No dataset found” and **Try demo** unless you add a data file via Secrets (see below). For judging, **Try demo** is enough to explore the UI.

---

## Option B: Deploy from your existing “Case Competition” repo

If your GitHub repo has the app in a subfolder: `Case Competition/submission/app.py`.

1. **Copy to repo root** (so Streamlit Cloud finds them):
   - Copy `submission/requirements.txt` to the repo root.
   - Copy the `submission/.streamlit` folder to the repo root (Community Cloud expects `.streamlit/config.toml` at root).

2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. **Repository:** your repo (e.g. `your-username/Case-Competition`).
4. **Branch:** e.g. `main`.
5. **Main file path:** `submission/app.py`

6. **Deploy.** Python will find `src` because the script lives in `submission/`. The app will show “No dataset found” and **Try demo** unless you add `case_competition.csv` at the repo root (or set secrets). For judging, demo is enough.

---

## Optional: Use real data on Cloud

To load `case_competition.csv` in the cloud (e.g. for judges with data):

1. In the Streamlit Cloud app dashboard, open **Settings** → **Secrets**.
2. Add a secret that makes the file available. For example, you can put the CSV content in a secret and have the app write it to a temp file and point `CASE_COMP_DATA` at it, or use a URL and download in `data_load.py`.  
   The default app only reads from `CASE_COMP_DATA` or `case_competition.csv` in the working directory; for Cloud you’d need to extend the loader (e.g. read from a URL or from secrets). For “demo only” judging, **Try demo** is enough.

---

## After deploy

- Copy the app URL (e.g. `https://research-collab-recommender-xxx.streamlit.app`).
- Put it in **SUBMISSION.md** and in your submission email so judges use this **open-access URL**.
