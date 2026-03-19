# Case Competition — Final Submission Cover Note

**What we’re submitting and where each item can be accessed:**

| Item | Where to access |
|------|-----------------|
| **Prototype (web app)** | **Local run** — We are providing the full working app code in this submission bundle. See the “How to run locally” steps below. |
| **Presentation slides** | **PDF attached to submission email.** |
| **Source code / GitHub**  | (https://github.com/jmagnuss1/research_collab_recommender/blob/main/SUBMISSION.md) |

**How to run locally (if you have the code):**

1. Download the repo and open a terminal in the folder that contains `app.py`.
2. Install dependencies: `pip install -r requirements.txt`
3. Start the app:
   - If your CSV is named `case_competition.csv` and placed next to `app.py`, run: `streamlit run app.py`
   - Otherwise, set `CASE_COMP_DATA` to your CSV path and run:
     - `export CASE_COMP_DATA="/path/to/case_competition.csv"`
     - `streamlit run app.py`
4. Open the URL shown (e.g. http://localhost:8501).

**One-line summary:**  
We are submitting a Research Collaboration Recommender for Illinois: a web app that helps students, donors, and leaders find faculty by topic, SDG, or paper, with explainability, evidence panels, and leadership views.

**Deployment note (privacy):**  
This app is straightforward to deploy as a web link, but we intentionally did not publish a public, open-access instance because the dataset includes school employee / faculty information.
