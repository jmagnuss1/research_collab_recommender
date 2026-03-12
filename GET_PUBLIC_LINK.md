# Get one link for judges (no code, no install)

Judges will only need to **click one link** and use the app in their browser. You do the setup once; they never open any folder or code.

**Shortcut:** From the **Case Competition** folder (the one that contains `submission`), you can run `./prepare_deploy_repo.sh` in Terminal. It creates a folder with all the right files so you don’t miss anything. Then do Part 2 (create repo on GitHub) and Part 3B (push from Terminal) and Part 4 (Streamlit) below.

---

## What you’re going to do (big picture)

1. Put the app code in a **GitHub repository** (so Streamlit can see it).
2. Use **Streamlit Community Cloud** to turn that repo into a **public webpage**.
3. Send judges that **webpage URL** in your submission email.

---

## Part 1: Create a GitHub account (if you don’t have one)

1. Go to **https://github.com**
2. Click **Sign up** and create a free account (email, password, username).
3. Verify your email if GitHub asks you to.

---

## Part 2: Create a new repository on GitHub

1. Log in to GitHub.
2. Click the **+** icon (top right) → **New repository**.
3. Fill in:
   - **Repository name:** e.g. `research-collab-recommender` (no spaces).
   - **Description:** optional, e.g. “Research collaboration recommender for Illinois.”
   - **Public** is selected.
   - **Do not** check “Add a README file” (we already have one).
4. Click **Create repository**.

5. You’ll see a page that says “Quick setup” and shows a URL like:
   `https://github.com/YOUR-USERNAME/research-collab-recommender.git`
   **Keep this page open** — you’ll need your repo URL in Part 3.

---

## Part 3: Put the app code into that repository

The repo must contain **only the contents of the `submission` folder** (not the folder itself). Easiest: use a new folder and copy everything in.

### Step A: Make a folder that’s ready to push

1. On your Mac, open **Finder**.
2. Go to your **Desktop** (or any place you like).
3. Create a **new folder** and name it the same as your repo, e.g. `research-collab-recommender`.
4. Open your **Case Competition** project and then the **submission** folder inside it.
5. **Select everything inside `submission`** (all files and folders):
   - `app.py`
   - `run_app.py`
   - `requirements.txt`
   - `README.md`
   - `DATA_USE.md`
   - `DEPLOY.md`
   - `SUBMISSION.md`
   - `GET_PUBLIC_LINK.md`
   - `.gitignore`
   - `src` (entire folder)
   - `.streamlit` (entire folder — you may need to show hidden files: **Finder → View → Show View Options → Show hidden files**, or press **Cmd + Shift + .**)
6. **Copy** (Cmd+C), then open your new folder `research-collab-recommender` and **Paste** (Cmd+V).  
   So inside `research-collab-recommender` you have: `app.py`, `run_app.py`, `src/`, `.streamlit/`, etc. **No** “submission” folder inside it.

### Step B: Push this folder to GitHub from Terminal

1. Open **Terminal** (Spotlight: Cmd+Space, type “Terminal”, Enter).
2. Go into your new folder (use your real path and folder name):
   ```bash
   cd ~/Desktop/research-collab-recommender
   ```
   If you put the folder somewhere else, use that path instead.

3. Turn it into a git repo and push (replace `YOUR-USERNAME` and `research-collab-recommender` if you used a different repo name):
   ```bash
   git init
   git add .
   git commit -m "Initial commit - Research Collaboration Recommender"
   git branch -M main
   git remote add origin https://github.com/YOUR-USERNAME/research-collab-recommender.git
   git push -u origin main
   ```
   When it asks for credentials, use your GitHub username and a **Personal Access Token** (not your GitHub password). To create one: GitHub → **Settings** → **Developer settings** → **Personal access tokens** → **Tokens (classic)** → **Generate new token**. Give it a name, check **repo**, generate, then copy the token and paste it when Terminal asks for a password.

4. Refresh your repo page on GitHub. You should see all the files (`app.py`, `src/`, `requirements.txt`, etc.).

---

## Part 4: Deploy the app so it gets a public URL

1. Go to **https://share.streamlit.io**
2. Click **Sign in with GitHub** and authorize Streamlit to use your GitHub account.
3. Click **New app**.
4. Fill in:
   - **Repository:** select `YOUR-USERNAME/research-collab-recommender`
   - **Branch:** `main`
   - **Main file path:** type `app.py`
   - **App URL:** you can leave the default (e.g. `research-collab-recommender`) or change it.
5. Click **Deploy**.

6. Wait a few minutes. The first time it has to download the model (Sentence-BERT); later deploys are faster. If you see errors, check that **Main file path** is exactly `app.py` and that your repo has `app.py` at the top level.
7. When it says “Your app is live”, click the URL (e.g. `https://research-collab-recommender-xxxxx.streamlit.app`). That’s your **one link**.

---

## Part 5: Give judges only that link

- In your submission email, write something like:  
  **“You can try the app here (no install required): [paste your Streamlit URL]”**
- Judges open the link in a browser. They’ll see “No dataset found” and a **Try demo** button — they click **Try demo** and use the app. They never need to open any folder or code.

---

## Quick checklist

- [ ] GitHub account created
- [ ] New repo created (e.g. `research-collab-recommender`), Public, no README
- [ ] New folder on Desktop (or elsewhere) with **contents** of `submission` copied in
- [ ] Terminal: `git init`, `git add .`, `git commit`, `git remote add origin`, `git push`
- [ ] share.streamlit.io → Sign in with GitHub → New app → pick repo, branch `main`, Main file `app.py` → Deploy
- [ ] Copy the live app URL and put it in your submission email and in SUBMISSION.md

If anything in Part 3 or 4 fails (e.g. “repository not found” or “main file not found”), double-check: repo name and username in the URL, and that `app.py` and `requirements.txt` are at the **root** of the repo (not inside a `submission` folder on GitHub).
