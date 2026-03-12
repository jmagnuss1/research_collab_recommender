#!/usr/bin/env python3
"""
Entrypoint for the Research Collaboration Recommender.
Runs the Streamlit app. Usage:
  python run_app.py
  # or
  streamlit run app.py
"""
import sys
from pathlib import Path

# Ensure project root is on path so "src" and data files are found
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def main():
    import streamlit.web.cli as stcli
    sys.argv = ["streamlit", "run", str(ROOT / "app.py"), "--server.headless", "true"]
    stcli.main()

if __name__ == "__main__":
    main()
