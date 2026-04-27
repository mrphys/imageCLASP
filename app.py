import sys
from pathlib import Path
import pandas as pd
import requests
import shutil
from platformdirs import user_data_dir

import streamlit as st


def resource_path(relative_path: str) -> Path:
    base_path = Path(getattr(sys, "_MEIPASS", Path.cwd()))
    return base_path / relative_path


APP_NAME = "ImageCLASP"
APP_AUTHOR = "ImageCLASP"

DATA_DIR = Path(user_data_dir(APP_NAME, APP_AUTHOR))
DATA_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------
# Copy bundled .streamlit config (read-only -> writable)
# ---------------------------
base_path = Path(getattr(sys, "_MEIPASS", Path.cwd()))
src = base_path / ".streamlit"
dst = DATA_DIR / ".streamlit"

if src.exists() and not dst.exists():
    shutil.copytree(src, dst)


# ---------------------------
# Orthanc setup
# ---------------------------
ORTHANC = "http://localhost:8042"
AUTH = ("orthanc", "orthanc")

SESSION = requests.Session()
SESSION.auth = AUTH
SESSION.trust_env = False


# ---------------------------
# Output directories (writable)
# ---------------------------
TABLES_DIR = DATA_DIR / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------
# Session state initialization
# ---------------------------
if "clasp.DB_PATH" not in st.session_state:
    st.session_state["clasp.DB_PATH"] = DATA_DIR / "image_clasp_db.json"
    st.session_state["clasp.ROUNDEL_PATH"] = DATA_DIR / "roundel"

    st.session_state["clasp.REFERENCE_PATH"] = resource_path("reference")
    st.session_state["clasp.OUT_PATH"] = TABLES_DIR

    st.session_state["clasp.DEMOGRAPHICS_PATH"] = TABLES_DIR / "demographics.csv"
    st.session_state["clasp.EXAMS_PATH"] = TABLES_DIR / "exams.csv"

    st.session_state["clasp.MASK_SCALER"] = 500
    st.session_state["clasp.MODELS_PATH"] = resource_path("models")

# ---------------------------
# Navigation
# ---------------------------
pg = st.navigation([
    st.Page("app_pages/0_Dashboard.py", title="Clasp Dashboard", icon=":material/link:", default=True),
    st.Page("app_pages/1_Roundel.py", title="Roundel", icon=":material/adjust:"),
    st.Page("app_pages/2_Data_Entry.py", title="Data Entry", icon=":material/note_alt:"),
    st.Page("app_pages/3_Query.py", title="Query Data", icon=":material/description:"),
])

pg.run()