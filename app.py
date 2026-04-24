import os, sys
import pandas as pd
import requests
import shutil

base_path = getattr(sys, "_MEIPASS", os.path.abspath("."))
src = os.path.join(base_path, ".streamlit")
dst = os.path.join(os.path.dirname(sys.executable), ".streamlit")

if os.path.exists(src) and not os.path.exists(dst):
    shutil.copytree(src, dst)

import streamlit as st


def resource_path(relative_path):
    base_path = getattr(sys, "_MEIPASS", os.path.abspath("."))
    return os.path.join(base_path, relative_path)

ORTHANC = "http://localhost:8042"
AUTH = ("orthanc", "orthanc")

SESSION = requests.Session()
SESSION.auth = AUTH
SESSION.trust_env = False

if not os.path.exists('tables'):
    os.makedirs('tables')

if 'clasp.DB_PATH' not in st.session_state:
    st.session_state['clasp.DB_PATH'] = "image_clasp_db.json"
    st.session_state['clasp.REFERENCE_PATH'] = resource_path('reference')
    st.session_state['clasp.OUT_PATH'] = 'tables'
    st.session_state['clasp.DEMOGRAPHICS_PATH'] = st.session_state['clasp.OUT_PATH'] + '/demographics.csv'
    st.session_state['clasp.EXAMS_PATH'] = st.session_state['clasp.OUT_PATH'] + '/exams.csv'
    st.session_state['clasp.MASK_SCALER'] = 500
    st.session_state['clasp.MODELS_PATH'] = resource_path('models')


pg = st.navigation([
    st.Page("app_pages/0_Dashboard.py", title="Clasp Dashboard", icon=":material/link:", default=True),
    st.Page("app_pages/1_Roundel.py", title="Roundel", icon=":material/adjust:"),
    st.Page("app_pages/2_Data_Entry.py", title="Data Entry", icon=":material/note_alt:"),
    st.Page("app_pages/3_Query.py", title="Query Data", icon=":material/description:"),
])

pg.run()