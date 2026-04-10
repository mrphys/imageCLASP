import os
import pandas as pd
import requests
import streamlit as st

ORTHANC = "http://localhost:8042"
AUTH = ("orthanc", "orthanc")

SESSION = requests.Session()
SESSION.auth = AUTH
SESSION.trust_env = False

pg = st.navigation([
    st.Page("tabs/0_Dashboard.py", title="Clasp Dashboard", icon=":material/link:", default=True),
    st.Page("tabs/1_Roundel.py", title="Roundel", icon=":material/adjust:"),
    st.Page("tabs/2_Data_Entry.py", title="Data Entry", icon=":material/note_alt:"),
    st.Page("tabs/3_Query.py", title="Query Data", icon=":material/description:")
])

pg.run()