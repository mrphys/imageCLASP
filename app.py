import streamlit as st

pg = st.navigation([
    st.Page("pages/0_Dashboard.py", title="Clasp Dashboard", icon="🖇", default=True),
    st.Page("pages/1_Roundel.py", title="Roundel", icon="⭕"),
    st.Page("pages/2_Data_Entry.py", title="Data Entry", icon="📝"),
    st.Page("pages/3_Query.py", title="Query Data", icon="📋")
])

pg.run()