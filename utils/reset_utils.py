import streamlit as st

def reset_app(key):
    try:
        st.session_state.pop(f'{key}.initialized')
    except:
        pass
