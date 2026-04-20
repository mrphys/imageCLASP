# utils/theme.py

import streamlit as st

def load_theme(
    secondary="#A94442",
    secondary_hover="#7A2F2F",
    secondary_active="#5C1F1F",
    base_font=16,
    label_font=16,
    button_font=16,
    form_font=16,
):

    st.markdown(f"""
    <style>

    :root {{
        --secondary-color: {secondary};
        --secondary-hover: {secondary_hover};
        --secondary-active: {secondary_active};
        --base-font: {base_font}px;
        --label-font: {label_font}px;
        --button-font: {button_font}px;
        --form-font: {form_font}px;
    }}

    /* ---------- GLOBAL TEXT ---------- */
    html, body, [class*="css"] {{
        font-size: var(--base-font) !important;
        font-weight: 500 !important;
    }}

    /* ---------- TABS ---------- */
    div[data-testid="stTabs"] button p {{
        font-size: var(--label-font) !important;
        font-weight: 500 !important;
        color: #155a8a !important;
    }}

    div[data-testid="stTabs"] button[aria-selected="true"] p {{
        font-weight: 650 !important;
        color: #A94442 !important;
    }}

    /* ---------- FORMS ---------- */
    [data-testid="stForm"] * {{
        font-size: var(--form-font) !important;
    }}

    div[data-testid="stFormSubmitButton"] {{
        margin-top: 18px !important;
    }}

    /* ---------- LABELS ---------- */
    div[data-testid="stSelectbox"] label p,
    div[data-testid="stTextInput"] label p,
    div[data-testid="stDateInput"] label p,
    div[data-testid="stNumberInput"] label p,
    div[data-testid="stTextArea"] label p {{
        font-size: var(--label-font) !important;
        font-weight: 600 !important;
        color: #155a8a !important;
    }}

    /* =========================================================
       FORM SUBMIT + PRIMARY BUTTONS
       ========================================================= */

    div[data-testid="stFormSubmitButton"] > button {{
        height: 1.6em !important;
        color: #155a8a !important;
        border-radius: 20px !important;
        font-weight: 500 !important;
        font-size: var(--button-font) !important;
    }}

    div[data-testid="stFormSubmitButton"] > button,
    div[data-testid="stFormSubmitButton"] > button *,
    div[data-testid="stFormSubmitButton"] > button p,
    div[data-testid="stFormSubmitButton"] > button span {{
        color: #155a8a !important;
        font-size: var(--button-font) !important;
    }}

    div[data-testid="stFormSubmitButton"] > button:hover {{
        background-color: #DFDBD2 !important;
    }}

    div[data-testid="stFormSubmitButton"] > button:active {{
        background-color: #304F40 !important;
    }}

    button[kind="primary"] {{
        height: 1.6em !important;
        background-color: #155a8a !important;
        color: white !important;
        border-radius: 20px !important;
        font-weight: 500 !important;
        font-size: var(--button-font) !important;
        border: none !important;
    }}

    button[kind="primary"],
    button[kind="primary"] * {{
        color: white !important;
        font-size: var(--button-font) !important;
    }}

    button[kind="primary"]:hover {{
        background-color: #35754C !important;
    }}

    button[kind="primary"]:active {{
        background-color: #304F40 !important;
    }}

    /* =========================================================
       SECONDARY BUTTONS
       ========================================================= */

    div[data-testid="stButton"] {{
        margin: 0 !important;
        padding: 0 !important;
    }}

    button[kind="secondary"] {{
        border-radius: 20px !important;
    }}

    button[kind="secondary"],
    button[kind="secondary"] * {{
        color: var(--secondary-color) !important;
        font-size: var(--button-font) !important;
    }}

    button[kind="secondary"]:hover,
    button[kind="secondary"]:hover * {{
        color: var(--secondary-hover) !important;
        background: none !important;
    }}

    button[kind="secondary"]:active,
    button[kind="secondary"]:active * {{
        color: var(--secondary-active) !important;
        background: none !important;
    }}
    


    

    </style>
    """, unsafe_allow_html=True)