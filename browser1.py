import os
import pandas as pd
import streamlit as st


def file_browser():
    st.divider()

    base_path = os.path.expanduser("~")

    @st.cache_data(show_spinner=False)
    def has_dcm_in_tree(path):
        for root, _, files in os.walk(path):
            for f in files:
                if f.lower().endswith(".dcm"):
                    return True
        return False

    @st.cache_data(show_spinner=False)
    def list_dirs(path):
        try:
            dirs = []
            for d in os.listdir(path):
                full = os.path.join(path, d)
                if os.path.isdir(full) and not d.startswith("."):
                    if has_dcm_in_tree(full):
                        dirs.append(d)
            return sorted(dirs)
        except PermissionError:
            return []

    def breadcrumbs(path):
        home = os.path.expanduser("~")
        path = os.path.abspath(path)

        rel = os.path.relpath(path, home)
        parts = [] if rel == "." else rel.split(os.sep)

        crumbs = [(home, home)]
        cur = home

        for p in parts:
            cur = os.path.join(cur, p)
            crumbs.append((p, cur))

        return crumbs

    if "dashboard.upload_path" not in st.session_state:
        st.session_state["dashboard.upload_path"] = base_path

    current_path = st.session_state["dashboard.upload_path"]

    st.markdown("### 📂 File Browser")

    crumbs = breadcrumbs(current_path)
    cols = st.columns(len(crumbs), gap="small")

    for i, (name, crumb_path) in enumerate(crumbs):
        with cols[i]:
            if crumb_path == current_path:
                st.button(name, disabled=True, use_container_width=True)
            else:
                if st.button(name, key=f"crumb_{crumb_path}", use_container_width=True):
                    st.session_state["dashboard.upload_path"] = crumb_path
                    st.rerun()

    st.markdown("---")

    dirs = list_dirs(current_path)

    if not dirs:
        st.caption("No subfolders with .dcm files.")
        return

    for d in dirs:
        full_path = os.path.join(current_path, d)

        if st.button(f"{d}", key=f"dir_{full_path}", use_container_width=True):
            st.session_state["dashboard.upload_path"] = full_path
            st.rerun()


file_browser()