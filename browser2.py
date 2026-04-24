import pandas as pd
import streamlit as st
import os

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

    df = pd.DataFrame({
        "folder": dirs,
        "select": [False] * len(dirs),
        "path": [os.path.join(current_path, d) for d in dirs],
    })

    edited = st.data_editor(
        df,
        hide_index=True,
        use_container_width=True,
        column_order=("folder", "select"),
        column_config={
            "folder": st.column_config.TextColumn("Folder", disabled=True),
            "select": st.column_config.CheckboxColumn("Select"),
            "path": None,
        },
        disabled=["folder"],
    )

    selected = edited[edited["select"]]

    if not selected.empty:
        st.session_state["dashboard.upload_path"] = selected.iloc[0]["path"]
        st.rerun()
    return st.session_state["dashboard.upload_path"]


import streamlit as st
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import sys
import glob
from tinydb import TinyDB
DB_PATH = 'image_clasp_db.json'


st.set_page_config(layout="wide")


if "dashboard.initialised" not in st.session_state:
    st.session_state['dashboard.initialised'] = True


# ---------- Hard-coded Orthanc settings ----------
def metric_box(label, value, color="#dfdbd2"):
    st.markdown(f"""
    <div style="
        background-color:{color};
        padding:20px;
        border-radius:16px;
        text-align:center;
        color:#155a8a;
        margin-bottom:20px;
    ">
        <div style="font-size:20px; font-weight:600;">{label}</div>
        <div style="font-size:35px; font-weight:700;">{value}</div>
    </div>
    """, unsafe_allow_html=True)

# ---------- to allow UI folder picking ----------
def pick_folder_external() -> str | None:
    script_path = (Path(__file__).parent / ".." / "utils" / "pick_folder.py").resolve()
    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        folder = result.stdout.strip()
        return folder or None

    return None

# ---------- Upload helpers ----------
def get_max_workers():
    cpu = os.cpu_count()
    if cpu is None:
        return 4
    return min(32, cpu + 4)


def upload_root_folder(root_folder: str):
    root = Path(root_folder)
    max_workers = get_max_workers()

    if not root.exists():
        raise FileNotFoundError(f"Folder does not exist: {root_folder}")

    if not root.is_dir():
        raise NotADirectoryError(f"Not a folder: {root_folder}")

    def iter_files():
        for p in root.rglob("*"):
            if p.is_file():
                yield p

    uploaded = 0
    failed = 0
    total = 0
    series_list = []
    study_list = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = set()

        for file_path in iter_files():
            total += 1

        progress_bar = st.progress(0, f"## **Uploading Files:**")

        completed = 0

        for future in as_completed(futures):
            try:
                series_id, study_id = future.result()
                if series_id is not None:
                    series_list.append(series_id)
                    study_list.append(study_id)
                    uploaded += 1
                else:
                    failed += 1
            except Exception:
                failed += 1

            completed += 1
            progress_bar.progress(
                completed / total, 
                f"## **Uploading Files:** {completed}/{total} ({completed/total:.1%})"
            )

    num_series = len(set(series_list))
    num_studies = len(set(study_list))
    return total, failed, num_series, num_studies


if "dashboard.selected_folder" not in st.session_state:
    st.session_state['dashboard.selected_folder'] = ""

if "dashboard.upload_summary" not in st.session_state:
    st.session_state['dashboard.upload_summary'] = None

db = TinyDB(DB_PATH)
total_patients = 1
num_roundelled = 1
num_segmented = 1
num_entered = 1
num_studies = 1
pn1, spacer, pn2 = st.columns([1.5, 0.1, 1.5])
with pn1:
    col1, col2 = st.columns([1, 1])

    with col1:
        metric_box("Patients", total_patients)

        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

        metric_box("Roundelled", f'{num_roundelled}/{num_segmented}')

    with col2:
        metric_box("Studies", num_studies)

        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

        metric_box("Data Entered", f'{num_entered}/{total_patients}')



with pn2:

    if st.button("Upload DICOM Folder", type="primary", use_container_width=True):
        folder = file_browser()

        if not folder:
            st.warning("No folder selected")
        else:
            st.session_state['dashboard.selected_folder'] = folder

            try:
                total, failed, num_series, num_studies = upload_root_folder(folder)

                st.session_state['dashboard.upload_summary'] = {
                    "Number of Studies":num_studies,
                    "Number of Series":num_series,
                    "Total Files": total,
                    "Failed Files": failed
                }

            except Exception as e:
                st.error(str(e))

    if st.session_state['dashboard.upload_summary'] is not None:
        summary = st.session_state['dashboard.upload_summary']
        summary_columns = st.columns(len(summary))

        for col, (key, value) in zip(summary_columns, summary.items()):
            with col:
                st.metric(label=key, value=value)

        # sync_orthanc_and_db()
        # run_pipelines()
        st.session_state['dashboard.upload_summary'] = None
        st.rerun()