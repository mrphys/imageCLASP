import streamlit as st
import requests
from utils.pipeline import *
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import sys, os, glob
import pandas as pd
from utils.theme_utils import *
from utils.reset_utils import *


st.set_page_config(layout="wide")
load_theme()
reset_app('data_entry')
reset_app('roundel')
st_header('CLASP Dashboard')


if "dashboard.initialised" not in st.session_state:
    sync_orthanc_and_db()
    # run_pipelines()
    st.session_state['dashboard.initialised'] = True



def file_browser():
    base_path = os.path.expanduser("~")

    def list_dirs(path):
        try:
            return sorted(
                d for d in os.listdir(path)
                if os.path.isdir(os.path.join(path, d)) and not d.startswith(".")
            )
        except PermissionError:
            return []

    def display_name(name):
        return name.replace("\\", "/") + "/"

    def breadcrumbs(path):
        path = os.path.abspath(path)
        drive, tail = os.path.splitdrive(path)

        parts = tail.strip(os.sep).split(os.sep) if tail.strip(os.sep) else []

        crumbs = []

        if drive:
            cur = drive + os.sep
            crumbs.append((drive, cur))
        else:
            cur = os.sep
            crumbs.append((os.sep, cur))

        for p in parts:
            cur = os.path.join(cur, p)
            crumbs.append((p, cur))

        return crumbs

    # session state
    if "dashboard.upload_path" not in st.session_state:
        st.session_state["dashboard.upload_path"] = base_path

    if "dashboard.selected_folders" not in st.session_state:
        st.session_state["dashboard.selected_folders"] = set()

    if "dashboard.prev_selected" not in st.session_state:
        st.session_state["dashboard.prev_selected"] = set()

    if "dashboard.multi_select_block" not in st.session_state:
        st.session_state["dashboard.multi_select_block"] = False

    current_path = st.session_state["dashboard.upload_path"]

    # bread# breadcrumbs UI
    crumbs = breadcrumbs(current_path)

    MAX_VISIBLE_CRUMBS = 4

    display_crumbs = []

    # Collapse leading crumbs if there are too many
    if len(crumbs) > MAX_VISIBLE_CRUMBS:
        hidden_count = len(crumbs) - MAX_VISIBLE_CRUMBS + 1

        grouped_name = " / ".join(
            name for name, _ in crumbs[:hidden_count]
        )
        grouped_path = crumbs[hidden_count - 1][1]

        display_crumbs.append((grouped_name, grouped_path))
        display_crumbs.extend(crumbs[hidden_count:])
    else:
        display_crumbs = crumbs

    cols = st.columns([len(crumb) for crumb in display_crumbs], gap="small")

    for i, (name, crumb_path) in enumerate(display_crumbs):
        with cols[i]:
            label = display_name(name)

            if crumb_path == current_path:
                st.button(label, disabled=True, use_container_width=True)
            else:
                if st.button(
                    label,
                    key=f"crumb_{crumb_path}",
                    use_container_width=True,
                ):
                    st.session_state["dashboard.upload_path"] = crumb_path
                    st.session_state["dashboard.prev_selected"] = set()
                    st.session_state["dashboard.multi_select_block"] = False
                    st.rerun()

    dirs = list_dirs(current_path)

    # leaf directory UI (do not return here)
    if not dirs:
        n_files = len(glob.glob(os.path.join(current_path, "*.dcm")))
        st.metric(label="DICOM files", value=n_files)

        # still show upload button below (handled later)
        df = pd.DataFrame(columns=["folder", "select", "path"])
        edited = None
        selected = set()

    else:
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
                "folder": st.column_config.TextColumn("Folder", disabled=True, width="medium"),
                "select": st.column_config.CheckboxColumn("Select", width="small"),
                "path": None,
            },
            disabled=["folder"],
        )

        selected = set(edited[edited["select"]]["path"].tolist())

    # selection state logic (only runs if data_editor exists)
    if edited is not None:
        prev_selected = st.session_state["dashboard.prev_selected"]
        multi_block = st.session_state["dashboard.multi_select_block"]

        if len(selected) > 1:
            st.session_state["dashboard.multi_select_block"] = True

        removed = prev_selected - selected

        if (
            len(prev_selected) == 1 and
            len(selected) == 0 and
            len(removed) == 1 and
            not multi_block
        ):
            target = list(removed)[0]
            st.session_state["dashboard.upload_path"] = target
            st.session_state["dashboard.prev_selected"] = set()
            st.session_state["dashboard.multi_select_block"] = False
            st.rerun()

        if len(selected) == 0:
            st.session_state["dashboard.multi_select_block"] = False

        st.session_state["dashboard.prev_selected"] = selected

    # upload controls always visible
    folder_message = "Upload Folder" if len(selected) == 0 else "Upload Selected Folders"

    if st.button(folder_message, use_container_width=True, type="primary"):
        st.session_state["dashboard.upload_path"] = current_path

        if len(selected) == 0:
            st.session_state["dashboard.selected_folders"] = set(df["path"].tolist())
        else:
            st.session_state["dashboard.selected_folders"] = selected

        st.rerun()

    return st.session_state["dashboard.selected_folders"]


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

# ---------- Upload helpers ----------
def get_max_workers():
    cpu = os.cpu_count()
    if cpu is None:
        return 4
    return min(32, cpu + 4)


def upload_folders(selected_folders: str):
    uploaded = 0
    failed = 0
    total = 0
    series_list = []
    study_list = []

    for root_folder in selected_folders:
        root = Path(root_folder)
        folder_name = root.name
        max_workers = get_max_workers()

        if not root.exists():
            raise FileNotFoundError(f"Folder does not exist: {root_folder}")

        if not root.is_dir():
            raise NotADirectoryError(f"Not a folder: {root_folder}")

        def iter_files():
            for p in root.rglob("*"):
                if p.is_file():
                    yield p


        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = set()
            folder_total = 0

            for file_path in iter_files():
                futures.add(executor.submit(upload_orthanc_file, file_path))
                total += 1
                folder_total += 1

            progress_bar = st.progress(0, f"## **Uploading Files from {folder_name}:**")

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
                    completed / folder_total, 
                    f"## **Uploading Files from {folder_name}:** {completed}/{folder_total} ({completed/folder_total:.1%})"
                )

    num_series = len(set(series_list))
    num_studies = len(set(study_list))
    return total, failed, num_series, num_studies


if "dashboard.selected_folder" not in st.session_state:
    st.session_state['dashboard.selected_folder'] = ""

if "dashboard.upload_summary" not in st.session_state:
    st.session_state['dashboard.upload_summary'] = None

db = TinyDB(DB_PATH)
studies = fetch_db_studies()

total_patients = len(np.unique([study.patient_id for study in studies]))
num_studies = len(studies)
num_entered = len(get_entered_patients())

num_segmented = 0
num_roundelled = 0
for study in studies:
    df = pd.DataFrame([series.__dict__ for series in study.series_dict.values()])
    num_segmented+=(df['dl_orthanc_id'].notna()).any()
    num_roundelled+=(df['roundel_orthanc_id'].notna()).any()


pn1, pn2 = st.columns([1.3,2])
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
    if "dashboard.upload_summary" not in st.session_state:
        st.session_state["dashboard.upload_summary"] = None

    if "dashboard.selected_folder" not in st.session_state:
        st.session_state["dashboard.selected_folder"] = None

    if "dashboard.show_upload_button" not in st.session_state:
        st.session_state["dashboard.show_upload_button"] = True

    if "dashboard.open_file_browser" not in st.session_state:
        st.session_state["dashboard.open_file_browser"] = False


    # STEP 1: open browser
    if st.session_state["dashboard.show_upload_button"]:
        if st.button("Upload DICOM Folder", type="primary", use_container_width=True):
            st.session_state["dashboard.open_file_browser"] = True
            st.session_state["dashboard.show_upload_button"] = False
            st.rerun()


    # STEP 2: show file browser
    if st.session_state["dashboard.open_file_browser"]:
        selected_folders = file_browser()
        if selected_folders:
            st.session_state["dashboard.selected_folders"] = set()
            total, failed, num_series, num_studies = upload_folders(selected_folders)

            st.session_state["dashboard.upload_summary"] = {
                "Number of Studies": num_studies,
                "Number of Series": num_series,
                "Total Files": total,
                "Failed Files": failed
            }
            st.session_state["dashboard.show_upload_button"] = True
            st.session_state["dashboard.open_file_browser"] = False


    if st.session_state['dashboard.upload_summary']:
        summary = st.session_state['dashboard.upload_summary']
        summary_columns = st.columns(len(summary))

        for col, (key, value) in zip(summary_columns, summary.items()):
            with col:
                st.metric(label=key, value=value)

        sync_orthanc_and_db()
        run_pipelines()
        st.session_state['dashboard.upload_summary'] = None
        st.rerun()