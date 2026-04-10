import streamlit as st
import requests
from utils.pipeline import *

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import sys
from utils.theme_utils import *
import glob

st.set_page_config(layout="wide")
load_theme()
st_header('Clasp Dashboard')


if "initialised" not in st.session_state:
    sync_orthanc_and_db()
    # run_pipelines()
    st.session_state['initialised'] = True

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

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = set()

        for file_path in iter_files():
            futures.add(executor.submit(upload_orthanc_file, file_path))
            total += 1

        progress_bar = st.progress(0, f"## **Uploading Files:**")

        completed = 0

        for future in as_completed(futures):
            try:
                series_id = future.result()
                if series_id is not None:
                    series_list.append(series_id)
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
    return total, failed, num_series


if "selected_folder" not in st.session_state:
    st.session_state.selected_folder = ""

if "upload_summary" not in st.session_state:
    st.session_state.upload_summary = None

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


pn1, spacer, pn2 = st.columns([1.5, 0.1, 1.5])
with pn1:
    col1, col2 = st.columns([1, 1])

    with col1:
        metric_box("Patients", total_patients)

        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

        metric_box("Roundelled", num_roundelled)

    with col2:
        metric_box("Studies", num_studies)

        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

        metric_box("Data Entered", num_entered)




with pn2:
    if st.button("Upload DICOM Folder", type="primary", use_container_width=True):
        folder = pick_folder_external()

        if not folder:
            st.warning("No folder selected")
        else:
            st.session_state.selected_folder = folder

            try:
                total, failed, num_series = upload_root_folder(folder)

                st.session_state.upload_summary = {
                    "Total Files": total,
                    "Failed Files": failed,
                    "Number of Series":num_series
                }

            except Exception as e:
                st.error(str(e))

    if st.session_state.upload_summary is not None:
        summary = st.session_state.upload_summary
        summary_columns = st.columns(len(summary))

        for col, (key, value) in zip(summary_columns, summary.items()):
            with col:
                st.metric(label=key, value=value)

        sync_orthanc_and_db()
        run_pipelines()
        st.session_state.upload_summary = None
        st.rerun()