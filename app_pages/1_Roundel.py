# --------------------------------------------------------------
# Configure Streamlit page
# --------------------------------------------------------------
from utils.roundel_utils import *
from utils.pipeline import *
from utils.theme_utils import *

st.set_page_config(page_title="Roundel", layout='wide')
load_theme()
reset_app('data_entry')


# -----------------------------
# Data paths and series info
# -----------------------------

st_header('Roundel')
db = TinyDB(DB_PATH)
studies = fetch_db_studies()
study_dict = {}
study_description = {}

for study in studies:
    df = pd.DataFrame([series.__dict__ for series in study.series_dict.values()])
    sax_dl_df = df[(df['dl_orthanc_id'].notna()) & (df['roundel_orthanc_id'].isna())]

    if len(sax_dl_df) > 0:
        patient_name = study.patient_name.split('^')
        last_name, first_name = patient_name
        description = sax_dl_df['series_description'].values[0]
        
        study_date = datetime.strptime(study.study_date, "%Y%m%d").strftime("%d/%m/%Y")
        study_dict[study.orthanc_study_id] = study
        study_description[f'{first_name} {last_name} | {study_date} | {description}'] = study.orthanc_study_id
        
num_remaining_studies = len(study_dict)

if num_remaining_studies == 0:
    st.success("🎉 All Roundel cases completed!")
    st.stop()


col1, col2 = st.columns([0.26, 0.7])
if "roundel.prev_study_id" not in st.session_state:
    st.session_state["roundel.prev_study_id"] = list(study_dict.keys())[0]

if "roundel.current_study_id" not in st.session_state:
    st.session_state["roundel.current_study_id"] = list(study_dict.keys())[0]

with col1:
    orthanc_study_id = st.selectbox(
        f"Select Study ({num_remaining_studies} Studies Left)",
        options=list(study_dict.keys()),
        key="roundel.current_study_id",
        format_func=lambda x: next(k for k, v in study_description.items() if v == x),
        on_change=restart_app
    )
    study = study_dict[orthanc_study_id]

# --------------------------------------------------------------
# App
# --------------------------------------------------------------


if 'roundel.initialized' not in st.session_state:
    initialize_app(study)

view = st.radio(
    "Tab",
    options=["EDV/ESV Finder 🔍", "Mask Editor 📝", "Final Result ✅"],
    index=["EDV/ESV Finder 🔍", "Mask Editor 📝", "Final Result ✅"].index(st.session_state["roundel.view"]),
    horizontal=True
)
st.session_state["roundel.view"] = view
st.divider()

# --------------------------------------------------------------
# EDV/ESV Finder 
# --------------------------------------------------------------
if view == "EDV/ESV Finder 🔍":
    edv_esv_view()


# --------------------------------------------------------------
# Mask Editor 
# --------------------------------------------------------------

if view == "Mask Editor 📝":
    mask_editor_view()


# --------------------------------------------------------------
# Final Result
# --------------------------------------------------------------
if view == "Final Result ✅":
    final_result_view()