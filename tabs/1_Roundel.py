# --------------------------------------------------------------
# Configure Streamlit page
# --------------------------------------------------------------
from utils.roundel_utils import *
from utils.pipeline import *
from utils.theme_utils import *

st.set_page_config(page_title="Roundel", layout='wide')
load_theme()

# -----------------------------
# Data paths and series info
# -----------------------------

# if len(sax_series_uid_list) == 0:
#     st.success("🎉 All Roundel cases completed!")
#     st.stop()

# # Sidebar dropdown
# st.write('# Roundel App (2D Biventricular)')

# col1, col2 = st.columns([0.3,0.7])
# with col1:
#     sax_series_uid = st.selectbox(
#         "Select SAX Series UID",
#         options=sax_series_uid_list,
#         index=0  # optional: preselect the first UID
#     )
db = TinyDB(DB_PATH)
studies = fetch_db_studies()
study = studies[0]

if 'initialized_roundel' not in st.session_state:
    initialize_app(study)

# with col2:

#     ## GABE IS THERE A WAY TO GET THIS INFORMATION FROM THE SAX_SERIES UID, ITS JUST FOR VISUALS
#     patient, study_date, description = 'AAA-IMATEST-1', '01-01-2020', 'short axis cine stack'

#     # Display metadata in the app
#     st.markdown(f"**SAX Series UID:** {sax_series_uid} | **Patient:** {patient} | **Study Date:** {study_date}")
#     st.markdown(f"**Description:** {description} | **Pixel Size**: {pixelspacing} x {pixelspacing}mm | **Slice Thickness**: {thickness} mm")

# --------------------------------------------------------------
# App
# --------------------------------------------------------------

view = st.segmented_control(
    "Tab",
    options=["EDV/ESV Finder 🔍", "Mask Editor 📝", "Final Result ✅"],
    default = "EDV/ESV Finder 🔍",
    label_visibility='hidden'
)
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