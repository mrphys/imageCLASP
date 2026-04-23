from utils.data_entry_utils import *
from utils.pipeline import *
from utils.theme_utils import *
from utils.reset_utils import *

reset_app("roundel")

load_theme(secondary="#155a8a",
    secondary_hover="#1F4264",
    secondary_active="#12324D"
    )


if not os.path.exists(st.session_state["clasp.DEMOGRAPHICS_PATH"]):
    st.warning("There are no patients in the database!")
    st.stop()

# ---------- UI ----------
st_header("Clinical Data Entry")

demo_df = load_demographics()
num_remaining_patients = len(demo_df)

names = [
    f"{row["first_name"]} {row["last_name"]} | {row["patient_id"]}"
    for _, row in demo_df.iterrows()
]

if "data_entry.current_patient" not in st.session_state:
    st.session_state["data_entry.current_patient"] = names[0]

if "data_entry.prev_patient" not in st.session_state:
    st.session_state["data_entry.prev_patient"] = names[0]
    
col1, col2 = st.columns([0.26, 0.7])

# ---------- Configure toggle ----------
with col2:
    selection = st.pills("", ['Configure'])

configure_mode = selection == False#'Configure'

# ---------- Left column ----------
with col1:
    if not configure_mode:
        selected_name = st.selectbox(
            f"Select Patient ({num_remaining_patients} Patients Left)",
            options=names,
            key="data_entry.current_patient",
            on_change=clear_on_patient_change
        )

        patient_idx = names.index(selected_name)
        st.session_state["data_entry.patient_idx"] = patient_idx

# ---------- Init ----------
if "data_entry.initialized" not in st.session_state:
    init_patient_from_csv()
    st.session_state["data_entry.initialized"] = True

# # ---------- Configure ----------
# with col1:
#     if configure_mode:
#         if "accessed" not in st.session_state:
#             st.session_state.accessed = False

#         if not st.session_state.accessed:
#             password = st.text_input("Enter a password", type="password")

#             if password == "admin":
#                 st.session_state.accessed = True
#                 st.rerun()
#         else:
#             table = st.selectbox("Tables", ["Diagnoses", "Events", "Procedures"], index=None)


# if st.session_state.accessed:
#     df = pd.DataFrame(
#         [{"Primary": [""], "Secondary": [""]}]
#     )

#     edited_df = st.data_editor(
#         df,
#         num_rows="dynamic",
#         use_container_width=True,
#         column_config={
#             "Primary": st.column_config.ListColumn(
#                 "Primary Diagnoses",
#             ),
#             "Secondary": st.column_config.ListColumn(
#                 "Secondary Diagnoses",
#             ),
#         },
#     )


if not configure_mode:
    tabs = st.tabs(CLINICAL_ENTRIES)
    # --- Tab 1: Demographics ---
    with tabs[0]:
        with st.form("patient_form"):
            c1, c2 = st.columns(2)

            with c1:
                st.text_input("First name", key="data_entry.first_name")
                st.text_input("Last name", key="data_entry.last_name")

            with c2:
                st.selectbox("Sex", ["F", "M", "Other"], key="data_entry.sex", index=None)
                st.date_input(
                    "Date of birth",
                    min_value=MIN_DATE,
                    max_value=date.today(),
                    key="data_entry.dob",
                    format="DD-MM-YYYY"

                )

            confirm_patient = st.form_submit_button("Confirm demographics")

    tab_index = 1

    for label in multi_forms.keys():
        with tabs[tab_index]:
            create_multi_form(
                label=label,
                forms=multi_forms[label]
            )
        tab_index += 1

    for label in single_forms.keys():
        with tabs[tab_index]:
            create_single_form(
                label,
                single_forms[label]
            )
        tab_index += 1

    # ---------- Global Save ----------
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # Initialize states
    if "confirm_save" not in st.session_state:
        st.session_state.confirm_save = False
    if "saved" not in st.session_state:
        st.session_state.saved = False

    c1, c2 = st.columns([0.5, 0.5])

    with c1:
        if st.button(
            "Save Record",
            key="data_entry.save_record_btn",
            type="primary",
            use_container_width=True
        ):
            st.session_state.confirm_save = True

    with c2:
        if st.session_state.confirm_save:

            col1, col2 = st.columns(2)

            with col1:
                if st.button(
                    "Yes, Save",
                    icon=":material/save:",
                    use_container_width=True
                ):
                    if not st.session_state["data_entry.patient_id"]:
                        st.error("Patient ID is missing.")
                    else:
                        save_data_entry()
                        clear_on_patient_change()
                        st.session_state.saved = True

                    st.session_state.confirm_save = False

            with col2:
                if st.button(
                    "Cancel",
                    icon=":material/cancel:",
                    use_container_width=True
                ):
                    st.session_state.confirm_save = False
                    st.rerun()
            
            if not st.session_state.saved:
                st.warning("Are you sure you want to save this record?")


        if st.session_state.saved:
            st.success("Saved successfully!")
            time.sleep(1)
            st.session_state.saved = False
            st.rerun()