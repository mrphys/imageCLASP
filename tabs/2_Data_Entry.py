from utils.data_entry_utils import *
from utils.pipeline import *
from utils.theme_utils import *

load_theme(secondary="#155a8a",
    secondary_hover="#1F4264",
    secondary_active="#12324D"
    )

# ---------- UI ----------
st_header("CLASP Clinical Data")

demo_df = load_demographics()

# ---------- Session State ----------
if 'patient_idx' not in st.session_state:
    st.session_state['patient_idx'] = 0


# ---------- Header ----------
top1, top2 = st.columns([3, 1])

load_current_patient_from_csv()
with top1:
    st.markdown(f"### Patient ID {st.session_state.patient_id} - {st.session_state.patient_idx + 1} of {len(demo_df)}")

with top2:
    col_a, col_b = st.columns(2)
    if col_a.button("⏮ Previous patient", key="prev"):
        st.session_state.pending_nav = "prev"

    if col_b.button("Next patient ⏭", key="next"):
        st.session_state.pending_nav = "next"
    process_pending_navigation()


# ---------- Tabs ----------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["Demographics", "Events", "Diagnoses", "Procedures", "Tests", "Medications"]
)

# --- Tab 1: Demographics ---
with tab1:
    with st.form("patient_form"):
        c1, c2 = st.columns(2)

        with c1:
            st.text_input("First name", key="first_name")
            st.text_input("Last name", key="last_name")

        with c2:
            st.selectbox("Sex", ["F", "M", "Other","Missing"], key="sex")
            st.date_input(
                "Date of birth",
                min_value=MIN_DATE,
                max_value=date.today(),
                key="dob",
                format="DD-MM-YYYY",
            )

        confirm_patient = st.form_submit_button("Confirm demographics")


# --- Tab 2: Events ---
with tab2:
    multi_group_form(
        label="Events",
        groups={
            "Major": major_events,
            "Minor": minor_events
        }
    )
    

# --- Tab 3: Diagnoses ---
with tab3:
    multi_group_form(
        label="Diagnoses",
        groups={
            "Primary": primary_options,
            "Secondary": secondary_options
        }
    )


# --- Tab 4: Procedures ---
with tab4:
    multi_group_form(
        label="Procedures",
        groups={
            "Surgical": surgery_options,
            "Interventional": intervention_options
        }
    )

# --- Tab 5: Tests ---
with tab5:
    test_kind = st.selectbox(
        "Test type",
        ["Blood test", "CPEX"],
        key="test_kind_select",
    )

    if test_kind == "Blood test":
        with st.form("blood_test_form", clear_on_submit=True):
            left, right = st.columns(2)

            with left:
                blood_test_type = st.selectbox(
                    "Blood test",
                    blood_test_options,
                    key="blood_test_type",
                )
                blood_test_date = st.date_input(
                    "Date of blood test",
                    value=date.today(),
                    min_value=MIN_DATE,
                    max_value=date.today(),
                    key="blood_test_date",
                    format="DD-MM-YYYY",
                )

            with right:
                blood_result = st.number_input("Result", key="blood_result")
                blood_units = st.selectbox("Unit", blood_units_options, key="blood_units")

            if st.form_submit_button("Add blood test"):
                if not st.session_state.patient_id:
                    st.error("Patient ID is required.")
                elif not blood_test_type or blood_result == 0:
                    st.warning("Select a blood test and enter a result.")
                else:
                    test_id = make_test_id()

                    st.session_state.current_tests.append(
                        {
                            "test_id": test_id,
                            "patient_id": st.session_state.patient_id,
                            "test_group": "blood",
                            "test_type": blood_test_type,
                            "test_date": str(blood_test_date),
                        }
                    )

                    st.session_state.current_test_values.append(
                        {
                            "test_id": test_id,
                            "parameter_name": "result",
                            "parameter_value": blood_result,
                            "units": blood_units,
                        }
                    )

    elif test_kind == "CPEX":
        with st.form("cpex_form", clear_on_submit=True):
            left, right = st.columns(2)

            with left:
                cpex_date = st.date_input(
                    "Date of CPEX",
                    value=date.today(),
                    min_value=MIN_DATE,
                    max_value=date.today(),
                    key="cpex_date",
                    format="DD-MM-YYYY",
                )
                cpex_vo2 = st.number_input("VO2", key="cpex_vo2")
                cpex_vevco2 = st.number_input("VE/VCO2", key="cpex_vevco2")

            with right:
                cpex_rer = st.number_input("RER", key="cpex_rer")
                cpex_peak_hr = st.number_input("Peak HR", key="cpex_peak_hr")
                cpex_peak_bp = st.number_input("Peak BP", key="cpex_peak_bp")

            if st.form_submit_button("Add CPEX"):
                if not st.session_state.patient_id:
                    st.error("Patient ID is required.")
                elif not (cpex_vo2 or cpex_vevco2 or cpex_rer or cpex_peak_hr or cpex_peak_bp):
                    st.warning("Enter at least one CPEX value.")
                else:
                    test_id = make_test_id()

                    st.session_state.current_tests.append(
                        {
                            "test_id": test_id,
                            "patient_id": st.session_state.patient_id,
                            "test_group": "cpex",
                            "test_type": "CPEX",
                            "test_date": str(cpex_date),
                        }
                    )

                    value_rows = [
                        ("VO2", cpex_vo2, "ml/kg/min"),
                        ("VE/VCO2", cpex_vevco2, ""),
                        ("RER", cpex_rer, ""),
                        ("Peak HR", cpex_peak_hr, "bpm"),
                        ("Peak BP", cpex_peak_bp, "mmHg"),
                    ]

                    for param_name, param_value, units in value_rows:
                        if param_value:
                            st.session_state.current_test_values.append(
                                {
                                    "test_id": test_id,
                                    "parameter_name": param_name,
                                    "parameter_value": param_value,
                                    "units": units,
                                }
                            )
    df = pd.DataFrame(st.session_state.current_tests)
    if not df.empty:
        st.dataframe(df)


# --- Tab 6: Medications ---
with tab6:
    with st.form("medication_form", clear_on_submit=True):
        left, right = st.columns(2)

        with left:
            medication_name = st.selectbox("Medication", medication_options, key="medication_name")
            medication_date = st.date_input(
                "Medication date",
                value=date.today(),
                min_value=MIN_DATE,
                max_value=date.today(),
                key="medication_date",
                format="DD-MM-YYYY",
            )

        with right:
            medication_dose = st.number_input("Dose", key="medication_dose")
            medication_frequency = st.selectbox(
                "Frequency",
                ["OD", "BD", "TDS", "QDS"],
                key="medication_frequency",
            )

        if st.form_submit_button("Add medication"):
            add_record(
                "current_medications",
                {
                    "patient_id": st.session_state.patient_id,
                    "medication": medication_name,
                    "medication_date": str(medication_date),
                    "dose": medication_dose,
                    "frequency": medication_frequency,
                },
                medication_name,
                "Enter a medication name.",
            )
    df = pd.DataFrame(st.session_state.current_medications)
    if not df.empty:
        st.dataframe(df.set_index('patient_id'))


# ---------- Global Save ----------
st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
save_record = st.button("Save Record", key="save_record_btn", type = 'primary')

if save_record:
    if not st.session_state.patient_id:
        st.error("Patient ID is missing.")
    else:
        save_data_entry()