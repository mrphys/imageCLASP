import os
import uuid
from datetime import date

import pandas as pd
import streamlit as st
from utils.pipeline import *




# ---------- Configuration ----------
REFERENCE_PATH = 'reference'
EVENTS_CSV = f"{REFERENCE_PATH}/events_list.csv"
DIAGNOSES_CSV = f"{REFERENCE_PATH}/diagnoses_list.csv"
PROCEDURES_CSV = f"{REFERENCE_PATH}/procedures_list.csv"
BLOOD_TESTS_CSV = f"{REFERENCE_PATH}/blood_tests_list.csv"
BLOOD_UNITS_CSV = f"{REFERENCE_PATH}/blood_test_unit.csv"
MEDICATIONS_CSV = f"{REFERENCE_PATH}/medications_list.csv"

OUT_PATH = 'tables'
OUT_DEMOGRAPHICS_CSV = f"{OUT_PATH}/demographics.csv"
OUT_EVENTS_CSV = f"{OUT_PATH}/events.csv"
OUT_DIAGNOSES_CSV = f"{OUT_PATH}/diagnoses.csv"
OUT_PROCEDURES_CSV = f"{OUT_PATH}/procedures.csv"
OUT_TESTS_CSV = f"{OUT_PATH}/tests.csv"
OUT_TEST_VALUES_CSV = f"{OUT_PATH}/test_values.csv"
OUT_MEDICATIONS_CSV = f"{OUT_PATH}/medications.csv"

MIN_DATE = date(1900, 1, 1)


# ---------- Helper Functions ----------
def append_csv(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, mode="a", index=False, header=not os.path.exists(path))


def upsert_demographics_csv(row_df: pd.DataFrame, path: str) -> None:
    if os.path.exists(path):
        existing = pd.read_csv(path)
        existing.columns = [c.strip() for c in existing.columns]

        if "patient_id" in existing.columns:
            existing["patient_id"] = existing["patient_id"].astype(str).str.strip()

        patient_id = str(row_df.iloc[0]["patient_id"]).strip()
        existing = existing[existing["patient_id"] != patient_id]
        updated = pd.concat([existing, row_df], ignore_index=True)
    else:
        updated = row_df.copy()

    updated.to_csv(path, index=False)


@st.cache_data
def load_demographics(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    required = ["patient_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in demographics CSV: {missing}")

    df["patient_id"] = df["patient_id"].astype(str).str.strip()

    for col in ["first_name", "last_name", "sex", "dob"]:
        if col not in df.columns:
            df[col] = ""

    return df


@st.cache_data
def load_options(
    path: str,
    value_col: str,
    group_col: str | None = None,
    group_value: str | None = None,
    include_blank: bool = True,
) -> list[str]:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    if value_col not in df.columns:
        raise ValueError(
            f"{path} is missing required column: {value_col}. "
            f"Found columns: {df.columns.tolist()}"
        )

    df[value_col] = df[value_col].astype(str).str.strip()

    if group_col and group_value:
        if group_col not in df.columns:
            raise ValueError(
                f"{path} is missing required column: {group_col}. "
                f"Found columns: {df.columns.tolist()}"
            )
        df[group_col] = df[group_col].astype(str).str.strip().str.lower()
        df = df[df[group_col] == group_value.lower()]

    options = df[value_col].dropna().tolist()
    return [""] + options if include_blank else options


def make_test_id() -> str:
    return str(uuid.uuid4())[:8]


def load_current_patient_from_csv() -> None:
    row = demo_df.iloc[st.session_state.patient_idx]

    patient_id = str(row.get("patient_id", "") or "")
    first_name = str(row.get("first_name", "") or "")
    last_name = str(row.get("last_name", "") or "")
    sex = str(row.get("sex", "F") or "F")

    dob_val = row.get("dob", "")
    if pd.notna(dob_val) and str(dob_val).strip():
        dob = pd.to_datetime(dob_val, dayfirst=True).date()
    else:
        dob = date(1990, 1, 1)

    st.session_state.patient_id = patient_id
    st.session_state.first_name = first_name
    st.session_state.last_name = last_name
    st.session_state.sex = sex
    st.session_state.dob = dob

    st.session_state.first_name_input = first_name
    st.session_state.last_name_input = last_name
    st.session_state.sex_input = sex if sex in ["F", "M", "Other"] else "F"
    st.session_state.dob_input = dob

    st.session_state.current_events = []
    st.session_state.current_diagnoses = []
    st.session_state.current_procedures = []
    st.session_state.current_tests = []
    st.session_state.current_test_values = []
    st.session_state.current_medications = []


def process_pending_navigation() -> None:
    pending_nav = st.session_state.get("pending_nav", None)

    if pending_nav == "next":
        if st.session_state.patient_idx < len(demo_df) - 1:
            st.session_state.patient_idx += 1
            load_current_patient_from_csv()
        st.session_state.pending_nav = None

    elif pending_nav == "prev":
        if st.session_state.patient_idx > 0:
            st.session_state.patient_idx -= 1
            load_current_patient_from_csv()
        st.session_state.pending_nav = None


def add_record(
    state_list_name: str,
    record_dict: dict,
    requirement_check,
    requirement_msg: str,
) -> None:
    if not st.session_state.patient_id:
        st.error("Patient ID is required.")
        return

    if not requirement_check:
        st.warning(requirement_msg)
        return

    st.session_state[state_list_name].append(record_dict)


# ---------- Session State ----------
DEFAULT_STATE = {
    "patient_idx": 0,
    "current_events": [],
    "current_diagnoses": [],
    "current_procedures": [],
    "current_tests": [],
    "current_test_values": [],
    "current_medications": [],
    "patient_id": "",
    "first_name": "",
    "last_name": "",
    "dob": date(1990, 1, 1),
    "sex": "F",
    "first_name_input": "",
    "last_name_input": "",
    "dob_input": date(1990, 1, 1),
    "sex_input": "F",
    "pending_nav": None,
    "show_save_toast": False,
}

for key, default_val in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = default_val


# ---------- Load Data ----------
demo_df = load_demographics(OUT_DEMOGRAPHICS_CSV)
if demo_df.empty:
    st.error("Demographics CSV is empty.")
    st.stop()

if not st.session_state.patient_id:
    load_current_patient_from_csv()

process_pending_navigation()

if st.session_state.get("show_save_toast", False):
    st.toast("Record saved")
    st.session_state.show_save_toast = False

major_events = load_options(EVENTS_CSV, "event_type", "event_group", "major")
minor_events = load_options(EVENTS_CSV, "event_type", "event_group", "minor")
primary_options = load_options(DIAGNOSES_CSV, "diagnosis_type", "group", "primary")
secondary_options = load_options(DIAGNOSES_CSV, "diagnosis_type", "group", "secondary")
surgery_options = load_options(PROCEDURES_CSV, "procedure_type", "procedure_group", "surgery")
intervention_options = load_options(PROCEDURES_CSV, "procedure_type", "procedure_group", "intervention")
blood_test_options = load_options(BLOOD_TESTS_CSV, "blood_test_name")
blood_units_options = load_options(BLOOD_UNITS_CSV, "units")
medication_options = load_options(MEDICATIONS_CSV, "medication_name")


# ---------- UI ----------
st.set_page_config(layout="wide")
st_header("Clinical Data Entry")

st.markdown("""
<style>
/* Tabs */
div[data-testid="stTabs"] button p {
    font-size: 18px !important;
    font-weight: 500 !important;
    color: #155a8a !important;
}

div[data-testid="stTabs"] button[aria-selected="true"] p {
    font-weight: 650 !important;
    color: #A94442 !important;
}

/* Forms */
[data-testid="stForm"] * {
    font-size: 16px !important;
}

div[data-testid="stFormSubmitButton"] {
    margin-top: 18px !important;
}

div[data-testid="stFormSubmitButton"] > button {
    width: 100%;
    height: 1.5em;
    background-color: #155a8a !important;
    color: white !important;
    border-radius: 18px;
    font-weight: 500;
    border: none;
}

/* Force all inner text nodes */
div[data-testid="stFormSubmitButton"] > button,
div[data-testid="stFormSubmitButton"] > button *,
div[data-testid="stFormSubmitButton"] > button p,
div[data-testid="stFormSubmitButton"] > button span {
    color: white !important;
}

div[data-testid="stFormSubmitButton"] > button:hover {
    background-color: #35754C !important;
}

div[data-testid="stFormSubmitButton"] > button:hover,
div[data-testid="stFormSubmitButton"] > button:hover *,
div[data-testid="stFormSubmitButton"] > button:hover p,
div[data-testid="stFormSubmitButton"] > button:hover span {
    color: white !important;
}

div[data-testid="stFormSubmitButton"] > button:active {
    background-color: #304F40 !important;
}

/* Plain st.button */

div.stButton > button {
    width: 100% !important;
    height: 1.9em !important;
    background: none !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    color: #155a8a !important;
}



div.stButton > button:hover {
    color: #1F4264 !important;
}

/* Visible button text */
div.stButton > button,
div.stButton > button *,
div.stButton > button p,
div.stButton > button span {
    font-size: 18px !important;
    font-weight: 500 !important;
    color: #155a8a !important;
    line-height: 1 !important;
}

div.stButton > button:hover,
div.stButton > button:hover *,
div.stButton > button:hover p,
div.stButton > button:hover span {
    color: #1F4264 !important;
}
            
</style>
""", unsafe_allow_html=True)

# ---------- Header ----------
top1, top2 = st.columns([3, 1])

with top1:
    st.markdown(
    f"""
    <div style="font-size:25px; font-weight:550;">
        Patient ID {st.session_state.patient_id} - {st.session_state.patient_idx + 1} of {len(demo_df)}
    </div>
    """,
    unsafe_allow_html=True,
)

with top2:
    col_a, col_b = st.columns(2)

    if col_a.button("⏮ Previous patient", key="prev"):
        st.session_state.pending_nav = "prev"
        st.rerun()

    if col_b.button("Next patient ⏭", key="next"):
        st.session_state.pending_nav = "next"
        st.rerun()


# ---------- Tabs ----------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["Demographics", "Events", "Diagnoses", "Procedures", "Tests", "Medications"]
)


# --- Tab 1: Demographics ---
with tab1:
    with st.form("patient_form"):
        c1, c2 = st.columns(2)

        with c1:
            st.text_input("First name", key="first_name_input")
            st.text_input("Last name", key="last_name_input")

        with c2:
            st.selectbox("Sex", ["F", "M", "Other"], key="sex_input")
            st.date_input(
                "Date of birth",
                min_value=MIN_DATE,
                max_value=date.today(),
                key="dob_input",
                format="DD-MM-YYYY",
            )

        confirm_patient = st.form_submit_button("Confirm demographics")

    if confirm_patient:
        st.session_state.first_name = st.session_state.first_name_input
        st.session_state.last_name = st.session_state.last_name_input
        st.session_state.sex = st.session_state.sex_input
        st.session_state.dob = st.session_state.dob_input


# --- Tab 2: Events ---
with tab2:
    left, right = st.columns(2)

    with left:
        with st.form("major_event_form", clear_on_submit=True):
            evt_type_1 = st.selectbox("Major event", major_events, key="major_event_type")
            evt_date_1 = st.date_input(
                "Date of major event",
                value=date.today(),
                min_value=MIN_DATE,
                max_value=date.today(),
                key="major_event_date",
                format="DD-MM-YYYY",
            )

            if st.form_submit_button("Add major event"):
                add_record(
                    "current_events",
                    {
                        "patient_id": st.session_state.patient_id,
                        "event_group": "major",
                        "event_type": evt_type_1,
                        "event_date": str(evt_date_1),
                    },
                    evt_type_1,
                    "Select a major event.",
                )

    with right:
        with st.form("minor_event_form", clear_on_submit=True):
            evt_type_2 = st.selectbox("Minor event", minor_events, key="minor_event_type")
            evt_date_2 = st.date_input(
                "Date of minor event",
                value=date.today(),
                min_value=MIN_DATE,
                max_value=date.today(),
                key="minor_event_date",
                format="DD-MM-YYYY",
            )

            if st.form_submit_button("Add minor event"):
                add_record(
                    "current_events",
                    {
                        "patient_id": st.session_state.patient_id,
                        "event_group": "minor",
                        "event_type": evt_type_2,
                        "event_date": str(evt_date_2),
                    },
                    evt_type_2,
                    "Select a minor event.",
                )


# --- Tab 3: Diagnoses ---
with tab3:
    left, right = st.columns(2)

    with left:
        with st.form("primary_diagnosis_form", clear_on_submit=True):
            diag_type = st.selectbox("Primary diagnosis", primary_options, key="primary_diagnosis_type")
            diag_date = st.date_input(
                "Date of primary diagnosis",
                value=date.today(),
                min_value=MIN_DATE,
                max_value=date.today(),
                key="primary_diagnosis_date",
                format="DD-MM-YYYY",
            )

            if st.form_submit_button("Add primary diagnosis"):
                add_record(
                    "current_diagnoses",
                    {
                        "patient_id": st.session_state.patient_id,
                        "diagnosis_group": "primary",
                        "diagnosis_type": diag_type,
                        "diagnosis_date": str(diag_date),
                    },
                    diag_type,
                    "Select a primary diagnosis.",
                )

    with right:
        with st.form("secondary_diagnosis_form", clear_on_submit=True):
            comorb_type = st.selectbox(
                "Secondary diagnosis/comorbidity",
                secondary_options,
                key="secondary_diagnosis_type",
            )
            comorb_date = st.date_input(
                "Date of secondary diagnosis/comorbidity",
                value=date.today(),
                min_value=MIN_DATE,
                max_value=date.today(),
                key="secondary_diagnosis_date",
                format="DD-MM-YYYY",
            )

            if st.form_submit_button("Add secondary diagnosis/comorbidity"):
                add_record(
                    "current_diagnoses",
                    {
                        "patient_id": st.session_state.patient_id,
                        "diagnosis_group": "comorbidity",
                        "diagnosis_type": comorb_type,
                        "diagnosis_date": str(comorb_date),
                    },
                    comorb_type,
                    "Select a comorbidity.",
                )


# --- Tab 4: Procedures ---
with tab4:
    left, right = st.columns(2)

    with left:
        with st.form("surgery_form", clear_on_submit=True):
            surg_type = st.selectbox("Surgery", surgery_options, key="surgery_type")
            surg_date = st.date_input(
                "Date of surgery",
                value=date.today(),
                min_value=MIN_DATE,
                max_value=date.today(),
                key="surgery_date",
                format="DD-MM-YYYY",
            )

            if st.form_submit_button("Add surgery"):
                add_record(
                    "current_procedures",
                    {
                        "patient_id": st.session_state.patient_id,
                        "procedure_group": "surgery",
                        "procedure_type": surg_type,
                        "procedure_date": str(surg_date),
                    },
                    surg_type,
                    "Select a surgery.",
                )

    with right:
        with st.form("intervention_form", clear_on_submit=True):
            interv_type = st.selectbox("Intervention", intervention_options, key="intervention_type")
            interv_date = st.date_input(
                "Date of intervention",
                value=date.today(),
                min_value=MIN_DATE,
                max_value=date.today(),
                key="intervention_date",
                format="DD-MM-YYYY",
            )

            if st.form_submit_button("Add intervention"):
                add_record(
                    "current_procedures",
                    {
                        "patient_id": st.session_state.patient_id,
                        "procedure_group": "intervention",
                        "procedure_type": interv_type,
                        "procedure_date": str(interv_date),
                    },
                    interv_type,
                    "Select an intervention.",
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


# ---------- Global Save ----------
st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
save_record = st.button("Save record", key="save_record_btn")

if save_record:
    if not st.session_state.patient_id:
        st.error("Patient ID is missing.")
    else:
        demographics_df = pd.DataFrame([{
            "patient_id": st.session_state.patient_id,
            "first_name": st.session_state.first_name_input,
            "last_name": st.session_state.last_name_input,
            "dob": str(st.session_state.dob_input),
            "sex": st.session_state.sex_input,
        }])

        upsert_demographics_csv(demographics_df, OUT_DEMOGRAPHICS_CSV)

        if st.session_state.current_events:
            append_csv(pd.DataFrame(st.session_state.current_events), OUT_EVENTS_CSV)

        if st.session_state.current_diagnoses:
            append_csv(pd.DataFrame(st.session_state.current_diagnoses), OUT_DIAGNOSES_CSV)

        if st.session_state.current_procedures:
            append_csv(pd.DataFrame(st.session_state.current_procedures), OUT_PROCEDURES_CSV)

        if st.session_state.current_tests:
            append_csv(pd.DataFrame(st.session_state.current_tests), OUT_TESTS_CSV)

        if st.session_state.current_test_values:
            append_csv(pd.DataFrame(st.session_state.current_test_values), OUT_TEST_VALUES_CSV)

        if st.session_state.current_medications:
            append_csv(pd.DataFrame(st.session_state.current_medications), OUT_MEDICATIONS_CSV)

        st.success("Record saved")