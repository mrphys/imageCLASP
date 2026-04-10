import os
import uuid
from datetime import date

import pandas as pd
import streamlit as st
from utils.pipeline import *
from utils.theme_utils import *

st.set_page_config(layout="wide")

load_theme(secondary="#155a8a",
    secondary_hover="#1F4264",
    secondary_active="#12324D"
    )

# ---------- Configuration ----------
REFERENCE_PATH = 'reference'
EVENTS_CSV = f"{REFERENCE_PATH}/events_list.csv"
DIAGNOSES_CSV = f"{REFERENCE_PATH}/diagnoses_list.csv"
PROCEDURES_CSV = f"{REFERENCE_PATH}/procedures_list.csv"
BLOOD_TESTS_CSV = f"{REFERENCE_PATH}/blood_tests_list.csv"
BLOOD_UNITS_CSV = f"{REFERENCE_PATH}/blood_test_unit.csv"
MEDICATIONS_CSV = f"{REFERENCE_PATH}/medications_list.csv"

OUT_PATH = 'tables'
clinical_entries = ['demographics','events', 'diagnoses', 'procedures','tests','test_values','medications']

MIN_DATE = date(1900, 1, 1)

# ---------- Helper Functions ----------
def append_csv(df: pd.DataFrame, path: str, subset=None) -> None:
    file_exists = os.path.exists(path)

    # append new data
    df.to_csv(path, mode="a", index=False, header=not file_exists)

    if file_exists:
        # reload full file and drop duplicates
        full = pd.read_csv(path)

        if subset is None:
            full = full.drop_duplicates()
        else:
            full = full.drop_duplicates(subset=subset)

        full.to_csv(path, index=False)

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


def load_demographics(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.loc[df['data_entered'] == False]
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
        dob = date(1900, 1, 1)

    st.session_state.patient_id = patient_id
    st.session_state.first_name = first_name
    st.session_state.last_name = last_name
    st.session_state.sex = sex if sex in ["F", "M", "Other", 'Missing'] else "Missing"
    st.session_state.dob = dob

    for entry in clinical_entries:
        st.session_state[f'current_{entry}'] = []

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

    current = st.session_state[state_list_name]

    if record_dict in current:
        return

    current.append(record_dict)

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
    "dob": date(1900, 1, 1),
    "sex": "-",
    "first_name": "",
    "last_name": "",
    "dob": date(1900, 1, 1),
    "sex": "Missing",
    "pending_nav": None,
}

for key, default_val in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = default_val


# ---------- Load Data ----------
demo_df = load_demographics(f'{OUT_PATH}/demographics.csv')
if demo_df.empty:
    for key in DEFAULT_STATE.keys():
        st.session_state.pop(key)
    st.success("All Patients Data Entered!")
    st.stop()

if not st.session_state.patient_id:
    load_current_patient_from_csv()

process_pending_navigation()

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
st_header("CLASP Clinical Data")

# ---------- Header ----------
top1, top2 = st.columns([3, 1])

with top1:
    st.markdown(f"### Patient ID {st.session_state.patient_id} - {st.session_state.patient_idx + 1} of {len(demo_df)}")

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

def multi_group_form(
    label,
    groups: dict,  # {"major": [...], "minor": [...]}
):
    
    state_key = f'current_{label.lower()}'
    group_key = f'{label.lower()}_group'
    type_field = f'{label.lower()}_type'
    date_field = f'{label.lower()}_date'
    cols = st.columns(len(groups))

    for col, (group_name, options) in zip(cols, groups.items()):
        with col:
            with st.form(f"{label}_{group_name}", clear_on_submit=True):
                entry_type = st.selectbox(
                    f"{group_name.capitalize()} {label}",
                    options,
                    key=f"{type_field}_{group_name}",
                )

                entry_date = st.date_input(
                    f"Date of {group_name} {label}",
                    value=date.today(),
                    min_value=MIN_DATE,
                    max_value=date.today(),
                    key=f"{date_field}_{group_name}",
                    format="DD-MM-YYYY",
                )

                if st.form_submit_button(f"Add {group_name} {label}"):
                    add_record(
                        state_key,
                        {
                            "patient_id": st.session_state.patient_id,
                            group_key: group_name,
                            type_field: entry_type,
                            date_field: str(entry_date),
                        },
                        entry_type,
                        f"Select a {group_name} {label}.",
                    )
    df = pd.DataFrame(st.session_state[state_key])
    if not df.empty:
        st.dataframe(df[['patient_id',type_field, date_field]].rename(columns={
            type_field: f"{label} Type",
            date_field: f"{label} Date"
        }).set_index(f"patient_id"))

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
        demographics_df = pd.DataFrame([{
            "patient_id": st.session_state.patient_id,
            "first_name": st.session_state.first_name,
            "last_name": st.session_state.last_name,
            "dob": str(st.session_state.dob),
            "sex": st.session_state.sex,
            "data_entered":True
        }])


        for entry in clinical_entries:
            if entry == 'demographics':
                upsert_demographics_csv(demographics_df, f'{OUT_PATH}/{entry}.csv')
            else:
                if len(st.session_state[f'current_{entry}'])>0:
                    append_csv(pd.DataFrame(st.session_state[f'current_{entry}']), f'{OUT_PATH}/{entry}.csv')
            st.session_state[f'current_{entry}'] = []

        st.success("Record saved")
        st.rerun()