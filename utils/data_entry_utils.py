import os
import uuid
from datetime import date
import pandas as pd
import streamlit as st

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

DEMOGRAPHICS_PATH = f"{OUT_PATH}/demographics.csv"

MIN_DATE = date(1900, 1, 1)

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

major_events = load_options(EVENTS_CSV, "event_type", "event_group", "major")
minor_events = load_options(EVENTS_CSV, "event_type", "event_group", "minor")
primary_options = load_options(DIAGNOSES_CSV, "diagnosis_type", "group", "primary")
secondary_options = load_options(DIAGNOSES_CSV, "diagnosis_type", "group", "secondary")
surgery_options = load_options(PROCEDURES_CSV, "procedure_type", "procedure_group", "surgery")
intervention_options = load_options(PROCEDURES_CSV, "procedure_type", "procedure_group", "intervention")
blood_test_options = load_options(BLOOD_TESTS_CSV, "blood_test_name")
blood_units_options = load_options(BLOOD_UNITS_CSV, "units")
medication_options = load_options(MEDICATIONS_CSV, "medication_name")

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


def load_demographics() -> pd.DataFrame:
    df = pd.read_csv(DEMOGRAPHICS_PATH)
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
    
    if df.empty:
        st.success("All Patients Data Entered!")
        st.stop()
    return df




def make_test_id() -> str:
    return str(uuid.uuid4())[:8]


def load_current_patient_from_csv() -> None:
    demo_df = load_demographics()
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

    st.session_state['patient_id'] = patient_id
    st.session_state['first_name'] = first_name
    st.session_state['last_name'] = last_name
    st.session_state['sex'] = sex if sex in ["F", "M", "Other", 'Missing'] else "Missing"
    st.session_state['dob'] = dob

    for entry in clinical_entries:
        st.session_state[f'current_{entry}'] = []

def process_pending_navigation() -> None:
    demo_df = load_demographics()
    pending_nav = st.session_state.get("pending_nav", None)
    if pending_nav == "next":
        if st.session_state.patient_idx < len(demo_df) - 1:
            st.session_state.patient_idx += 1

        load_current_patient_from_csv()
        st.session_state.pending_nav = None
        st.rerun()

    elif pending_nav == "prev":
        if st.session_state.patient_idx > 0:
            st.session_state.patient_idx -= 1
        load_current_patient_from_csv()
        st.session_state.pending_nav = None
        st.rerun()


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

def save_data_entry():
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