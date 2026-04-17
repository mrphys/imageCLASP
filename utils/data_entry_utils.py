import os
import uuid
from datetime import date
import pandas as pd
import streamlit as st
import time
# ---------- Configuration ----------
REFERENCE_PATH = st.session_state['clasp.REFERENCE_PATH'] 
EVENTS_CSV = f"{REFERENCE_PATH}/events_list.csv"
DIAGNOSES_CSV = f"{REFERENCE_PATH}/diagnoses_list.csv"
PROCEDURES_CSV = f"{REFERENCE_PATH}/procedures_list.csv"
BLOOD_TESTS_CSV = f"{REFERENCE_PATH}/blood_tests_list.csv"
BLOOD_UNITS_CSV = f"{REFERENCE_PATH}/blood_test_unit.csv"
MEDICATIONS_CSV = f"{REFERENCE_PATH}/medications_list.csv"

OUT_PATH = st.session_state['clasp.OUT_PATH']

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
medication_options = load_options(MEDICATIONS_CSV, "medication_name")


multi_forms = {
    "Events": {
        "Major": major_events,
        "Minor": minor_events,
    },
    "Diagnoses": {
        "Primary": primary_options,
        "Secondary": secondary_options,
    },
    "Procedures": {
        "Surgical": surgery_options,
        "Interventional": intervention_options,
    }
}

single_forms = {
    "Medications":{
        "Medication": medication_options,
        "Dose": 0,
        "Frequency": ["OD", "BD", "TDS", "QDS"]
    }, 

    "Catheter":{
        "RA Area": 0,
        "RV Pressure": 0,
        "PA Pressure": 0,
        "Cardiac Output": 0,
        "Cardiac Index": 0,
        "TR Vmax": 0,
        "PCWP": 0,
        "LVEDP": 0,
        "Shunt Present": [0, 1],
        "Shunt Type": ["None", "ASD", "VSD", "PDA", "Other"]
    }, 

    "Echo":{
        "LAVi": 0,
        "LV EF": 0,
        "LVEDD": 0,
        "LVESD": 0,
        "RA Area": 0,
        "RV Basal Diameter": 0,
        "TAPSE": 0,
        "S' Wave": 0,
        "E/e' Ratio": 0,
        "TR Vmax": 0,
        "RVSP": 0,
        "Diastolic Function": ["Normal", "Grade I", "Grade II", "Grade III"],
        "Valvular Disease": ["None", "Mild", "Moderate", "Severe"],
        "Regional Wall Motion": ["Normal", "Hypokinesia", "Akinesia", "Dyskinesia"]
    },
    "Bloods": {
        "Test": ['HB (g/L)','WBC'], # numerical
        "Value": 0
    },

}


CLINICAL_ENTRIES = ['Demographics'] + list(multi_forms.keys()) + list(single_forms.keys())
DEMOGRAPHICS_PATH = st.session_state['clasp.DEMOGRAPHICS_PATH']

MIN_DATE = date(1900, 1, 1)

def clear_on_patient_change():
    prev = st.session_state["data_entry.prev_patient"]
    curr = st.session_state["data_entry.current_patient"]

    if prev != curr:
        for key in list(st.session_state.keys()):
            if key.startswith("data_entry.") and key not in {
                "data_entry.current_patient",
                "data_entry.prev_patient",
            }:
                st.session_state.pop(key)

    st.session_state["data_entry.prev_patient"] = curr
    try:
        st.session_state.pop('data_entry.initialized')
    except:
        pass



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
        st.success("🎉 All data entries completed!")
        st.stop()
    return df


def make_test_id() -> str:
    return str(uuid.uuid4())[:8]


def init_patient_from_csv() -> None:
    demo_df = load_demographics()
    row = demo_df.iloc[st.session_state['data_entry.patient_idx']]

    patient_id = str(row.get("patient_id", "") or "")
    first_name = str(row.get("first_name", "") or "")
    last_name = str(row.get("last_name", "") or "")
    sex = str(row.get("sex", "F") or "F")

    dob_val = row.get("dob", "")
    if pd.notna(dob_val) and str(dob_val).strip():
        dob = pd.to_datetime(dob_val, dayfirst=True).date()
    else:
        dob = date(1900, 1, 1)

    st.session_state['data_entry.patient_id'] = patient_id
    st.session_state['data_entry.first_name'] = first_name
    st.session_state['data_entry.last_name'] = last_name
    st.session_state['data_entry.sex'] = sex if sex in ["F", "M", "Other", 'Missing'] else "Missing"
    st.session_state['data_entry.dob'] = dob

    for entry in CLINICAL_ENTRIES:
        st.session_state[f'data_entry.current_{entry.lower()}'] = []

def add_record(
        state_list_name: str,
        record_dict: dict,
        requirement_check,
        requirement_msg = '',
    ) -> None:

    if not st.session_state['data_entry.patient_id']:
        st.error("Patient ID is required.")
        return

    if not requirement_check:
        st.warning(requirement_msg)
        return

    current = st.session_state[f'{state_list_name}']

    if record_dict in current:
        return

    current.append(record_dict)




def save_data_entry():
    demographics_df = pd.DataFrame([{
        "patient_id": st.session_state['data_entry.patient_id'],
        "first_name": st.session_state['data_entry.first_name'],
        "last_name": st.session_state['data_entry.last_name'],
        "dob": str(st.session_state['data_entry.dob']),
        "sex": st.session_state['data_entry.sex'],
        "data_entered":True
    }])


    for entry in CLINICAL_ENTRIES:
        if entry == 'demographics':
            upsert_demographics_csv(demographics_df, f'{OUT_PATH}/{entry.lower()}.csv')
        else:
            if len(st.session_state[f'data_entry.current_{entry.lower()}'])>0:
                df = pd.DataFrame(st.session_state[f'data_entry.current_{entry.lower()}'])
                columns = {c: c.replace("_", " ").title().replace("Id", "ID") for c in df.columns}
                df = df.rename(columns=columns).set_index("Patient ID")
                df = df.groupby(["Patient ID", f"{entry} Date"]).last().reset_index()
                append_csv(df, f'{OUT_PATH}/{entry.lower()}.csv')


def create_multi_form(
        label,
        forms: dict,  # {"major": [...], "minor": [...]}
    ):
    
    state_key = f'data_entry.current_{label.lower()}'
    type_field = f'{label.lower()}_type'
    value_field = f'{label.lower()}_value'
    date_field = f'{label.lower()}_date'
    cols = st.columns(len(forms))

    for col, (form_name, options) in zip(cols, forms.items()):
        with col:
            with st.form(f"{label}_{form_name}", clear_on_submit=True):

                entry_date = st.date_input(
                    f"Date of {form_name} {label}", 
                    value=None,
                    min_value=MIN_DATE,
                    max_value=date.today(),
                    key=f"data_entry.{date_field}_{form_name}",
                    format="DD-MM-YYYY",
                )

                if options == 0:
                    entry_type = st.number_input(
                        f"{form_name} {label}",
                        key=f"data_entry.{value_field}_{form_name}", 
                        value=None, 
                        placeholder="Enter a value"
                    )
                else:
                    entry_type = st.selectbox(
                        f"{form_name} {label}",
                        options,
                        key=f"data_entry.{value_field}_{form_name}", index=None
                    )

                if st.form_submit_button(f"Add {form_name} {label}"):
                    add_record(
                        state_key,
                        {
                            "patient_id": st.session_state['data_entry.patient_id'],
                            date_field: str(entry_date),
                            type_field: form_name,
                            value_field: entry_type,
                        },
                        entry_type,
                        f"Select a {form_name} {label}.",
                    )
    df = pd.DataFrame(st.session_state[f'{state_key}'])
    if not df.empty:
        columns = {col:col.replace('_',' ').title().replace('Id','ID') for col in df.columns}
        st.dataframe(df.rename(columns=columns).set_index('Patient ID'), use_container_width=True)

def create_single_form(label, forms: dict, num_cols=3):

    state_key = f"data_entry.current_{label.lower()}"
    type_field = f"{label.lower()}_type"
    value_field = f"{label.lower()}_value"
    date_field = f"{label.lower()}_date"

    with st.form(f"{label}_form", clear_on_submit=True):

        entry_date = st.date_input(
            f"Date of {label}",
            min_value=MIN_DATE,
            max_value=date.today(),
            key=f"data_entry.{date_field}",
            format="DD-MM-YYYY", 
            value=None
        )

        forms_items = list(forms.items())
        n = len(forms_items)

        num_cols = 2 if n < 3 else (3 if n % 2 == 1 else 4)

        num_rows = (n + num_cols - 1) // num_cols
        for i in range(num_rows):
            cols = st.columns(num_cols)
            row_items = forms_items[i*num_cols:(i+1)*num_cols]

            for col, (form_name, options) in zip(cols, row_items):
                with col:
                    key = f"data_entry.{value_field}_{form_name}"

                    if options is None or options == 0:
                        st.number_input(f"{form_name}", key=key, value=None, placeholder="Enter a value")
                    else:
                        st.selectbox(f"{form_name}", options, key=key, index=None)

        submitted = st.form_submit_button(f"Add {label}")

    if submitted:
        record = {
            "patient_id": st.session_state["data_entry.patient_id"],
            date_field: str(entry_date),
        }
        for form_name, options in forms_items:
            key = f"data_entry.{value_field}_{form_name}"
            entry_type = st.session_state.get(key)
            record[form_name] = st.session_state[f"data_entry.{value_field}_{form_name}"]
            
        add_record(
            state_key,
            record,
            requirement_check = True
        )

    df = pd.DataFrame(st.session_state.get(state_key, []))
    if not df.empty:
        columns = {c: c.replace("_", " ").title().replace("Id", "ID") for c in df.columns}
        df = df.rename(columns=columns).set_index("Patient ID")
        df = df.groupby(["Patient ID", f"{label} Date"]).last()
        st.dataframe(df, use_container_width=True)