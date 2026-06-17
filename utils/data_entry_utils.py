import os
import uuid
from datetime import date
import pandas as pd
import streamlit as st
import time, json
from pathlib import Path

# ---------- Configuration ----------
REFERENCE_PATH = st.session_state['clasp.REFERENCE_PATH'] 
OUT_PATH = st.session_state['clasp.OUT_PATH']


def load_options(label: str, group_value: str | None = None) -> list[str]:
    path = REFERENCE_PATH / f"{label.lower()}_reference.csv"

    df = pd.read_csv(path)

    parameter_col = "parameter"
    group_col = "entry_type"

    df[parameter_col] = df[parameter_col].astype(str).str.strip()

    # only filter if both a group value AND the column exist
    if group_value is not None and group_col in df.columns:
        df[group_col] = (
            df[group_col]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        df = df[df[group_col] == group_value.lower()]

    options = df[parameter_col].dropna().tolist()

    return options


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
    df = pd.read_csv(DEMOGRAPHICS_PATH, dtype={"patient_id": str})
    df.columns = [c.strip() for c in df.columns]
    df = df.loc[df['data_entered'] == False]

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
    st.session_state['data_entry.sex'] = sex if sex in ["F", "M", "Other"] else None
    st.session_state['data_entry.dob'] = dob 

    for entry in st.session_state['data_entry.CLINICAL_ENTRIES']:
        st.session_state[f'data_entry.current_{entry.lower()}'] = []


def add_record(
        state_key: str,
        record: dict,
        full_requirement_check: bool = True
    ) -> None:
    # normalize keys: replace spaces with underscores
    record = {
        (k.replace(" ", "_") if isinstance(k, str) else k): v.replace(" ", "_") if isinstance(v, str) else v
        for k, v in record.items()
    }


    patient_id = st.session_state.get("data_entry.patient_id")
    if not patient_id:
        st.error("Patient ID is required.")
        return

    date_key = next((k for k in record if "date" in k.lower()), None)
    date_value = record.get(date_key)

    if date_value is None or date_value == "" or pd.isna(date_value) or str(date_value) == "None":
        st.warning("Date is required")
        return

    excluded_keys = {date_key, "patient_id"}

    other_values = {
        k: v for k, v in record.items()
        if k not in excluded_keys
    }

    if full_requirement_check:
        missing = [k for k, v in other_values.items() if v is None or v == "" or str(v) == "None"]
        if missing:
            st.warning("Missing entries")
            return
    else:
        has_at_least_one_other = any(
            v is not None and v != "" and str(v) != "None"
            for v in other_values.values()
        )
        if not has_at_least_one_other:
            st.warning("Missing entries")
            return

    current = st.session_state.get(state_key, [])

    if record in current:
        return

    current.append(record)
    st.session_state[state_key] = current

    

def save_data_entry():
    demographics_df = pd.DataFrame([{
        "patient_id": st.session_state['data_entry.patient_id'],
        "first_name": st.session_state['data_entry.first_name'],
        "last_name": st.session_state['data_entry.last_name'],
        "dob": str(st.session_state['data_entry.dob']),
        "sex": st.session_state['data_entry.sex'],
        "data_entered":True
    }])


    for entry in st.session_state['data_entry.CLINICAL_ENTRIES']:
        entry = entry.lower()
        if entry == 'demographics':
            upsert_demographics_csv(demographics_df, OUT_PATH /f'{entry.lower()}.csv')
        else:
            if len(st.session_state[f'data_entry.current_{entry.lower()}'])>0:
                df = pd.DataFrame(st.session_state[f'data_entry.current_{entry.lower()}'])
                group_cols = ["patient_id", f"{entry}_date", f"{entry}_type"]

                if f"{entry}_type" in df.columns:
                    df = df.groupby(group_cols).last().reset_index()
                else:
                    df = df.groupby(["patient_id", f"{entry}_date"]).last().reset_index()

                append_csv(df, OUT_PATH /f'{entry.lower()}.csv')



def choose_entry_format(parameter, entry_type, value, label, value_field, key):
    if entry_type == 'numeric':
        entry = st.number_input(
            parameter,
            key=key,
            value=None,
            placeholder="Enter a value"
        )
    
    elif entry_type == 'list':
        entry = st.selectbox(
            parameter,
            value,
            key=key,
            index=None
        )

    return entry

def create_multi_form(
        label,
        forms: dict,  # {"major": [...], "minor": [...]}
    ):
    
    state_key = f'data_entry.current_{label.lower()}'
    type_field = f'{label.lower()}_type'
    value_field = f'{label.lower()}_value'
    date_field = f'{label.lower()}_date'
    cols = st.columns(len(forms))

    for col, form_name in zip(cols, forms):
        display_name = form_name.capitalize()
        with col:
            with st.form(f"{label}_{form_name}", clear_on_submit=True):

                options = load_options(label, form_name)
                entry = st.selectbox(
                    form_name,
                    options,
                    key=f"data_entry.{value_field}_{form_name}",
                    index=None
                )
                

                entry_date = st.date_input(
                    f"Date of {display_name} {label}", 
                    value=None,
                    min_value=MIN_DATE,
                    max_value=date.today(),
                    key=f"data_entry.{date_field}_{form_name}",
                    format="DD-MM-YYYY",
                )

                if st.form_submit_button(f"Add {display_name} {label}"):
                    record = {
                        "patient_id": st.session_state['data_entry.patient_id'],
                        date_field: str(entry_date),
                        type_field: form_name,
                        value_field: entry,
                    }
                    add_record(
                        state_key = state_key,
                        record = record,
                        full_requirement_check=True
                    )
    df = pd.DataFrame(st.session_state[f'{state_key}'])
    if not df.empty:
        columns = {col:col.replace('_',' ').title().replace('Id','ID') for col in df.columns}
        st.dataframe(df.rename(columns=columns).set_index('Patient ID'), use_container_width=True)


def create_single_form(label, num_cols=3):

    state_key = f"data_entry.current_{label.lower()}"
    value_field = f"{label.lower()}_value"
    date_field = f"{label.lower()}_date"

    file_path = REFERENCE_PATH/ f"{label.lower()}_reference.json"

    with open(file_path, "r") as f:
        forms_items = json.load(f)

    with st.form(f"{label}_form", clear_on_submit=True):

        entry_date = st.date_input(
            f"Date of {label}",
            min_value=MIN_DATE,
            max_value=date.today(),
            key=f"data_entry.{date_field}",
            format="DD-MM-YYYY", 
            value=None
        )

        n = len(forms_items)

        num_cols = 2 if n < 3 else 3

        num_rows = (n + num_cols - 1) // num_cols
        for i in range(num_rows):
            cols = st.columns(num_cols)
            row_items = forms_items[i*num_cols:(i+1)*num_cols]

            for col, row_item in zip(cols, row_items):
                parameter, entry_type, value = row_item.values()
                key = f"data_entry.{value_field}_{parameter}"

                with col:
                    entry = choose_entry_format(parameter, entry_type, value, label, value_field, key)

        if st.form_submit_button(f"Add {label}"):
            record = {
                "patient_id": st.session_state["data_entry.patient_id"],
                date_field: str(entry_date),
            }
            for form_item in forms_items:
                parameter, entry_type, value = form_item.values()
                key = f"data_entry.{value_field}_{parameter}"
                record[parameter] = st.session_state[f"data_entry.{value_field}_{parameter}"]
                
            add_record(
                state_key = state_key,
                record = record,
                full_requirement_check = False
            )

    df = pd.DataFrame(st.session_state.get(state_key, []))

    if not df.empty:
        date_col = f"{label.lower()}_date"

        df = df.groupby(["patient_id", date_col]).last().reset_index()

        df = df.rename(columns={
            "patient_id": "Patient ID",
            date_col: f"{label} Date"
        })

        df = df.set_index("Patient ID")
        st.dataframe(df, use_container_width=True)