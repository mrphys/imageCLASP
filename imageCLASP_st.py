import streamlit as st
import pandas as pd
from tinydb import TinyDB, Query
import plotly.express as px
import plotly.graph_objects as go
from math import nan
import pydicom as dicom
import numpy as np
import requests
import io
from pydicom.uid import generate_uid, ExplicitVRLittleEndian
from datetime import datetime


DB_PATH = "/Users/vivekmuthurangu/Downloads/image_clasp_db.json"

my_palette = [
    px.colors.qualitative.Plotly[9],
    px.colors.qualitative.Plotly[5],
    px.colors.qualitative.Plotly[2],
    px.colors.qualitative.Plotly[3],
    px.colors.qualitative.Plotly[4],
    px.colors.qualitative.Plotly[5],
    px.colors.qualitative.Plotly[6],
    px.colors.qualitative.Plotly[7],
    px.colors.qualitative.Plotly[8],
]

def send_series_to_orthanc(new_dcm, old_dcm, ORTHANC, AUTH):
    new_series_uid = generate_uid()
    for i in range(len(new_dcm)):
        old_dcm[i].SeriesInstanceUID = new_series_uid
        old_dcm[i].SeriesDescription = 'Processed Series'
        old_dcm[i].PixelData = new_dcm[i].astype(np.uint16).tobytes()
        old_dcm[i].SOPInstanceUID = generate_uid()

        old_dcm[i].file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        old_dcm[i].is_little_endian = True
        old_dcm[i].is_implicit_VR = False
        
        buffer = io.BytesIO()
        old_dcm[i].save_as(buffer)
        buffer.seek(0)
        upload = requests.post(f"{ORTHANC}/instances",data=buffer.read(),auth=AUTH,headers={"Content-Type": "application/dicom", "Expect": ""})
        upload.raise_for_status()
    return new_series_uid


def get_orthanc_series_data_from_uid(series_uid, ORTHANC, AUTH):
    payload = {
        "Level": "Series",
        "Expand": True,
        "Query": {
            "SeriesInstanceUID": series_uid
        }
    }

    r = requests.post(f"{ORTHANC}/tools/find", json=payload, auth=AUTH)
    r.raise_for_status()
    results = r.json()

    if not results:
        return None

    return results[0]


def query_orthanc(DB_PATH, ORTHANC, AUTH):
    db = TinyDB(DB_PATH)
    Study = Query()

    payload = {
        "Level": "Study",
        "Expand": True,
        "Query": {}
    }

    r = requests.post(f"{ORTHANC}/tools/find", json=payload, auth=AUTH)
    r.raise_for_status()

    studies = r.json()
    target_list = ['short', 'SAX', 'KT']

    db_studies_present = []
    for study in db:
        db_studies_present.append(study["orthanc_study_id"])

    for study in studies:
        if study["ID"] not in db_studies_present:
            print(study["ID"])
            dob = study["PatientMainDicomTags"].get("PatientBirthDate")
            dos = study["MainDicomTags"].get("StudyDate")
            if dob is None or dos is None:
                age = 'None'
            else:
                age = datetime.strptime(dos, "%Y%m%d") - datetime.strptime(dob, "%Y%m%d")
                age = str(np.round(age.days/365.25))
            record = {
                "orthanc_study_id": study["ID"],
                "study_uid": study["MainDicomTags"].get("StudyInstanceUID"),
                "patient_name": study["PatientMainDicomTags"].get("PatientName"),
                "patient_id": study["PatientMainDicomTags"].get("PatientID"),
                "patient_sex": study["PatientMainDicomTags"].get("PatientSex"),
                "patient_dob": dob,
                "patient_age": age,
                "study_date": dos,
                "series": []
            }

            # Get series belonging to this study
            study_id = study["ID"]
            series_list = requests.get(f"{ORTHANC}/studies/{study_id}/series",auth=AUTH).json()

            for i in range(len(series_list)):
                series_info = series_list[i]
                series_desc = series_info["MainDicomTags"].get("SeriesDescription")
                series_record = {
                    "orthanc_series_id": series_info["ID"],
                    "series_uid": series_info["MainDicomTags"].get("SeriesInstanceUID"),
                    "series_description": series_desc,
                    "is_target": False,
                    "DL_processed": False,
                }

                # this is where a piipeline would be inserted
                if any(w in series_desc for w in target_list):
                    series_record["is_target"] = True
                    series_record["DL_processed"] = True
                    instances = requests.get(f"{ORTHANC}/series/{series_info['ID']}/instances",auth=AUTH).json()

                    new_series_ims = []
                    old_dcms = []
                    for inst in instances:

                        instance_id = inst["ID"]

                        r = requests.get(f"{ORTHANC}/instances/{instance_id}/file", auth=AUTH)

                        # Read DICOM in memory
                        ds = dicom.dcmread(io.BytesIO(r.content))
                        #this is where DL segmentation would be inserted
                        new_im = abs(np.float16(ds.pixel_array)-np.max(ds.pixel_array))#DL segmentation
                        new_series_ims.append(new_im)
                        old_dcms.append(ds)
                    
                    processed_uid = send_series_to_orthanc(new_series_ims, old_dcms, ORTHANC, AUTH)
                    new_series_json = get_orthanc_series_data_from_uid(processed_uid, ORTHANC, AUTH)
                    new_record = {
                    "orthanc_series_id": new_series_json["ID"],
                    "series_uid": new_series_json["MainDicomTags"].get("SeriesInstanceUID"),
                    "series_description": new_series_json["MainDicomTags"].get("SeriesDescription"),
                    "is_target": True,
                    "DL_processed": True,
                    "roundel_processed": False,
                    "associated_original_series_uid": series_record["series_uid"]
                    }
                    record["series"].append(new_record)
                    
                db.upsert(record, Study.study_uid == record["study_uid"])
                    

def load_db_rows():
    db = TinyDB(DB_PATH)
    rows = []

    for study in db:
        series = study.get("series", [])
        row = {
            "patient_id": study.get("patient_id"),
            "patient_sex": study.get("patient_sex"),
            "age": study.get("patient_age"),
            "n_series": len(series),
            "DL_processed": any(s.get("DL_processed", False) for s in series),
            "roundel_processed": any(s.get("roundel_processed", False) for s in series),
        }
        rows.append(row)

    db.close()
    df = pd.DataFrame(rows)
    df = df.replace("", pd.NA)
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    return df


def hist(df, var, bin_start, bin_end, bin_size, colors):
    fig = go.Figure(
        go.Histogram(
            x=df[var].dropna(),
            xbins=dict(start=bin_start, end=bin_end, size=bin_size),
            marker_color=colors,
        )
    )
    fig.update_xaxes(range=[bin_start, bin_end])
    fig.update_xaxes(tickfont=dict(size=18))
    fig.update_yaxes(tickfont=dict(size=18))
    fig.update_layout(margin=dict(t=10), height=300)
    return fig


def pie(df, var, colors):
    counts = df[var].fillna("Unknown").value_counts().reset_index()
    counts.columns = [var, "count"]

    if var == "patient_sex":
        counts[var] = counts[var].replace({
            "M": "Male",
            "F": "Female"
        })

    if var == "DL_processed" or var == "roundel_processed":
        counts[var] = counts[var].replace({
            True: "Yes",
            False: "No"
        })
        
    fig = px.pie(counts, names=var, values="count")
    fig.update_traces(textposition="inside", textinfo="percent+label",textfont=dict(size=14,family="Helvetica"))
    fig.update_traces(marker_colors=colors)
    fig.update_layout(showlegend=False, margin=dict(t=10), height=300)
    return fig



ORTHANC = "http://localhost:8042"
AUTH = ("orthanc","orthanc")

if "df" not in st.session_state:
    query_orthanc(DB_PATH, ORTHANC, AUTH)
    st.session_state["df"] = load_db_rows()

with st.sidebar:
    st.header("Filtering")
    
    if st.button("Query Orthanc"):
        query_orthanc(DB_PATH, ORTHANC, AUTH)
        

    if st.button("Query CLASP"):
        st.session_state["df"] = load_db_rows()
        st.rerun()

    df = st.session_state["df"]

    sex = st.selectbox("Sex", ["All", "M", "F", "Unknown"])
    processed_only = st.selectbox("Processed only", ["All", "Yes", "No"])
    roundel_only = st.selectbox("Roundel only", ["All", "Yes", "No"])
    age_min, age_max = st.slider("Age range", 0, 100, (0, 100))

    filtered_df = df.copy()

    if sex != "All":
        filtered_df = filtered_df[filtered_df["patient_sex"] == sex]

    if processed_only == "Yes":
        filtered_df = filtered_df[filtered_df["DL_processed"]]
    elif processed_only == "No":
        filtered_df = filtered_df[~filtered_df["DL_processed"]]

    if roundel_only == "Yes":
        filtered_df = filtered_df[filtered_df["roundel_processed"]]
    elif roundel_only == "No":
        filtered_df = filtered_df[~filtered_df["roundel_processed"]]

    filtered_df = filtered_df[
        (filtered_df["age"].isna()) |
        ((filtered_df["age"] >= age_min) & (filtered_df["age"] <= age_max))
    ]

    st.metric("Filtered Studies", len(filtered_df))

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    "<h1 style='text-align: center;margin-bottom: 40px;'>ImageCLASP dashboard</h1>",
    unsafe_allow_html=True
)

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown(
        "<h4 style='text-align: center; margin-bottom: 0px;'>Summary</h4>",
        unsafe_allow_html=True
    )
    st.metric("Total Studies", len(df))
    median_age = df["age"].median()
    age_min = df["age"].min()
    age_max = df["age"].max()
    st.metric("Median Age",f"{median_age:.1f} ({age_min:.0f}-{age_max:.0f})")
    st.metric("Median Series", df["n_series"].median())
    

with col2:
    st.markdown(
        "<h4 style='text-align: center; margin-bottom: 0px;'>Age Distribution</h4>",
        unsafe_allow_html=True
    )
    st.plotly_chart(hist(df, "age", 0, 50, 5, my_palette[3]))

col3, col4, col5 = st.columns([1, 1, 1], gap="large")

with col3:
    st.markdown(
        "<h4 style='text-align: center; margin-bottom: 0px;'>Sex Balance</h4>",
        unsafe_allow_html=True
    )
    st.plotly_chart(pie(df, "patient_sex", my_palette))

with col4:
    st.markdown(
        "<h4 style='text-align: center; margin-bottom: 0px;'>DL processed</h4>",
        unsafe_allow_html=True
    )
    st.plotly_chart(pie(df, "DL_processed", my_palette))

with col5:
    st.markdown(
        "<h4 style='text-align: center; margin-bottom: 0px;'>Roundel</h4>",
        unsafe_allow_html=True
    )
    st.plotly_chart(pie(df, "roundel_processed", my_palette))