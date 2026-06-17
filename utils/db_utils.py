from tinydb import TinyDB, Query
import pandas as pd
import numpy as np
import os
import glob
import streamlit as st
DB_PATH = st.session_state['clasp.DB_PATH']
DEMOGRAPHICS_PATH = st.session_state["clasp.DEMOGRAPHICS_PATH"]



def fetch_db_studies():
    db = TinyDB(DB_PATH)
    studies = db.all()
    studies = [Study.from_dict(study) for study in studies]
    for study in studies:
        series_dict = {sid:Series.from_dict(series) for sid, series in study.series_dict.items()}
        study.series_dict = series_dict
    return studies

def fetch_db_study(study_id):
    db = TinyDB(DB_PATH)
    study = [study for study in db.all() if study['orthanc_study_id'] == study_id][0] 
    study = Study.from_dict(study)
    study.series_dict = {sid:Series.from_dict(series) for sid, series in study.series_dict.items()}
    return study

def fetch_db_series(study, series_orthanc_id):
    return [series for sid, series in study.series_dict.items() if sid == series_orthanc_id][0]

def get_entered_patients():
    demo_path = DEMOGRAPHICS_PATH
    if os.path.exists(demo_path):
        df = pd.read_csv(demo_path)
        entered_patients = (df.loc[df['data_entered'] == True]).patient_id
    else:
        entered_patients = []
    return entered_patients

class Series:
    def __init__(self, orthanc_series_info=None, **kwargs):
        if orthanc_series_info is not None:
            base = {
                "orthanc_series_id": orthanc_series_info["ID"],
                "series_uid": orthanc_series_info["MainDicomTags"].get("SeriesInstanceUID"),
                "series_description": orthanc_series_info["MainDicomTags"].get("SeriesDescription"),
            }
            self.__dict__.update(base)

        self.__dict__.update(kwargs)

        self.series_type = getattr(self, "series_type", None)
        self.series_group = getattr(self, "series_group", None)
        self.series_orientation = getattr(self, "series_orientation", None)
        self.dl_orthanc_id = getattr(self, "dl_orthanc_id", None)
        self.roundel_orthanc_id = getattr(self, "roundel_orthanc_id", None)

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)

    def to_dict(self):
        return {
            "orthanc_series_id": self.orthanc_series_id,
            "series_uid": self.series_uid,
            "series_description": self.series_description,
            "series_type": self.series_type,
            "series_group": self.series_group,
            "series_orientation": self.series_orientation,
            "dl_orthanc_id": self.dl_orthanc_id,
            "roundel_orthanc_id": self.roundel_orthanc_id,
        }

class Study:
    def __init__(self, orthanc_study_info=None, **kwargs):
        # handle series separately
        self.series_dict = kwargs.pop("series_dict", {})

        # populate from study_info if provided
        if orthanc_study_info is not None:

            patient_id = orthanc_study_info["PatientMainDicomTags"].get("PatientID", 0)
            patient_id = 0 if patient_id == "" else patient_id

            base = {
                "orthanc_study_id": orthanc_study_info["ID"],
                "study_uid": orthanc_study_info["MainDicomTags"].get("StudyInstanceUID"),
                "patient_name": orthanc_study_info["PatientMainDicomTags"].get("PatientName"),
                "patient_id": str(int(patient_id)),
                "patient_sex": orthanc_study_info["PatientMainDicomTags"].get("PatientSex"),
                "patient_dob": orthanc_study_info["PatientMainDicomTags"].get("PatientBirthDate"),
                "study_date": orthanc_study_info["MainDicomTags"].get("StudyDate"),
            }
            self.__dict__.update(base)

        # apply kwargs (override or extend)
        self.__dict__.update(kwargs)

        # ensure attributes exist
        self.orthanc_study_id = getattr(self, "orthanc_study_id", None)
        self.study_uid = getattr(self, "study_uid", None)
        self.patient_name = getattr(self, "patient_name", None)
        self.patient_id = getattr(self, "patient_id", None)
        self.patient_sex = getattr(self, "patient_sex", None)
        self.patient_dob = getattr(self, "patient_dob", None)
        self.study_date = getattr(self, "study_date", None)

        # derived field
        self.patient_age = self.compute_age(
            self.study_date,
            self.patient_dob,
        )

    @staticmethod
    def compute_age(dos, dob):
        if not dos or not dob:
            return np.nan
        dos_dt = pd.to_datetime(dos, format="%Y%m%d", errors="coerce")
        dob_dt = pd.to_datetime(dob, format="%Y%m%d", errors="coerce")
        if pd.isna(dos_dt) or pd.isna(dob_dt):
            return np.nan
        return (dos_dt - dob_dt).days // 365

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)

    def to_dict(self):
        return {
            "orthanc_study_id": self.orthanc_study_id,
            "study_uid": self.study_uid,
            "patient_name": self.patient_name,
            "patient_id": self.patient_id,
            "patient_sex": self.patient_sex,
            "patient_dob": self.patient_dob,
            "study_date": self.study_date,
            "series_dict": {
                k: (v.to_dict() if hasattr(v, "to_dict") else v)
                for k, v in self.series_dict.items()
            },
        }
   