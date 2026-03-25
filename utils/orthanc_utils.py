import io
import numpy as np
import requests
import pydicom
from pydicom.uid import ExplicitVRLittleEndian, generate_uid
from tinydb import TinyDB, Query
import pandas as pd

DB_PATH = "./image_clasp_db.json"
ORTHANC = "http://localhost:8042"
AUTH = ("orthanc","orthanc")

SESSION = requests.Session()
SESSION.auth = AUTH
SESSION.trust_env = False


class Series:
    def __init__(self, series_info, series_label = None, sax_processed = None, roundel_processed = None):
        self.orthanc_id = series_info["ID"]
        self.uid = series_info["MainDicomTags"].get("SeriesInstanceUID")
        self.description = series_info["MainDicomTags"].get("SeriesDescription")
        self.series_label = series_label
        self.sax_processed = sax_processed
        self.roundel_processed = roundel_processed

    def to_dict(self):
        record = {
            "orthanc_series_id": self.orthanc_id,
            "series_uid": self.uid,
            "series_description": self.description,
            "series_label":self.series_label,
            "sax_processed":self.sax_processed,
            "roundel_processed":self.roundel_processed
        }
        return record


class Study:
    def __init__(self, study_info):
        self.orthanc_id = study_info["ID"]
        self.uid = study_info["MainDicomTags"].get("StudyInstanceUID")
        self.patient_name = study_info["PatientMainDicomTags"].get("PatientName")
        self.patient_id = study_info["PatientMainDicomTags"].get("PatientID")
        self.patient_sex = study_info["PatientMainDicomTags"].get("PatientSex")
        self.patient_dob = study_info["PatientMainDicomTags"].get("PatientBirthDate")
        self.study_date = study_info["MainDicomTags"].get("StudyDate")
        self.age = self.compute_age(self.study_date, self.patient_dob)
        self.series_list = []

    @staticmethod
    def compute_age(dos, dob):
        if not dos or not dob:
            return np.nan
        dos_dt = pd.to_datetime(dos, format="%Y%m%d", errors="coerce")
        dob_dt = pd.to_datetime(dob, format="%Y%m%d", errors="coerce")
        if pd.isna(dos_dt) or pd.isna(dob_dt):
            return np.nan
        return (dos_dt - dob_dt).days // 365

    def to_dict(self):
        return {
            "orthanc_study_id": self.orthanc_id,
            "study_uid": self.uid,
            "patient_name": self.patient_name,
            "patient_id": self.patient_id,
            "patient_sex": self.patient_sex,
            "patient_dob": self.patient_dob,
            "patient_age": self.age,
            "study_date": self.study_date,
            "series": [s.to_dict() for s in self.series_list]
        }

def fetch_studies():
    payload = {"Level": "Study", "Expand": True, "Query": {}}
    r = SESSION.post(f"{ORTHANC}/tools/find", json=payload, auth=AUTH)
    r.raise_for_status()
    return r.json()

def fetch_series_for_study(study_id):
    r = SESSION.get(f"{ORTHANC}/studies/{study_id}/series", auth=AUTH)
    r.raise_for_status()
    return r.json()

def fetch_instances_for_series(series_id):
    r = SESSION.get(f"{ORTHANC}/series/{series_id}/instances", auth=AUTH)
    r.raise_for_status()
    return r.json()

def fetch_dicom(instance_id):
    r = SESSION.get(f"{ORTHANC}/instances/{instance_id}/file", auth=AUTH)
    r.raise_for_status()
    return pydicom.dcmread(io.BytesIO(r.content))

# def upload_processed_series(new_images, old_dcms):
#     new_series_uid = generate_uid()
#     for i, ds in enumerate(old_dcms):
#         ds.SeriesInstanceUID = new_series_uid
#         ds.SeriesDescription = "Processed Series"
#         ds.PixelData = new_images[i].astype(np.uint16).tobytes()
#         ds.SOPInstanceUID = generate_uid()
#         ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
#         ds.is_little_endian = True
#         ds.is_implicit_VR = False
#         buffer = io.BytesIO()
#         ds.save_as(buffer)
#         buffer.seek(0)
#         r = SESSION.post(
#             f"{ORTHANC}/instances",
#             data=buffer.read(),
#             auth=AUTH,
#             headers={"Content-Type": "application/dicom", "Expect": ""}
#         )
#         r.raise_for_status()
#     return new_series_uid

def update_orthanc():
    """Updates"""
    db = TinyDB(DB_PATH)
    StudyQuery = Query()
    existing_ids = [s["orthanc_study_id"] for s in db]

    studies = fetch_studies()
    for study_info in studies:
        if study_info["ID"] in existing_ids:
            continue
        study = Study(study_info)
        series_list = fetch_series_for_study(study.orthanc_id)
        for series_info in series_list:
            series = Series(series_info)
            if series.series_label is None: 
                # add mindmap
                print('Mindmap...')
                pass
            if not series.sax_processed: 
                # add segmentation
                print('Segmentation...')
                pass
            if series.sax_processed and not series.roundel_processed: 
                # add roundel
                print('Roundelling...')
                pass
            study.series_list.append(series)
        db.upsert(study.to_dict(), StudyQuery.study_uid == study.uid)
    db.close()