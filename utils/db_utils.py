from tinydb import TinyDB, Query
import pandas as pd
import numpy as np

class Series:
    def __init__(self, series_info, series_type = None, series_orientation = None, sax_processed = False, roundel_processed = False, associated_uid = None):
        self.orthanc_id = series_info["ID"]
        self.uid = series_info["MainDicomTags"].get("SeriesInstanceUID")
        self.description = series_info["MainDicomTags"].get("SeriesDescription")
        self.series_type = series_type
        self.series_orientation = series_orientation
        self.sax_processed = sax_processed
        self.roundel_processed = roundel_processed
        self.associated_uid = associated_uid

    def to_dict(self):
        record = {
            "orthanc_series_id": self.orthanc_id,
            "series_uid": self.uid,
            "series_description": self.description,
            "series_type":self.series_type,
            "series_orientation":self.series_orientation,
            "sax_processed":self.sax_processed,
            "roundel_processed":self.roundel_processed,
            "associated_orthanc_id":self.associated_uid
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

def load_db_rows(DB_PATH):
    # Open TinyDB database
    db = TinyDB(DB_PATH)
    rows = []

    # Iterate over all studies in the database
    for study in db:
        series = study.get("series", [])

        # Construct a summary row per study
        rows.append({
            "patient_id": study.get("patient_id"),
            "patient_sex": study.get("patient_sex"),
            "age": study.get("patient_age"),
            "n_series": len(series),  # Number of series in the study
            "sax_processed": any(s.get("sax_processed", False) for s in series),  # True if any series processed
            "roundel_processed": any(s.get("roundel_processed", False) for s in series),  # True if any series roundel processed
        })

    # Close database
    db.close()

    # Convert list of rows to a DataFrame
    df = pd.DataFrame(rows)
    return df