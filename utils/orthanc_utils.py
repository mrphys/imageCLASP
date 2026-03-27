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

def fetch_dicoms_for_series(series_id):
    dicoms = [
        fetch_dicom(d['ID'])
        for d in fetch_instances_for_series(series_id)
    ]
    return dicoms

def get_orthanc_series_data_from_uid(series_uid, ORTHANC, AUTH):
    payload = {
        "Level": "Series",
        "Expand": True,
        "Query": {
            "SeriesInstanceUID": series_uid
        }
    }

    r = SESSION.post(f"{ORTHANC}/tools/find", json=payload, auth=AUTH)
    r.raise_for_status()
    results = r.json()

    if not results:
        return None

    return results[0]   

def send_series_to_orthanc(new_array, old_dcm, new_description):
    new_series_uid = generate_uid()
    for i in range(len(old_dcm)):
        old_dcm[i].SeriesInstanceUID = new_series_uid
        old_dcm[i].SeriesDescription = new_description 
        old_dcm[i].PixelData = new_array[i].astype(np.uint16).tobytes()
        old_dcm[i].SOPInstanceUID = generate_uid()
        old_dcm[i].file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        old_dcm[i].is_little_endian = True
        old_dcm[i].is_implicit_VR = False
        buffer = io.BytesIO()
        old_dcm[i].save_as(buffer)
        buffer.seek(0)
        upload = SESSION.post(f"{ORTHANC}/instances",data=buffer.read(),auth=AUTH,headers={"Content-Type": "application/dicom", "Expect": ""})
        upload.raise_for_status()
    return new_series_uid

