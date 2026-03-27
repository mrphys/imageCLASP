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

def upload_processed_series(new_images, old_dcms):
    new_series_uid = generate_uid()
    new_images = np.uint16(new_images) * 500
    for i, ds in enumerate(old_dcms):
        ds.SeriesInstanceUID = new_series_uid
        ds.SeriesDescription = "Processed Series"
        ds.PixelData = new_images[:,:,i].astype(np.uint16).tobytes()
        ds.SOPInstanceUID = generate_uid()
        ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds.is_little_endian = True
        ds.is_implicit_VR = False
        buffer = io.BytesIO()
        ds.save_as(buffer)
        buffer.seek(0)
        r = SESSION.post(
            f"{ORTHANC}/instances",
            data=buffer.read(),
            auth=AUTH,
            headers={"Content-Type": "application/dicom", "Expect": ""}
        )
        r.raise_for_status()
    return new_series_uid
