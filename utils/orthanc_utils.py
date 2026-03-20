from tinydb import TinyDB, Query
import pydicom 
import numpy as np
import requests
import io
from pydicom.uid import generate_uid, ExplicitVRLittleEndian
from datetime import datetime

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
            age = compute_age(dos, dob)
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
                        ds = pydicom.dcmread(io.BytesIO(r.content))
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
                    
def compute_age(dos, dob):
    if dob is None or dos is None:
        return np.nan
    else:
        try:
            age = datetime.strptime(dos, "%Y%m%d") - datetime.strptime(dob, "%Y%m%d")
            return age.days // 365
        except ValueError:
            return np.nan