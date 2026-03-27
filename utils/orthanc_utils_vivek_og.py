from tinydb import TinyDB, Query
import pydicom 
import numpy as np
import requests
import io
from pydicom.uid import generate_uid, ExplicitVRLittleEndian
from datetime import datetime



def send_series_to_orthanc(SESSION, new_dcm, old_dcm, ORTHANC, AUTH):
    # Generate a new SeriesInstanceUID for the processed series
    new_series_uid = generate_uid()

    # Iterate over all slices / instances in the series
    for i in range(len(new_dcm)):
        # Update DICOM metadata to reflect new derived series
        old_dcm[i].SeriesInstanceUID = new_series_uid
        old_dcm[i].SeriesDescription = 'Processed Series'

        # Replace pixel data with processed image (cast to uint16 and convert to bytes)
        old_dcm[i].PixelData = new_dcm[i].astype(np.uint16).tobytes()

        # Assign a new SOPInstanceUID for each instance
        old_dcm[i].SOPInstanceUID = generate_uid()

        # Ensure correct transfer syntax for writing
        old_dcm[i].file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        old_dcm[i].is_little_endian = True
        old_dcm[i].is_implicit_VR = False
        
        # Write modified DICOM to in-memory buffer
        buffer = io.BytesIO()
        old_dcm[i].save_as(buffer)
        buffer.seek(0)

        # Upload instance to Orthanc
        upload = SESSION.post(
            f"{ORTHANC}/instances",
            data=buffer.read(),
            auth=AUTH,
            headers={"Content-Type": "application/dicom", "Expect": ""}
        )
        upload.raise_for_status()  # Ensure upload succeeded

    # Return UID of the newly created series
    return new_series_uid


def get_orthanc_series_data_from_uid(SESSION, series_uid, ORTHANC, AUTH):
    # Query Orthanc for series using SeriesInstanceUID
    payload = {
        "Level": "Series",
        "Expand": True,
        "Query": {
            "SeriesInstanceUID": series_uid
        }
    }

    # Send query request
    r = SESSION.post(f"{ORTHANC}/tools/find", json=payload, auth=AUTH)
    r.raise_for_status()

    # Parse response
    results = r.json()

    # Return None if no matching series found
    if not results:
        return None

    return results[0]


def query_orthanc(SESSION, DB_PATH, ORTHANC, AUTH):
    # Initialise TinyDB database and query object
    db = TinyDB(DB_PATH)
    Study = Query()

    # Define Orthanc /tools/find payload to retrieve all studies with expanded metadata
    payload = {
        "Level": "Study",
        "Expand": True,
        "Query": {}
    }

    # Send request to Orthanc server
    r = SESSION.post(f"{ORTHANC}/tools/find", json=payload, auth=AUTH)
    r.raise_for_status()  # Raise exception if request failed

    # Parse returned studies
    studies = r.json()

    # Keywords used to identify target series
    target_list = ['short', 'SAX', 'KT']

    # Collect study IDs already present in local database
    db_studies_present = []
    for study in db:
        db_studies_present.append(study["orthanc_study_id"])

    # Iterate through studies retrieved from Orthanc
    for study in studies:
        # Process only studies not already stored in DB
        if study["ID"] not in db_studies_present:
            print(study["ID"])

            # Extract patient and study metadata
            dob = study["PatientMainDicomTags"].get("PatientBirthDate")
            dos = study["MainDicomTags"].get("StudyDate")

            # Compute patient age at study date
            age = compute_age(dos, dob)

            # Construct study-level record
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

            # Retrieve all series belonging to the current study
            study_id = study["ID"]
            series_list = requests.get(
                f"{ORTHANC}/studies/{study_id}/series",
                auth=AUTH
            ).json()

            # Iterate through series in the study
            for i in range(len(series_list)):
                series_info = series_list[i]

                # Extract series description
                series_desc = series_info["MainDicomTags"].get("SeriesDescription")

                # Initialise series record
                series_record = {
                    "orthanc_series_id": series_info["ID"],
                    "series_uid": series_info["MainDicomTags"].get("SeriesInstanceUID"),
                    "series_description": series_desc,
                    "is_target": False,
                    "DL_processed": False,
                }

                # Identify target series based on keywords
                # This is where a processing pipeline would typically be inserted
                if any(w in series_desc for w in target_list):
                    series_record["is_target"] = True
                    series_record["DL_processed"] = True

                    # Retrieve all instances (DICOM files) for the series
                    instances = requests.get(
                        f"{ORTHANC}/series/{series_info['ID']}/instances",
                        auth=AUTH
                    ).json()

                    new_series_ims = []  # Processed image arrays
                    old_dcms = []        # Original DICOM datasets

                    # Iterate through instances in the series
                    for inst in instances:
                        instance_id = inst["ID"]

                        # Download raw DICOM file
                        r = requests.get(
                            f"{ORTHANC}/instances/{instance_id}/file",
                            auth=AUTH
                        )

                        # Load DICOM into memory
                        ds = pydicom.dcmread(io.BytesIO(r.content))

                        # Placeholder for DL segmentation step
                        # Current operation: simple intensity inversion
                        new_im = abs(np.float16(ds.pixel_array) - np.max(ds.pixel_array))

                        # Store processed image and original DICOM
                        new_series_ims.append(new_im)
                        old_dcms.append(ds)

                    # Send processed series back to Orthanc
                    processed_uid = send_series_to_orthanc(
                        SESSION, new_series_ims, old_dcms, ORTHANC, AUTH
                    )

                    # Retrieve metadata for newly created series
                    new_series_json = get_orthanc_series_data_from_uid(
                        SESSION, processed_uid, ORTHANC, AUTH
                    )

                    # Construct record for processed series
                    new_record = {
                        "orthanc_series_id": new_series_json["ID"],
                        "series_uid": new_series_json["MainDicomTags"].get("SeriesInstanceUID"),
                        "series_description": new_series_json["MainDicomTags"].get("SeriesDescription"),
                        "is_target": True,
                        "DL_processed": True,
                        "roundel_processed": False,
                        "associated_original_series_uid": series_record["series_uid"]
                    }

                    # Append processed series to study record
                    record["series"].append(new_record)

                # Insert or update study record in TinyDB
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