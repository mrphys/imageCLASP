import io
import numpy as np
import requests
import pydicom
from pydicom.uid import ExplicitVRLittleEndian, generate_uid
from tinydb import TinyDB, Query
import pandas as pd
import re 
from datetime import datetime
import time
from clasp_utils import crop_pad_hw, run_inference_on_scan
from scipy.ndimage import zoom
import copy

DB_PATH = "/Users/vivekmuthurangu/Downloads/CLASPv2/image_clasp_db.json"
ORTHANC = "http://localhost:8042"
AUTH = ("orthanc","orthanc")

SESSION = requests.Session()
SESSION.auth = AUTH
SESSION.trust_env = False


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

def extract_series_dcm_DL(series_info):
    instance_list = fetch_instances_for_series(series_info)
    new_ims = []
    old_dcms = []
    for inst in instance_list:
        ds = fetch_dicom(inst['ID'])
        image_array = ds.pixel_array
        old_dcms.append(ds)
        new_ims.append(image_array)
    
    pixel_spacing = ds.PixelSpacing
    image_size = image_array.shape
    all_ims = np.transpose(np.array(new_ims), (1,2,0))
    mask_h_w_z = run_inference_on_scan(all_ims, pixel_spacing)
    mask_h_w_z = zoom(mask_h_w_z, (1/pixel_spacing[0], 1/pixel_spacing[1], 1), order=1)
    mask_h_w_z = crop_pad_hw(mask_h_w_z, image_size[0], image_size[1])

    return mask_h_w_z, old_dcms

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


class MRI_Sorter:
    def __init__(self, study):
        self.dicom_info = pd.concat([self.extract_series_df(s) for s in fetch_series_for_study(study.orthanc_id)])

        dicom_info_3D = self.dicom_info.loc[self.dicom_info['Dimension'] == 3]
        sort_dict_3D = self.sort_3D(dicom_info_3D)

        dicom_info_2D = self.dicom_info.loc[self.dicom_info['Dimension'] == 2]
        sort_dict_2D = self.sort_2D(dicom_info_2D) 

        sort_dict = {**sort_dict_2D, **sort_dict_3D}
        sort_df = pd.DataFrame(sort_dict).T
        # sort_df.index.name = 'SeriesUID'
        # sort_df = sort_df.reset_index()
        # sort_df['patient'] = patient
        self.sort_df = sort_df

    def orientation_id(self, iop):
        """Get ImageOrientationPatient of dicom object.

        Returns one of
            Transverse
            Coronal
            Sagittal
            NA  (if ImageOrientationPatient not available)
        """
        iop_rounded = [round(x,1) for x in iop]
        plane_cross = np.cross(iop_rounded[0:3], iop_rounded[3:6])
        plane = [abs(x) for x in plane_cross]

        if plane[0] == 1:
            return 'Sagittal'
        elif plane[1] == 1:
            return 'Coronal'
        elif plane[2] == 1:
            return 'Axial'
        else:
            return 'NA'
        

    def update_sort_dict(self, sort_dict, dimension, all_series_list, series_type, flow_flag, cine_flag, stack_flag):
        for group_tag, all_series_df in enumerate(all_series_list):
            for series_uid, series_df in all_series_df.groupby('SeriesUID'):
                sort_dict[series_uid].update({
                    'Dimension': dimension,
                    'Flow': flow_flag,
                    'Cine': cine_flag,
                    'Stack':stack_flag,
                    'Plane': self.orientation_id(series_df.iloc[0].ImageOrientationPatient),
                    'Type': series_type,
                    'Slices': series_df['SliceLocation'].nunique(),
                    'Frames': int(len(series_df) / series_df['SliceLocation'].nunique()),
                    'Group': f'{series_type}: {group_tag}'
                })
        return sort_dict

    def extract_series_df(self, series_info):
        instance_list = fetch_instances_for_series(series_info['ID'])
        if not instance_list:
            return pd.DataFrame()

        # read first DICOM for shared series-level fields and RR
        ds0 = fetch_dicom(instance_list[0]['ID'])
        if not ("PixelData" in ds0 and "ImageOrientationPatient" in ds0):
            return pd.DataFrame()

        image_shape = (ds0.Rows, ds0.Columns)
        series_fields = {
            "SeriesUID": series_info['ID'],
            "SeriesDescription": getattr(ds0, "SeriesDescription", None),
            "PixelSpacing": getattr(ds0, "PixelSpacing")[0],
            "SliceThickness": getattr(ds0, "SpacingBetweenSlices", getattr(ds0, "SliceThickness", np.nan)),
            "Dimension": int(getattr(ds0, "MRAcquisitionType", "0").replace("D", "")) if hasattr(ds0, "MRAcquisitionType") else np.nan,
            "N_timesteps": int(getattr(ds0, "CardiacNumberOfImages", 0)),
            "ImageShape": image_shape,
            "venc": 0,
        }

        # compute RR only from first DICOM
        try:
            rr_ni = float(ds0.NominalInterval)
        except Exception:
            rr_ni = 0.0
        try:
            rr_hr = 60000.0 / float(ds0.HeartRate)
        except Exception:
            rr_hr = 0.0
        rr = max(rr_ni, rr_hr)
        series_fields["rr"] = rr if 100 < rr < 3000 else 0.0

        # build records using only MainDicomTags
        records = [
            {
                **series_fields,
                "ID": inst['ID'],
                "ImageOrientationPatient": tuple(float(x) for x in inst['MainDicomTags']['ImageOrientationPatient'].split('\\')),
                "ImagePositionPatient": tuple(float(x) for x in inst['MainDicomTags']['ImagePositionPatient'].split('\\')),
                "SliceLocation": float(inst['MainDicomTags']['ImagePositionPatient'].split('\\')[2]),
                "InstanceNumber": int(inst['MainDicomTags']['InstanceNumber']),
                "TriggerTime": int(inst['MainDicomTags']['InstanceNumber']),
            }
            for inst in instance_list
        ]

        return pd.DataFrame(records).sort_values("InstanceNumber")

    # def extract_series_df(self, series_info):
    #     instance_list = fetch_instances_for_series(series_info['ID'])
    #     if not instance_list:
    #         return pd.DataFrame()

    #     ds0 = fetch_dicom(instance_list[0]['ID'])
    #     if not ("PixelData" in ds0 and "ImageOrientationPatient" in ds0):
    #         return pd.DataFrame()

    #     image_shape = (ds0.Rows, ds0.Columns)

    #     series_fields = {
    #         "SeriesUID": series_info['ID'],
    #         "SeriesDescription": getattr(ds0, "SeriesDescription", None),
    #         "PixelSpacing": getattr(ds0, "PixelSpacing")[0],
    #         "SliceThickness": getattr(ds0, "SpacingBetweenSlices", getattr(ds0, "SliceThickness", np.nan)),
    #         "Dimension": int(getattr(ds0, "MRAcquisitionType", "0").replace("D", "")) if hasattr(ds0, "MRAcquisitionType") else np.nan,
    #         "N_timesteps": int(getattr(ds0, "CardiacNumberOfImages", 0)),
    #         "ImageShape": image_shape,
    #         "venc": 0
    #     }

    #     # fetch all dicoms once
    #     dcms = {inst['ID']: fetch_dicom(inst['ID']) for inst in instance_list}

        

    #     records = [
    #         {
    #             **series_fields,
    #             "ID": inst['ID'],
    #             "ImageOrientationPatient": tuple(map(float, inst['MainDicomTags']['ImageOrientationPatient'].split('\\'))),
    #             "ImagePositionPatient": tuple(map(float, inst['MainDicomTags']['ImagePositionPatient'].split('\\'))),
    #             "SliceLocation": float(inst['MainDicomTags']['ImagePositionPatient'].split('\\')[2]),
    #             "InstanceNumber": int(inst['MainDicomTags']['InstanceNumber']),
    #             "TriggerTime": getattr(dcms[inst['ID']], "TriggerTime", np.nan),
    #             "rr": compute_rr(dcms[inst['ID']])
    #         }
    #         for inst in instance_list
    #     ]   

    #     return pd.DataFrame(records).sort_values("InstanceNumber")
    
    def sort_cine_stacks(self, cine_non_flow_list):
        """
        Sort the 2D DICOMs into cine stacks and update the sort dictionary.
        """
        cine_non_flow_df = pd.concat(cine_non_flow_list)
        cine_stack_list = []
        
        # Process each unique orientation
        for unique_or in cine_non_flow_df['ImageOrientationPatient'].dropna().unique():
            # Filter by orientation and sort by pixel spacing, thickness, and timesteps
            unique_df = cine_non_flow_df[cine_non_flow_df['ImageOrientationPatient'] == unique_or].reset_index()
            unique_df = unique_df.set_index(['PixelSpacing', 'SliceThickness', 'N_timesteps']).sort_index()

            for unique_idx, stack_df in unique_df.groupby(level=[0, 1, 2]):
                # Process each unique image shape
                for unique_imshape, shape_group in stack_df.groupby('ImageShape'):
                    shape_group = shape_group.reset_index()
                    
                    if len(shape_group) > 1:  # Ensure there are multiple entries
                        separated = False
                        
                        # Process each unique series
                        for uni_series, series_df in shape_group.groupby('SeriesUID'):
                            if len(series_df) > 50:  # Large series: separate based on SliceLocation
                                if series_df['SliceLocation'].nunique() > 1:
                                    cine_stack_list.append(series_df)
                                    separated = True
                        
                        # If not separated, add the entire group if it has multiple slice locations
                        if not separated and shape_group['SliceLocation'].nunique() > 1:
                            cine_stack_list.append(shape_group)
        
        if cine_stack_list:
            cine_stack_ = pd.concat(cine_stack_list)['SeriesUID'].unique()
            cine_single_list = [df for df in cine_non_flow_list if df['SeriesUID'].iloc[0] not in cine_stack_SeriesUIDs] # get the single cines
        else:
            cine_single_list = cine_non_flow_list
        return cine_stack_list, cine_single_list


    def get_venc(self, cine_single_list):
        for series_df in cine_single_list:
            max_venc = 0

            idxs = (0, len(series_df)//2, len(series_df)-1)
            for i in idxs:
                ds = fetch_dicom(series_df.iloc[i]['ID'])
                manufacturer = (getattr(ds, 'Manufacturer', '') or '').lower()

                venc = 0
                if 'siemens' in manufacturer:
                    val = getattr(ds.get((0x0051, 0x1014)), 'value', None)
                    if val:
                        nums = re.findall(r'\d+', str(val))
                        if nums:
                            venc = float(max(map(int, nums)))
                    if not venc:
                        val = getattr(ds.get((0x0018, 0x0024)), 'value', '')
                        m = re.search(r'v(\d+)in', str(val))
                        if m:
                            venc = float(m.group(1))

                elif 'ge' in manufacturer:
                    val = getattr(ds.get((0x0019, 0x10cc)), 'value', None)
                    if val:
                        venc = val / 10

                elif 'philips' in manufacturer:
                    seq = getattr(ds, 'RealWorldValueMappingSequence', None)
                    if seq:
                        intercept = getattr(seq[0], 'RealWorldValueIntercept', None)
                        if intercept is not None:
                            venc = abs(intercept)
                    if not venc:
                        intercept = getattr(ds, 'RescaleIntercept', None)
                        if intercept is not None:
                            venc = abs(intercept)

                max_venc = max(max_venc, venc)

            series_df['venc'] = max_venc

        return cine_single_list
   
    def sort_2D(self, dicom_info_2D):

        sort_dict_2D = {}

        for series_uid, series_df in dicom_info_2D.groupby('SeriesUID'):
            # Initialize the dictionary for each series
            sort_dict_2D[series_uid] = {
                'Description': series_df.iloc[0]['SeriesDescription'],
                'N': len(series_df),
                'Dimension': 2
            }

        # Initialize lists to store results
        series_df_list = []

        # Process unique ImageOrientationPatients
        for unique_or in dicom_info_2D['ImageOrientationPatient'].dropna().unique():
            # Filter and sort DataFrame
            unique_df = dicom_info_2D[dicom_info_2D['ImageOrientationPatient'] == unique_or]
            unique_df = unique_df.set_index(['PixelSpacing', 'SliceThickness']).sort_index()

            # Group by unique indices (PixelSpacing, SliceThickness)
            for unique_idx, series_df in unique_df.groupby(level=[0, 1]):
                # Group by image shape
                for unique_imshape, shape_group in series_df.groupby('ImageShape'):
                    # Filter for series with more than one entry
                    for uni_series, series_df in shape_group.groupby('SeriesUID'):
                        if len(series_df) > 0:
                            series_df_list.append(series_df)

            stack_df_list = [df for df in series_df_list if df['SliceLocation'].nunique() > 1]
            single_df_list = [df for df in series_df_list if df['SliceLocation'].nunique() == 1]

            static_stack_list = [
                        df
                        for df in stack_df_list
                        if df.loc[df['SliceLocation'] == df['SliceLocation'].unique()[0]]['InstanceNumber'].nunique() == 1
                    ]

            cine_stack_list = [
                        df
                        for df in stack_df_list
                        if df.loc[df['SliceLocation'] == df['SliceLocation'].unique()[0]]['InstanceNumber'].nunique() > 1
                    ]


            static_single_list = [
                        df
                        for df in single_df_list
                        if df.loc[df['SliceLocation'] == df['SliceLocation'].unique()[0]]['InstanceNumber'].nunique() == 1
                    ]

            cine_single_list = [
                        df
                        for df in single_df_list
                        if df.loc[df['SliceLocation'] == df['SliceLocation'].unique()[0]]['InstanceNumber'].nunique() > 1 
                    ]

            cine_single_list = self.get_venc(cine_single_list)

            cine_2d_flow_list = [df for df in cine_single_list if (df['venc'] > 0).any()]
            cine_single_list = [df for df in cine_single_list if not (df['venc'] > 0).any()]



        sort_dict_2D = self.update_sort_dict(sort_dict_2D, 2, static_single_list, 'Static Single', flow_flag=0, cine_flag=0, stack_flag = 0)
        sort_dict_2D = self.update_sort_dict(sort_dict_2D, 2, static_stack_list, 'Static Stack', flow_flag=0, cine_flag=0, stack_flag = 1)
        
        sort_dict_2D = self.update_sort_dict(sort_dict_2D, 2, cine_single_list, 'Cine Single', flow_flag=0, cine_flag=0, stack_flag = 0)
        sort_dict_2D = self.update_sort_dict(sort_dict_2D, 2, cine_stack_list, 'Cine Stack', flow_flag=0, cine_flag=0, stack_flag = 1)

        sort_dict_2D = self.update_sort_dict(sort_dict_2D, 2, cine_2d_flow_list, '2D Flow', flow_flag=1, cine_flag=1, stack_flag = 0)

        return sort_dict_2D
    
    def sort_3D(self, dicom_info_3D):
        '''
        sort the 3D dicoms
        '''
        sort_dict_3D = {}
        for series_uid, series_df in dicom_info_3D.groupby('SeriesUID'):
            sort_dict_3D[series_uid] = {
                'Description': series_df.iloc[0]['SeriesDescription'],
                'N': len(series_df),
                'Dimension': 3
            }

        # Initialize lists to store results
        stack_df_list = []

        # Process unique orientations
        for unique_or in dicom_info_3D['ImageOrientationPatient'].dropna().unique():
            # Filter and sort DataFrame
            unique_df = dicom_info_3D[dicom_info_3D['ImageOrientationPatient'] == unique_or]
            unique_df = unique_df.set_index(['PixelSpacing', 'SliceThickness']).sort_index()

            # Group by unique indices (pixelspacing, thickness)
            for unique_idx, stack_df in unique_df.groupby(level=[0, 1]):
                # Group by image shape
                for unique_imshape, shape_group in stack_df.groupby('ImageShape'):
                    if len(shape_group) > 1:
                        stack_df_list.append(shape_group)

        # Classify single-phase and multi-phase datasets
        stack_df_list = self.get_venc(stack_df_list)

        flow_4d = [df for df in stack_df_list if (df['venc'] > 0).any()]
        single_phase_list = [df for df in stack_df_list if df['SliceLocation'].nunique() == len(df) and not (df['venc'] != 0).any()]
        multi_phase_list = [df for df in stack_df_list if df['SliceLocation'].nunique() != len(df) and not (df['venc'] != 0).any()]

        single_mra = [df for df in single_phase_list if df['rr'].max() == 0]
        single_whole_heart = [df for df in single_phase_list if df['rr'].max() > 0]
        multi_mra = [df for df in multi_phase_list if df['rr'].max() == 0]
        cine_4d = [df for df in multi_phase_list if (df['rr'].max() > 0) & (df['TriggerTime'].nunique() > 1)]

        # Update the sort_dict_3D 
        sort_dict_3D = self.update_sort_dict(sort_dict_3D, 3, single_mra, 'Single MRA', flow_flag=0, cine_flag=0, stack_flag = 'NA')
        sort_dict_3D = self.update_sort_dict(sort_dict_3D, 3, single_whole_heart, 'Single Whole-Heart', flow_flag=0, cine_flag=0, stack_flag = 'NA')
        sort_dict_3D = self.update_sort_dict(sort_dict_3D, 3, multi_mra, 'Multi MRA', flow_flag=0, cine_flag=0, stack_flag = 'NA')
        sort_dict_3D = self.update_sort_dict(sort_dict_3D, 3, flow_4d, '4D Flow', flow_flag=1, cine_flag=1, stack_flag = 'NA')
        sort_dict_3D = self.update_sort_dict(sort_dict_3D, 3, cine_4d, '4D Cine', flow_flag=0, cine_flag=1, stack_flag = 'NA')
        return sort_dict_3D

def update_orthanc():
    """Updates"""

    db = TinyDB(DB_PATH)
    StudyQuery = Query()
    existing_ids = [s["orthanc_study_id"] for s in db]
    
    target_list = ['short', 'SAX', 'KT', 'ML cine']
    
    studies = fetch_studies()
    for study_info in studies:
        if study_info["ID"] in existing_ids:
            continue
        
        study = Study(study_info)
        mri_sorter = MRI_Sorter(study)
        series_type_df = mri_sorter.sort_df
        series_list = fetch_series_for_study(study.orthanc_id)

        #sl = 0
        for series_info in series_list:
            series = Series(series_info)
            if series.series_type is None: 
                series_id = series_info['ID']
                if series_id in series_type_df.index:
                    series.series_type = series_type_df.loc[series_id]['Type']
                else:
                    series.series_type = 'non-image'

            if not series.sax_processed:
                 
                 if any(w in series.description for w in target_list):#probably need to add type here as well
                    print(series.orthanc_id)
                    series.series_orientation = 'SAX'
                    series.sax_processed = True
                    mask, old_dcms = extract_series_dcm_DL(series.orthanc_id)
                    #np.save(f'/Users/vivekmuthurangu/Downloads/test_mask_{sl}.npy', mask)
                    new_uid = upload_processed_series(mask,old_dcms)
                    new_orthanc_id = get_orthanc_series_data_from_uid(new_uid, ORTHANC, AUTH)
                    new_series = copy.deepcopy(series)
                    new_series.description = 'DL Processed'
                    new_series.orthanc_id = new_orthanc_id['ID']
                    new_series.associated_uid = series.orthanc_id
                    study.series_list.append(new_series)
                    pass
                
            if series.sax_processed and not series.roundel_processed: 
                # add roundel
                pass
            study.series_list.append(series)
        db.upsert(study.to_dict(), StudyQuery.study_uid == study.uid)
    db.close()