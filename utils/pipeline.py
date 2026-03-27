from utils.orthanc_utils import *
from utils.db_utils import *
from utils.plot_utils import *
from utils.mri_sorter import MRI_Sorter
from utils.model_utils import *
from scipy.ndimage import zoom
import copy

DB_PATH = "image_clasp_db.json"
ORTHANC = "http://localhost:8042"
METRICS_PATH = 'clasp_metrics.csv'
AUTH = ("orthanc","orthanc")

SESSION = requests.Session()
SESSION.auth = AUTH
SESSION.trust_env = False

columns = [
    'orthanc_study_id', "lv_edv", "lv_esv", "lv_ef", "lv_mass",
    "rv_edv", "rv_esv", "rv_ef", "rv_mass"
]

if os.path.exists(METRICS_PATH):
    metrics_df = pd.read_csv(METRICS_PATH)
else:
    metrics_df = pd.DataFrame(columns=columns)
metrics_df.to_csv(METRICS_PATH)

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
    mask_h_w_z = zoom(mask_h_w_z, (1/pixel_spacing[0], 1/pixel_spacing[1], 1), order=0)
    mask_h_w_z = crop_pad_hw(mask_h_w_z, image_size[0], image_size[1])
    mask_h_w_z = np.uint16(mask_h_w_z) * 500
    mask_h_w_z = np.transpose(np.array(mask_h_w_z), (2,0,1))
    return mask_h_w_z, old_dcms

def update_orthanc():
    """Updates"""

    db = TinyDB(DB_PATH)
    StudyQuery = Query()
    existing_ids = [s["orthanc_study_id"] for s in db]
    
    target_list = ['ML cine']
    
    studies = fetch_studies()
    for study_info in studies:
        if study_info["ID"] in existing_ids:
            continue
        
        study = Study(study_info)
        mri_sorter = MRI_Sorter(study)
        series_type_df = mri_sorter.sort_df
        series_list = fetch_series_for_study(study.orthanc_id)

        for series_info in series_list:
            series = Series(series_info)
            if series.series_type is None: 
                series_id = series_info['ID']
                if series_id in series_type_df.index:
                    series.series_type = series_type_df.loc[series_id]['Type']
                else:
                    series.series_type = 'non-image'

            if series.dl_orthanc_id is None:
                 
                 if any(w in series.description for w in target_list): # probably need to add type here as well
                    series.series_orientation = 'SAX'
                    mask, old_dcms = extract_series_dcm_DL(series.orthanc_id)
                    new_uid = send_series_to_orthanc(mask, old_dcms, new_description="DL Processed Series")
                    new_orthanc_id = get_orthanc_series_data_from_uid(new_uid, ORTHANC, AUTH)
                    series.dl_orthanc_id = new_orthanc_id['ID']
                
            if series.dl_orthanc_id and not series.roundel_orthanc_id: 
                # add roundel
                pass
            study.series_list.append(series)
        db.upsert(study.to_dict(), StudyQuery.study_uid == study.uid)
    db.close()