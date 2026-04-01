from utils.orthanc_utils import *
from utils.db_utils import *
from utils.plot_utils import *
from utils.mri_sorter import MRI_Sorter
from utils.model_utils import *
from scipy.ndimage import zoom
import copy
from stqdm import stqdm

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

def sync_orthanc_and_db():
    db = TinyDB(DB_PATH)
    StudyQuery = Query()
    existing_ids = [s["orthanc_study_id"] for s in db]

    orthanc_studies = fetch_orthanc_studies()
    for study_info in orthanc_studies:
        if study_info["ID"] in existing_ids:
            continue
        study = Study(study_info)
        series_list = fetch_orthanc_series_for_study(study.orthanc_study_id)
        for series_info in series_list:
            series = Series(series_info)
            study.series_dict[series.orthanc_series_id] = series
        db.upsert(study.to_dict(), StudyQuery.study_uid == study.orthanc_study_id)
    db.close()

def update_studies(db, studies):
    query = Query()

    for study in studies:
        db.upsert(
            study.to_dict(),
            query.orthanc_study_id == study.orthanc_study_id
        )

def mri_sorting_pipeline(studies):
    for study in studies:
        mri_sorter = MRI_Sorter(study)
        sort_df = mri_sorter.sort_df
        for sid, series in study.series_dict.items():
            if series.series_type is not None:
                continue
            if sid in sort_df.index:
                series.series_type = sort_df.loc[sid]["Type"]
                series.series_orientation = sort_df.loc[sid]["Orientation"]
            else:
                series.series_type = "non-image"


def sax_segmentation_pipeline(studies):
    for study in studies:
        for sid, series in stqdm(study.series_dict.items(), desc = 'Segmenting SAX'):
            if series.dl_orthanc_id is not None:
                continue

            elif series.dl_orthanc_id is None and series.series_orientation == 'SAX' and series.series_type == "Cine Stack":

                old_dcms = fetch_orthanc_dicoms_for_series(series.orthanc_series_id)
                mask = run_inference_on_scan(old_dcms)

                new_uid = send_series_to_orthanc(
                    mask,
                    old_dcms,
                    new_description="SAX DL Segmented"
                )

                new_series_info = get_orthanc_series_data_from_uid(new_uid)
                series.dl_orthanc_id = new_series_info["ID"]


def roundel_pipeline(studies):
    pass


def run_pipelines():
    db = TinyDB(DB_PATH)

    studies = fetch_db_studies()

    pipelines = [
        mri_sorting_pipeline,
        # sax_segmentation_pipeline
        # roundel_pipeline,
    ]

    for pipeline in pipelines:
        pipeline(studies)
        update_studies(db, studies)