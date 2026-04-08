from utils.orthanc_utils import *
from utils.db_utils import *
from utils.plot_utils import *
from utils.mri_sorter import MRI_Sorter
from utils.sax_dl_utils import *
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

def update_study(db, study):
    query = Query()

    db.upsert(
        study.to_dict(),
        query.orthanc_study_id == study.orthanc_study_id
    )

def mri_sorting_pipeline(study):
    mri_sorter = MRI_Sorter(study)
    sort_df = mri_sorter.sort_df
    if not sort_df.empty:
        for sid, series in study.series_dict.items():
            if series.series_type is not None:
                continue
            if sid in sort_df.index:
                series.series_type = sort_df.loc[sid]["Type"]
                series.series_orientation = sort_df.loc[sid]["Orientation"]
                series.series_group = sort_df.loc[sid]["Group"]
            else:
                series.series_type = "non-image"


def sax_segmentation_pipeline(study):
    df = pd.DataFrame([series.__dict__ for series in study.series_dict.values()])

    sax_df = df.loc[
        (df['series_type'] == 'Cine Stack') &
        (df['series_orientation'] == 'SAX')
    ]

    if sax_df['dl_orthanc_id'].notna().any():
        return

    sax_df = sax_df[sax_df['dl_orthanc_id'].isna()]

    if sax_df.empty:
        return

    series_group = sax_df['series_group'].value_counts().index[0]
    sax_df = sax_df[sax_df['series_group'] == series_group]
    series_orthanc_ids = set(sax_df['orthanc_series_id'])

    for sid, series in study.series_dict.items():
        if sid in series_orthanc_ids:
            old_dcms = fetch_orthanc_dicoms_for_series(sid)
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

    pipelines = {
        "MRI Sorting": mri_sorting_pipeline,
        "SAX Segmentation": sax_segmentation_pipeline,
        # "Roundel": roundel_pipeline,
    }

    studies = fetch_db_studies()

    for pipeline_idx, (pipeline_name, pipeline) in enumerate(pipelines.items()):
        for study in stqdm(
            studies,
            desc=f"Pipeline {pipeline_idx}/{len(pipelines)}: {pipeline_name}. Studies",
        ):
            pipeline(study)
            update_study(db, study)