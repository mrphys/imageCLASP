from utils.orthanc_utils import *
from utils.db_utils import *
from utils.mri_sorter import MRI_Sorter
from utils.sax_dl_utils import *
from scipy.ndimage import zoom
import random
import uuid
import copy

DB_PATH = st.session_state['clasp.DB_PATH'] 
DEMOGRAPHICS_PATH = st.session_state['clasp.DEMOGRAPHICS_PATH']

_ANON_ADJECTIVES = [
    'Brave', 'Calm', 'Cheerful', 'Clever', 'Daring', 'Eager', 'Fierce',
    'Gentle', 'Grumpy', 'Happy', 'Jolly', 'Kind', 'Lively', 'Mellow',
    'Noble', 'Peppy', 'Quiet', 'Rowdy', 'Sleepy', 'Sneaky', 'Speedy',
    'Spunky', 'Stern', 'Stoic', 'Swift', 'Timid', 'Witty', 'Zany',
]
_ANON_ANIMALS = [
    'Badger', 'Bear', 'Bison', 'Crane', 'Dingo', 'Falcon', 'Ferret',
    'Gecko', 'Hedgehog', 'Hippo', 'Ibis', 'Jackal', 'Lemur', 'Lynx',
    'Marmot', 'Meerkat', 'Narwhal', 'Otter', 'Pangolin', 'Rabbit',
    'Raven', 'Salamander', 'Sloth', 'Tapir', 'Vulture', 'Walrus',
    'Wombat', 'Yak', 'Zebu',
]

def _random_anon_name():
    patient_name = random.choice(_ANON_ANIMALS) + "^" + random.choice(_ANON_ADJECTIVES)
    return patient_name

def _random_anon_id():
    return f"anon_{uuid.uuid4()}"

def load_or_create_csv(path, columns):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if os.path.exists(path):
        df = pd.read_csv(path)
    else:
        df = pd.DataFrame(columns=columns)

    df.to_csv(path, index=False)
    return df

def sync_orthanc_and_db():
    db = TinyDB(DB_PATH)
    StudyQuery = Query()

    existing_study_ids = {s["orthanc_study_id"] for s in db}
    orthanc_studies = fetch_orthanc_studies()

    # load demographics once
    if os.path.exists(DEMOGRAPHICS_PATH):
        demo_df = pd.read_csv(DEMOGRAPHICS_PATH)
    else:
        demo_df = pd.DataFrame(columns=[
            "patient_id", "first_name", "last_name",
            "sex", "dob", "data_entered"
        ])

    existing_patient_ids = set(demo_df["patient_id"]) if not demo_df.empty else set()

    new_rows = []

    for study_info in orthanc_studies:
        if study_info["ID"] in existing_study_ids:
            continue

        study = Study(study_info)

        if not study.patient_id or study.patient_id == "None":
            study.patient_id = _random_anon_id()
            print(f"Warning: No patient ID for scan. Assigning pseudoid: {study.patient_id}")

        series_list = fetch_orthanc_series_for_study(study.orthanc_study_id)
        for series_info in series_list:
            series = Series(series_info)
            study.series_dict[series.orthanc_series_id] = series

        db.upsert(
            study.to_dict(),
            StudyQuery.study_uid == study.orthanc_study_id
        )

        if study.patient_id in existing_patient_ids:
            continue
        
        patient_name = getattr(study, "patient_name", None)

        if patient_name:
            parts = patient_name.split("^")
        else:
            anon = _random_anon_name()
            parts = anon.split("^")
            print(f"Warning: No patient name for scan. Assigning random name: {anon}")
        last_name = parts[0] if len(parts) > 0 else ""
        first_name = parts[1] if len(parts) > 1 else ""

        new_rows.append({
            "patient_id": study.patient_id,
            "first_name": first_name,
            "last_name": last_name,
            "sex": study.patient_sex if study.patient_sex else "",
            "dob": pd.to_datetime(study.patient_dob, format="%Y%m%d").strftime("%Y-%m-%d") if study.patient_dob else "",
            "data_entered": False
        })

        existing_patient_ids.add(study.patient_id)

    if new_rows:
        demo_df = pd.concat([demo_df, pd.DataFrame(new_rows)], ignore_index=True)
        print(f"DEBUG - demo_df: {demo_df}")
        os.makedirs(os.path.dirname(DEMOGRAPHICS_PATH), exist_ok=True)
        demo_df.to_csv(DEMOGRAPHICS_PATH, index=False)

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
    sax_orthanc_ids = list(sax_df['orthanc_series_id'])  # list preserves order

    # Get total number of frames from first series
    first_dcms = fetch_orthanc_dicoms_for_series(sax_orthanc_ids[0])
    total_frames = len(first_dcms)
    progress_bar = st.progress(0, f"## **Segmentating SAX:**")
    completed = 0

    # Collect all 2D+T series into an ordered dict and build 4D array
    old_dcm_list = {}
    image_2dts = {}
    slice_locations = {}
    pixel_spacing = None

    for sid in sax_orthanc_ids:
        old_dcms = fetch_orthanc_dicoms_for_series(sid)
        old_dcm_list[sid] = old_dcms
        slice_locations[sid] = old_dcms[0].SliceLocation
        if pixel_spacing is None:
            pixel_spacing = old_dcms[0].PixelSpacing
        old_dcms_sorted = sorted(old_dcms, key=lambda d: int(d.InstanceNumber))
        image_2dts[sid] = np.array([ds.pixel_array for ds in old_dcms_sorted])  # T, H, W
        old_dcm_list[sid] = old_dcms_sorted

    # Sort slices by ascending slice location
    sax_orthanc_ids = sorted(sax_orthanc_ids, key=lambda sid: slice_locations[sid])

    # CREATE 4D IMAGE ARRAY (S, T, H, W)
    image_4d = np.array([image_2dts[sid] for sid in sax_orthanc_ids])  # S, T, H, W
    mask_4d = np.zeros_like(image_4d)
    ## Optional - save a gif of the 4d image to check slice and timestep ordering
    # print(f"image 4d shape before inference: {image_4d.shape}")
    # slice_locs = [slice_locations[sid] for sid in sax_orthanc_ids]
    # save_4d_gif(image_4d, str(PLOTS_PATH / f"{study.orthanc_study_id}_sax.gif"), slice_locations=slice_locs)

    ### START OF SECTION TO INTEGRATE YOUR OWN SEGMENTATION MODEL ###
    ### Slice the 4D image as needed for inputs to your model (e.g. 2D, 2D+T, 3D, 4D) and edit the run_inference_on_scan function to run your model ###
    ### Remember to reassign the output mask to the correct location in the 4D mask array so that it gets uploaded to the correct series in Orthanc! ###

    # Loop through each frame of the 4D array and predict on 3D SAX
    for frame in range(image_4d.shape[1]):
        image_3d = image_4d[:, frame, :, :] # S, H, W
        image_3d = np.transpose((image_3d), (1,2,0))  # H,W,S for run inference()
        mask_4d[:, frame, :, :] = run_inference_on_scan(image_3d, pixel_spacing, frame) * st.session_state['clasp.MASK_SCALER']
        completed += 1
        progress_bar.progress(completed / total_frames, text=f"## **SAX Segmentation:** {completed}/{total_frames} frames ({completed/total_frames:.1%})")

    ### END OF SECTION TO INTEGRATE YOUR OWN SEGMENTATION MODEL ###
    
    # Upload each 2D+T mask back to Orthanc
    for idx, sid in enumerate(sax_orthanc_ids):
        old_dcms = old_dcm_list[sid]
        mask_2dt = mask_4d[idx, ...]
        new_orthanc_id = send_series_to_orthanc(
            mask_2dt,
            old_dcms,
            new_description="SAX DL Segmented"
        )
        study.series_dict[sid].dl_orthanc_id = new_orthanc_id

        

# def stupid_roundel_pipeline(study):
#     df = pd.DataFrame([series.__dict__ for series in study.series_dict.values()])
#     sax_dl_df = df[(df['dl_orthanc_id'].notna()) & (df['roundel_orthanc_id'].isna())]

#     if sax_dl_df.empty:
#         return

#     for series_id, dl_series_id in zip(sax_dl_df["orthanc_series_id"], sax_dl_df["dl_orthanc_id"]):
#         series = study.series_dict[series_id]
#         image_dicoms = fetch_orthanc_dicoms_for_series(series_id)
#         mask_dicoms = fetch_orthanc_dicoms_for_series(dl_series_id)

#         masked_images = [image.pixel_array * (mask.pixel_array > 0) for image, mask in zip(image_dicoms, mask_dicoms) ]
#         new_orthanc_id = send_series_to_orthanc(masked_images, image_dicoms, new_description='Roundel Processed')
#         series.roundel_orthanc_id = new_orthanc_id

#     metrics = get_volumes(fetch_orthanc_dicoms_for_series_list(df["dl_orthanc_id"].dropna().unique()))
#     metrics['orthanc_study_id'] = study.orthanc_study_id
#     metrics['patient_id'] = study.patient_id
#     metrics['study_date'] = study.study_date

#     metrics_df = pd.read_csv(EXAMS_PATH)
#     metrics_df = pd.concat([metrics_df, pd.DataFrame([metrics])]).set_index('orthanc_study_id')
#     metrics_df.to_csv(EXAMS_PATH)



def run_pipelines():
    db = TinyDB(DB_PATH)

    pipelines = {
        "MRI Sorting 🧹": mri_sorting_pipeline,
        "SAX Segmentation 🫀": sax_segmentation_pipeline,
        # "Stupid Roundel ⭕": stupid_roundel_pipeline
    }

    studies = fetch_db_studies()

    for pipeline_idx, (pipeline_name, pipeline) in enumerate(pipelines.items()):
        for study in studies:
        # stqdm(
        #     studies,
        #     desc=f"Pipeline {pipeline_idx+1}/{len(pipelines)}: {pipeline_name}. Studies",
        # ):
            pipeline(study)
            update_study(db, study)