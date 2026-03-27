from utils.orthanc_utils import *
from utils.db_utils import *
from utils.plot_utils import *
from utils.mri_sorter import MRI_Sorter
import streamlit as st
from scipy.ndimage import zoom
from model_utils import crop_pad_hw, run_inference_on_scan
import copy

st.set_page_config(page_title="ImageCLASP", page_icon="🖇", layout='wide')

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

def update_orthanc():
    """Updates"""

    db = TinyDB(DB_PATH)
    StudyQuery = Query()
    existing_ids = [s["orthanc_study_id"] for s in db]
    
    target_list = ['short', 'SAX', 'KT', 'ML cine','sa']
    
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

            if not series.sax_processed:
                 
                if any(w in series.description for w in target_list):#probably need to add type here as well
                    print(series.orthanc_id)
                    series.series_orientation = 'SAX'
                    series.sax_processed = True
                    mask, old_dcms = extract_series_dcm_DL(series.orthanc_id)
                    new_uid = upload_processed_series(mask,old_dcms)
                    new_orthanc_id = get_orthanc_series_data_from_uid(new_uid, ORTHANC, AUTH)
                    new_series = copy.deepcopy(series)
                    new_series.description = 'DL Processed'
                    new_series.orthanc_id = new_orthanc_id['ID']
                    new_series.associated_uid = series.orthanc_id
                    study.series_list.append(new_series)
                    
                
            if series.sax_processed and not series.roundel_processed: 
                # add roundel
                pass
            study.series_list.append(series)
        db.upsert(study.to_dict(), StudyQuery.study_uid == study.uid)
        print('Done')

    db.close()

##################
# Dashboard
##################

st.set_page_config(page_icon="🖇", layout='wide')
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    "<h1 style='text-align: center;margin-bottom: 40px;'>CLASP dashboard</h1>",
    unsafe_allow_html=True
)

if "df" not in st.session_state:
    update_orthanc()
    st.session_state["df"] = load_db_rows(DB_PATH)

if len(st.session_state["df"]) == 0:
    st.warning('There are no studies in Orthanc')
    
else:
    with st.sidebar:

        if st.button("Check Orthanc"):
            update_orthanc()
        

        if st.button("Update CLASP"):
            st.session_state["df"] = load_db_rows(DB_PATH)
            st.rerun()
            
        st.header("Filtering")
        df = st.session_state["df"]

        sex = st.selectbox("Sex", ["All", "M", "F", "Unknown"])
        processed_only = st.selectbox("Processed only", ["All", "Yes", "No"])
        roundel_only = st.selectbox("Roundel only", ["All", "Yes", "No"])
        age_min, age_max = st.slider("Age range", 0, 100, (0, 100))

        filtered_df = df.copy()

        if sex != "All":
            filtered_df = filtered_df[filtered_df["patient_sex"] == sex]

        if processed_only == "Yes":
            filtered_df = filtered_df[filtered_df["sax_processed"]]
        elif processed_only == "No":
            filtered_df = filtered_df[~filtered_df["sax_processed"]]

        if roundel_only == "Yes":
            filtered_df = filtered_df[filtered_df["roundel_processed"]]
        elif roundel_only == "No":
            filtered_df = filtered_df[~filtered_df["roundel_processed"]]

        filtered_df = filtered_df[
            (filtered_df["age"].isna()) |
            ((filtered_df["age"] >= age_min) & (filtered_df["age"] <= age_max))
        ]

        st.metric("Filtered Studies", len(filtered_df))


    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown('### Summary')
        st.metric("Total Studies", len(df))
        median_age = df["age"].median()
        age_min = df["age"].min()
        age_max = df["age"].max()
        st.metric("Median Age",f"{median_age:.1f} ({age_min:.0f}-{age_max:.0f})")
        st.metric("Median Series", df["n_series"].median())
        

    with col2:
        st.plotly_chart(hist(df, "age", 0, 50, 5, my_palette[3]))

    col3, col4, col5 = st.columns([1, 1, 1], gap="large")

    with col3:
        st.plotly_chart(pie(df, "patient_sex", my_palette))

    with col4:
        st.plotly_chart(pie(df, "sax_processed", my_palette))

    with col5:
        st.plotly_chart(pie(df, "roundel_processed", my_palette))
