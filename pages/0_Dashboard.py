import streamlit as st
from utils.pipeline import *

st.set_page_config(page_icon="🖇", layout='wide')
st_header('CLASP Dashboard')

if "df" not in st.session_state:
    sync_orthanc_and_db()
    run_pipelines()
    st.session_state["df"] = load_db_rows(DB_PATH)

if len(st.session_state["df"]) == 0:
    st.warning('There are no studies in Orthanc')
    
else:

    # if st.button("Check Orthanc"):
    #     sync_orthanc_and_db()
    #     run_pipelines()

    #     st.session_state["df"] = load_db_rows(DB_PATH)
    #     st.rerun()

    # if st.button("Update CLASP"):
    #     metrics_df = pd.read_csv(METRICS_PATH)
    #     st.session_state["df"] = load_db_rows(DB_PATH)
    #     st.rerun()
        
    # st.header("Filtering")
    df = st.session_state["df"]

    # sex = st.selectbox("Sex", ["All", "M", "F", "Unknown"])
    # processed_only = st.selectbox("Processed only", ["All", "Yes", "No"])
    # roundel_only = st.selectbox("Roundel only", ["All", "Yes", "No"])
    # age_min, age_max = st.slider("Age range", 0, 100, (0, 100))

    # filtered_df = df.copy()

    # if sex != "All":
    #     filtered_df = filtered_df[filtered_df["patient_sex"] == sex]

    # if processed_only == "Yes":
    #     filtered_df = filtered_df[filtered_df["sax_processed"]]
    # elif processed_only == "No":
    #     filtered_df = filtered_df[~filtered_df["sax_processed"]]

    # if roundel_only == "Yes":
    #     filtered_df = filtered_df[filtered_df["roundel_processed"]]
    # elif roundel_only == "No":
    #     filtered_df = filtered_df[~filtered_df["roundel_processed"]]

    # filtered_df = filtered_df[
    #     (filtered_df["age"].isna()) |
    #     ((filtered_df["age"] >= age_min) & (filtered_df["age"] <= age_max))
    # ]

    # st.metric("Filtered Studies", len(filtered_df))


    col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1, 1, 1, 1], gap="medium")

    with col1:
        st.plotly_chart(hist(metrics_df, "lv_edv", my_palette[2], "LVEDV"))

    with col2:
        st.plotly_chart(hist(metrics_df, "lv_esv", my_palette[2], "LVESV"))
    
    with col3:
        st.plotly_chart(hist(metrics_df, "lv_ef", my_palette[2], "LVEF"))

    with col4:
        st.plotly_chart(hist(metrics_df, "rv_edv", my_palette[1], "RVEDV"))

    with col5:
        st.plotly_chart(hist(metrics_df, "rv_esv", my_palette[1], "RVESV"))
    
    with col6:
        st.plotly_chart(hist(metrics_df, "rv_ef", my_palette[1], "RVEF"))
    
    
    col7, col8, col9 = st.columns([1, 1, 1], gap="large")

    with col7:
        st.plotly_chart(pie(df, "patient_sex", my_palette))

    with col8:
        st.plotly_chart(pie(df, "sax_processed", my_palette))

    with col9:
        st.plotly_chart(pie(df, "roundel_processed", my_palette))