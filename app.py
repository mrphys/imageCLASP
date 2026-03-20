from orthanc_utils import *
from db_utils import *
from plot_utils import *
import streamlit as st


DB_PATH = "./image_clasp_db.json"
ORTHANC = "http://localhost:8042"
AUTH = ("orthanc","orthanc")

if "df" not in st.session_state:
    query_orthanc(DB_PATH, ORTHANC, AUTH)
    st.session_state["df"] = load_db_rows(DB_PATH)

with st.sidebar:
    st.header("Filtering")
    
    if st.button("Query Orthanc"):
        with st.spinner('Querying...'):
            query_orthanc(DB_PATH, ORTHANC, AUTH)
        

    if st.button("Query CLASP"):
        st.session_state["df"] = load_db_rows(DB_PATH)
        st.rerun()

    df = st.session_state["df"]

    sex = st.selectbox("Sex", ["All", "M", "F", "Unknown"])
    processed_only = st.selectbox("Processed only", ["All", "Yes", "No"])
    roundel_only = st.selectbox("Roundel only", ["All", "Yes", "No"])
    age_min, age_max = st.slider("Age range", 0, 100, (0, 100))

    filtered_df = df.copy()

    if sex != "All":
        filtered_df = filtered_df[filtered_df["patient_sex"] == sex]

    if processed_only == "Yes":
        filtered_df = filtered_df[filtered_df["DL_processed"]]
    elif processed_only == "No":
        filtered_df = filtered_df[~filtered_df["DL_processed"]]

    if roundel_only == "Yes":
        filtered_df = filtered_df[filtered_df["roundel_processed"]]
    elif roundel_only == "No":
        filtered_df = filtered_df[~filtered_df["roundel_processed"]]

    filtered_df = filtered_df[
        (filtered_df["age"].isna()) |
        ((filtered_df["age"] >= age_min) & (filtered_df["age"] <= age_max))
    ]

    st.metric("Filtered Studies", len(filtered_df))

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
    "<h1 style='text-align: center;margin-bottom: 40px;'>ImageCLASP dashboard</h1>",
    unsafe_allow_html=True
)

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown(
        "<h4 style='text-align: center; margin-bottom: 0px;'>Summary</h4>",
        unsafe_allow_html=True
    )
    st.metric("Total Studies", len(df))
    median_age = df["age"].median()
    age_min = df["age"].min()
    age_max = df["age"].max()
    st.metric("Median Age",f"{median_age:.1f} ({age_min:.0f}-{age_max:.0f})")
    st.metric("Median Series", df["n_series"].median())
    

with col2:
    st.markdown(
        "<h4 style='text-align: center; margin-bottom: 0px;'>Age Distribution</h4>",
        unsafe_allow_html=True
    )
    st.plotly_chart(hist(df, "age", 0, 50, 5, my_palette[3]))

col3, col4, col5 = st.columns([1, 1, 1], gap="large")

with col3:
    st.markdown(
        "<h4 style='text-align: center; margin-bottom: 0px;'>Sex Balance</h4>",
        unsafe_allow_html=True
    )
    st.plotly_chart(pie(df, "patient_sex", my_palette))

with col4:
    st.markdown(
        "<h4 style='text-align: center; margin-bottom: 0px;'>DL processed</h4>",
        unsafe_allow_html=True
    )
    st.plotly_chart(pie(df, "DL_processed", my_palette))

with col5:
    st.markdown(
        "<h4 style='text-align: center; margin-bottom: 0px;'>Roundel</h4>",
        unsafe_allow_html=True
    )
    st.plotly_chart(pie(df, "roundel_processed", my_palette))