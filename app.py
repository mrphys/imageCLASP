from utils.orthanc_utils import *
from utils.db_utils import *
from utils.plot_utils import *
import streamlit as st


DB_PATH = "./image_clasp_db.json"
ORTHANC = "http://localhost:8042"
AUTH = ("orthanc","orthanc")

SESSION = requests.Session()
SESSION.auth = AUTH
SESSION.trust_env = False

st.set_page_config(page_title="ImageCLASP", page_icon="🖇", layout='wide')
st.write('# ImageCLASP')

if "df" not in st.session_state:
    update_orthanc()
    st.session_state["df"] = load_db_rows(DB_PATH)

if len(st.session_state["df"]) == 0:
    st.warning('There are no studies in Orthanc')
    
else:
    with st.sidebar:
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

if st.button("Update Orthanc", use_container_width=True, type = 'primary'):
    update_orthanc()
    st.session_state["df"] = load_db_rows(DB_PATH)
    st.rerun()