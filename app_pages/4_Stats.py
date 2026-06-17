import streamlit as st
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import zipfile
from utils.stats_util import*
from utils.theme_utils import *
from scipy.stats import shapiro

st.set_page_config(layout="wide")

load_theme(secondary="#155a8a",
    secondary_hover="#1F4264",
    secondary_active="#12324D"
    )

# Load CSV

if "df" not in st.session_state:
    st.session_state.df = None

st.markdown(
    "<h1 style='text-align: center; margin-bottom: 40px;'>CLASP Statistics Dashboard</h1>",
    unsafe_allow_html=True,
)



# ---------- Tabs ----------
tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Upload CSV", "Summaries", "Kaplan-Meier", "Cox Regression", "Linear Regression", "Group Comparisons"]
)

if "df" not in st.session_state:
    st.session_state.df = None

with tab0:
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        st.session_state.df = pd.read_csv(uploaded_file)
        st.success("CSV uploaded")
        st.dataframe(st.session_state.df.head(), hide_index=True)

df = st.session_state.df

if df is None:
    st.info("Upload a CSV file in the Upload CSV tab first.")
    st.stop()

variables = df.columns.tolist() 

with tab1:
    with st.form("summary_form"):
        cl1, cl2 = st.columns(2)

        with cl1:
            select_sum_variables = st.multiselect(
                "Choose continuous variables",
                variables,
                placeholder="Choose an option (type to filter)"
            )
        with cl2:
            select_cat_variables = st.multiselect(
                "Choose categorical variables",
                variables,
                placeholder="Choose an option (type to filter)"
            )

        confirm = st.form_submit_button(
            "Confirm selections"
        )

    if confirm and select_sum_variables:

        summary = (df[select_sum_variables].describe(include="all").T)

        normality = []

        for col in select_sum_variables:

            if pd.api.types.is_numeric_dtype(df[col]):

                x = df[col].dropna()

                if len(x) >= 3:
                    p = shapiro(x)[1]

                    normality.append(
                        "Yes" if p > 0.05 else "No"
                    )
                else:
                    normality.append("Insufficient data")

            else:
                normality.append("N/A")

        summary["Normal"] = normality
        
        st.subheader("Continuous variables")
        st.dataframe(summary)
        csv_sum = summary.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download continuous results",
            data=csv_sum,
            file_name="continuous_summary_results.txt",
            mime="text/csv"
        )

    if confirm and select_cat_variables:

        cat_summary = []

        for col in select_cat_variables:

            counts = (
                df[col]
                .fillna("Missing")
                .value_counts()
            )

            categories = " / ".join(
                [str(x) for x in counts.index]
            )

            n_values = " / ".join(
                [str(x) for x in counts.values]
            )

            percents = " / ".join(
                [
                    f"{100*x/len(df):.1f}%"
                    for x in counts.values
                ]
            )

            cat_summary.append({
                "Variable": col,
                "Categories": categories,
                "N": n_values,
                "%": percents
            })

        cat_summary = pd.DataFrame(cat_summary)

        st.subheader("Categorical variables")

        st.dataframe(cat_summary, hide_index=True)
        csv_cat_summary = cat_summary.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download categorical results",
            data=csv_cat_summary,
            file_name="categorical_summary_results.txt",
            mime="text/csv"
        )

with tab2:
    with st.form("KM form"):
        c1, c2 = st.columns(2)
        zip_buffer = BytesIO()

        with c1:
            selected_time = st.selectbox("Choose time variable",variables,index=None, placeholder="Choose an option (type to filter)")
            selected_death = st.selectbox("Choose death variable",variables,index=None, placeholder="Choose an option (type to filter)")

        with c2:
            selected_variables = st.multiselect("Choose test variables",variables, placeholder="Choose an option (type to filter)")
            cut_type = st.selectbox("Choose cutpoint method", ["Median", "Tertile", "Quartile", "Maxstat"], index=None,
                                    placeholder="Choose an option"
            )

        
        confirm = st.form_submit_button("Confirm selections")

    if confirm:

        n = len(selected_variables)
        ncols = 2
        nrows = math.ceil(n / ncols)
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(10, 4*nrows)
        )

        # convert to flat list

        axes = np.array(axes).flatten()

        zip_buffer = BytesIO()

        with zipfile.ZipFile(
            zip_buffer,
            "w",
            zipfile.ZIP_DEFLATED
        ) as zip_file:

            for i, selected_variable in enumerate(selected_variables):


                _, ax, p_value = plot_km(
                    df,
                    time_col=selected_time,
                    event_col=selected_death,
                    value_col=selected_variable,
                    threshold_type=cut_type,
                    ax=axes[i]
                )
                

                axes[i].set_title(
                    f"{selected_variable}\n"
                    f"p={p_value:.4f}"
                )

            plt.tight_layout()

            st.pyplot(
                fig,
                use_container_width=False
            )

            # save complete figure
            img_buffer = BytesIO()

            fig.patch.set_facecolor("white")
            fig.patch.set_alpha(1)

            for ax in fig.get_axes():
                ax.set_facecolor("white")
                for txt in ax.texts:
                    txt.set_color("white")

            fig.savefig(
                img_buffer,
                format="png",
                dpi=300,
                bbox_inches="tight",
                facecolor="white",
                edgecolor="white",
                transparent=False
            )

            img_buffer.seek(0)


            zip_file.writestr(
                "all_KM_plots.png",
                img_buffer.getvalue()
            )

        zip_buffer.seek(0)

        st.download_button(
            label="Download all figures",
            data=zip_buffer,
            file_name="KM_figures.zip",
            mime="application/zip"
        )

            
            

    
            
    
with tab3:

    with st.form("cox_form"):
        c1, c2 = st.columns(2)

        with c1:
            uni_time = st.selectbox("Choose time variable",variables,index=None, placeholder="Choose an option (type to filter)")
            uni_death = st.selectbox("Choose death variable",variables,index=None, placeholder="Choose an option (type to filter)")


        with c2:
            uni_variables = st.multiselect(
                "Choose test variables",
                variables,
                placeholder="Choose an option (type to filter)"
            )
            uni_adjust_variables = st.multiselect(
                "Choose variables to adjust",
                variables,
                placeholder="Choose an option (type to filter)"
            )

        confirm_uni = st.form_submit_button("Run univariate Cox", use_container_width=False)


    with st.form("multivariate_cox_form"):
        d1, d2, d3 = st.columns(3)

        with d1:
            multi_time = st.selectbox("Choose time variable", variables,index=None, placeholder="Choose an option (type to filter)")

        with d2:
            multi_death = st.selectbox("Choose death variable", variables,index=None, placeholder="Choose an option (type to filter)")


        with d3:
            multi_variables = st.multiselect(
                "Choose test variables",
                variables,
                placeholder="Choose an option (type to filter)"
            )

        confirm_multi = st.form_submit_button("Run multivariate Cox", use_container_width=False)


    if confirm_uni:
        results = univariate_cox(
            df,
            uni_time,
            uni_death,
            uni_variables,
            uni_adjust_variables
        )

        st.dataframe(results, hide_index=True)
        header_text = (
            "Univariate Cox Results\n"
            "Generated from CLASP\n"
            f"Adjusted Variables: {', '.join(uni_adjust_variables)}\n"
            "\n"
        )

        csv = (
            header_text
            + results.to_csv(index=False)
        ).encode("utf-8")

        st.download_button(
            label="Download univariate results",
            data=csv,
            file_name="univariate_results.txt",
            mime="text/csv"
        )


    if confirm_multi:
        results = multivariate_cox(
            df,
            multi_time,
            multi_death,
            multi_variables
        )

        st.dataframe(results, hide_index=True)
        header_text = (
            "Multiple Cox Results\n"
            "Generated from CLASP\n"
            f"Test Variables: {', '.join(multi_variables)}\n"
            "\n"

        )

        txt = (
            header_text
            + "\n"
            + results.to_string(index=False)
        )

        st.download_button(
            label="Download multivariate results",
            data=txt.encode("utf-8"),
            file_name="multivariate_results.txt",
            mime="text/plain"
        )

with tab4:
    with st.form("Linear_form"):
        c1, c2 = st.columns(2)
        zip_buffer = BytesIO()

        with c1:
            selected_dep = st.selectbox("Choose dependent variable",variables, index=None, placeholder="Choose an option (type to filter)")
            

        with c2:
            selected_variables = st.multiselect("Choose independent variables",variables, placeholder="Choose an option (type to filter)")
        confirm = st.form_submit_button("Confirm selections")

    if confirm:

        n = len(selected_variables)
        ncols = 2
        nrows = math.ceil(n / ncols)
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(10, 4*nrows)
        )

        # convert to flat list

        axes = np.array(axes).flatten()

        zip_buffer = BytesIO()

        with zipfile.ZipFile(
            zip_buffer,
            "w",
            zipfile.ZIP_DEFLATED
        ) as zip_file:

            for i, selected_variable in enumerate(selected_variables):

                _, _, p_value = plot_regression(df, x_col=selected_variable, y_col=selected_dep, ax=axes[i])

            plt.tight_layout() 
            st.pyplot(
                fig,
                use_container_width=False
            )

            # save complete figure
            img_buffer = BytesIO()

            fig.patch.set_facecolor("white")
            fig.patch.set_alpha(1)

            for ax in fig.get_axes():
                ax.set_facecolor("white")
                for txt in ax.texts:
                    txt.set_color("white")

            fig.savefig(
                img_buffer,
                format="png",
                dpi=300,
                bbox_inches="tight",
                facecolor="white",
                edgecolor="white",
                transparent=False
            )

            img_buffer.seek(0)

            zip_file.writestr(
                "all_regression_plots.png",
                img_buffer.getvalue()
            )
        zip_buffer.seek(0)
    

        st.download_button(
            label="Download all figures",
            data=zip_buffer,
            file_name="ML_figures.zip",
            mime="application/zip"
        )

with tab5:
    with st.form("group_form"):
        gcl1, gcl2 = st.columns(2)

        with gcl1:
            select_gp= st.selectbox("Choose group variable", variables,index=None, placeholder="Choose an option (type to filter)")
        with gcl2:
            select_gp_variables = st.multiselect(
                "Choose test variables",
                variables,
                placeholder="Choose an option (type to filter)"
            )

        confirm = st.form_submit_button(
            "Confirm selections"
        )

    if confirm and select_gp_variables:
        results = desc_by_grp(select_gp_variables, select_gp, 2, df)
        st.dataframe(results, hide_index=True)
        csv_gp = (results.to_csv(index=False)).encode("utf-8")
        st.download_button(
            label="Download group results",
            data=csv_gp,
            file_name="group_summary_results.txt",
            mime="text/csv"
        )