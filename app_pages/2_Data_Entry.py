from utils.data_entry_utils import *
from utils.pipeline import *
from utils.theme_utils import *
from utils.reset_utils import *

if 'data_entry.CLINICAL_ENTRIES' not in st.session_state:
    st.session_state['data_entry.MAIN_TABLES'] = {'Diagnoses':['primary','secondary'], 
                                                    'Events':['major','minor'], 
                                                    'Procedures':['surgery','intervention']}
    st.session_state['data_entry.TEST_TABLES'] = [p.name.removesuffix("_reference.json").capitalize()
                                                    for p in REFERENCE_PATH.glob("*_reference.json")]
    st.session_state['data_entry.CLINICAL_ENTRIES'] = ['Demographics'] + list(st.session_state['data_entry.MAIN_TABLES'].keys()) + st.session_state['data_entry.TEST_TABLES']


REFERENCE_PATH = st.session_state["clasp.REFERENCE_PATH"]
reset_app("roundel")

load_theme(secondary="#155a8a",
    secondary_hover="#1F4264",
    secondary_active="#12324D"
    )

if not os.path.exists(st.session_state["clasp.DEMOGRAPHICS_PATH"]):
    st.warning("There are no patients in the database!")
    st.stop()

# ---------- UI ----------
st_header("Clinical Data Entry")

demo_df = load_demographics()
num_remaining_patients = len(demo_df)

names = [
    f"{row['first_name']} {row['last_name']} | {row['patient_id']}"
    for _, row in demo_df.iterrows()
]



col1, col2 = st.columns(2)

# ---------- Configure toggle ----------

if 'data_entry.configure_mode' not in st.session_state:
    st.session_state['data_entry.configure_mode'] = False

with col2:
    configure_mode = st.session_state['data_entry.configure_mode']

    configure_mode = st.pills("Admin", ['Configurator'], label_visibility='hidden')
    st.session_state['data_entry.configure_mode'] = True if configure_mode == 'Configurator' else False


#########################################################
###################  Data Entry #######################
#########################################################


if not configure_mode:
    if "data_entry.current_patient" not in st.session_state:
        st.session_state["data_entry.current_patient"] = names[0]

    if "data_entry.prev_patient" not in st.session_state:
        st.session_state["data_entry.prev_patient"] = names[0]
        
    with col1:
        selected_name = st.selectbox(
            f"Select Patient ({num_remaining_patients} Patients Left)",
            options=names,
            key="data_entry.current_patient",
            on_change=clear_on_patient_change
        )

        patient_idx = names.index(selected_name)
        st.session_state["data_entry.patient_idx"] = patient_idx

    # ---------- Init ----------
    if "data_entry.initialized" not in st.session_state:
        init_patient_from_csv()
        st.session_state["data_entry.initialized"] = True

    tabs = st.tabs(st.session_state['data_entry.CLINICAL_ENTRIES'])
    # --- Tab 1: Demographics ---
    with tabs[0]:
        with st.form("patient_form"):
            c1, c2 = st.columns(2)

            with c1:
                st.text_input("First name", key="data_entry.first_name")
                st.text_input("Last name", key="data_entry.last_name")

            with c2:
                st.selectbox("Sex", ["F", "M", "Other"], key="data_entry.sex", index=None)
                st.date_input(
                    "Date of birth",
                    min_value=MIN_DATE,
                    max_value=date.today(),
                    key="data_entry.dob",
                    format="DD-MM-YYYY"

                )

            confirm_patient = st.form_submit_button("Confirm demographics")

    tab_index = 1

    for label in st.session_state['data_entry.MAIN_TABLES'].keys():
        with tabs[tab_index]:
            create_multi_form(label=label, forms=st.session_state['data_entry.MAIN_TABLES'][label])
        tab_index += 1

    for label in st.session_state['data_entry.TEST_TABLES']:
        with tabs[tab_index]:
            create_single_form(label)
        tab_index += 1

    # ---------- Global Save ----------
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # Initialize states
    if "confirm_save" not in st.session_state:
        st.session_state.confirm_save = False
    if "saved" not in st.session_state:
        st.session_state.saved = False

    c1, c2 = st.columns([0.5, 0.5])

    with c1:
        if st.button(
            "Save Record",
            key="data_entry.save_record_btn",
            type="primary",
            use_container_width=True
        ):
            st.session_state.confirm_save = True

    with c2:
        if st.session_state.confirm_save:

            col1, col2 = st.columns(2)

            with col1:
                if st.button(
                    "Yes, Save",
                    icon=":material/save:",
                    use_container_width=True
                ):
                    if not st.session_state["data_entry.patient_id"]:
                        st.error("Patient ID is missing.")
                    else:
                        save_data_entry()
                        clear_on_patient_change()
                        st.session_state.saved = True

                    st.session_state.confirm_save = False

            with col2:
                if st.button(
                    "Cancel",
                    icon=":material/cancel:",
                    use_container_width=True
                ):
                    st.session_state.confirm_save = False
                    st.rerun()
            
            if not st.session_state.saved:
                st.warning("Are you sure you want to save this record?")


        if st.session_state.saved:
            st.success("Saved successfully!")
            time.sleep(1)
            st.session_state.saved = False
            st.rerun()




#########################################################
###################  Configurator #######################
#########################################################
if configure_mode:
    reset_app('data_entry')

    # --------------------------------------------------
    # apply pending selectbox value BEFORE widget creation
    # --------------------------------------------------
    if "pending_table" in st.session_state:

        st.session_state["data_entry.tables"] = (
            st.session_state["pending_table"]
        )

        del st.session_state["pending_table"]

    # --------------------------------------------------
    # refresh tables
    # --------------------------------------------------
    tables_list = sorted([
        f.stem.replace('_reference', '')
        for f in REFERENCE_PATH.glob("*_reference.*")
    ])

    select_options = (
        ['-']
        + tables_list
        + ['➕ Add Test']
        + ['🗑️ Delete Test']
    )

    # initialize
    if "data_entry.tables" not in st.session_state:
        st.session_state["data_entry.tables"] = '-'

    # --------------------------------------------------
    # select table
    # --------------------------------------------------
    with col1:

        selected_table = st.selectbox(
            "Select Table",
            options=select_options,
            key="data_entry.tables"
        )

    # --------------------------------------------------
    # no selection
    # --------------------------------------------------
    if selected_table == '-':

        st.write('Select Table to Configure')

    # --------------------------------------------------
    # add test
    # --------------------------------------------------
    elif selected_table == '➕ Add Test':

        add_test_name = (
            st.text_input('Enter New Test Name')
            .replace(' ', '_')
            .lower()
        )

        if st.button(
            "➕ Add Test",
            type='primary',
            use_container_width=True
        ):

            if add_test_name == '':

                st.warning('Enter New Test Name')

            else:
                st.session_state['data_entry.CLINICAL_ENTRIES'].append(add_test_name)
                st.session_state['data_entry.TEST_TABLES'].append(add_test_name)

                add_test_dict = [{
                    "parameter": None,
                    "entry_type": None,
                    "value": None
                }]

                with open(
                    REFERENCE_PATH / f"{add_test_name}_reference.json",
                    "w"
                ) as f:

                    json.dump(add_test_dict, f, indent=2)

                st.success(f'Added {add_test_name}')

                # queue selectbox update
                st.session_state["pending_table"] = add_test_name

                time.sleep(0.5)
                st.rerun()

    # --------------------------------------------------
    # delete test
    # --------------------------------------------------
    elif selected_table == '🗑️ Delete Test':

        delete_test_name = st.selectbox(
            "Select Test to Delete",
            options=[
                table
                for table in st.session_state['data_entry.TEST_TABLES']
                if table.lower() != 'medication'
            ]
        )

        if st.button(
            "🗑️ Delete Test",
            type='primary',
            use_container_width=True
        ):
            st.session_state['data_entry.CLINICAL_ENTRIES'].remove(delete_test_name.lower())
            st.session_state['data_entry.TEST_TABLES'].remove(delete_test_name.lower())
            os.remove(
                REFERENCE_PATH
                / f"{delete_test_name.lower()}_reference.json"
            )

            st.success(f'Deleted {delete_test_name}')

            # reset selection
            st.session_state["pending_table"] = '-'

            time.sleep(0.5)
            st.rerun()

    # --------------------------------------------------
    # csv tables
    # --------------------------------------------------
    elif selected_table in [k.lower() for k in st.session_state['data_entry.MAIN_TABLES'].keys()]:

        selected_table_key = next(k for k in st.session_state['data_entry.MAIN_TABLES'] if k.lower() == selected_table.lower())

        table_REFERENCE_PATH = (
            REFERENCE_PATH
            / f"{selected_table.lower()}_reference.csv"
        )

        df = pd.read_csv(table_REFERENCE_PATH)

        if selected_table == 'Medication':

            edited_df = st.data_editor(
                df,
                num_rows="dynamic",
                use_container_width=True,
                key='data_entry.table_editor'
            )

        else:

            edited_df = st.data_editor(
                df,
                column_config={
                    "entry_type": st.column_config.SelectboxColumn(
                        "entry_type",
                        options=list(st.session_state['data_entry.MAIN_TABLES'][selected_table_key]),
                        required=True,
                    )
                },
                num_rows="dynamic",
                use_container_width=True,
                key='data_entry.table_editor'
            )

        if st.button(
            "Save changes",
            key=f"save_{selected_table}",
            type='primary',
            use_container_width=True
        ):

            edited_df.to_csv(table_REFERENCE_PATH, index=False)

            st.success("Saved!")

    # --------------------------------------------------
    # json tables
    # --------------------------------------------------
    else:

        reference_file = (
            REFERENCE_PATH
            / f"{selected_table.lower()}_reference.json"
        )

        with open(reference_file, "r") as f:
            data = json.load(f)

        df = pd.DataFrame(data)

        edited_df = st.data_editor(
            df.drop(columns=["value"], errors="ignore"),
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "parameter": st.column_config.TextColumn(
                    "parameter"
                ),
                "entry_type": st.column_config.SelectboxColumn(
                    "entry_type",
                    options=["numeric", "list"],
                    required=True,
                ),
            },
            key="main_editor",
        )

        # restore original values
        updated_df = edited_df.copy()
        updated_df["value"] = df["value"]

        # --------------------------------------------------
        # list editors
        # --------------------------------------------------
        list_rows = (
            updated_df[
                updated_df["entry_type"] == "list"
            ].index
        )

        cols = (
            st.columns(max(len(list_rows), 1))
            if len(list_rows) > 0
            else []
        )

        for col, i in zip(cols, list_rows):

            with col:

                param = updated_df.at[i, "parameter"]
                raw_value = updated_df.at[i, "value"]

                # normalize list values
                if isinstance(raw_value, list):

                    values = raw_value

                elif isinstance(raw_value, str):

                    try:

                        parsed = json.loads(raw_value)

                        values = (
                            parsed
                            if isinstance(parsed, list)
                            else []
                        )

                    except Exception:

                        values = []

                else:

                    values = []

                sub_df = pd.DataFrame({
                    param: values if values else [""]
                })

                edited_sub_df = st.data_editor(
                    sub_df,
                    num_rows="dynamic",
                    use_container_width=True,
                    key=f"list_editor_{i}",
                    hide_index=True,
                )

                cleaned_values = [
                    v
                    for v in edited_sub_df[param].tolist()
                    if isinstance(v, str)
                    and v.strip() != ""
                ]

                updated_df.at[i, "value"] = cleaned_values

        # --------------------------------------------------
        # save
        # --------------------------------------------------
        if st.button(
            "Save changes",
            use_container_width=True,
            type='primary'
        ):

            df_to_save = updated_df.copy()

            def serialize(v):

                if isinstance(v, (list, dict)):
                    return v

                return v

            df_to_save["value"] = (
                df_to_save["value"].apply(serialize)
            )

            with open(reference_file, "w") as f:

                json.dump(
                    df_to_save.to_dict(orient="records"),
                    f,
                    indent=2
                )

            st.success("Saved!")