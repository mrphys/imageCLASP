import os
import streamlit as st
import duckdb

from utils.pipeline import *
from utils.theme_utils import *
from utils.reset_utils import *

reset_app("data_entry")
reset_app("roundel")

if not os.path.exists(st.session_state["clasp.DEMOGRAPHICS_PATH"]):
    st.warning("There are no patients in the database!")
    st.stop()

st.set_page_config(layout="wide")

load_theme(
    secondary="#A94442",
    secondary_hover="#7A2F2F",
    secondary_active="#5C1F1F",
)

con = duckdb.connect()

# ---------- Register raw tables ----------
OUT_PATH = "tables"
OUT_DEMOGRAPHICS_CSV = f"{OUT_PATH}/demographics.csv"
OUT_EVENTS_CSV = f"{OUT_PATH}/events.csv"
OUT_DIAGNOSES_CSV = f"{OUT_PATH}/diagnoses.csv"
OUT_PROCEDURES_CSV = f"{OUT_PATH}/procedures.csv"
OUT_TESTS_CSV = f"{OUT_PATH}/tests.csv"
OUT_TEST_VALUES_CSV = f"{OUT_PATH}/test_values.csv"
OUT_MEDICATIONS_CSV = f"{OUT_PATH}/medications.csv"
OUT_EXAMS_CSV = f"{OUT_PATH}/exams.csv"

con.execute(f"""
CREATE OR REPLACE VIEW demographics AS
SELECT * FROM read_csv_auto('{OUT_DEMOGRAPHICS_CSV}')
""")

con.execute(f"""
CREATE OR REPLACE VIEW events AS
SELECT * FROM read_csv_auto('{OUT_EVENTS_CSV}')
""")

con.execute(f"""
CREATE OR REPLACE VIEW diagnoses AS
SELECT * FROM read_csv_auto('{OUT_DIAGNOSES_CSV}')
""")

con.execute(f"""
CREATE OR REPLACE VIEW procedures AS
SELECT * FROM read_csv_auto('{OUT_PROCEDURES_CSV}')
""")

con.execute(f"""
CREATE OR REPLACE VIEW exams AS
SELECT * FROM read_csv_auto('{OUT_EXAMS_CSV}')
""")

con.execute(f"""
CREATE OR REPLACE VIEW tests AS
SELECT * FROM read_csv_auto('{OUT_TESTS_CSV}')
""")

con.execute(f"""
CREATE OR REPLACE VIEW test_values AS
SELECT * FROM read_csv_auto('{OUT_TEST_VALUES_CSV}')
""")

con.execute(f"""
CREATE OR REPLACE VIEW medications AS
SELECT * FROM read_csv_auto('{OUT_MEDICATIONS_CSV}')
""")

# ---------- Tables exposed to UI ----------
TABLES = [
    "demographics",
    "events",
    "diagnoses",
    "procedures",
    "exams",
    "tests",
    "test_values",
    "medications",
]

# One filter per table
FILTER_COLUMN_MAP = {
    "demographics": "",
    "events": "events_type",
    "diagnoses": "diagnosis_type",
    "procedures": "procedures_type",
    "tests": "test_type",
    "medications": "medication",
}

FILTER_LABEL_MAP = {
    "demographics": "Filter",
    "events": "Event",
    "diagnoses": "Diagnosis",
    "procedures": "Procedure type",
    "exams": "Exam type",
    "tests": "Test type",
    "test_values": "Parameter",
    "medications": "Medication",
}


# ---------- Helpers ----------
def get_columns(table: str) -> list[str]:
    return con.execute(f"DESCRIBE {table}").df()["column_name"].tolist()


def get_distinct_values(table: str, column: str) -> list[str]:
    if not column or column not in get_columns(table):
        return []

    df = con.execute(f"""
        SELECT DISTINCT CAST({column} AS VARCHAR) AS value
        FROM {table}
        WHERE {column} IS NOT NULL
        ORDER BY value
    """).df()

    if df.empty:
        return []

    return df["value"].tolist()


def sql_quote(value) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def get_filter_column(table: str) -> str:
    col = FILTER_COLUMN_MAP.get(table, "")
    return col if col in get_columns(table) else ""


def get_filter_label(table: str) -> str:
    return FILTER_LABEL_MAP.get(table, "Filter")


def get_default_date_col(table: str) -> str:
    cols = get_columns(table)
    preferred = [
        "event_date",
        "procedure_date",
        "exam_date",
        "test_date",
        "diagnosis_date",
        "medication_date",
        "date",
    ]
    for col in preferred:
        if col in cols:
            return col
    return cols[0] if cols else ""


def build_where_clause(base_table: str, joins: list[dict]) -> str:
    clauses = []

    # Base table filter
    base_filter_col = st.session_state.get("base_filter_col", "")
    base_filter_value = st.session_state.get("base_filter_value", "")
    if base_filter_col and base_filter_value:
        clauses.append(f"t0.{base_filter_col} = {sql_quote(base_filter_value)}")

    # Joined table filters
    for i, join in enumerate(joins, start=1):
        filter_col = join.get("filter_col", "")
        filter_values = join.get("filter_values", [])
        if filter_col and filter_values:
            clauses.append(f"t{i}.{filter_col} = {sql_quote(filter_values[0])}")

    return "\nWHERE " + "\n  AND ".join(clauses) if clauses else ""


def build_query(base_table: str, joins: list[dict]) -> str:
    query = [
        "SELECT *",
        f"FROM {base_table} t0",
    ]

    base_cols = get_columns(base_table)
    base_has_patient_id = "patient_id" in base_cols

    for i, join in enumerate(joins, start=1):
        alias = f"t{i}"
        right_table = join["table"]
        right_cols = get_columns(right_table)
        right_has_patient_id = "patient_id" in right_cols
        date_mode = join.get("date_mode", "All rows")

        if not base_has_patient_id or not right_has_patient_id:
            continue

        if date_mode == "All rows":
            query.append(
                f"""LEFT JOIN {right_table} {alias}
ON t0.patient_id = {alias}.patient_id"""
            )

        elif date_mode == "Latest":
            right_date_col = join.get("right_date_col", "")
            if not right_date_col:
                continue

            query.append(
                f"""LEFT JOIN LATERAL (
    SELECT *
    FROM {right_table} src
    WHERE t0.patient_id = src.patient_id
      AND src.{right_date_col} IS NOT NULL
    ORDER BY TRY_CAST(src.{right_date_col} AS DATE) DESC NULLS LAST
    LIMIT 1
) {alias} ON TRUE"""
            )

        elif date_mode == "Nearest to reference date":
            left_date_col = join.get("left_date_col", "")
            right_date_col = join.get("right_date_col", "")
            if not left_date_col or not right_date_col:
                continue

            query.append(
                f"""LEFT JOIN LATERAL (
    SELECT *
    FROM {right_table} src
    WHERE t0.patient_id = src.patient_id
      AND t0.{left_date_col} IS NOT NULL
      AND src.{right_date_col} IS NOT NULL
    ORDER BY ABS(
        DATEDIFF(
            'day',
            TRY_CAST(t0.{left_date_col} AS DATE),
            TRY_CAST(src.{right_date_col} AS DATE)
        )
    ) ASC
    LIMIT 1
) {alias} ON TRUE"""
            )

    query.append(build_where_clause(base_table, joins))
    query.append("LIMIT 50")

    return "\n".join([line for line in query if line.strip()])


# ---------- Session state ----------
if "joins" not in st.session_state:
    st.session_state.joins = []

if "query_result" not in st.session_state:
    st.session_state.query_result = None

if "base_filter_col" not in st.session_state:
    st.session_state.base_filter_col = ""

if "base_filter_value" not in st.session_state:
    st.session_state.base_filter_value = ""


# ---------- UI ----------
st_header("Query and Export Data")

base_table = st.selectbox("Base table", TABLES, key="base_table")
base_cols = get_columns(base_table)

if "patient_id" not in base_cols:
    st.warning(f"Base table '{base_table}' does not contain patient_id, so patient-level joins will not work.")

# ---------- Base table filter ----------
base_filter_col = get_filter_column(base_table)
base_filter_label = get_filter_label(base_table)

if base_filter_col:
    base_filter_options = get_distinct_values(base_table, base_filter_col)
    st.session_state.base_filter_col = base_filter_col
    st.session_state.base_filter_value = st.selectbox(
        f"{base_filter_label} ({base_table})",
        options=[""] + base_filter_options,
        key="base_filter_value_select",
    )
else:
    st.session_state.base_filter_col = ""
    st.session_state.base_filter_value = ""

if st.button("Add Table", type="primary"):
    default_table = next((t for t in TABLES if t != base_table), TABLES[0])
    st.session_state.joins.append(
        {
            "table": default_table,
            "date_mode": "All rows",
            "left_date_col": get_default_date_col(base_table),
            "right_date_col": get_default_date_col(default_table),
            "filter_col": "",
            "filter_values": [],
        }
    )
    st.rerun()

# ---------- Join configuration ----------
for i, join in enumerate(st.session_state.joins):
    if i > 0:
        st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

    with st.container(border=True):
        top1, top2, top3 = st.columns(3)

        with top1:
            available_tables = [t for t in TABLES if t != base_table]
            if not available_tables:
                available_tables = TABLES

            if join["table"] not in available_tables:
                join["table"] = available_tables[0]

            join["table"] = st.selectbox(
                "Additional table",
                available_tables,
                index=available_tables.index(join["table"]),
                key=f"table_{i}",
            )

        filter_col = get_filter_column(join["table"])
        filter_label = get_filter_label(join["table"])

        with top2:
            if filter_col:
                filter_options = get_distinct_values(join["table"], filter_col)
                selected_value = st.selectbox(
                    filter_label,
                    options=[""] + filter_options,
                    index=0 if not join.get("filter_values") else (
                        ([""] + filter_options).index(join["filter_values"][0])
                        if join["filter_values"][0] in filter_options else 0
                    ),
                    key=f"filter_value_{i}",
                )
                join["filter_col"] = filter_col
                join["filter_values"] = [selected_value] if selected_value else []
            else:
                st.selectbox(
                    "Filter",
                    options=["No filter available"],
                    index=0,
                    disabled=True,
                    key=f"filter_disabled_{i}",
                )
                join["filter_col"] = ""
                join["filter_values"] = []

        with top3:
            join["date_mode"] = st.selectbox(
                "Date filter",
                ["All rows", "Latest", "Nearest to reference date"],
                index=["All rows", "Latest", "Nearest to reference date"].index(join["date_mode"])
                if join["date_mode"] in ["All rows", "Latest", "Nearest to reference date"] else 0,
                key=f"date_mode_{i}",
            )

        right_cols = get_columns(join["table"])

        if join["date_mode"] == "Nearest to reference date":
            c1, c2 = st.columns(2)

            with c1:
                join["left_date_col"] = st.selectbox(
                    f"Reference date in {base_table}",
                    base_cols,
                    index=base_cols.index(join["left_date_col"]) if join["left_date_col"] in base_cols else 0,
                    key=f"left_date_col_{i}",
                )

            with c2:
                join["right_date_col"] = st.selectbox(
                    f"Date in {join['table']}",
                    right_cols,
                    index=right_cols.index(join["right_date_col"]) if join["right_date_col"] in right_cols else 0,
                    key=f"right_date_col_{i}",
                )

        elif join["date_mode"] == "Latest":
            join["right_date_col"] = st.selectbox(
                f"Date in {join['table']}",
                right_cols,
                index=right_cols.index(join["right_date_col"]) if join["right_date_col"] in right_cols else 0,
                key=f"right_date_col_{i}",
            )

        btn1, spacer, btn2 = st.columns([1, 4, 1])

        with btn1:
            if st.button("Add Table", type="primary", key=f"add_after_{i}"):
                default_table = next((t for t in TABLES if t != base_table), TABLES[0])
                st.session_state.joins.insert(
                    i + 1,
                    {
                        "table": default_table,
                        "date_mode": "All rows",
                        "left_date_col": get_default_date_col(base_table),
                        "right_date_col": get_default_date_col(default_table),
                        "filter_col": "",
                        "filter_values": [],
                    },
                )
                st.rerun()

        with btn2:
            if st.button("Remove", key=f"remove_{i}", type="secondary"):
                st.session_state.joins.pop(i)
                st.rerun()

# ---------- SQL preview ----------
query = build_query(base_table, st.session_state.joins)

st.subheader("Generated SQL")
st.code(query, language="sql")

# ---------- Run query ----------
if st.button("Run query", type="primary"):
    try:
        st.session_state.query_result = con.execute(query).df()
    except Exception as e:
        st.error(str(e))

# ---------- Results ----------
if st.session_state.query_result is not None:
    df = st.session_state.query_result
    st.dataframe(df, use_container_width=True)

    st.download_button(
        "Download CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="clasp_export.csv",
        mime="text/csv",
        type="primary",
    )