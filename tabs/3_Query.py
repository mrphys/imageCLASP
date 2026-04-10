import streamlit as st
import duckdb
from utils.pipeline import *
from utils.theme_utils import *

st.set_page_config(layout="wide")

load_theme(secondary="#A94442",
    secondary_hover="#7A2F2F",
    secondary_active="#5C1F1F")



con = duckdb.connect()

# ---------- Register raw tables ----------
OUT_PATH = 'tables'
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
CREATE OR REPLACE VIEW procedures AS
SELECT * FROM read_csv_auto('{OUT_DIAGNOSES_CSV}')
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


# ---------- Helpers ----------
def get_columns(table: str) -> list[str]:
    return con.execute(f"DESCRIBE {table}").df()["column_name"].tolist()


def make_pivot_view(
    source_table: str,
    pivot_view_name: str,
    id_col: str,
    category_col: str,
    value_col: str,
) -> None:
    source_cols = get_columns(source_table)
    if not all(col in source_cols for col in [id_col, category_col, value_col]):
        return

    values = con.execute(f"""
        SELECT DISTINCT {category_col}
        FROM {source_table}
        WHERE {category_col} IS NOT NULL
        ORDER BY {category_col}
    """).df()[category_col].tolist()

    if not values:
        return

    pivot_cols = ", ".join([f"'{v}'" for v in values])

    con.execute(f"""
        CREATE OR REPLACE VIEW {pivot_view_name} AS
        SELECT *
        FROM (
            SELECT {id_col}, {category_col}, {value_col}
            FROM {source_table}
        )
        PIVOT (
            MAX({value_col})
            FOR {category_col} IN ({pivot_cols})
        )
    """)


def alias_to_table(base_table: str, joins: list[dict], alias: str) -> str:
    if alias == "t0":
        return base_table
    return joins[int(alias[1:]) - 1]["table"]


def alias_to_display_name(base_table: str, joins: list[dict], alias: str) -> str:
    if alias == "t0":
        return f"{base_table} (base)"
    idx = int(alias[1:]) - 1
    return joins[idx]["table"]


def shared_columns(left_table: str, right_table: str) -> list[str]:
    left_cols = set(get_columns(left_table))
    right_cols = set(get_columns(right_table))
    return sorted(left_cols.intersection(right_cols))


def build_query(base_table: str, joins: list[dict]) -> str:
    query = [f"SELECT *", f"FROM {base_table} t0"]

    for i, join in enumerate(joins, start=1):
        alias = f"t{i}"
        left_alias = join["left_alias"]
        join_key = join["join_key"]
        right_table = join["table"]
        date_mode = join.get("date_mode", "All rows")

        if not join_key:
            continue

        if date_mode == "All rows":
            query.append(
                f"""LEFT JOIN {right_table} {alias}
                    ON {left_alias}.{join_key} = {alias}.{join_key}"""
                )

        elif date_mode == "Latest":
            right_date_col = join["right_date_col"]
            if not right_date_col:
                continue

            query.append(
                f"""LEFT JOIN LATERAL (
                    SELECT *
                    FROM {right_table} src
                    WHERE {left_alias}.{join_key} = src.{join_key}
                    AND src.{right_date_col} IS NOT NULL
                    ORDER BY TRY_CAST(src.{right_date_col} AS DATE) DESC NULLS LAST
                    LIMIT 1
                ) {alias} ON TRUE"""
                            )

        elif date_mode == "Nearest to exam/reference date":
            left_date_col = join["left_date_col"]
            right_date_col = join["right_date_col"]
            if not left_date_col or not right_date_col:
                continue

            query.append(
                f"""LEFT JOIN LATERAL (
                    SELECT *
                    FROM {right_table} src
                    WHERE {left_alias}.{join_key} = src.{join_key}
                    AND src.{right_date_col} IS NOT NULL
                    AND {left_alias}.{left_date_col} IS NOT NULL
                    ORDER BY ABS(
                        DATEDIFF(
                            'day',
                            TRY_CAST({left_alias}.{left_date_col} AS DATE),
                            TRY_CAST(src.{right_date_col} AS DATE)
                        )
                    ) ASC
                    LIMIT 1
                ) {alias} ON TRUE"""
            )

    query.append("LIMIT 50")
    return "\n".join(query)


# ---------- Create pivot views ----------
make_pivot_view(
    source_table="events",
    pivot_view_name="events_pivot",
    id_col="patient_id",
    category_col="event_type",
    value_col="event_date",
)

make_pivot_view(
    source_table="procedures",
    pivot_view_name="procedures_pivot",
    id_col="patient_id",
    category_col="procedure_type",
    value_col="procedure_date",
)

make_pivot_view(
    source_table="test_values",
    pivot_view_name="test_values_pivot",
    id_col="test_id",
    category_col="parameter_name",
    value_col="parameter_value",
)


# ---------- Tables exposed to UI ----------
TABLES = [
    "demographics",
    "events",
    "events_pivot",
    "procedures",
    "procedures_pivot",
    "exams",
    "tests",
    "test_values",
    "test_values_pivot",
]


# ---------- Session state ----------
if "joins" not in st.session_state:
    st.session_state.joins = []

if "query_result" not in st.session_state:
    st.session_state.query_result = None



# ---------- UI ----------
st_header('Query and Export Data')

base_table = st.selectbox("Base table", TABLES, key="base_table")

if st.button("Add Table", type="primary"):
    st.session_state.joins.append(
        {
            "table": TABLES[0],
            "left_alias": "t0",
            "join_key": "",
            "date_mode": "All rows",
            "left_date_col": "",
            "right_date_col": "",
        }
    )

for i, join in enumerate(st.session_state.joins):
    if i > 0:
        st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        join["table"] = st.selectbox(
            "Additional table",
            TABLES,
            index=TABLES.index(join["table"]),
            key=f"table_{i}",
        )

        join["date_mode"] = st.selectbox(
            "Date filter",
            ["All rows", "Latest", "Nearest to exam/reference date"],
            index=["All rows", "Latest", "Nearest to exam/reference date"].index(join["date_mode"])
            if join["date_mode"] in ["All rows", "Latest", "Nearest to exam/reference date"] else 0,
            key=f"date_mode_{i}",
        )

    valid_left_aliases = ["t0"] + [f"t{j+1}" for j in range(i)]
    alias_labels = [
        alias_to_display_name(base_table, st.session_state.joins, alias)
        for alias in valid_left_aliases
    ]

    current_label = alias_to_display_name(
        base_table,
        st.session_state.joins,
        join["left_alias"] if join["left_alias"] in valid_left_aliases else "t0",
    )

    with col3:
        selected_label = st.selectbox(
            "Join to",
            alias_labels,
            index=alias_labels.index(current_label) if current_label in alias_labels else 0,
            key=f"left_alias_label_{i}",
        )
        join["left_alias"] = valid_left_aliases[alias_labels.index(selected_label)]

    left_table = alias_to_table(base_table, st.session_state.joins, join["left_alias"])
    left_cols = get_columns(left_table)
    right_cols = get_columns(join["table"])
    common_cols = shared_columns(left_table, join["table"])

    if not common_cols:
        common_cols = [""]

    with col2:
        join["join_key"] = st.selectbox(
            "Link ID",
            common_cols,
            index=common_cols.index(join["join_key"]) if join["join_key"] in common_cols else 0,
            key=f"join_key_{i}",
        )

        if join["date_mode"] == "Nearest to exam/reference date":
            join["left_date_col"] = st.selectbox(
                "Scan/reference date",
                left_cols,
                index=left_cols.index(join["left_date_col"]) if join["left_date_col"] in left_cols else 0,
                key=f"left_date_col_{i}",
            )

    with col3:
        if join["date_mode"] in ["Latest", "Nearest to exam/reference date"]:
            join["right_date_col"] = st.selectbox(
                "Secondary date",
                right_cols,
                index=right_cols.index(join["right_date_col"]) if join["right_date_col"] in right_cols else 0,
                key=f"right_date_col_{i}",
            )
    btn1, spacer, btn2 = st.columns([1, 4, 1], vertical_alignment="center")

    with btn1:
        if st.button("Add Table", type="primary", key=f"add_after_{i}"):
            st.session_state.joins.insert(
                i + 1,
                {
                    "table": TABLES[0],
                    "left_alias": "t0",
                    "join_key": "",
                    "date_mode": "All rows",
                    "left_date_col": "",
                    "right_date_col": "",
                },
            )
            st.rerun()

    with btn2:
        # push button to right inside the column
        inner_spacer, right = st.columns([1, 1])
        with right:
            if st.button("Remove", key=f"remove_{i}", type="secondary"):
                st.session_state.joins.pop(i)
                st.rerun()

query = build_query(base_table, st.session_state.joins)

st.subheader("Generated SQL")
st.code(query, language="sql")

if st.button("Run query", type="primary"):
    try:
        st.session_state.query_result = con.execute(query).df()
    except Exception as e:
        st.error(str(e))

if st.session_state.query_result is not None:
    df = st.session_state.query_result
    st.dataframe(df)

    st.download_button(
        "Download CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="clasp_export.csv",
        mime="text/csv",
        type="primary",
    )