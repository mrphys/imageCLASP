import os
import glob
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

OUT_PATH = "tables"

files = sorted(
    glob.glob(f"{OUT_PATH}/*.csv"),
    key=lambda f: (0 if "exams" in f.lower() else 1 if "demographics" in f.lower() else 2, f),
)

TABLES = []
for file in files:
    table_name = os.path.splitext(os.path.basename(file))[0]
    if os.path.exists(file):
        con.execute(f"""
            CREATE OR REPLACE VIEW {table_name} AS
            SELECT * FROM read_csv_auto('{file}')
        """)
    TABLES.append(table_name)

DATE_MODES = ["All rows", "Latest", "Earliest", "Nearest to reference date"]

# ---------- Helpers ----------

def format_text(text: str) -> str:
    return text.replace("_", " ").title()


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
    return [] if df.empty else df["value"].tolist()


def get_default_date_col(table: str) -> str:
    return "dob" if table == "demographics" else f"{table}_date"


def get_pivot_category_values(table: str, pivot_col: str) -> list[str]:
    if not pivot_col or pivot_col not in get_columns(table):
        return []
    df = con.execute(f"""
        SELECT DISTINCT CAST({pivot_col} AS VARCHAR) AS value
        FROM {table}
        WHERE {pivot_col} IS NOT NULL
        ORDER BY value
    """).df()
    return [] if df.empty else df["value"].tolist()


def is_long_format(table: str) -> bool:
    cols = get_columns(table)
    return f"{table}_value" in cols and f"{table}_date" in cols


# ---------- Query builder ----------

def build_query(base_table: str, joins: list[dict]) -> str:
    base_cols = get_columns(base_table)
    base_has_patient_id = "patient_id" in base_cols

    ctes: list[str] = []
    join_clauses: list[str] = []
    select_cols: list[str] = [f"t0.{col}" for col in base_cols]
    alias_columns: dict[str, list[str]] = {}

    for i, join in enumerate(joins, start=1):
        alias = f"t{i}"
        right_table = join["table"]
        right_cols = get_columns(right_table)
        date_mode = join.get("date_mode", "All rows")

        if not base_has_patient_id or "patient_id" not in right_cols:
            continue

        output_cols = []

        # ── All rows ──────────────────────────────────────────────────────
        if date_mode == "All rows":
            join_clauses.append(
                f"LEFT JOIN {right_table} {alias}\n"
                f"    ON t0.patient_id = {alias}.patient_id"
            )
            output_cols = right_cols

        # ── Latest / Earliest ─────────────────────────────────────────────
        elif date_mode in ("Latest", "Earliest"):
            right_date_col = join.get("right_date_col", "")
            if not right_date_col:
                continue

            order_dir = "DESC" if date_mode == "Latest" else "ASC"

            if is_long_format(right_table):
                pivot_col = f"{right_table}_value"
                cat_values = get_pivot_category_values(right_table, pivot_col)
                if not cat_values:
                    continue

                pivot_cols_sql = ", ".join([f"'{v}'" for v in cat_values])
                cte_name = f"cte_{alias}"
                ctes.append(f"""
    {cte_name} AS (
        SELECT *
        FROM (
            SELECT patient_id, {pivot_col}, {right_date_col}
            FROM (
                SELECT *,
                    ROW_NUMBER() OVER (
                        PARTITION BY patient_id, {pivot_col}
                        ORDER BY TRY_CAST({right_date_col} AS DATE) {order_dir} NULLS LAST
                    ) AS _rn
                FROM {right_table}
                WHERE {right_date_col} IS NOT NULL
            ) _ranked
            WHERE _rn = 1
        )
        PIVOT (
            MAX({right_date_col})
            FOR {pivot_col} IN ({pivot_cols_sql})
        )
    )""")
                join_clauses.append(
                    f"LEFT JOIN {cte_name} {alias}\n"
                    f"    ON t0.patient_id = {alias}.patient_id"
                )
                output_cols = ["patient_id"] + cat_values

            else:
                join_clauses.append(f"""LEFT JOIN LATERAL (
    SELECT *
    FROM {right_table} src
    WHERE t0.patient_id = src.patient_id
        AND src.{right_date_col} IS NOT NULL
    ORDER BY TRY_CAST(src.{right_date_col} AS DATE) {order_dir} NULLS LAST
    LIMIT 1
) {alias} ON TRUE""")
                output_cols = right_cols

        # ── Nearest to reference date ─────────────────────────────────────
        elif date_mode == "Nearest to reference date":
            left_date_col = join.get("left_date_col", "")
            right_date_col = join.get("right_date_col", "")
            if not left_date_col or not right_date_col:
                continue

            if is_long_format(right_table):
                pivot_col = f"{right_table}_value"
                cat_values = get_pivot_category_values(right_table, pivot_col)
                if not cat_values:
                    continue

                select_parts = [
                    f"""(
            SELECT src.{right_date_col}
            FROM {right_table} src
            WHERE src.patient_id = t0.patient_id
              AND src.{pivot_col} = '{v}'
              AND src.{right_date_col} IS NOT NULL
              AND t0.{left_date_col} IS NOT NULL
            ORDER BY ABS(DATEDIFF(
                'day',
                TRY_CAST(t0.{left_date_col} AS DATE),
                TRY_CAST(src.{right_date_col} AS DATE)
            )) ASC
            LIMIT 1
        ) AS "{v}" """
                    for v in cat_values
                ]
                join_clauses.append(
                    f"LEFT JOIN LATERAL (\n    SELECT\n        {', '.join(select_parts)}\n) {alias} ON TRUE"
                )
                output_cols = cat_values

            else:
                join_clauses.append(f"""LEFT JOIN LATERAL (
    SELECT *
    FROM {right_table} src
    WHERE src.patient_id = t0.patient_id
        AND t0.{left_date_col} IS NOT NULL
        AND src.{right_date_col} IS NOT NULL
    ORDER BY ABS(DATEDIFF(
        'day',
        TRY_CAST(t0.{left_date_col} AS DATE),
        TRY_CAST(src.{right_date_col} AS DATE)
    )) ASC
    LIMIT 1
) {alias} ON TRUE""")
                output_cols = right_cols

        alias_columns[alias] = output_cols

    # ── Build SELECT ──────────────────────────────────────────────────────
    for alias, cols in alias_columns.items():
        for col in cols:
            if col == "patient_id":
                continue
            select_cols.append(f'{alias}."{col}"')

    select_sql = ",\n    ".join(select_cols)
    cte_block = "WITH " + ",\n".join(ctes) + "\n" if ctes else ""
    joins_sql = "\n".join(join_clauses)

    return f"""{cte_block}SELECT
    {select_sql}
FROM {base_table} t0
{joins_sql}"""


# ---------- Session state ----------

if "joins" not in st.session_state:
    st.session_state.joins = []

if "query_result" not in st.session_state:
    st.session_state.query_result = None

if "base_filter_value" not in st.session_state:
    st.session_state.base_filter_value = ""


# ---------- UI ----------

st_header("Query and Export Data")

base_table = st.selectbox("Base table", TABLES, key="base_table")
base_cols = get_columns(base_table)

if "patient_id" not in base_cols:
    st.warning(
        f"Base table '{base_table}' does not contain patient_id, "
        "so patient-level joins will not work."
    )


if st.button("Add Table", type="primary"):
    default_table = next((t for t in TABLES if t != base_table), TABLES[0])
    st.session_state.joins.append({
        "table": default_table,
        "date_mode": "All rows",
        "left_date_col": get_default_date_col(base_table),
        "right_date_col": get_default_date_col(default_table),
        "filter_values": [],
    })
    st.rerun()

# ---------- Join configuration ----------

for i, join in enumerate(st.session_state.joins):
    if i > 0:
        st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

    with st.container(border=True):
        top1, top2 = st.columns(2)

        with top1:
            available_tables = [t for t in TABLES if t != base_table] or TABLES
            if join["table"] not in available_tables:
                join["table"] = available_tables[0]
            join["table"] = st.selectbox(
                "Additional Table",
                available_tables,
                index=available_tables.index(join["table"]),
                key=f"table_{i}",
            )

        with top2:
            join["date_mode"] = st.selectbox(
                "Date Filter",
                DATE_MODES,
                index=DATE_MODES.index(join["date_mode"])
                if join["date_mode"] in DATE_MODES else 0,
                key=f"date_mode_{i}",
            )

        if join["date_mode"] == "Nearest to reference date":
            join["left_date_col"] = get_default_date_col(base_table)
            join["right_date_col"] = get_default_date_col(join["table"])
        elif join["date_mode"] in ("Latest", "Earliest"):
            join["right_date_col"] = get_default_date_col(join["table"])

        btn1, spacer, btn2 = st.columns([1, 4, 1])

        with btn1:
            if st.button("Add Table", type="primary", key=f"add_after_{i}"):
                default_table = next((t for t in TABLES if t != base_table), TABLES[0])
                st.session_state.joins.insert(i + 1, {
                    "table": default_table,
                    "date_mode": "All rows",
                    "left_date_col": get_default_date_col(base_table),
                    "right_date_col": get_default_date_col(default_table),
                    "filter_values": [],
                })
                st.rerun()

        with btn2:
            if st.button("Remove", key=f"remove_{i}", type="secondary"):
                st.session_state.joins.pop(i)
                st.rerun()

# ---------- SQL preview ----------

query = build_query(base_table, st.session_state.joins)

# ---------- Run query ----------

if st.button("Run Query", type="primary"):
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