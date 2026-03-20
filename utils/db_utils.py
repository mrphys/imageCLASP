from tinydb import TinyDB, Query
import pandas as pd

def load_db_rows(DB_PATH):
    db = TinyDB(DB_PATH)
    rows = []
    for study in db:
        series = study.get("series", [])
        rows.append({
            "patient_id": study.get("patient_id"),
            "patient_sex": study.get("patient_sex"),
            "age": study.get("patient_age"),
            "n_series": len(series),
            "DL_processed": any(s.get("DL_processed", False) for s in series),
            "roundel_processed": any(s.get("roundel_processed", False) for s in series),
        })
    db.close()
    df = pd.DataFrame(rows)
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    return df