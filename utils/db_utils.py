from tinydb import TinyDB, Query
import pandas as pd

def load_db_rows(DB_PATH):
    # Open TinyDB database
    db = TinyDB(DB_PATH)
    rows = []

    # Iterate over all studies in the database
    for study in db:
        series = study.get("series", [])

        # Construct a summary row per study
        rows.append({
            "patient_id": study.get("patient_id"),
            "patient_sex": study.get("patient_sex"),
            "age": study.get("patient_age"),
            "n_series": len(series),  # Number of series in the study
            "sax_processed": any(s.get("sax_processed", False) for s in series),  # True if any series processed
            "roundel_processed": any(s.get("roundel_processed", False) for s in series),  # True if any series roundel processed
        })

    # Close database
    db.close()

    # Convert list of rows to a DataFrame
    df = pd.DataFrame(rows)
    return df