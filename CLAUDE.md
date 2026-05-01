# imageCLASP

Semi-automated cardiac MRI analysis platform. Connects an Orthanc DICOM server to a segmentation pipeline and clinical data collection workflow.

**Run:** Orthanc must be running on `localhost:8042`, then `streamlit run app.py` (port 8502).

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit + streamlit-drawable-canvas |
| DICOM server | Orthanc REST API (localhost:8042, auth: orthanc/orthanc) |
| Study/series metadata | TinyDB → `image_clasp_db.json` |
| Clinical data | CSV files in `tables/` |
| Analytics queries | DuckDB (in-memory SQL over CSVs) |
| Segmentation model | PyTorch 2D U-Net (`models/example_2d_model.pth`) |
| Series classifier | PyTorch CNN (`models/Classifier-22.pth`) |

## Directory Structure

```
app.py                        # Entry point: session state init, navigation
app_pages/
  0_Dashboard.py              # DICOM upload, pipeline trigger, aggregate metrics
  1_Roundel.py                # 3-tab mask editing workflow
  2_Data_Entry.py             # Clinical form data collection
  3_Query.py                  # SQL query builder on CSV tables
utils/
  orthanc_utils.py            # Orthanc REST API wrappers
  pipeline.py                 # Orchestration: sync, MRI sort, segmentation
  sax_dl_utils.py             # U-Net inference pipeline
  roundel_utils.py            # Canvas mask editor (1463 lines, most complex)
  mri_sorter.py               # Rules-based + CNN series classification
  db_utils.py                 # Study/Series dataclasses + TinyDB serialization
  data_entry_utils.py         # Dynamic form generation + CSV I/O
  classifier.py               # CNN classifier architecture
  classifier_utils.py         # Classifier inference helpers
  UNet.py                     # Generic N-D U-Net architecture
  theme_utils.py              # CSS theming via st.markdown
  plot_utils.py               # GIF/curve visualization helpers
  reset_utils.py              # Session state reset between pages
  pick_folder.py              # Native OS folder picker dialog
models/
  example_2d_model.pth        # SAX segmentation U-Net weights
  Classifier-22.pth           # Series plane classifier weights
reference/
  data_entry_forms.json       # Form field definitions (drives Data Entry page)
  diagnoses_reference.csv
  medication_reference.csv
  procedures_reference.csv
  events_reference.csv
tables/                       # Generated at runtime
  demographics.csv
  exams.csv
  (other per-form CSVs)
results/temp/                 # Generated GIF overlays (transient)
```

## End-to-End Data Flow

```
DICOM files on disk
  → [Dashboard] upload to Orthanc
  → [pipeline.sync_orthanc_and_db()] mirror to TinyDB + demographics.csv
  → [pipeline.mri_sorting_pipeline()] classify series type/orientation
  → [pipeline.sax_segmentation_pipeline()] U-Net inference on SAX cine stacks
  → masks uploaded to Orthanc (stored as dl_orthanc_id on Series)
  → [Roundel] manual review/edit of masks
  → edited masks uploaded to Orthanc (roundel_orthanc_id) + metrics to exams.csv
  → [Data Entry] clinical metadata saved to per-form CSVs
  → [Query] DuckDB joins across CSV tables → export
```

## Pages

### 0_Dashboard.py
- Shows aggregate counts: patients, studies, segmented, roundelled
- Folder picker → multi-threaded upload to Orthanc → triggers all pipelines
- Calls `run_pipelines()` from `pipeline.py` on completion

### 1_Roundel.py — Mask Editor (3 tabs)
- **Tab 1: EDV/ESV Finder** — slider per ventricle (LV/RV) to identify end-diastolic and end-systolic frame indices; shows animated GIF preview
- **Tab 2: Mask Editor** — 3-column layout: controls | canvas | preview GIF. Freehand drawing/erasing per structure (LV pool, LV myo, RV pool, RV myo). Stroke post-processing: dilation, fill, erosion, smooth
- **Tab 3: Final Result** — metrics comparison (EDV, ESV, EF%, mass) raw vs edited; save button uploads edited masks to Orthanc and writes exams.csv

### 2_Data_Entry.py
- Patient selector dropdown
- Tabs generated dynamically from `reference/data_entry_forms.json`
- Demographics tab (name, sex, DOB) + multi-value tabs (diagnoses, meds, procedures, events)
- Saves to individual CSVs with deduplication via `data_entry_utils.append_csv()`

### 3_Query.py
- Select base table, add LEFT JOINs with date-filter options (all/latest/earliest/nearest)
- Builds DuckDB SQL with CTEs; executes and shows results dataframe
- Download CSV export

## Key Utilities

### orthanc_utils.py
Thin wrapper around Orthanc REST API using `requests.Session`. Key functions:
- `fetch_orthanc_studies()` — POST `/tools/find`
- `fetch_orthanc_dicoms_for_series(series_id)` — downloads raw DICOM bytes → pydicom objects
- `send_series_to_orthanc(dicoms, description)` — modifies DICOM UIDs, POSTs to `/instances`

### pipeline.py
- `sync_orthanc_and_db()` — adds new Orthanc studies to TinyDB, writes demographics.csv
- `mri_sorting_pipeline()` — runs MRI_Sorter, populates series_type/orientation/group
- `sax_segmentation_pipeline()` — filters SAX cine stacks without dl_orthanc_id, runs inference, uploads

### sax_dl_utils.py
- Loads 2D U-Net, auto-detects device (CUDA/MPS/CPU)
- `crop_pad_scan()` — resize to pixel spacing, then crop/pad to 256×256
- `run_inference_on_scan(dicoms)` → uint16 mask array (H, W, Z) with 5 classes

### roundel_utils.py
- Color map: LV pool = red (255,10,10), LV myo = cyan (0,255,255), RV pool = yellow (255,190,10), RV myo = green (0,200,10)
- Canvas stroke → RGB color → L2 distance match → channel index → binary dilation → mask update
- `edv_esv_view()`, `mask_editor_view()`, and final result view are the 3 tab renderers
- Generates GIF overlays with PIL alpha compositing

### mri_sorter.py (MRI_Sorter class)
- Uses ImageOrientationPatient cross product for plane detection (SAX/4CH/2CH/Axial)
- Detects 2D vs 3D from MRAcquisitionType; velocity encoding for flow series
- Falls back to CNN classifier for ambiguous cases

### db_utils.py
- `Study`: orthanc_study_id, study_uid, patient_*, study_date, series_dict
- `Series`: orthanc_series_id, series_uid, series_description, series_type, series_orientation, series_group, dl_orthanc_id, roundel_orthanc_id
- Both have `.to_dict()` / `.from_dict()` for TinyDB storage

## Segmentation Mask Format

5-channel uint16, shape (H, W, Z):
- Channel 0: background
- Channel 1: LV blood pool
- Channel 2: RV blood pool
- Channel 3: LV myocardium
- Channel 4: RV myocardium

Stored in Orthanc as a DICOM series (modified pixel data). `dl_orthanc_id` = raw model output; `roundel_orthanc_id` = after manual editing.

## Session State Keys (app.py)

```python
st.session_state.DB_PATH          # path to image_clasp_db.json
st.session_state.REFERENCE_PATH   # path to reference/
st.session_state.OUT_PATH         # path to results/temp/
st.session_state.DEMOGRAPHICS_PATH  # path to tables/demographics.csv
st.session_state.EXAMS_PATH       # path to tables/exams.csv
st.session_state.MASK_SCALER      # pixel spacing normalisation factor
```

## Notes

- Orthanc credentials are hardcoded (`orthanc:orthanc`) in `orthanc_utils.py`
- The `.streamlit/config.toml` disables telemetry and sets port 8502
- `clear_all.py` resets session state for debugging
- `image_clasp_db.json` is gitignored (untracked) — it's generated at runtime
- Test notebooks in repo root: `mri_sorter_test.ipynb`, `roundel_test.ipynb`, `check_db.ipynb`
