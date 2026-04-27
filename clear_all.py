import requests
import os
import glob
import shutil
from platformdirs import user_data_dir
from pathlib import Path


APP_NAME = "ImageCLASP"
APP_AUTHOR = "ImageCLASP"
DATA_DIR = Path(user_data_dir(APP_NAME, APP_AUTHOR))
TABLES_DIR = DATA_DIR / "tables"
ROUNDEL_DIR = DATA_DIR / "roundel"

files = [DATA_DIR/ "image_clasp_db.json"]
folders = [TABLES_DIR, ROUNDEL_DIR]

for file in files:
    if os.path.exists(file):
        os.remove(file)
for folder in folders:
    try:
        shutil.rmtree(folder)
        print(f'cleared {folder}')
    except:
        
        pass





ORTHANC = "http://localhost:8042"
AUTH = ("orthanc", "orthanc")

session = requests.Session()
session.auth = AUTH
session.trust_env = False

studies = session.get(f"{ORTHANC}/studies").json()

for study_id in studies:
    session.delete(f"{ORTHANC}/studies/{study_id}")