import requests
import os
import glob
import shutil
try:
    if os.path.exists('image_clasp_db.json'):
        os.remove('image_clasp_db.json')
    files = glob.glob(f'tables/*')
    for file in files:
        os.remove(file)
    shutil.rmtree('utils/roundel')
    
except Exception as e:
    print(e)

ORTHANC = "http://localhost:8042"
AUTH = ("orthanc", "orthanc")

session = requests.Session()
session.auth = AUTH
session.trust_env = False

studies = session.get(f"{ORTHANC}/studies").json()

for study_id in studies:
    session.delete(f"{ORTHANC}/studies/{study_id}")