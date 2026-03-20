python3.13 -m venv clasp

source activate clasp

pip install --upgrade pip
pip install -r requirements.txt

Install orthanc
sudo systemctl start orthanc

http://localhost:8042/app/explorer.html
