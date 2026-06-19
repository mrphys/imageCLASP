# CLASP

[![CLASP](https://img.youtube.com/vi/F4cwBQrVG_A/maxresdefault.jpg)](https://youtu.be/F4cwBQrVG_A)

**CLASP** (Content Linking with AI and Supervised Processing) is a medical imaging platform for managing MRI data and extracting clinically relevant biomarkers using AI driven analysis pipelines.

The platform is designed to support federated analysis workflows, enabling hospitals and research institutions to perform statistical analysis without transferring sensitive imaging data outside institutional infrastructure.

## Features

* MRI data management and organization
* AI based biomarker extraction
* Local data storage for privacy preserving research 
* Integration with Orthanc PACS server
* Clinical and research-oriented imaging pipelines 
* Statistics package added

## Distribution

CLASP is distributed as a standalone Windows `.exe` installer.

The installer includes:

* CLASP
* Orthanc installation and configuration

## System Requirements

### Windows Release

* Windows 10 or later
* Administrative privileges for installation
* Recommended: NVIDIA GPU for accelerated AI processing

### Development Environment

* Linux recommended
* Python 3.13
* Orthanc

## Installation for Development

Create and activate a Python virtual environment:

```bash
python3.13 -m venv clasp
source clasp/bin/activate
```

Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Install and start Orthanc:

```bash
sudo apt install orthanc
sudo systemctl start orthanc
```

Download the view classification and segmentation models:
If not using the exe distribution, the ML models for automatically identifying short-axis (SAX) Cine scans, and performing biventricular segmentation need to be downloaded and place in a models/ folder in the root directory. 

The models can be downloaded from this link: https://drive.google.com/file/d/1wawDKOSKjV_KwBvrvZA2xtA3zKGyfbV5/view?usp=sharing

The zip file should be unzipped, and the following models placed into a models/ folder: 
- view_classification-35.pth
- SAX-Seg-186.pth


### Orthanc Integration

ImageCLASP uses Orthanc as the underlying DICOM server for MRI data storage and communication. Orthanc is an open-source, lightweight DICOM server designed to facilitate medical imaging workflows and interoperability. For further details, see [The Orthanc Ecosystem for Medical Imaging](https://link.springer.com/article/10.1007/s10278-018-0082-y).

Default Orthanc service management:

```bash
sudo systemctl status orthanc
sudo systemctl restart orthanc
```


## Running ImageCLASP

After activating the environment:

```bash
streamlit run app.py
```
