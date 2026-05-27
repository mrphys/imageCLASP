# ImageCLASP

[![CLASP](https://markdown-videos-api.jasoncameron.dev/youtube/F4cwBQrVG_A)](https://youtu.be/F4cwBQrVG_A)

**ImageCLASP** (Image Content Linking with AI and Supervised Processing) is a medical imaging platform for managing MRI data and extracting clinically relevant biomarkers using AI driven analysis pipelines.

The platform is designed to support federated analysis workflows, enabling hospitals and research institutions to perform statistical analysis without transferring sensitive imaging data outside institutional infrastructure.

## Features

* MRI data management and organization
* AI based biomarker extraction
* Local data storage for privacy preserving research 
* Integration with Orthanc PACS server
* Clinical and research-oriented imaging pipelines

## Distribution

ImageCLASP is distributed as a standalone Windows `.exe` installer.

The installer includes:

* ImageCLASP
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

### Orthanc Integration

ImageCLASP uses Orthanc as the underlying DICOM server for MRI data storage and communication.

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
