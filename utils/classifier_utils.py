import numpy as np
import torch
import albumentations as A
import torch.nn as nn
import torch
from .classifier import *
import streamlit as st

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
CLASSIFIER_PATH = f"{st.session_state['clasp.MODELS_PATH']}/view_classification-33.pth"
model = ResNet18Classifier2d(num_classes=3) # cine_sax=0, cine_lax_4ch=1, all other types=2
state_dict = torch.load(CLASSIFIER_PATH, map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()


label_dict = {0:'SAX', 
              1:'4CH',
              2:'Other'}

resize_transform = A.Compose([
    A.Resize(height=256, width=256, p=1.0)
])

def square_crop(image):
    h, w = image.shape
    side = min(h, w)
    top = (h - side) // 2
    left = (w - side) // 2
    return image[top:top+side, left:left+side]

def percentile_clip(image, lower=1, upper=99):
    lo = np.percentile(image, lower)
    hi = np.percentile(image, upper)
    image = np.clip(image, lo, hi)
    return image

def normalize(image):
    return (image - image.min()) / (image.max() - image.min() + 1e-8)

# Function to normalise images with z-score normalisation
def z_normalise_image_classifier(image):
    """
    Normalise the image data using z-score normalisation.
    
    Args:
        image (numpy array): Image data to be normalised.
    
    Returns:
        numpy array: Normalised image data.
    """
    mean = np.mean(image)
    std = np.std(image)
    image -= mean
    image /= (max(std, 1e-8))
    
    return image


def preprocess(image):
    image = square_crop(image)
    image = percentile_clip(image)
    image = normalize(image).astype(np.float32)

    image = resize_transform(image=image)["image"]
    image = image.astype(np.float32)

    image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
    return image


def preprocess_resnet_classifier(image):
    '''
    Preprocess an image for feeding to view classification model. 
    '''
    image = resize_transform(image=image)["image"]
    image = image.astype(np.float32)

    image = percentile_clip(image)
    image = z_normalise_image_classifier(image).astype(np.float32)

    image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
    
    return image
