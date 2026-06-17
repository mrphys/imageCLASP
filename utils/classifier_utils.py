import numpy as np
import torch
import albumentations as A
import torch.nn as nn
import torch
from .classifier import *
import streamlit as st

# Load pre-trained view classification model
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
CLASSIFIER_PATH = f"{st.session_state['clasp.MODELS_PATH']}/view_classification-35.pth"
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

# Function to pad a 2D image to make it square
def pad_to_square(image):
    """
    Pad a 2D image to make it square by adding zeros to the shorter dimension.

    Args:
        image (numpy array): 2D image array of shape (H, W).

    Returns:
        numpy array: Padded square image array of shape (max(H, W), max(H, W)).
    """
    height, width = image.shape
    if height == width:
        return image  # Already square

    if height > width:
        pad_width = (height - width) // 2
        padded_image = np.pad(image, ((0, 0), (pad_width, pad_width)), mode='constant', constant_values=0)
    else:
        pad_height = (width - height) // 2
        padded_image = np.pad(image, ((pad_height, pad_height), (0, 0)), mode='constant', constant_values=0)

    # Check that the padded image is now square
    new_height, new_width = padded_image.shape
    if new_height != new_width:
        # add the difference to the end of the shorter dimension if the original difference was odd
        if new_height < new_width:
            padded_image = np.pad(padded_image, ((0, new_width - new_height), (0, 0)), mode='constant', constant_values=0)
        else:
            padded_image = np.pad(padded_image, ((0, 0), (0, new_height - new_width)), mode='constant', constant_values=0)

    return padded_image



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
    # Pad image to make it square 
    image = pad_to_square(image)
    
    image = resize_transform(image=image)["image"]
    image = image.astype(np.float32)

    image = percentile_clip(image)
    image = z_normalise_image_classifier(image).astype(np.float32)

    image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
    
    return image