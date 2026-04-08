import numpy as np
import torch
import albumentations as A
import torch.nn as nn
import torch
from .classifier import CNNClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSIFIER_PATH = 'models/Classifier-22.pth'
model = CNNClassifier(num_classes=3)
state_dict = torch.load(CLASSIFIER_PATH)
model.load_state_dict(state_dict)
model.eval()


label_dict = {0:'LV_LAX', 
              1:'4CH',
              2:'SAX'}

resize_transform = A.Compose([
    A.Resize(height=128, width=128, p=1.0)
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

def preprocess(image):
    image = square_crop(image)
    image = percentile_clip(image)
    image = normalize(image).astype(np.float32)

    image = resize_transform(image=image)["image"]
    image = image.astype(np.float32)

    image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
    return image

