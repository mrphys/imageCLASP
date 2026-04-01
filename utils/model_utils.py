import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import zoom
import time

from .UNet import UNet

# ---- User-configurable paths ----
MODEL_PATH = "models/example_2d_model.pth"
TARGET_SHAPE = (256, 256)
NUM_CLASSES = 5
BATCH_SIZE = 16

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

model = UNet(
    in_channels=1,
    out_channels=NUM_CLASSES,
    filters=[16, 32, 64, 128, 256],
    kernel_size=3,
    conv_blocks_per_level=2,
    rank=2,
    activation="leaky_relu",
    norm_type="BatchNorm",
    dropout_rate=None,
    final_activation="softmax",
    pool_size=2,
    upsample_size=2,
)


state_dict = torch.load(MODEL_PATH, map_location="cpu")
model.load_state_dict(state_dict)

model.to(device)
model.eval()



 # Preprocessing helpers copied to match the training preprocessing logic

def _crop_or_pad_axis(arr: np.ndarray, axis: int, target_size: int) -> np.ndarray:
    current_size = arr.shape[axis]

    if current_size > target_size:
        offset = (current_size - target_size) // 2
        slices = [slice(None)] * arr.ndim
        slices[axis] = slice(offset, offset + target_size)
        arr = arr[tuple(slices)]
    elif current_size < target_size:
        pad_before = (target_size - current_size) // 2
        pad_after = (target_size - current_size) - pad_before
        pad_width = [(0, 0)] * arr.ndim
        pad_width[axis] = (pad_before, pad_after)
        arr = np.pad(arr, pad_width, mode="constant", constant_values=0)

    return arr


def resize_with_crop_or_pad(image: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
    image = _crop_or_pad_axis(image, axis=0, target_size=target_height)
    image = _crop_or_pad_axis(image, axis=1, target_size=target_width)
    return image


def crop_pad_scan(image: np.ndarray, target_shape: tuple = (256, 256)) -> np.ndarray:
    num_layers = image.shape[2]
    processed_image_list = []

    for layer_idx in range(num_layers):
        image_layer = image[:, :, layer_idx].astype(np.float32)
        processed_image_layer = resize_with_crop_or_pad(
            image_layer,
            target_height=target_shape[0],
            target_width=target_shape[1],
        )
        processed_image_list.append(processed_image_layer)

    processed_image = np.array(processed_image_list)
    processed_image = np.transpose(processed_image, (1, 2, 0))
    return processed_image


def preprocess_scan_like_training(image_data, img_zooms, target_shape=(256, 256)):

    zoom_all = (np.float32(img_zooms[0]), np.float32(img_zooms[1]), 1)
    image = zoom(image_data, zoom=zoom_all, order=1)
    image = crop_pad_scan(image, target_shape=target_shape)

    return image

def z_normalise_image(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    mean = np.mean(image)
    std = np.std(image)
    image -= mean
    image /= max(std, 1e-8)
    return image

def crop_pad_hw(arr, target_h, target_w):
    """
    arr: (H, W, N)
    returns: (target_h, target_w, N)
    """
    H, W, N = arr.shape

    # --- Crop ---
    start_h = max((H - target_h) // 2, 0)
    start_w = max((W - target_w) // 2, 0)

    end_h = start_h + min(H, target_h)
    end_w = start_w + min(W, target_w)

    cropped = arr[start_h:end_h, start_w:end_w, :]

    # --- Pad ---
    pad_h = target_h - cropped.shape[0]
    pad_w = target_w - cropped.shape[1]

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top

    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    padded = np.pad(
        cropped,
        ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
        mode="constant",  # or "edge"/"reflect"
        constant_values=0
    )

    return padded

class InferenceSliceDataset(Dataset):
    def __init__(self, preprocessed_image_3d: np.ndarray):
        # expected shape: (H, W, Z)
        self.image_3d = preprocessed_image_3d
        self.num_slices = preprocessed_image_3d.shape[2]

    def __len__(self):
        return self.num_slices

    def __getitem__(self, idx):
        image_slice = self.image_3d[:, :, idx]
        image_slice = z_normalise_image(image_slice)
        image_tensor = torch.from_numpy(image_slice[..., np.newaxis]).permute(-1, 0, 1)
        return image_tensor.to(torch.float32), idx

def run_inference_on_scan(old_dcms):
    ims = [ds.pixel_array for ds in old_dcms]
    image_size = ims[0].shape
    ds0 = old_dcms[0]
    pixel_spacing = ds0.PixelSpacing

    images = np.transpose(np.array(ims), (1,2,0))

    preprocessed_image_3d = preprocess_scan_like_training(images, pixel_spacing, target_shape=TARGET_SHAPE)

    test_dataset = InferenceSliceDataset(preprocessed_image_3d)
    num_workers = min(4, os.cpu_count())
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory = (device.type == "cuda")
    )

    num_slices = preprocessed_image_3d.shape[2]
    start = time.time()

    pred_mask_3d = np.zeros((num_slices, TARGET_SHAPE[0], TARGET_SHAPE[1]), dtype=np.uint8)

    with torch.no_grad():
        for batch_images, batch_slice_indices in test_loader:
            batch_images = batch_images.to(device, non_blocking=True)
            logits = model(batch_images)
            pred_class = torch.argmax(logits, dim=1).cpu().numpy().astype(np.uint8)  # (B, H, W)

            for i, slice_idx in enumerate(batch_slice_indices.numpy().tolist()):
                pred_mask_3d[slice_idx] = pred_class[i]


    # Convert to (H, W, Z) to match image convention in preprocessing
    mask = np.transpose(pred_mask_3d, (1, 2, 0))
    end = time.time()

    mask = zoom(mask, (1/pixel_spacing[0], 1/pixel_spacing[1], 1), order=0)
    mask = crop_pad_hw(mask, image_size[0], image_size[1])
    mask = np.uint16(mask) * 500
    mask = np.transpose(np.array(mask), (2,0,1))


    print(f"Time: {end - start:.4f} seconds")
    return mask



