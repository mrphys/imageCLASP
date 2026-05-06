import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import zoom
import time

from .UNet import UNet
from .sliding_window_inference import *
import streamlit as st

# ---- User-configurable paths ----
MODEL_PATH = f"{st.session_state['clasp.MODELS_PATH']}/initial_unet_model.pth"
PLOTS_PATH = Path(st.session_state['clasp.MODELS_PATH']).parent / "plots"
TARGET_SHAPE = (256, 256)
NUM_CLASSES = 5
BATCH_SIZE = 16
TARGET_SPACING = 1.3671900033950806 # Median X/Y Pixdims from ACDC/MMS training set

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"[sax_dl_utils] Device: {device} | TTA: {device.type != 'cpu'}")

model = UNet(
            in_channels=1, out_channels=5,
            filters=[32, 64, 128, 256, 320, 320, 320],
            kernel_sizes=[(1,3,3),(1,3,3),(3,3,3),(3,3,3),(3,3,3),(3,3,3),(3,3,3)],
            strides=[(1,1,1),(1,2,2),(1,2,2),(2,2,2),(1,2,2),(1,2,2),(1,2,2)],
            conv_blocks_per_level=2, 
            rank=3,
            activation='leaky_relu',
            norm_type='InstanceNorm',
            final_activation='softmax',
            deep_supervision=True,
            num_ds_outputs=4,
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

def resample_volume(image, native_spacing, target_spacing=TARGET_SPACING):
    """Resample H and W to target spacing; leave D unchanged. Uses bilinear (order=1)."""
    native_spacing_y = native_spacing[0]
    native_spacing_x = native_spacing[1]
    zoom_factor_y = native_spacing_y / target_spacing
    zoom_factor_x = native_spacing_x / target_spacing
    return zoom(image, (zoom_factor_y, zoom_factor_x, 1.0), order=1)

def resample_mask(mask, native_spacing, target_spacing=TARGET_SPACING):
    """Resample integer label mask back to native spacing. Uses nearest-neighbour (order=0)."""
    native_spacing_y = native_spacing[0]
    native_spacing_x = native_spacing[1]
    zoom_factor_y = target_spacing / native_spacing_y
    zoom_factor_x = target_spacing / native_spacing_x
    return zoom(mask.astype(np.float32), (zoom_factor_y, zoom_factor_x, 1.0), order=0).astype(np.uint8)


def crop_pad_image_only(image, target_shape=(256, 256)):
    if (image.shape[0] < image.shape[1]) and (image.shape[0] < image.shape[2]):
        image = np.transpose(image, (1, 2, 0))
    orig_shape = image.shape
    H, W, D = orig_shape
    tH, tW = target_shape
    if H >= tH:
        start = (H - tH) // 2
        image = image[start:start + tH, :, :]
    else:
        pad = tH - H
        image = np.pad(image, ((pad // 2, pad - pad // 2), (0, 0), (0, 0)))
    if W >= tW:
        start = (W - tW) // 2
        image = image[:, start:start + tW, :]
    else:
        pad = tW - W
        image = np.pad(image, ((0, 0), (pad // 2, pad - pad // 2), (0, 0)))
    meta = {"orig_shape": orig_shape}
    return image, meta

def z_normalise_image(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    mean = np.mean(image)
    std = np.std(image)
    image -= mean
    image /= max(std, 1e-8)
    return image

def preprocess_scan_like_training(image_data, pixel_spacing, target_shape=(256, 256)):
    
    image_resampled = resample_volume(image_data, pixel_spacing)
    image_cropped, meta = crop_pad_image_only(image_resampled, target_shape=target_shape)
    image_norm = z_normalise_image(image_cropped.copy())

    return image_norm, meta

def reverse_crop_pad(processed, meta):
    orig_y, orig_x, orig_z = meta["orig_shape"]
    py, px, pz = processed.shape
    reconstructed = np.zeros((orig_y, orig_x, orig_z), dtype=processed.dtype)
    start_y_proc = max((py - orig_y) // 2, 0)
    start_x_proc = max((px - orig_x) // 2, 0)
    start_y_orig = max((orig_y - py) // 2, 0)
    start_x_orig = max((orig_x - px) // 2, 0)
    copy_y = min(py, orig_y)
    copy_x = min(px, orig_x)
    reconstructed[
        start_y_orig:start_y_orig + copy_y,
        start_x_orig:start_x_orig + copy_x,
        :
    ] = processed[
        start_y_proc:start_y_proc + copy_y,
        start_x_proc:start_x_proc + copy_x,
        :
    ]
    return reconstructed

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

def save_4d_gif(image_4d, save_path, fps=8, slice_locations=None):
    """
    image_4d: (S, T, H, W) — slices × timesteps × height × width
    slice_locations: optional list of floats (length S), displayed above each subplot.
    Saves a GIF cycling over timesteps with all slices tiled in a grid.
    """
    S, T, _, _ = image_4d.shape
    grid_rows = max(1, int(np.sqrt(S) + 0.5))
    grid_cols = (S + grid_rows - 1) // grid_rows

    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(grid_cols * 3, grid_rows * 3), squeeze=False)
    fig.patch.set_facecolor('black')
    axes_flat = axes.flatten()
    for ax in axes_flat:
        ax.axis('off')
        ax.patch.set_facecolor('black')

    if slice_locations is not None:
        for s in range(S):
            axes_flat[s].set_title(f"{slice_locations[s]:.1f} mm", color='white', fontsize=9, pad=2)

    vmin, vmax = image_4d.min(), image_4d.max()

    frames = []
    for t in range(T):
        artists = []
        ttl = axes_flat[0].text(
            0.5, 1.08, f't = {t + 1}/{T}',
            ha='center', va='bottom', transform=axes_flat[0].transAxes,
            fontsize=11, color='white',
        )
        artists.append(ttl)
        for s in range(S):
            im = axes_flat[s].imshow(image_4d[s, t], cmap='gray', vmin=vmin, vmax=vmax, animated=True)
            artists.append(im)
        frames.append(artists)

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    ani = animation.ArtistAnimation(fig, frames, interval=1000 // fps, blit=True)
    ani.save(save_path, fps=fps, writer='pillow')
    plt.close(fig)


def run_inference_on_scan(image_3d, pixel_spacing, timestep):

    preprocessed_image_3d, meta = preprocess_scan_like_training(image_3d, pixel_spacing, target_shape=TARGET_SHAPE)

    # # Optional - Save plot of the 3d image before inference
    # n_slices = preprocessed_image_3d.shape[2]
    # fig, axes = plt.subplots(1, n_slices, figsize=(3 * n_slices, 3))
    # if n_slices == 1:
    #     axes = [axes]
    # for i, ax in enumerate(axes):
    #     ax.imshow(preprocessed_image_3d[:, :, i], cmap='gray')
    #     ax.set_title(f'Slice {i}')
    #     ax.axis('off')
    # fig.suptitle(f'Preprocessed image (t={timestep})')
    # fig.savefig(f'/Users/Ruaraidh/Documents/UCL_CDT/PhD_Year1/cMRI_projects/imageCLASP/plots/debug_og_image_time{timestep}.png', bbox_inches='tight')
    # plt.close(fig)
    
    X = preprocessed_image_3d.transpose(2, 0, 1)[np.newaxis, np.newaxis, ...].astype(np.float32)
    
    start = time.time()
    prob_map, _ = monai_sliding_window_inference_3d(
            model, X,
            patch_size=(256, 256, 10),
            overlap=0.5,
            apply_softmax=False,
            out_channels=5,
            # tta=device.type != 'cpu', 
            tta=False,
            deep_supervision=True,
        )
    
    pred_mask = np.argmax(prob_map, axis=-1).astype(np.uint8)
    pred_mask = reverse_crop_pad(pred_mask, meta)
    pred_mask = resample_mask(pred_mask, pixel_spacing)
    end = time.time()
    
    pred_mask = np.uint16(pred_mask)
    pred_mask = np.transpose(np.array(pred_mask), (2,0,1))

    print(f"Time for inference on SAX frame: {end - start:.4f} seconds")
    return pred_mask