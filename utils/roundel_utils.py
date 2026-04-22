import numpy as np
import pandas as pd
from utils.pipeline import *
import os
from pathlib import Path
import nibabel as nib
import numpy as np
import imageio.v2 as imageio
from PIL import Image, ImageSequence, ImageDraw, ImageFont
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from scipy.ndimage import (
    binary_fill_holes,
    binary_dilation,
    binary_erosion,
    gaussian_filter
) 
from skimage.measure import find_contours
import cv2
import json
from datetime import datetime
import copy
from utils.reset_utils import *
import shutil

root_path = Path(__file__).resolve().parent
data_path = str(root_path / "roundel/data")
results_path = str(root_path / "roundel/results")
models_path = str(root_path / "roundel/models")

blank_gif_path = f'{results_path}/temp/blank'
full_edited_gif_path = f'{results_path}/temp/edited'
preprocessed_gif_path = f'{results_path}/temp/preprocessed'
edv_esv_gif_path = f'{results_path}/temp/edv_esv'
edited_gif_path = f'{results_path}/temp/edited_edv_esv'
raw_curve_path = f'{results_path}/temp/raw_metrics.png'
edited_curve_path = f'{results_path}/temp/edited_metrics.png'
dicom_mask_path = f'{results_path}/masks/dicoms/'
nifti_mask_path = f'{results_path}/masks/nifti/'

cache_dir = str(root_path / "roundel/cache")
final_dir = f'{results_path}/results.zip'

os.makedirs(f'{data_path}', exist_ok=True)
os.makedirs(f'{results_path}/temp', exist_ok=True)
os.makedirs(f'{results_path}/gifs', exist_ok=True)
os.makedirs(f'{results_path}/edited_sax_df', exist_ok=True)
os.makedirs(cache_dir, exist_ok=True)

GIF_W = 150
DISPLAY_W = 400

labels = {
  "background": 0,
  "LV": 1,
  "RV": 2,
  "LV_myo": 3,
  "RV_myo": 4
}

background_idx = labels['background']
lv_idx = labels['LV']
rv_idx = labels['RV']
lv_myo_idx = labels['LV_myo']
rv_myo_idx = labels['RV_myo']


BACKGROUND_COLOR = (10, 10, 10, 0) # THIS HAS TO BE NON-ZERO
RV_MYO_COLOR = (0, 200, 10, 50)    # Green
RV_COLOR = (255, 190, 10, 50)      # Yellow
LV_MYO_COLOR =  (0, 255, 255, 50)  # Blue
LV_COLOR = (255, 10, 10, 50)       # Red



OVERLAY_COLORS = {
    background_idx: BACKGROUND_COLOR,
    rv_idx: RV_COLOR,
    rv_myo_idx: RV_MYO_COLOR,
    lv_myo_idx: LV_MYO_COLOR,
    lv_idx: LV_COLOR,
}


BRUSH_LABELS = {
    rv_myo_idx: 'RV Myocardium 🟢',
    rv_idx: 'RV Blood Pool 🟡',
    lv_myo_idx: 'LV Myocardium 🔵',
    lv_idx: 'LV Blood Pool 🔴',
}

VENTRICLE_CHANNEL = {'lv':[lv_idx, lv_myo_idx],
                     'rv':[rv_idx, rv_myo_idx]}


BRUSH_LABELS = dict(
    sorted(
        BRUSH_LABELS.items(),
        key=lambda item: 0 if 'myocardium' in item[1].lower() else 1
    )
)


def restart_app():
    prev = st.session_state["roundel.prev_study_id"]
    curr = st.session_state["roundel.current_study_id"]

    if prev != curr:
        for key in list(st.session_state.keys()):
            if key.startswith("roundel.") and key not in {
                "roundel.prev_study_id",
                "roundel.current_study_id",
            }:
                st.session_state.pop(key)

    st.session_state["roundel.prev_study_id"] = curr


def get_4d_array(instances):
    rows = [
        {
            "OrthancSeriesID":inst.get('ParentSeries'),
            "OrthancInstanceID": inst["ID"],
            "SliceLocation": (ds := fetch_orthanc_dicom(inst["ID"])).SliceLocation,
            "InstanceNumber": ds.InstanceNumber,
            "PixelArray": ds.pixel_array,
        }
        for inst in instances
    ]
    
    sax_df = pd.DataFrame(rows).sort_values(['SliceLocation','InstanceNumber']).reset_index(drop = True)
    array_4d = []
    for _, slice_df in sax_df.groupby("SliceLocation"):
        time_array = []
        for _, time_df in slice_df.groupby("InstanceNumber"):
            pixel_array = time_df["PixelArray"].iloc[0]
            time_array.append(pixel_array)

        slice_array = np.stack(time_array, axis=-1)
        array_4d.append(slice_array)

    array_4d = np.stack(array_4d, axis=-2)
    return sax_df, array_4d

def flatten_4d_array(array_4d):
    array_flat = [
        array_4d[:, :, sl, t]
        for sl in range(array_4d.shape[2])
        for t in range(array_4d.shape[3])
    ]
    return array_flat



def load_font(size):
    # Try Linux font
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except:
        pass
    # Try Windows font
    try:
        return ImageFont.truetype("C:/Windows/Fonts/arial.ttf", size)
    except:
        pass
    # Fallback (non scalable)
    return ImageFont.load_default()


def save_cached_mask(mask, save_path):
    np.save(save_path, mask)

def load_cached_mask(save_path):
    return np.load(save_path)

def save_config(config, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(config, f, indent=2)

def load_config(path) :
    path = Path(path)
    with path.open("r") as f:
        return json.load(f)

def save_mask(mask, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    nib_mask = nib.Nifti1Image(mask, affine=np.eye(4), dtype='uint8')
    nib.save(nib_mask, save_path)

def save_image(image, save_path):
    nib_image = nib.Nifti1Image(image, affine=np.eye(4), dtype='float32')
    nib.save(nib_image, save_path)

def normalize(image):
    image = (image - np.min(image))/(np.max(image) - np.min(image))
    return image

def merge_masks(lv_mask, rv_mask):
    combined_mask = lv_mask + rv_mask
    combined_mask = np.argmax(combined_mask, -1)
    combined_mask = np.eye(st.session_state['roundel.N'], dtype=np.uint8)[combined_mask]
    return combined_mask


def cv_zoom(images, zoom, interpolation=cv2.INTER_CUBIC):
    """
    Resize height and width of a 4D or 5D array using OpenCV. Only H and W are scaled.

    Args:
        images (numpy.ndarray): Array of shape (H, W, D, T) or (H, W, D, T, C)
        zoom_factors (list or tuple): Zoom factors for (H, W, D, T, C). Only H and W > 1
        interpolation (int): OpenCV interpolation method (default: cv2.INTER_CUBIC)

    Returns:
        numpy.ndarray: Resized array with height and width scaled, other dimensions unchanged
    """
    h_zoom, w_zoom = zoom[0], zoom[1]

    if images.ndim == 4:
        h, w, d, t = images.shape
        resized = np.zeros((int(h*h_zoom), int(w*w_zoom), d, t), dtype=images.dtype)
        for z in range(d):
            for tau in range(t):
                resized[..., z, tau] = cv2.resize(images[..., z, tau], (int(w*w_zoom), int(h*h_zoom)), interpolation=interpolation)
    elif images.ndim == 5:
        h, w, d, t, c = images.shape
        resized = np.zeros((int(h*h_zoom), int(w*w_zoom), d, t, c), dtype=images.dtype)
        for z in range(d):
            for tau in range(t):
                for ch in range(c):
                    resized[..., z, tau, ch] = cv2.resize(images[..., z, tau, ch], (int(w*w_zoom), int(h*h_zoom)), interpolation=interpolation)
    else:
        raise ValueError("Input must be 4D or 5D array.")

    return resized


def load_nii(nii_path):
    file = nib.load(nii_path)
    data = file.get_fdata(caching='unchanged')
    return data

def cv_zoom_mask(
    mask,
    zoom,
    sigma=2.0,
    interpolation=cv2.INTER_CUBIC,
):
    """
    mask: H,W,D,T,C
    returns: H,W,D,T,C one hot
    """
    zoomed = cv_zoom(mask.astype(np.float32), zoom, interpolation=interpolation)

    H, W, D, T, _ = zoomed.shape
    labels = np.zeros((H, W, D, T), dtype=np.uint8)

    ventricles = [
        (lv_idx, lv_myo_idx),
        (rv_idx, rv_myo_idx),
    ]

    for endo_idx, myo_idx in ventricles:
        endo = (zoomed[..., endo_idx] > 0.5).astype(np.float32)
        myo  = (zoomed[..., myo_idx] > 0.5).astype(np.float32)

        epi = np.zeros_like(myo, dtype=bool)
        for d in range(D):
            for t in range(T):
                epi[..., d, t] = binary_fill_holes(
                    myo[..., d, t].astype(np.uint8)
                )

        epi = gaussian_filter(
            epi.astype(np.float32), sigma=(sigma, sigma, 0, 0)
        ) > 0.5

        endo = gaussian_filter(
            endo.astype(np.float32), sigma=(sigma, sigma, 0, 0)
        ) > 0.5

        labels[epi] = myo_idx
        labels[endo] = endo_idx

    return np.eye(st.session_state['roundel.N'], dtype=np.uint8)[labels]

def format_delta(value, raw_value, suffix="", round_digits=None):
    if round_digits is not None:
        value = round(value, round_digits)
        raw_value = round(raw_value, round_digits)
    return None if value == raw_value else f"{value - raw_value:.1f}{suffix}"


def find_crop_box(mask, crop_factor):
    '''
    Calculated a bounding box that contains the masks inside.

    Parameters:
    mask: np.array
        A binary mask array, which should be the flattened 3D multislice mask, where the pixels in the z-dimension are summed
    crop_factor: float
        A scaling factor for the bounding box
    Returns:
    list
        A list containing the coordinates of the bounding box [x_min, y_min, x_max, y_max]. These co-ordinates can be used to crop each slice of the input multislice image.
    '''
    # Check shape of the input is 2D
    if len(mask.shape) != 2:
        raise ValueError("Input mask must be a 2D array")

    if np.max(mask) == 0:
        x_min, x_max = 0, mask.shape[0]
        y_min, y_max = 0, mask.shape[1]
        return [x_min, y_min, x_max, y_max]

    else:
        y = np.sum(mask, axis=1) # sum the masks across columns of array, returns a 1D array of row totals
        x = np.sum(mask, axis=0) # sum the masks across rows of array, returns a 1D array of column totals

        top = np.min(np.nonzero(y)) - 1 # Returns the indices of the elements in 1d row totals array that are non-zero, then finds the minimum value and subtracts 1 (i.e. top extent of mask)
        bottom = np.max(np.nonzero(y)) + 1 # Returns the indices of the elements in 1d row totals array that are non-zero, then finds the maximum value and adds 1 (i.e. bottom extent of mask)

        left = np.min(np.nonzero(x)) - 1 # Returns the indices of the elements in 1d column totals array that are non-zero, then finds the minimum value and subtracts 1 (i.e. left extent of mask)
        right = np.max(np.nonzero(x)) + 1 # Returns the indices of the elements in 1d column totals array that are non-zero, then finds the maximum value and adds 1 (i.e. right extent of mask)
        if abs(right - left) > abs(top - bottom):
            largest_side = abs(right - left) # Find the largest side of the bounding box
        else:
            largest_side = abs(top - bottom)

        
        x_mid = round((left + right) / 2) # Find the mid-point of the x-length of mask
        y_mid = round((top + bottom) / 2) # Find the mid-point of the y-length of mask
        half_largest_side = round(largest_side * crop_factor / 2) # Find half the largest side of the bounding box (crop factor scales the largest side to ensure whole heart and some surrounding is captured)
        x_max, x_min = round(x_mid + half_largest_side), round(x_mid - half_largest_side) # Find the maximum and minimum x-values of the bounding box
        y_max, y_min = round(y_mid + half_largest_side), round(y_mid - half_largest_side) # Find the maximum and minimum y-values of the bounding box
        if x_min < 0:
            x_max -= x_min # if x_min less than zero, expand the x_max value by the absolute value of x_min, to ensure bounding box is same size
            x_min = 0

        if y_min < 0:
            y_max -= y_min # if y_min less than zero, expand the y_max value by the absolute value of y_min, to ensure bounding box is same size
            y_min = 0

        if largest_side < 20:
            x_min, x_max = 0, mask.shape[0]
            y_min, y_max = 0, mask.shape[1]
        return [x_min, y_min, x_max, y_max]


def make_video(image, mask, save_file, ventricle = 'all', mask_frames = 'all',scale=1):
    N = st.session_state['roundel.N']
    if ventricle == 'rv':
        channels = [rv_idx, rv_myo_idx]
    elif ventricle == 'lv':
        channels = [lv_idx, lv_myo_idx]
    else:
        channels = [n for n in np.arange(N) if n != background_idx]

    if mask.shape[-1]!=N:
        mask = np.eye(N, dtype=np.uint8)[mask]

    position = image.shape[2]
    timesteps = image.shape[3]

    grid_rows = int(np.sqrt(position) + 0.5)
    grid_cols = (position + grid_rows - 1) // grid_rows

    H, W = image.shape[:2]
    GIF_H = H*GIF_W/W
    H_scaled, W_scaled = round(GIF_H * scale), round(GIF_W * scale)

    try:
        font = load_font(int(20 * scale))
    except:
        font = ImageFont.load_default()

    frames = []
    if mask_frames == 'all':
        mask_frames = np.arange(timesteps)

    for t in mask_frames:
        canvas = Image.new(
            "RGBA",
            (grid_cols * W_scaled, grid_rows * H_scaled),
            color=(0, 0, 0, 255)
        )

        draw_canvas = ImageDraw.Draw(canvas)

        for idx in range(position):
            row, col = divmod(idx, grid_cols)
        
            img_slice = image[:, :, idx, t]

            p1, p99 = np.percentile(img_slice, [0.5, 99.5]) # improve contrast
            img_slice = np.clip(img_slice, p1, p99)

            # Convert to RGB
            img_slice_norm = ((img_slice - img_slice.min()) / (img_slice.max() - img_slice.min() + 1e-9) * 255).astype(np.uint8)
            img_rgb = np.stack([img_slice_norm] * 3, axis=-1)
            img_pil = Image.fromarray(img_rgb, mode="RGB").convert("RGBA")

            # Resize slice
            img_pil = img_pil.resize((W_scaled, H_scaled), resample=Image.NEAREST)

            overlay = np.zeros((H, W, 4), dtype=np.uint8)
            for ch in channels:
                ch_mask = mask[:,:,idx,t,ch]
                if np.any(ch_mask):
                    color = np.array(OVERLAY_COLORS[ch], dtype=np.uint8)
                    overlay[ch_mask > 0] = color
            overlay_pil = Image.fromarray(overlay, mode="RGBA").resize((W_scaled, H_scaled), resample=Image.NEAREST)
            img_pil.alpha_composite(overlay_pil)

            draw_tile = ImageDraw.Draw(img_pil)
            draw_tile.rectangle([0,0,int(28*scale), int(22*scale)], fill=(211,211,211,255))
            draw_tile.text((3*scale,2*scale), f"{idx}", fill=(0,0,0,255), font=font)

            canvas.paste(img_pil, (col * W_scaled, row * H_scaled), img_pil)

        draw_canvas.rectangle(
            [canvas.width - int(60*scale), canvas.height - int(20*scale),
             canvas.width, canvas.height],
            fill=(211,211,211,255)
        )
        draw_canvas.text(
            (canvas.width - int(55*scale), canvas.height - int(20*scale)),
            f"{t:02}/{timesteps - 1:02}",
            fill=(0,0,0,255),
            font=font
        )

        frames.append(canvas.convert("RGB"))

    if len(mask_frames) < 5:
        fps = len(mask_frames)/2
    else:
        fps = np.clip(len(mask_frames) / 2, 8, 15)

    save_file = save_file.replace('.gif','')
    imageio.mimsave(f'{save_file}.gif', frames, fps=fps, loop=0)


def calculate_sax_metrics(mask, blood_pool_idx, myo_idx, dia_idx, sys_idx):
    voxel_size = st.session_state['roundel.pixelspacing'] ** 2 * st.session_state['roundel.thickness'] / 1000
    volume = np.sum(mask[..., blood_pool_idx], axis=(0,1,2)) * voxel_size
    masses = np.sum(mask[..., myo_idx], axis=(0,1,2)) * voxel_size * 1.05
    mass = masses[dia_idx]
    edv = volume[dia_idx]
    esv = volume[sys_idx]
    sv = edv - esv
    ef = (sv) * 100 / edv
    
    edv = round(edv, 2)
    esv = round(esv, 2)
    sv = round(sv, 2)
    ef = round(ef, 1)
    mass = round(mass, 2)
    return volume, masses, edv, esv, sv, ef, mass


def thicken_close_fill_and_smooth(strokes, stroke_width):
    if strokes is None or not strokes.any():
        return strokes

    # Use power-law scaling for dilation
    dilation_factor = max(1, int(10 / (stroke_width ** 2)))

    # Detect contours to check for nested shapes
    dilated = binary_dilation(strokes, iterations=dilation_factor)
    contours = find_contours(dilated, 0.5)

    has_ring = False
    for i, c1 in enumerate(contours):
        for j, c2 in enumerate(contours):
            if i == j:
                continue
            y1, x1 = c1[:, 0], c1[:, 1]
            y2, x2 = c2[:, 0], c2[:, 1]
            if (y2.min() > y1.min() and y2.max() < y1.max() and
                x2.min() > x1.min() and x2.max() < x1.max()):
                has_ring = True
                break
        if has_ring:
            break

    if has_ring:
        # Dilation + fill + erosion
        closed = binary_dilation(strokes, iterations=dilation_factor)
        filled = binary_fill_holes(closed)
        filled = binary_erosion(filled, iterations=dilation_factor)
        return filled.astype('uint8')
    else:
        return strokes.astype('uint8')


def wrap(key, min_val, max_val):
    if st.session_state[f'{key}'] > max_val:
        st.session_state[f'{key}'] = min_val
    elif st.session_state[f'{key}'] < min_val:
        st.session_state[f'{key}'] = max_val



def frame_index_slider(
    T,
    frames,
    initial_idx,
    label,
    disabled_flag,
    key
):
    idx = st.slider(
        f"{label} | *{initial_idx}*",
        -1,
        T,
        value=initial_idx,
        key = key,
        on_change=wrap,
        args=(key, 0, T-1),
        disabled=disabled_flag
    )
    st.image(frames[idx], use_container_width=True)
    return idx

def copy_frames_channels(mask_name, dia_idx, sys_idx, blood_idx, myo_idx):
    frames = [dia_idx, sys_idx]
    channels = [blood_idx, myo_idx]

    mask = st.session_state[f'roundel.{mask_name}']
    smooth_mask = st.session_state['roundel.preprocessed']["smooth_mask"]

    # Loop over frames and channels to ensure proper assignment
    for f in frames:
        for c in channels:
            mask[:, :, :, f, c] = smooth_mask[:, :, :, f, c]

def confirm_selection(lv_dia_idx, lv_sys_idx,rv_dia_idx, rv_sys_idx):
    """Store confirmed EDV/ESV indices in session state."""
    st.session_state['roundel.edv_esv_selected'].update({
        "lv_dia_idx": lv_dia_idx,
        "lv_sys_idx": lv_sys_idx,
        "rv_dia_idx": rv_dia_idx,
        "rv_sys_idx": rv_sys_idx,
        "confirmed": True
    })

    save_config(st.session_state['roundel.edv_esv_selected'], st.session_state['roundel.cache_config_path'])

    # LV
    copy_frames_channels('edited_mask_lv', lv_dia_idx, lv_sys_idx, lv_idx, lv_myo_idx)

    # RV
    copy_frames_channels('edited_mask_rv', rv_dia_idx, rv_sys_idx, rv_idx, rv_myo_idx)

    save_cached_mask(merge_masks(st.session_state['roundel.edited_mask_lv'],
                                 st.session_state['roundel.edited_mask_rv']),
                                 save_path=st.session_state['roundel.cache_mask_path'])

    make_video(
        st.session_state['roundel.preprocessed']['smooth_image'],
        st.session_state['roundel.edited_mask_lv'],
        mask_frames = [lv_dia_idx, lv_sys_idx],
        save_file=f'{edited_gif_path}_lv',

    )

    make_video(
        st.session_state['roundel.preprocessed']['smooth_image'],
        st.session_state['roundel.edited_mask_rv'],
        mask_frames = [rv_dia_idx, rv_sys_idx],
        save_file=f'{edited_gif_path}_rv',
        ventricle = 'rv'
    )

    gif = Image.open(f'{edited_gif_path}_lv.gif')
    lv_frames = [frame.copy() for frame in ImageSequence.Iterator(gif)]
    st.session_state['roundel.lv_frames'] = lv_frames

    gif = Image.open(f'{edited_gif_path}_rv.gif')
    rv_frames = [frame.copy() for frame in ImageSequence.Iterator(gif)]
    st.session_state['roundel.rv_frames'] = rv_frames

def resize_to_original(edited_mask, raw_mask, crop_box, dia_idx, sys_idx, ventricle):
    """
    Place the edited mask back into the original full-size mask array.
    Assumes edited_mask has shape (H_crop, W_crop, C, 2, num_classes)
    """
    x_min, y_min, x_max, y_max = crop_box
    final_mask_2d = np.zeros_like(raw_mask)

    channels = [rv_idx, rv_myo_idx] if ventricle == 'rv' else [lv_idx, lv_myo_idx]

    for ch in channels:
        final_mask_2d[y_min:y_max, x_min:x_max, ch, dia_idx, :] = edited_mask[:, :, ch, dia_idx, :]
        final_mask_2d[y_min:y_max, x_min:x_max, ch, sys_idx, :] = edited_mask[:, :, ch, sys_idx, :]

    final_mask_2d = np.argmax(final_mask_2d, axis=-1)
    print(np.unique(final_mask_2d))
    return final_mask_2d



def cv_zoom(images, zoom=[4,4,1,1,1], interpolation=cv2.INTER_CUBIC):
    """
    Resize height and width of a 4D or 5D array using OpenCV. Only H and W are scaled.

    Args:
        images (numpy.ndarray): Array of shape (H, W, D, T) or (H, W, D, T, C)
        zoom_factors (list or tuple): Zoom factors for (H, W, D, T, C). Only H and W > 1
        interpolation (int): OpenCV interpolation method (default: cv2.INTER_CUBIC)

    Returns:
        numpy.ndarray: Resized array with height and width scaled, other dimensions unchanged
    """
    h_zoom, w_zoom = zoom[0], zoom[1]

    if images.ndim == 4:
        h, w, d, t = images.shape
        resized = np.zeros((int(h*h_zoom), int(w*w_zoom), d, t), dtype=images.dtype)
        for z in range(d):
            for tau in range(t):
                resized[..., z, tau] = cv2.resize(images[..., z, tau], (int(w*w_zoom), int(h*h_zoom)), interpolation=interpolation)
    elif images.ndim == 5:
        h, w, d, t, c = images.shape
        resized = np.zeros((int(h*h_zoom), int(w*w_zoom), d, t, c), dtype=images.dtype)
        for z in range(d):
            for tau in range(t):
                for ch in range(c):
                    resized[..., z, tau, ch] = cv2.resize(images[..., z, tau, ch], (int(w*w_zoom), int(h*h_zoom)), interpolation=interpolation)
    else:
        raise ValueError("Input must be 4D or 5D array.")

    return resized

def smooth_zoom(mask, zoom=[4,4,1,1,1], sigma=5.0, to_discrete=True):
    """
    Zoom a 4D or 5D categorical mask and smooth edges for visual appearance.

    Args:
        mask (np.ndarray): Input mask of shape H,W,D,T or H,W,D,T,C
        zoom (list): Zoom factors for H,W,D,T,(C). Only H and W >1
        sigma (float): Gaussian blur sigma
        to_discrete (bool): If True, round blurred mask back to original integer labels

    Returns:
        np.ndarray: Zoomed and smoothed mask
    """
    # Step 1: Zoom with nearest-neighbor to preserve labels
    zoomed = cv_zoom(mask.astype(np.float32), zoom, interpolation=cv2.INTER_CUBIC)
    dims = zoomed.ndim
    if dims == 4:
        H,W,D,T = zoomed.shape
        for z in range(D):
            for t in range(T):
                zoomed[..., z, t] = cv2.GaussianBlur(zoomed[..., z, t], (0,0), sigmaX=sigma, sigmaY=sigma)
    elif dims == 5:
        H,W,D,T,C = zoomed.shape
        for z in range(D):
            for t in range(T):
                for c in range(C):
                    zoomed[..., z, t, c] = cv2.GaussianBlur(zoomed[..., z, t, c], (0,0), sigmaX=sigma, sigmaY=sigma)
    else:
        raise ValueError("Mask must be 4D or 5D")

    # Step 2: Optionally convert back to integer labels
    if to_discrete:
        zoomed = np.rint(zoomed).astype(mask.dtype)

    return zoomed


# --------------------------------------------------------------
# Initialization
# --------------------------------------------------------------
def initialize_app(study):
    st.divider()
    st.session_state['roundel.N'] = len(labels.keys())

    progress_bar = st.progress(0, text=f"## **Loading Roundel**")

    def step(p, msg):
        progress_bar.progress(p, text=f'## **{msg}**')

    st.session_state['roundel.orthanc_study_id'] = study.orthanc_study_id
    st.session_state['roundel.patient_id'] = study.patient_id
    st.session_state['roundel.study_date'] = study.study_date

    df = pd.DataFrame([s.__dict__ for s in study.series_dict.values()])
    sax_dl_df = df[(df["dl_orthanc_id"].notna()) & (df["roundel_orthanc_id"].isna())]


    image_instances = fetch_orthanc_instances_for_series_list(
        sax_dl_df["orthanc_series_id"].dropna().unique()
    )
    mask_instances = fetch_orthanc_instances_for_series_list(
        sax_dl_df["dl_orthanc_id"].dropna().unique()
    )

    step(1/5, "Loading Roundel")

    sax_df, raw_image = get_4d_array(image_instances)
    _, raw_mask = get_4d_array(mask_instances)
    raw_mask = (raw_mask / st.session_state['clasp.MASK_SCALER']).astype("uint8")

    st.session_state['roundel.sax_df'] = sax_df

    d0 = fetch_orthanc_dicom(image_instances[0]['ID'])
    st.session_state["roundel.pixelspacing"] = float(d0.PixelSpacing[0])
    st.session_state["roundel.thickness"] = float(
        getattr(d0, "SpacingBetweenSlices",
        getattr(d0, "SliceThickness", np.nan))
    )

    raw_mask = np.eye(st.session_state["roundel.N"], dtype=np.uint8)[raw_mask]
    raw_shape = raw_image.shape

    lv_volume = np.sum(raw_mask[..., lv_idx], axis=(0, 1, 2))
    rv_volume = np.sum(raw_mask[..., rv_idx], axis=(0, 1, 2))

    if np.max(lv_volume) == 0:
        raw_lv_dia_idx = 0
        raw_lv_sys_idx = 15
        raw_rv_dia_idx = 0
        raw_rv_sys_idx = 15
    else:
        nz_lv = np.where(lv_volume != 0)[0]
        nz_rv = np.where(rv_volume != 0)[0]

        raw_lv_dia_idx = int(np.argmax(lv_volume))
        raw_lv_sys_idx = int(nz_lv[np.argmin(lv_volume[nz_lv])])

        raw_rv_dia_idx = int(np.argmax(rv_volume))
        raw_rv_sys_idx = int(nz_rv[np.argmin(rv_volume[nz_rv])])

    st.session_state['roundel.raw'] = {
        "image": raw_image,
        "mask": raw_mask,
        "shape": raw_shape,
        "raw_lv_dia_idx": raw_lv_dia_idx,
        "raw_lv_sys_idx": raw_lv_sys_idx,
        "raw_rv_dia_idx": raw_rv_dia_idx,
        "raw_rv_sys_idx": raw_rv_sys_idx,
    }

    if "roundel.edv_esv_selected" not in st.session_state:
        st.session_state['roundel.edv_esv_selected'] = {
            "lv_dia_idx": None,
            "lv_sys_idx": None,
            "rv_dia_idx": None,
            "rv_sys_idx": None,
            "confirmed": False,
        }

    mask_channels = [i for i in range(st.session_state['roundel.N']) if i != background_idx]

    x_min, y_min, x_max, y_max = find_crop_box(
        np.max(raw_mask[..., mask_channels], axis=(-1, -2, -3)),
        crop_factor=1.5,
    )

    st.session_state["roundel.subpixel_resolution"] = 2

    preprocessed_image = raw_image[y_min:y_max, x_min:x_max, :, :]
    preprocessed_mask = raw_mask[y_min:y_max, x_min:x_max, :, :, :].astype("uint8")

    H, W, D, T, N = preprocessed_mask.shape

    step(2/5, "Loading Roundel")

    has_masks = np.where(
        np.sum(preprocessed_mask[..., mask_channels], axis=(0, 1, 3, -1)) > 0
    )[0]

    if len(has_masks) == 0:
        has_masks = np.array([1, 2, 3, 4, 5, 6])

    mid_slice = len(has_masks) // 2

    zoom = [
        st.session_state["roundel.subpixel_resolution"],
        st.session_state["roundel.subpixel_resolution"],
        1,
        1,
    ]

    smoothed_image = cv_zoom(preprocessed_image, zoom=zoom)

    st.session_state["roundel.cache_config_path"] = (
        f"{cache_dir}/config___{st.session_state['roundel.orthanc_study_id']}.json"
    )
    st.session_state["roundel.cache_mask_path"] = (
        f"{cache_dir}/masks___{st.session_state['roundel.orthanc_study_id']}.npy"
    )

    step(3/5, "Loading Roundel")

    if (
        os.path.exists(st.session_state["roundel.cache_config_path"])
        and os.path.exists(st.session_state["roundel.cache_mask_path"])
    ):
        smoothed_mask = load_cached_mask(
            st.session_state["roundel.cache_mask_path"]
        ).astype("uint8")
        cached = True
    else:
        smoothed_mask = cv_zoom_mask(
            preprocessed_mask,
            zoom=zoom + [1],
            interpolation=cv2.INTER_NEAREST,
        )
        cached = False

    make_video(
        smoothed_image[:, :, has_masks[mid_slice - 3 : mid_slice + 3], :],
        smoothed_mask[:, :, has_masks[mid_slice - 3 : mid_slice + 3], :, :] * 0,
        save_file=edv_esv_gif_path,
    )

    step(4/5, "Loading Roundel")

    make_video(
        smoothed_image,
        smoothed_mask * 0,
        save_file=blank_gif_path,
    )

    step(5/5, "Loading Roundel")

    gif = Image.open(f"{edv_esv_gif_path}.gif")

    st.session_state['roundel.preprocessed'] = {
        "image": preprocessed_image,
        "mask": preprocessed_mask,
        "smooth_image": smoothed_image,
        "smooth_mask": smoothed_mask,
        "H": H,
        "W": W,
        "D": D,
        "T": T,
        "N": N,
        "edv_esv_frames": [f.copy() for f in ImageSequence.Iterator(gif)],
        "crop_box": [x_min, y_min, x_max, y_max],
    }

    st.session_state['roundel.edited_mask_lv'] = np.zeros_like(
        st.session_state['roundel.preprocessed']["smooth_mask"]
    )
    st.session_state['roundel.edited_mask_rv'] = np.zeros_like(
        st.session_state['roundel.preprocessed']["smooth_mask"]
    )

    if cached:
        config = load_config(st.session_state["roundel.cache_config_path"])
        confirm_selection(
            lv_dia_idx=config["lv_dia_idx"],
            rv_dia_idx=config["rv_dia_idx"],
            lv_sys_idx=config["lv_sys_idx"],
            rv_sys_idx=config["rv_sys_idx"],
        )

    st.session_state['roundel.lv_frames'] = None
    st.session_state['roundel.rv_frames'] = None
    st.session_state['roundel.view_mode'] = "Static"
    st.session_state['roundel.brush_mode'] = "Draw ✐"
    st.session_state['roundel.stroke_width'] = "thin"
    st.session_state['roundel.edit_made'] = False
    st.session_state['roundel.cached'] = cached
    st.session_state['roundel.saved'] = False
    st.session_state['roundel.initialized'] = True
    st.session_state["roundel.view"] = 'EDV/ESV Finder 🔍'
    st.session_state['roundel.edv_esv_selected'] = {"lv_dia_idx": None, "lv_sys_idx": None, "rv_dia_idx": None, "rv_sys_idx": None,"confirmed": False}
    progress_bar.empty()



def edv_esv_view():
    """Full EDV/ESV Finder view layout."""
    if not st.session_state['roundel.initialized']:
        st.error("Select and confirm EDV/ESV first.")
        st.stop()

    H, W, D, T, N = [st.session_state['roundel.preprocessed'][k] for k in ["H","W","D","T","N"]]
    edv_esv_frames= st.session_state['roundel.preprocessed']['edv_esv_frames']


    if st.session_state['roundel.edv_esv_selected']['confirmed']:
        display_lv_dia_idx=st.session_state['roundel.edv_esv_selected']['lv_dia_idx']
        display_rv_dia_idx=st.session_state['roundel.edv_esv_selected']['rv_dia_idx']
        display_lv_sys_idx=st.session_state['roundel.edv_esv_selected']['lv_sys_idx']
        display_rv_sys_idx=st.session_state['roundel.edv_esv_selected']['rv_sys_idx']
    else:
        display_lv_dia_idx=st.session_state['roundel.raw']['raw_lv_dia_idx']
        display_rv_dia_idx=st.session_state['roundel.raw']['raw_rv_dia_idx'] 
        display_lv_sys_idx=st.session_state['roundel.raw']['raw_lv_sys_idx'] 
        display_rv_sys_idx=st.session_state['roundel.raw']['raw_rv_sys_idx'] 

    disabled_flag = st.session_state['roundel.edv_esv_selected']["confirmed"]

    col_lv, col_rv = st.columns(2)

    with col_lv:
        st.markdown('#### Left Ventricle')
        col_edv, col_esv = st.columns(2)

        with col_edv:
            lv_dia_idx = frame_index_slider(T, edv_esv_frames, display_lv_dia_idx, 'LV End-Diastolic Index', disabled_flag, key = 'roundel.lv_edv')

        with col_esv:
            lv_sys_idx = frame_index_slider(T, edv_esv_frames, display_lv_sys_idx, 'LV End-Systolic Index',disabled_flag, key = 'roundel.lv_esv')

    with col_rv:
        st.markdown('#### Right Ventricle')
        col_edv, col_esv = st.columns(2)
        with col_edv:
            rv_dia_idx = frame_index_slider(T, edv_esv_frames, display_rv_dia_idx, 'RV End-Diastolic Index', disabled_flag, key = 'roundel.rv_edv')

        with col_esv:
            rv_sys_idx = frame_index_slider(T, edv_esv_frames, display_rv_sys_idx, 'RV End-Systolic Index',disabled_flag, key = 'roundel.rv_esv')


    st.write('')
    if not disabled_flag:
        st.button(
            "Confirm EDV | ESV",
            on_click=lambda: confirm_selection(lv_dia_idx, lv_sys_idx, rv_dia_idx, rv_sys_idx),
            type="primary",
            use_container_width=True
        )
    else:
        st.success("EDV | ESV Confirmed! 🔍")



def slice_navigation(D):
    if "roundel.slice_idx" not in st.session_state:
        st.session_state['roundel.slice_idx'] = 0
    if "roundel.previous_slice_idx" not in st.session_state:
        st.session_state['roundel.previous_slice_idx'] = st.session_state['roundel.slice_idx']

    # Store previous slice
    previous_d = st.session_state['roundel.previous_slice_idx']

    # Slider (updates slice_idx immediately)
    st.slider("Slice Index", 0, D - 1, key= "roundel.slice_idx")

    col_prev, col_next = st.columns(2)
    with col_prev:
        st.button(
            "Previous",
            on_click=lambda: st.session_state.update(
                **{"roundel.slice_idx": max(0, st.session_state["roundel.slice_idx"] - 1)}
            ),
            use_container_width=True,
        )
    with col_next:
        st.button(
            "Next",
            on_click=lambda: st.session_state.update(
                **{"roundel.slice_idx": min(D-1, st.session_state["roundel.slice_idx"] + 1)}
            ),
            use_container_width=True,
        )

    # Determine if canvas needs reset
    previous_objects = st.session_state.get('roundel.canvas', {}).get('previous_objects', [])
    reset_canvas = previous_d != st.session_state['roundel.slice_idx'] and bool(previous_objects)

    # Update previous slice for next rerun
    st.session_state['roundel.previous_slice_idx'] = st.session_state['roundel.slice_idx']

    return st.session_state['roundel.slice_idx'], reset_canvas



def get_overlay(image_slice, mask_state, H, W, N, OVERLAY_COLORS, ventricle):
    if ventricle == 'rv':
        channels = [rv_idx, rv_myo_idx]
    elif ventricle == 'lv':
        channels = [lv_idx, lv_myo_idx]
    else:
        channels = np.arange(N)

    overlay = Image.fromarray(np.stack([image_slice]*3, axis=-1)).convert("RGBA")
    for i in channels:
        ch_mask = mask_state[:, :, i]
        if np.any(ch_mask):
            mask_img = np.zeros((H*st.session_state['roundel.subpixel_resolution'], W*st.session_state['roundel.subpixel_resolution'], 4), dtype=np.uint8)
            mask_img[ch_mask > 0] = OVERLAY_COLORS[i]
            overlay = Image.alpha_composite(overlay, Image.fromarray(mask_img))
    return overlay



def select_brush(N, ventricle):
    
    """Brush selection UI for channel, action, and stroke width."""

    st.markdown(
        "<div style='color:#155a8a; margin-bottom:-10px; margin-top:20px; font-weight:600;'>Choose Brush</div>",
        unsafe_allow_html=True
    )
    action = st.pills(
        "Type", 
        options=["Draw ✐", "Erase ✂"],  
        selection_mode="single",
        default=st.session_state['roundel.brush_mode']
    )

    if action is not None:
        st.session_state['roundel.brush_mode'] = action

    brush_mode = st.session_state['roundel.brush_mode']


    stroke_width_map = {"thin":6,"medium":20,"thick":40}

    stroke_width_sel = st.pills(
        "Thickness", 
        options=list(stroke_width_map.keys()),  
        selection_mode="single",
        default=st.session_state["roundel.stroke_width"]
    )

    if stroke_width_sel is not None:
        st.session_state['roundel.stroke_width'] = stroke_width_sel

    stroke_width = st.session_state['roundel.stroke_width']

    if ventricle == 'lv':
        valid_channels = [lv_myo_idx, lv_idx]
    elif ventricle == 'rv':
        valid_channels = [rv_myo_idx, rv_idx]
    else:
        valid_channels = [i for i in range(N) if i != background_idx]

    if action == "Draw ✐":
        default_channel = rv_myo_idx if ventricle == 'rv' else lv_myo_idx

        if "channel" not in st.session_state:
            st.session_state["channel"] = default_channel
            st.session_state["channel_ventricle"] = ventricle

        if st.session_state["channel_ventricle"] != ventricle:
            st.session_state["channel"] = default_channel
            st.session_state["channel_ventricle"] = ventricle

        channel_val = st.pills(
            "Structure",
            options=valid_channels,
            format_func=lambda x: BRUSH_LABELS[x],
            selection_mode="single",
            default=st.session_state["channel"],
        )

        if channel_val is not None:
            st.session_state["channel"] = channel_val

        channel = st.session_state["channel"]
    else:
        channel = 0
    stroke_width = stroke_width_map[stroke_width_sel]
    return channel, action, stroke_width



def mask_editor_view():
    """Full Mask Editor layout."""
    if not st.session_state['roundel.edv_esv_selected']["confirmed"]:
        st.error("Select and confirm EDV/ESV first.")
        st.stop()

    col1, col2, col3 = st.columns([1,1.5,1.5])

    H, W, D, T, N = [st.session_state['roundel.preprocessed'][k] for k in ["H","W","D","T","N"]]
    image=st.session_state['roundel.preprocessed']["smooth_image"]

    with col1:
        st.markdown(
            "<div style='color:#155a8a; margin-bottom:-25px; font-weight:600;'>Choose Ventricle</div>",
            unsafe_allow_html=True
        )
        options = ["Left Ventricle", "Right Ventricle"]

        if "ventricle_label" not in st.session_state:
            st.session_state.ventricle_label = options[0]

        val = st.pills("", options, selection_mode="single", default=st.session_state.ventricle_label)

        if val is not None:
            st.session_state.ventricle_label = val

        ventricle_label = st.session_state.ventricle_label

        ventricle = 'lv' if 'left' in ventricle_label.lower() else 'rv'
        channel, action, stroke_width = select_brush(N, ventricle)

        st.markdown(
            "<div style='color:#155a8a; margin-bottom:-10px; margin-top:20px; font-weight:600;'>Choose Position</div>",
            unsafe_allow_html=True
        )
        frame_options = ["End-Diastole", "End-Systole"]

        if "idx_label" not in st.session_state:
            st.session_state.idx_label = frame_options[0]

        val = st.pills("Frame", options=frame_options, selection_mode="single", default=st.session_state.idx_label)

        if val is not None:
            st.session_state.idx_label = val

        idx_label = st.session_state.idx_label


        d, reset_canvas = slice_navigation(D)


        edited_mask=st.session_state[f'roundel.edited_mask_{ventricle}']
        dia_idx=st.session_state['roundel.edv_esv_selected'][f"{ventricle}_dia_idx"]
        sys_idx=st.session_state['roundel.edv_esv_selected'][f"{ventricle}_sys_idx"]


    idx = dia_idx if idx_label=="End-Diastole" else sys_idx
    image_slice = image[:,:,d,idx]
    image_slice = (normalize(image_slice) * 255).astype(np.uint8)
    mask_slice = edited_mask[:,:,d,idx,:]

    with col2:
        st.markdown(
            "<div style='color:#155a8a; margin-bottom:-10px; font-weight:600;'>Segmentation Editor</div>",
            unsafe_allow_html=True
        )
        val = st.pills(
            "",
            options=["Hide masks"],
            selection_mode="single",
        )

        edit_mode = val 
        stroke_color = f"rgba{OVERLAY_COLORS[background_idx][:3]+(0.7,)}" if action == "Erase ✂️" else f"rgba{OVERLAY_COLORS[channel][:3]+(0.65,)}"
        if edit_mode == 'Hide masks':
            st.image(image_slice, width=DISPLAY_W)
        else:
            # Initialize canvas state
            if 'roundel.canvas' not in st.session_state:
                st.session_state['roundel.canvas'] = {
                    'canvas_key': f'editor_{d}',
                    'previous_d': d,
                    'previous_objects': []
                }
                        
            if reset_canvas:
                st.session_state['roundel.canvas']['canvas_key'] = f'editor_{d}'
                st.session_state['roundel.canvas']['previous_objects'] = []

            st.session_state['roundel.canvas']['previous_d'] = d

            canvas_result = st_canvas(
                stroke_width=stroke_width,
                stroke_color=stroke_color,
                background_image=get_overlay(image_slice, mask_slice, H, W, N, OVERLAY_COLORS, ventricle),
                update_streamlit=True,
                # height = H*DISPLAY_W/W,
                # width=DISPLAY_W,
                drawing_mode='freedraw',
                key=st.session_state['roundel.canvas']['canvas_key']+ ventricle
            )


            # Track current objects
            current_objects = []
            if canvas_result and canvas_result.json_data:
                current_objects = canvas_result.json_data.get("objects", [])
            st.session_state['roundel.canvas']['previous_objects'] = current_objects

            # Save / clear buttons (trigger rerun only here)
            col_save, col_clear = st.columns([1, 0.3])
            edited_mask = st.session_state[f'roundel.edited_mask_{ventricle}']
            
            with col_save:
                save_contour = st.button('Save Contour', type='primary', use_container_width=True)
                if save_contour and canvas_result and canvas_result.image_data is not None and current_objects:
                    brush_data = np.array(canvas_result.image_data)
                    rgb = brush_data[:, :, :3].astype(np.float32)
                    alpha = brush_data[:, :, 3].astype(np.float32) / 255.0

                    overlay_colors_list = np.array([color[:3] for color in OVERLAY_COLORS.values()], dtype=np.float32)
                    overlay_channels = list(OVERLAY_COLORS.keys())

                    h, w, _ = rgb.shape
                    rgb_flat = rgb.reshape(-1, 3)
                    alpha_flat = alpha.flatten()
                    distances = np.linalg.norm(rgb_flat[:, None, :] - overlay_colors_list[None, :, :], axis=-1)
                    closest_idx = np.argmin(distances, axis=1)

                    mask_flat = np.zeros((h*w, len(overlay_channels)), dtype=np.uint8)
                    for idx_color, ch in enumerate(overlay_channels):
                        mask_flat[:, idx_color] = ((closest_idx == idx_color) & (alpha_flat > 0)).astype(np.uint8)

                    masks = []
                    for idx_color, ch in enumerate(overlay_channels):
                        mask_bool = mask_flat[:, idx_color].reshape(h, w)
                        mask_bool = thicken_close_fill_and_smooth(mask_bool, stroke_width)
                        masks.append(mask_bool)

                    combined_mask = np.stack(masks, axis=-1)
                    for idx_color, ch in enumerate(overlay_channels):
                        resized_mask = np.array(
                            Image.fromarray(combined_mask[:, :, idx_color]).resize(
                                (W*st.session_state['roundel.subpixel_resolution'], H*st.session_state['roundel.subpixel_resolution']),
                                resample=Image.NEAREST
                            )
                        )
                        edited_mask[:, :, d, idx, :][resized_mask > 0] = 0
                        edited_mask[:, :, d, idx, ch][resized_mask > 0] = 1

                    st.session_state['roundel.edit_made'] = True
                    combined_mask = merge_masks(st.session_state[f'roundel.edited_mask_lv'] , st.session_state[f'roundel.edited_mask_rv'])
                    save_cached_mask(combined_mask, save_path=st.session_state['roundel.cache_mask_path'])
                    st.rerun()

            with col_clear:
                if st.button('❌', use_container_width=True):
                    edited_mask[:, :, d, idx, :] = 0
                    combined_mask = merge_masks(st.session_state[f'roundel.edited_mask_lv'] , st.session_state[f'roundel.edited_mask_rv'])
                    save_cached_mask(combined_mask, save_path=st.session_state['roundel.cache_mask_path'])

                    st.session_state['roundel.edit_made'] = True
                    st.rerun()

            st.session_state[f'roundel.edited_mask_{ventricle}'] = edited_mask
            


    # ---------- right column preview ----------
    with col3:
        
        st.markdown(
            "<div style='color:#155a8a; margin-bottom:40px; font-weight:600;'>All Slices</div>",
            unsafe_allow_html=True
        )
        
        if st.session_state[f'roundel.{ventricle}_frames'] is None or st.session_state['roundel.edit_made']:
            make_video(
                image,
                st.session_state[f'roundel.edited_mask_{ventricle}'],
                save_file=f'{edited_gif_path}_{ventricle}',
                mask_frames = [dia_idx, sys_idx],
                ventricle = ventricle
            )

            gif = Image.open(f'{edited_gif_path}_{ventricle}.gif')
            st.session_state[f'roundel.{ventricle}_frames'] = [frame.copy() for frame in ImageSequence.Iterator(gif)]
            st.session_state['roundel.edit_made'] = False

        
        view_image = st.session_state[f'roundel.{ventricle}_frames'][0 if idx_label == "End-Diastole" else 1]
        width = int(DISPLAY_W * 1.5)
        

        st.image(view_image, width = width)



def final_result_view():
    raw = st.session_state['roundel.raw']
    preprocessed = st.session_state['roundel.preprocessed']
    pixelspacing = st.session_state['roundel.pixelspacing']
    thickness = st.session_state['roundel.thickness']

    raw_image = raw["image"]
    raw_mask = raw["mask"]
    preprocessed_image = preprocessed["image"]

    H, W, D, T, N = [preprocessed[k] for k in ["H","W","D","T","N"]]

    crop_box = preprocessed['crop_box']
    
    if not st.session_state['roundel.edv_esv_selected']["confirmed"]:
        st.error("Select and confirm EDV/ESV first.")
        st.stop()

    raw_lv_dia_idx = raw["raw_lv_dia_idx"]
    raw_lv_sys_idx = raw["raw_lv_sys_idx"]
    raw_rv_dia_idx = raw["raw_rv_dia_idx"]
    raw_rv_sys_idx = raw["raw_rv_sys_idx"]

    lv_dia_idx = st.session_state['roundel.edv_esv_selected']["lv_dia_idx"]
    lv_sys_idx = st.session_state['roundel.edv_esv_selected']["lv_sys_idx"]
    rv_dia_idx = st.session_state['roundel.edv_esv_selected']["rv_dia_idx"]
    rv_sys_idx = st.session_state['roundel.edv_esv_selected']["rv_sys_idx"]
    orthanc_study_id = st.session_state['roundel.orthanc_study_id']
    patient_id = st.session_state['roundel.patient_id']
    study_date = st.session_state['roundel.study_date']

    final_lv_gif_path = f"{results_path}/gifs/{orthanc_study_id}_lv.gif"
    final_rv_gif_path = f"{results_path}/gifs/{orthanc_study_id}_rv.gif"

    lv_mask = cv_zoom(st.session_state['roundel.edited_mask_lv'], zoom = [1/st.session_state['roundel.subpixel_resolution'],1/st.session_state['roundel.subpixel_resolution'],1,1])
    rv_mask = cv_zoom(st.session_state['roundel.edited_mask_rv'], zoom = [1/st.session_state['roundel.subpixel_resolution'],1/st.session_state['roundel.subpixel_resolution'],1,1])

    combined_mask = merge_masks(lv_mask, rv_mask)


    # Calculate LV metrics
    lv_volume, lv_masses, lv_edv, lv_esv, lv_sv, lv_ef, lv_mass = calculate_sax_metrics(
        mask=combined_mask,
        blood_pool_idx=lv_idx,
        myo_idx=lv_myo_idx,
        dia_idx=lv_dia_idx,
        sys_idx=lv_sys_idx
    )
    raw_lv_volume, raw_lv_masses, raw_lv_edv, raw_lv_esv, raw_lv_sv, raw_lv_ef, raw_lv_mass = calculate_sax_metrics(
        mask=raw_mask,
        blood_pool_idx=lv_idx,
        myo_idx=lv_myo_idx,
        dia_idx=raw_lv_dia_idx,
        sys_idx=raw_lv_sys_idx
    )

    # Calculate RV metrics
    rv_volume, rv_masses, rv_edv, rv_esv, rv_sv, rv_ef, rv_mass = calculate_sax_metrics(
        mask=combined_mask,
        blood_pool_idx=rv_idx,
        myo_idx=rv_myo_idx,
        dia_idx=rv_dia_idx,
        sys_idx=rv_sys_idx
    )
    raw_rv_volume, raw_rv_masses, raw_rv_edv, raw_rv_esv, raw_rv_sv, raw_rv_ef, raw_rv_mass = calculate_sax_metrics(
        mask=raw_mask,
        blood_pool_idx=rv_idx,
        myo_idx=rv_myo_idx,
        dia_idx=raw_rv_dia_idx,
        sys_idx=raw_rv_sys_idx
    )


    x_min, y_min, x_max, y_max = crop_box
    final_mask_2d = np.zeros_like(raw_mask)
    final_mask_2d[y_min:y_max, x_min:x_max, :, :, :] = combined_mask
    final_mask_2d = np.argmax(final_mask_2d, axis=-1)


    make_video(preprocessed_image, 
               final_mask_2d[y_min:y_max,x_min:x_max,:,:], 
               save_file=final_lv_gif_path, 
               mask_frames=[lv_dia_idx,lv_sys_idx],
               ventricle='all')
    
    make_video(preprocessed_image, 
               final_mask_2d[y_min:y_max,x_min:x_max,:,:], 
               save_file=final_rv_gif_path, 
               mask_frames=[rv_dia_idx,rv_sys_idx],
               ventricle='all')
    

    col_lv, _, col_rv = st.columns([1,0.05,1])
    with col_lv:
        st.markdown('#### Left Ventricle')

        col1, col2 = st.columns([0.3,0.7])
        with col1:
            st.caption("LV Metrics")
            st.metric("EDV", f"{lv_edv:.1f}mL", delta=format_delta(lv_edv, raw_lv_edv, "mL"))
            st.metric("ESV", f"{lv_esv:.1f}mL", delta=format_delta(lv_esv, raw_lv_esv, "mL"))
            st.metric("EF", f"{lv_ef:.1f}%", delta=format_delta(lv_ef, raw_lv_ef, "%", round_digits=1))
            st.metric("Mass", f"{lv_mass:.1f}g", delta=format_delta(lv_mass, raw_lv_mass, "g"))

        with col2:
            st.caption("Final LV Mask")
            st.image(final_lv_gif_path)
        
    with col_rv:
        st.markdown('#### Right Ventricle')

        col1, col2 = st.columns([0.3,0.7])
        with col1:
            st.caption("RV Metrics")
            st.metric("EDV", f"{rv_edv:.1f}mL", delta=format_delta(rv_edv, raw_rv_edv, "mL"))
            st.metric("ESV", f"{rv_esv:.1f}mL", delta=format_delta(rv_esv, raw_rv_esv, "mL"))
            st.metric("EF", f"{rv_ef:.1f}%", delta=format_delta(rv_ef, raw_rv_ef, "%", round_digits=1))
            st.metric("Mass", f"{rv_mass:.1f}g", delta=format_delta(rv_mass, raw_rv_mass, "g"))


        with col2:
            st.caption("Final RV Mask")
            st.image(final_rv_gif_path)

    save_button = st.button('Save Masks and Metrics 💾', type='primary', use_container_width=True)


    if save_button:
        with st.spinner('Saving...'):
            final_mask_2d_flat = flatten_4d_array(final_mask_2d * st.session_state['clasp.MASK_SCALER'])

            new_sax_df = st.session_state['roundel.sax_df'].copy()
            new_sax_df['PixelArray'] = final_mask_2d_flat 

            study = fetch_db_study(st.session_state['roundel.current_study_id'])
            for series_orthanc_id, series_df in new_sax_df.groupby('OrthancSeriesID'):
                old_dcms = [fetch_orthanc_dicom(id) for id in series_df.OrthancInstanceID]
                new_masks = [mask for mask in series_df.PixelArray]
                new_orthanc_id = send_series_to_orthanc(new_masks, old_dcms, new_description='Roundel')
                series = fetch_db_series(study, series_orthanc_id)
                series.roundel_orthanc_id = new_orthanc_id
                study.series_dict[series_orthanc_id] = series
        
            db = TinyDB(st.session_state['clasp.DB_PATH'])
            update_study(db, study)


            # save_mask(final_mask_2d, f'{nifti_mask_path}/{st.session_state['roundel.patient_name']}.nii.gz')
            # save_mask_as_dicom_series(final_mask_2d, f'{dicom_mask_path}/{st.session_state['roundel.patient_name']}')

            combined_df = pd.DataFrame({
                "patient_id": [patient_id],
                "orthanc_study_id": [orthanc_study_id],
                "exams_date": [pd.to_datetime(study_date, dayfirst=True).date()],
                "lv_edv": [lv_edv],
                "lv_esv": [lv_esv],
                "lv_sv": [lv_sv],
                "lv_ef": [lv_ef],
                "rv_mass": [rv_mass],
                "rv_edv": [rv_edv],
                "rv_esv": [rv_esv],
                "rv_sv": [rv_sv],
                "rv_ef": [rv_ef],
                "rv_mass": [rv_mass],
            })

            EXAMS_PATH = st.session_state["clasp.EXAMS_PATH"]

            if os.path.exists(EXAMS_PATH):
                exams_df = pd.read_csv(EXAMS_PATH)
                exams_df = pd.concat([exams_df, combined_df], ignore_index=True)
                exams_df = exams_df.drop_duplicates(subset="orthanc_study_id", keep="last")
            else:
                exams_df = combined_df

            exams_df.to_csv(EXAMS_PATH, index=False)
            st.session_state["roundel.saved"] = True
        
        if st.session_state.get("roundel.saved", False):
            st.success('Masks and Metrics Overwritten! ✅')
        else:
            st.success('Masks and Metrics Saved! ✅')
        

    elif st.session_state.get("roundel.saved", False):
        st.info('Masks and Metrics Previously Saved! ✅')

    

    if st.session_state["roundel.saved"] and st.button('Next Patient ➡️', use_container_width=True):
        reset_app('roundel')
        for filename in os.listdir(cache_dir):
            file_path = os.path.join(cache_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        st.rerun()
        