# Import standard libraries
import numpy as np
import torch
import warnings
from scipy.ndimage import gaussian_filter


# ---------- small utilities ----------
# Define function to generate one-hot encoding of the predicted channels 
def get_one_hot(indices, num_classes):
    """Convert indices to one-hot encoded array."""
    one_hot_array = np.eye(num_classes)[indices]
    one_hot_array = one_hot_array.astype(np.uint8)
    return one_hot_array

def _stable_softmax(x, axis=-1):
    """
    Compute softmax values for each set of scores in x.
    
    Args:
        x: Input array of shape (batch_size, num_classes) or (num_classes,)
        
    Returns:
        Softmax probabilities of same shape as input
    """
    # For numerical stability, subtract the maximum value from each input vector
    # This prevents overflow when calculating exp(x)
    x = np.asarray(x) # ensure input is a numpy array
    shifted_x = x - np.max(x, axis=axis, keepdims=True)
    
    # Calculate exp(x) for each element
    exp_x = np.exp(shifted_x)
    
    # Calculate the sum of exp(x) for normalization
    sum_exp_x = np.sum(exp_x, axis=axis, keepdims=True)
    
    # Normalize to get probabilities
    probabilities = exp_x / sum_exp_x
    
    return probabilities

def _make_gaussian_importance_map(
    patch_size,             # Tuple[int, int, int], e.g. (128,128,16)
    sigma_scale=1/8,        # same meaning as nnU-Net
    value_scaling_factor=1, # peak value after normalisation
    dtype=np.float32
):
    """
    Replicates nnU-Net's compute_gaussian (NumPy version):
    - place 1 at the center of patch
    - apply gaussian_filter with per-axis sigmas
    - scale so max == value_scaling_factor
    - replace zeros with min non-zero to avoid division by zero in stitching
    """
    patch_size = tuple(int(p) for p in patch_size)
    tmp = np.zeros(patch_size, dtype=np.float32)
    center_coords = tuple([p // 2 for p in patch_size])
    tmp[center_coords] = 1.0
    sigmas = [p * float(sigma_scale) for p in patch_size]

    g = gaussian_filter(tmp, sigma=sigmas, order=0, mode='constant', cval=0.0)

    maxv = g.max() # get max in filter before scaling
    if maxv > 0:
        g = g / (maxv / float(value_scaling_factor)) # scale to desired peak value

    # ensure strictly positive (nnU-Net replaces zeros with min non-zero to avoid nans)
    zero_mask = (g == 0)
    if np.any(zero_mask):
        nonzero_vals = g[~zero_mask]
        if nonzero_vals.size == 0:
            # degenerate fallback: uniform map
            g[...] = float(value_scaling_factor)
        else:
            g[zero_mask] = np.min(nonzero_vals)

    return g.astype(dtype)

def _compute_steps(im_size, patch_size, overlap):
    """
    Compute sliding window start indices with the last window anchored to the end.
    overlap in [0, 1). e.g. 0.5 -> stride = patch_size // 2
    """
    ps = np.array(patch_size, dtype=int)
    sz = np.array(im_size, dtype=int)
    stride = np.maximum((ps * (1.0 - overlap)).astype(int), 1)  # clamp stride to at least 1 to avoid infinite loop
    steps = []
    for i in range(len(sz)):
        s = []
        pos = 0
        while True:
            s.append(pos)
            if pos + ps[i] >= sz[i]:
                break # if the patch + size exceeds the image size, do not move further
            pos += stride[i] # if the patch + size does not exceed the image size, move by stride
            if pos + ps[i] > sz[i]:  # if the next position + patch size exceeds the image size, change next position to be the image size minus the patch size, so that the last patch is always anchored to the end
                pos = sz[i] - ps[i]
        steps.append(s)
    return steps  # list of lists of start positions for each axis

def _pad_to_patch_size(vol, patch_size):
    """
    Pad constantly so that each spatial dim is at least patch_size.
    Returns padded volume and the padding applied.
    """
    vol = np.asarray(vol)
    spatial = vol.shape[:3]
    pads = []
    for d in range(3):
        need = max(0, patch_size[d] - spatial[d])
        pads.append((need // 2, need - need // 2))
    # no padding for channels
    pad_width = (pads[0], pads[1], pads[2], (0, 0))
    vol_padded = np.pad(vol, pad_width, mode='reflect')
    return vol_padded, pads

def _crop_from_pad(vol, pads):
    """Crop back to original shape given symmetric pads for first 3 dims."""
    y0, y1 = pads[0]
    x0, x1 = pads[1]
    z0, z1 = pads[2]
    Y, X, Z = vol.shape[:3]
    return vol[y0:Y - y1 if y1 > 0 else Y, 
               x0:X - x1 if x1 > 0 else X, 
               z0:Z - z1 if z1 > 0 else Z, ...]

def _flip3d(vol, axes_mask):
    """
    Flip a 4D volume [Y,X,Z,C] over any combination of axes (0,1,2) based on axes_mask (tuple of bools).
    """
    vol_f = vol
    for ax, do in enumerate(axes_mask):
        if do:
            vol_f = np.flip(vol_f, axis=ax)
    return np.ascontiguousarray(vol_f)

def _iter_tta_axes(do_tta):
    """Generator that yields axes masks for TTA. If do_tta=False -> only (False, False, False)."""
    if not do_tta:
        yield (False, False, False)
        return
    for a in (False, True):
        for b in (False, True):
            for c in (False, True):
                yield (a, b, c) # 8 combinations for the 3 axes (False, False, False) to (True, True, True)

# ---------- main function ----------

def sliding_window_inference_3d(
    model,
    volume,                   # np.ndarray [batchsize,C,Z,Y,X] already preprocessed & normalised
    patch_size=(128, 128, 16),
    overlap=0.5,
    apply_softmax=True,       # set False if your model already uses softmax activation
    out_channels=None,    # if known, can be set to avoid dry run
    tta=True,                 # Whether to use test-time augmentation (flips)
    gaussian_sigma_scale=1/8, # controls how peaked the Gaussian is
    deep_supervision=True
):
    """
    Returns:
        prob_map: np.ndarray [Y,X,Z,C] softmax probabilities
        seg:      np.ndarray [Y,X,Z]     argmax labels (int)
    """

    # print("Input volume shape:", volume.shape)
    assert volume.ndim == 5 and volume.shape[1] >= 1, "Expected [B,C,Z,Y,X]"
    # remove batch dim and transpose from [C,Z,Y,X] -> [Y,X,Z,C] for internal processing
    volume = np.squeeze(volume, axis=0)        # [C,Z,Y,X]
    volume = np.transpose(volume, (2, 3, 1, 0))  # [Y,X,Z,C]
    Y0, X0, Z0, C_in = volume.shape
    # print("Input volume shape after removing batch:", volume.shape)

    # Pad so patching works even if the input is smaller than the patch.
    vol_pad, pads = _pad_to_patch_size(volume, patch_size)
    # print("Padded volume shape:", vol_pad.shape, "Pads applied:", pads)

    # Compute the steps needed to cover input image with patches
    Y, X, Z, _ = vol_pad.shape
    steps = _compute_steps((Y, X, Z), patch_size, overlap)
    # print("Number of steps (y,x,z):", (len(steps[0]), len(steps[1]), len(steps[2])))
    # print("Steps (y):", steps[0])
    # print("Steps (x):", steps[1])
    # print("Steps (z):", steps[2])
    # Compute the gaussian importance map
    gauss = _make_gaussian_importance_map(patch_size, sigma_scale=gaussian_sigma_scale).astype(np.float32)
    gauss = gauss[..., np.newaxis]  # [py, px, pz, 1]
    # print("Gaussian importance map shape:", gauss.shape)

    # Create variable to hold the accumulated probabilities over test-time augmentations
    prob_accum_all_tta = np.zeros((Y, X, Z, out_channels), dtype=np.float32)

    # TTA loop
    for axes_mask in _iter_tta_axes(tta): # axes_mask is a tuple of 3 bools indicating whether to flip along each axis
        # flip input
        vol_aug = _flip3d(vol_pad, axes_mask)

        prob_accum = np.zeros((Y, X, Z, out_channels), dtype=np.float32)
        weight_accum = np.zeros((Y, X, Z, 1), dtype=np.float32)

        # sliding window over the flipped volume
        for y in steps[0]:
            for x in steps[1]:
                for z in steps[2]:
                    patch = vol_aug[
                        y:y+patch_size[0],
                        x:x+patch_size[1],
                        z:z+patch_size[2],
                        :
                    ]
                    # print("Patch position (y,x,z):", (y, x, z))
                    # print("Patch shape:", patch.shape)
                    # model inference (batch size = 1)
                    # patch is [py, px, pz, C_in]; convert to [1, C_in, pz, py, px] for PyTorch
                    patch_t = torch.from_numpy(
                        np.ascontiguousarray(np.transpose(patch, (3, 2, 0, 1))[np.newaxis, ...])  # [1, C_in, pz, py, px]
                    ).to(next(model.parameters()).device)
                    with torch.no_grad():
                        pred = model(patch_t)  # [1, C_out, pz, py, px] or list of such tensors if deep supervision
                    if isinstance(pred, (list, tuple)) and deep_supervision:
                        pred = pred[0]  # index 0 = highest-resolution output
                    # convert back to [py, px, pz, C_out]
                    pred = pred[0].cpu().numpy()          # [C_out, pz, py, px]
                    pred = np.transpose(pred, (2, 3, 1, 0))  # [py, px, pz, C_out]

                    if apply_softmax:
                        pred = _stable_softmax(pred, axis=-1)

                    # weight and accumulate
                    w = gauss  # [py, px, pz, 1]
                    prob_accum[y:y+patch_size[0], x:x+patch_size[1], z:z+patch_size[2], :] += pred * w # add weight probabilities for each pixel in the patch to the probability_accumulator
                    weight_accum[y:y+patch_size[0], x:x+patch_size[1], z:z+patch_size[2], :] += w # add weights for each pixel to the weight_accumulator

        # normalise by the accumulated weights
        prob_aug = prob_accum / np.clip(weight_accum, 1e-8, None) # clip the weights to a minimum value to avoid division by zero

        # unflip back to original orientation
        prob_unflipped = _flip3d(prob_aug, axes_mask)
        # print("Accumulated prob map shape after unflip:", prob_unflipped.shape)
        # for channel in range(prob_unflipped.shape[-1]):
        #     print(f"Channel {channel}: min {prob_unflipped[..., channel].min()}, max {prob_unflipped[..., channel].max()}")
        # print("Sum of probabilities per voxel (should be close to 1): min", prob_unflipped.sum(axis=-1).min(), "max", prob_unflipped.sum(axis=-1).max())
        prob_accum_all_tta += prob_unflipped

    # average across TTA variants
    # print("Accumulated prob map shape after all TTA:", prob_accum_all_tta.shape)
    # print("Sum of probabilities per voxel after all TTA before average: min", prob_accum_all_tta.sum(axis=-1).min(), "max", prob_accum_all_tta.sum(axis=-1).max())
    num_aug = 8 if tta else 1
    prob_padded = prob_accum_all_tta / float(num_aug)
    # print("Sum of probabilities per voxel after all TTA after average: min", prob_padded.sum(axis=-1).min(), "max", prob_padded.sum(axis=-1).max())

    # crop back to original size
    prob_map = _crop_from_pad(prob_padded, pads)  # [Y0,X0,Z0,C_out]
    # print("Final prob map shape after cropping:", prob_map.shape)
    # print(f"Min, median and max value in each channel of final prob map: {[(c, prob_map[..., c].min(), np.median(prob_map[..., c]), prob_map[..., c].max()) for c in range(prob_map.shape[-1])]}")
    seg = get_one_hot(np.argmax(prob_map, axis=-1).astype(np.int16), out_channels)  # [Y0,X0,Z0,C_out]
    # print("Final segmentation shape after arg-max and one-hot:", seg.shape)
    # print("Min, median and max value in each channel of final seg map: ", [(c, seg[..., c].min(), np.median(seg[..., c]), seg[..., c].max()) for c in range(seg.shape[-1])])

    return prob_map, seg

# ========== MONAI-based sliding window inference with TTA ==========

def monai_sliding_window_inference_3d(
    model,
    volume,                   # np.ndarray [batchsize,C,Z,Y,X] already preprocessed & normalised
    patch_size=(128, 128, 16),
    overlap=0.5,
    apply_softmax=True,       # set False if your model already uses softmax activation
    out_channels=None,        # number of output channels
    tta=True,                 # Whether to use test-time augmentation (flips)
    gaussian_sigma_scale=1/8, # controls how peaked the Gaussian is
    deep_supervision=True,
    sw_batch_size=1,          # batch size for sliding window (reduce if OOM)
):
    """
    MONAI-based implementation of sliding window inference with TTA support.
    Provides similar interface and functionality to sliding_window_inference_3d.
    
    Args:
        model: PyTorch model
        volume: np.ndarray [B,C,Z,Y,X]
        patch_size: Tuple[int, int, int] (Y, X, Z)
        overlap: float in [0, 1), overlap ratio for patching
        apply_softmax: bool, whether to apply softmax to outputs
        out_channels: int, number of output channels
        tta: bool, whether to use test-time augmentation
        gaussian_sigma_scale: float, Gaussian kernel scale (same as nnU-Net)
        deep_supervision: bool, whether model uses deep supervision
        sw_batch_size: int, batch size for sliding window inference
    
    Returns:
        prob_map: np.ndarray [Y,X,Z,C] softmax probabilities
        seg:      np.ndarray [Y,X,Z,C] one-hot encoded labels
    """
    from monai.inferers import sliding_window_inference as monai_swi

    # print("MONAI-based sliding window inference")
    # print("Input volume shape:", volume.shape)
    assert volume.ndim == 5 and volume.shape[1] >= 1, "Expected [B,C,Z,Y,X]"
    
    # volume is [B,C,Z,Y,X] - MONAI expects this format
    # patch_size is (Y, X, Z) - need to convert to MONAI format (Z, Y, X)
    roi_size = (patch_size[2], patch_size[0], patch_size[1])  # (Z, Y, X)
    
    device = next(model.parameters()).device
    if isinstance(volume, torch.Tensor):
        volume_t = volume.to(device).float()
    else:
        volume_t = torch.from_numpy(np.ascontiguousarray(volume)).to(device).float()
    
    # Create accumulator for TTA
    # Output will be [B, C_out, Z, Y, X]
    prob_accum_all_tta = None
    num_tta_passes = 0
    
    # TTA loop
    for axes_mask in _iter_tta_axes(tta):
        # axes_mask is (flip_Y, flip_X, flip_Z) over spatial dims of [B,C,Z,Y,X]
        # tensor dims are: Z=2, Y=3, X=4
        
        volume_aug = volume_t.clone()
        if axes_mask[0]:  # flip Y
            volume_aug = torch.flip(volume_aug, dims=[3])
        if axes_mask[1]:  # flip X
            volume_aug = torch.flip(volume_aug, dims=[4])
        if axes_mask[2]:  # flip Z
            volume_aug = torch.flip(volume_aug, dims=[2])
        
        # Define inferer that handles model output (accounting for deep supervision)
        def model_fn(x):
            with torch.no_grad():
                pred = model(x)
            # Handle deep supervision: index 0 = highest-resolution output
            if isinstance(pred, (list, tuple)) and deep_supervision:
                pred = pred[0]
            # Apply softmax if needed (MONAI doesn't do this automatically)
            if apply_softmax:
                pred = torch.softmax(pred, dim=1)
            return pred
        
        # Run MONAI sliding window inference
        with torch.no_grad():
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"Using a non-tuple sequence for multidimensional indexing is deprecated.*",
                    category=UserWarning,
                    module=r"monai\.inferers\.utils",
                )
                pred_aug = monai_swi(
                    inputs=volume_aug,
                    roi_size=roi_size,
                    sw_batch_size=sw_batch_size,
                    predictor=model_fn,
                    overlap=overlap,
                    mode='gaussian',
                    sigma_scale=gaussian_sigma_scale,
                    padding_mode='reflect',
                )  # [B, C_out, Z, Y, X]
        
        # Unflip back to original orientation (reverse the flips)
        pred_unflipped = pred_aug.clone()
        if axes_mask[0]:  # flip Y back
            pred_unflipped = torch.flip(pred_unflipped, dims=[3])
        if axes_mask[1]:  # flip X back
            pred_unflipped = torch.flip(pred_unflipped, dims=[4])
        if axes_mask[2]:  # flip Z back
            pred_unflipped = torch.flip(pred_unflipped, dims=[2])
        
        # Accumulate
        if prob_accum_all_tta is None:
            prob_accum_all_tta = pred_unflipped.clone()
        else:
            prob_accum_all_tta += pred_unflipped
        
        num_tta_passes += 1
    
    # Average across TTA variants
    prob_final = prob_accum_all_tta / float(num_tta_passes)  # [B, C_out, Z, Y, X]
    
    # Convert to numpy and reformat to [Y, X, Z, C_out]
    prob_final = prob_final[0].cpu().numpy()  # [C_out, Z, Y, X]
    prob_map = np.transpose(prob_final, (2, 3, 1, 0))  # [Y, X, Z, C_out]
    
    # print("Final prob map shape:", prob_map.shape)
    # print(f"Min, median and max value in each channel of final prob map: {[(c, prob_map[..., c].min(), np.median(prob_map[..., c]), prob_map[..., c].max()) for c in range(prob_map.shape[-1])]}")
    
    if out_channels is None:
        out_channels = prob_map.shape[-1]
    seg = get_one_hot(np.argmax(prob_map, axis=-1).astype(np.int16), out_channels)  # [Y,X,Z,C_out]
    # print("Final segmentation shape:", seg.shape)
    # print("Min, median and max value in each channel of final seg map:", [(c, seg[..., c].min(), np.median(seg[..., c]), seg[..., c].max()) for c in range(seg.shape[-1])])
    
    return prob_map, seg



# # ── Compare Custom vs MONAI sliding window inference ────────────────────────
# import time

# def compare_sliding_window_implementations(model, device, sample_paths, patch_size=(256, 256, 10),
#                                            num_segmentation_classes=5, overlap=0.5, num_timing_runs=10):
#     """
#     Comprehensive comparison of custom and MONAI sliding window inference implementations.
#     Tests both timing and output agreement.
#     """
#     sample_path = random.choice(sample_paths)  # Select a random sample from the test set
#     test_dataset = ImageDataset_withPriorMask([sample_path], cohort='test',
#                                               num_segmentation_classes=num_segmentation_classes)
#     test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)

#     model.eval()
#     X, _ = next(iter(test_loader))
#     X_device = X.to(device, non_blocking=True)
#     X_numpy = X.cpu().numpy()

#     # ──────────────────────────────────────────────────────────────────────────
#     # TIMING: Custom Implementation
#     # ──────────────────────────────────────────────────────────────────────────
#     print("\n" + "="*80)
#     print("TIMING TESTS")
#     print("="*80)
    
#     custom_times = []
#     custom_prob = None
#     custom_seg = None
    
#     print(f"\nRunning Custom sliding_window_inference_3d ({num_timing_runs} runs)...")
#     for run_idx in range(num_timing_runs):
#         if torch.cuda.is_available():
#             torch.cuda.synchronize(device)
#         start_time = time.perf_counter()
        
#         custom_prob, custom_seg = sliding_window_inference_3d(
#             model,
#             X_numpy,
#             patch_size=patch_size,
#             overlap=overlap,
#             apply_softmax=True,
#             out_channels=num_segmentation_classes,
#             tta=True,
#             gaussian_sigma_scale=1/8,
#             deep_supervision=False,
#         )
        
#         if torch.cuda.is_available():
#             torch.cuda.synchronize(device)
#         end_time = time.perf_counter()
#         custom_times.append(end_time - start_time)
#         print(f"  Run {run_idx+1}: {custom_times[-1]:.4f}s")

#     # ──────────────────────────────────────────────────────────────────────────
#     # TIMING: MONAI Implementation
#     # ──────────────────────────────────────────────────────────────────────────
#     monai_times = []
#     monai_prob = None
#     monai_seg = None
    
#     print(f"\nRunning MONAI sliding_window_inference_3d ({num_timing_runs} runs)...")
#     for run_idx in range(num_timing_runs):
#         if torch.cuda.is_available():
#             torch.cuda.synchronize(device)
#         start_time = time.perf_counter()
        
#         monai_prob, monai_seg = monai_sliding_window_inference_3d(
#             model,
#             X_device,
#             patch_size=patch_size,
#             overlap=overlap,
#             apply_softmax=True,
#             out_channels=num_segmentation_classes,
#             tta=True,
#             gaussian_sigma_scale=1/8,
#             deep_supervision=True,
#             sw_batch_size=1,
#         )
        
#         if torch.cuda.is_available():
#             torch.cuda.synchronize(device)
#         end_time = time.perf_counter()
#         monai_times.append(end_time - start_time)
#         print(f"  Run {run_idx+1}: {monai_times[-1]:.4f}s")

#     # ──────────────────────────────────────────────────────────────────────────
#     # TIMING SUMMARY
#     # ──────────────────────────────────────────────────────────────────────────
#     print("\n" + "="*80)
#     print("TIMING SUMMARY")
#     print("="*80)
    
#     custom_mean = np.mean(custom_times)
#     custom_std = np.std(custom_times)
#     monai_mean = np.mean(monai_times)
#     monai_std = np.std(monai_times)
    
#     print(f"\nCustom sliding_window_inference_3d:")
#     print(f"  Min:  {min(custom_times):.4f}s")
#     print(f"  Max:  {max(custom_times):.4f}s")
#     print(f"  Mean: {custom_mean:.4f}s ± {custom_std:.4f}s")
    
#     print(f"\nMONAI sliding_window_inference_3d:")
#     print(f"  Min:  {min(monai_times):.4f}s")
#     print(f"  Max:  {max(monai_times):.4f}s")
#     print(f"  Mean: {monai_mean:.4f}s ± {monai_std:.4f}s")
    
#     speedup_ratio = monai_mean / custom_mean
#     faster_impl = "MONAI" if speedup_ratio < 1 else "Custom"
#     speedup_pct = abs(speedup_ratio - 1) * 100
    
#     print(f"\n{faster_impl} is {speedup_pct:.1f}% {'faster' if speedup_ratio != 1 else 'the same'}")
#     print(f"Speedup ratio (MONAI/Custom): {speedup_ratio:.2f}x")

#     # ──────────────────────────────────────────────────────────────────────────
#     # OUTPUT COMPARISON
#     # ──────────────────────────────────────────────────────────────────────────
#     print("\n" + "="*80)
#     print("OUTPUT COMPARISON")
#     print("="*80)
    
#     print(f"\nCustom prob map shape:  {custom_prob.shape}")
#     print(f"MONAI prob map shape:   {monai_prob.shape}")
    
#     # Probability map differences
#     prob_diff = np.abs(custom_prob - monai_prob)
#     print(f"\nProbability Map Differences:")
#     print(f"  Max absolute difference:  {prob_diff.max():.6f}")
#     print(f"  Mean absolute difference: {prob_diff.mean():.6f}")
#     print(f"  Median absolute difference: {np.median(prob_diff):.6f}")
    
#     # Per-channel statistics
#     print(f"\n  Per-channel differences:")
#     for c in range(num_segmentation_classes):
#         c_diff = prob_diff[..., c]
#         print(f"    Channel {c}: max={c_diff.max():.6f}, mean={c_diff.mean():.6f}, median={np.median(c_diff):.6f}")
    
#     # Argmax segmentation agreement
#     custom_seg_argmax = np.argmax(custom_prob, axis=-1)  # [Y, X, Z]
#     monai_seg_argmax = np.argmax(monai_prob, axis=-1)    # [Y, X, Z]
    
#     agreement = (custom_seg_argmax == monai_seg_argmax).mean() * 100
#     disagreement_count = (custom_seg_argmax != monai_seg_argmax).sum()
#     total_voxels = custom_seg_argmax.size
    
#     print(f"\nSegmentation Agreement (argmax):")
#     print(f"  Agreement: {agreement:.2f}% ({total_voxels - disagreement_count}/{total_voxels} voxels)")
#     print(f"  Disagreement: {100 - agreement:.2f}% ({disagreement_count}/{total_voxels} voxels)")
    
#     # Class-wise agreement
#     print(f"\n  Per-class agreement:")
#     for c in range(num_segmentation_classes):
#         class_mask = (custom_seg_argmax == c) | (monai_seg_argmax == c)
#         if class_mask.sum() > 0:
#             class_agreement = (custom_seg_argmax[class_mask] == monai_seg_argmax[class_mask]).mean() * 100
#             print(f"    Class {c}: {class_agreement:.2f}%")

#     print("\n" + "="*80 + "\n")
    
#     return {
#         'custom_prob': custom_prob,
#         'custom_seg': custom_seg,
#         'monai_prob': monai_prob,
#         'monai_seg': monai_seg,
#         'custom_times': custom_times,
#         'monai_times': monai_times,
#         'agreement': agreement,
#     }


# # Run comparison on the first test scan
# print("Running comparison of sliding window implementations...")
# comparison_results = compare_sliding_window_implementations(
#     model, device, test_paths,
#     patch_size=(256, 256, 10),
#     num_segmentation_classes=5,
#     num_timing_runs=50,
# )
