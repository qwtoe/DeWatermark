"""
DeWatermark — Video Watermark Removal with ProPainter
Extreme memory optimization for low-VRAM GPUs (tested on 6 GB).
"""

import os

# Must be set before import torch to prevent OOM caused by VRAM fragmentation
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import re
import sys
import gc
import subprocess
import tempfile
import shutil
import argparse
import warnings
from pathlib import Path
from typing import Tuple, List, Optional, Dict

warnings.filterwarnings("ignore")

import numpy as np
import cv2
import torch
from tqdm import tqdm


def get_ffmpeg_path() -> str:
    """Get FFmpeg executable path, preferring system PATH, then imageio-ffmpeg built-in"""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg

    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass

    raise RuntimeError(
        "FFmpeg not found. Please do one of the following:\n"
        "  1. Install FFmpeg and add it to system PATH\n"
        "  2. pip install imageio-ffmpeg (included in install.bat)"
    )


def extract_frames(
    video_path: str,
    output_dir: str,
    quality: int = 95,
) -> Tuple[List[str], float, Tuple[int, int], int]:
    """
    Extract video frames to disk as JPEG using FFmpeg.
    Never loads the entire video into Python memory.

    Args:
        video_path: Input video path
        output_dir: Frame output directory
        quality: JPEG quality (1-100)

    Returns:
        (frame_paths, fps, (width, height), total_frames)
    """
    ffmpeg = get_ffmpeg_path()

    # First get video info
    probe_cmd = [
        ffmpeg, "-i", video_path,
        "-f", "null", "-"
    ]
    try:
        result = subprocess.run(
            probe_cmd,
            capture_output=True, text=True,
            encoding="utf-8", errors="replace"
        )
        stderr = result.stderr
    except Exception:
        raise RuntimeError(f"Cannot read video file: {video_path}")

    # Parse resolution
    res_match = re.search(r'(\d{2,5})x(\d{2,5})', stderr)
    if not res_match:
        raise RuntimeError(f"Cannot parse video resolution: {video_path}")
    width, height = int(res_match.group(1)), int(res_match.group(2))

    # Parse FPS
    fps_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:fps|FPS)', stderr)
    fps = float(fps_match.group(1)) if fps_match else 30.0

    # Parse total frames (estimated from nb_frames or duration * fps)
    nb_frames_match = re.search(r'Nb_frames:\s*(\d+)', stderr)
    if nb_frames_match:
        total_frames = int(nb_frames_match.group(1))
    else:
        dur_match = re.search(r'Duration:\s*(\d+):(\d+):(\d+\.\d+)', stderr)
        if dur_match:
            h, m, s = int(dur_match.group(1)), int(dur_match.group(2)), float(dur_match.group(3))
            duration = h * 3600 + m * 60 + s
            total_frames = int(duration * fps)
        else:
            total_frames = 0

    print(f"Video info: {width}x{height}, {fps:.2f} FPS, ~{total_frames} frames")

    # FFmpeg extract frames as JPEG
    output_pattern = os.path.join(output_dir, "frame_%06d.jpg")
    extract_cmd = [
        ffmpeg,
        "-i", video_path,
        "-q:v", str(max(1, min(31, int((100 - quality) * 31 / 100) + 1))),
        "-start_number", "0",
        "-hide_banner", "-loglevel", "error",
        "-y",
        output_pattern
    ]

    print("Extracting frames from video to disk...")
    subprocess.run(extract_cmd, check=True)

    # Collect all frame paths
    frame_paths = sorted([
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if f.endswith('.jpg')
    ])
    actual_count = len(frame_paths)
    print(f"Extraction complete: {actual_count} frames -> {output_dir}")

    return frame_paths, fps, (width, height), actual_count


def generate_mask(
    first_frame_path: str,
    mask_path: str,
    flow_mask_path: str,
    width: int,
    height: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pop up a window for the user to select the watermark region on the first frame, auto-generating Mask and Flow Mask.

    Args:
        first_frame_path: First frame JPEG path
        mask_path: Path to save the mask
        flow_mask_path: Path to save the flow_mask
        width: Video width
        height: Video height

    Returns:
        (mask, flow_mask): (H, W) binary image, watermark region=255, others=0
    """
    import scipy.ndimage

    frame = cv2.imread(first_frame_path)
    if frame is None:
        raise RuntimeError(f"Cannot read first frame: {first_frame_path}")

    print("\n" + "=" * 60)
    print("Please select the watermark region in the popup window")
    print("Instructions:")
    print("  - Drag mouse to select watermark region")
    print("  - Press SPACE/ENTER to confirm selection")
    print("  - Press ESC to skip (no processing)")
    print("  - You can select multiple watermark regions")
    print("=" * 60 + "\n")

    mask = np.zeros((height, width), dtype=np.uint8)

    while True:
        roi = cv2.selectROI("Select Watermark - SPACE=confirm, ESC=finish", frame, showCrosshair=True, fromCenter=False)

        if roi[2] == 0 or roi[3] == 0:
            break

        x, y, w, h = roi
        mask[y:y + h, x:x + w] = 255
        print(f"  Marked watermark region: x={x}, y={y}, w={w}, h={h}")

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.destroyAllWindows()

    if mask.max() == 0:
        print("\nWARNING: No watermark region selected, skipping watermark removal")
        flow_mask = mask.copy()
    else:
        # Generate mask (dilated 5 times, for inpainting)
        mask_dilated = scipy.ndimage.binary_dilation(mask > 128, iterations=5).astype(np.uint8) * 255
        cv2.imwrite(mask_path, mask_dilated)
        print(f"\nMask saved: {mask_path}")

        # Generate flow_mask (dilated 8 times, for optical flow completion, larger area ensures flow stability)
        flow_mask = scipy.ndimage.binary_dilation(mask > 128, iterations=8).astype(np.uint8) * 255
        cv2.imwrite(flow_mask_path, flow_mask)
        print(f"Flow Mask saved: {flow_mask_path}")

        mask = mask_dilated

    return mask, flow_mask


def load_or_generate_mask(
    first_frame_path: str,
    mask_path: str,
    flow_mask_path: str,
    width: int,
    height: int,
    force_regenerate: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load existing Mask/Flow Mask or generate new Mask via interactive selection.

    Args:
        first_frame_path: First frame path (needed to generate new mask)
        mask_path: Mask save/load path
        flow_mask_path: Flow Mask save/load path
        width: Video width
        height: Video height
        force_regenerate: Force regeneration

    Returns:
        (mask, flow_mask): (H, W) binary image
    """
    if os.path.exists(mask_path) and os.path.exists(flow_mask_path) and not force_regenerate:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        flow_mask = cv2.imread(flow_mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is not None and flow_mask is not None:
            print(f"Loaded existing Mask: {mask_path}")
            print(f"Loaded existing Flow Mask: {flow_mask_path}")
            return mask, flow_mask

    return generate_mask(first_frame_path, mask_path, flow_mask_path, width, height)


def get_ref_index(
    mid_neighbor_id: int,
    neighbor_ids: List[int],
    length: int,
    ref_stride: int = 10,
    ref_num: int = -1,
) -> List[int]:
    """
    Select reference frame indices to provide global context for the ProPainter Transformer.

    Reference frames are frames outside the current neighborhood, used to help the model understand global scene information.

    Args:
        mid_neighbor_id: Center frame index of the current neighborhood
        neighbor_ids: List of frame indices in the current neighborhood
        length: Total number of video frames
        ref_stride: Reference frame stride
        ref_num: Maximum number of reference frames (-1 means unlimited)

    Returns:
        List of reference frame indices
    """
    ref_index = []
    neighbor_set = set(neighbor_ids)

    if ref_num == -1:
        for i in range(0, length, ref_stride):
            if i not in neighbor_set:
                ref_index.append(i)
    else:
        start_idx = max(0, mid_neighbor_id - ref_stride * (ref_num // 2))
        end_idx = min(length, mid_neighbor_id + ref_stride * (ref_num // 2))
        for i in range(start_idx, end_idx, ref_stride):
            if i not in neighbor_set:
                if len(ref_index) >= ref_num:
                    break
                ref_index.append(i)

    return ref_index


# ============================================================
# ProPainter model loading module
# ============================================================

PROPAINTER_REPO_URL = "https://github.com/sczhou/ProPainter.git"
PROPAINTER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ProPainter")
PRETRAIN_URL_BASE = "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/"

WEIGHT_FILES = {
    "raft-things.pth": "RAFT optical flow model",
    "recurrent_flow_completion.pth": "Recurrent flow completion model",
    "ProPainter.pth": "ProPainter inpainting model",
}


def ensure_propainter() -> str:
    """
    Ensure ProPainter source code exists, if not, auto git clone.
    Returns the ProPainter directory path.
    """
    if os.path.exists(os.path.join(PROPAINTER_DIR, "inference_propainter.py")):
        print(f"ProPainter source already exists: {PROPAINTER_DIR}")
        return PROPAINTER_DIR

    # Directory exists but is incomplete (previous clone failed), delete and retry
    if os.path.exists(PROPAINTER_DIR):
        print(f"Detected incomplete ProPainter directory, cleaning up: {PROPAINTER_DIR}")
        shutil.rmtree(PROPAINTER_DIR, ignore_errors=True)

    print(f"First run, downloading ProPainter source...")
    print(f"Source: {PROPAINTER_REPO_URL}")

    git_path = shutil.which("git")
    if git_path:
        cmd = [git_path, "clone", "--depth", "1", PROPAINTER_REPO_URL, PROPAINTER_DIR]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True, encoding="utf-8", errors="replace")
            print("ProPainter source download complete.")
            return PROPAINTER_DIR
        except subprocess.CalledProcessError as e:
            print(f"Git clone failed: {e.stderr}")

    raise RuntimeError(
        "Cannot download ProPainter source. Please run manually:\n"
        f"  git clone --depth 1 {PROPAINTER_REPO_URL} \"{PROPAINTER_DIR}\""
    )


def download_weights(weights_dir: str, force: bool = False) -> dict:
    """
    Download ProPainter pretrained weights.
    Returns {name: path} dict.
    """
    import requests

    os.makedirs(weights_dir, exist_ok=True)
    weight_paths = {}

    for fname, desc in WEIGHT_FILES.items():
        local_path = os.path.join(weights_dir, fname)
        if os.path.exists(local_path) and not force:
            print(f"  [Exists] {fname} ({desc})")
            weight_paths[fname] = local_path
            continue

        url = PRETRAIN_URL_BASE + fname
        print(f"  Downloading {fname} ({desc})...")
        print(f"    {url}")

        try:
            resp = requests.get(url, stream=True, timeout=300)
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))

            with open(local_path, "wb") as f:
                with tqdm(total=total, unit="B", unit_scale=True, desc=f"    {fname}") as pbar:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))

            weight_paths[fname] = local_path
            print(f"    {fname} download complete.")
        except Exception as e:
            print(f"    Download {fname} failed: {e}")
            if os.path.exists(local_path):
                os.remove(local_path)
            raise RuntimeError(f"Weight download failed: {fname}\nPlease manually download to {weights_dir}/")

    return weight_paths


def load_propainter_models(
    weights_dir: str = "weights",
    use_fp16: bool = True,
    device: str = "cuda",
):
    """
    Load the three core ProPainter models with FP16 half-precision support.

    Returns:
        (fix_raft, fix_flow_complete, model):
            - fix_raft: RAFT_bi optical flow model (FP32, RAFT is more stable with FP32)
            - fix_flow_complete: RecurrentFlowCompleteNet
            - model: InpaintGenerator (ProPainter inpainting network)
    """
    # Ensure ProPainter source is available
    propainter_root = ensure_propainter()
    if propainter_root not in sys.path:
        sys.path.insert(0, propainter_root)

    # Download weights
    weights_dir = os.path.join(PROPAINTER_DIR, "weights") if weights_dir == "weights" else weights_dir
    weight_paths = download_weights(weights_dir)

    device_obj = torch.device(device)
    print(f"\nLoading ProPainter models to {device_obj}...")

    # 1. RAFT optical flow model (keep FP32, flow precision is sensitive)
    print("  [1/3] Loading RAFT optical flow model...")
    from model.modules.flow_comp_raft import RAFT_bi
    fix_raft = RAFT_bi(weight_paths["raft-things.pth"], device_obj)
    fix_raft.eval()
    print(f"    RAFT loaded (FP32)")

    # 2. Recurrent flow completion network
    print("  [2/3] Loading recurrent flow completion network...")
    from model.recurrent_flow_completion import RecurrentFlowCompleteNet
    fix_flow_complete = RecurrentFlowCompleteNet(weight_paths["recurrent_flow_completion.pth"])
    for p in fix_flow_complete.parameters():
        p.requires_grad = False
    fix_flow_complete.to(device_obj)
    fix_flow_complete.eval()

    # 3. ProPainter inpainting network
    print("  [3/3] Loading ProPainter inpainting network...")
    from model.propainter import InpaintGenerator
    model = InpaintGenerator(model_path=weight_paths["ProPainter.pth"]).to(device_obj)
    model.eval()

    # FP16 conversion (except RAFT)
    if use_fp16 and device != "cpu":
        print("  Enabling FP16 half-precision...")
        fix_flow_complete = fix_flow_complete.half()
        model = model.half()
    else:
        print("  Using FP32 full precision")

    torch.cuda.empty_cache()
    print(f"  Model loading complete.")

    return fix_raft, fix_flow_complete, model


def preprocess_frames_for_propainter(
    frame_pils: list,
    mask: np.ndarray,
    flow_mask: np.ndarray,
    resize_to: Optional[int] = None,
    num_local_frames: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[int, int], Tuple[int, int]]:
    """
    Preprocess from PIL image list to ProPainter input format.

    Supports reference frames: first num_local_frames are local frames (with watermark mask),
    the rest are reference frames (mask is zero, meaning no watermark).

    Args:
        frame_pils: List of PIL Images (local frames + reference frames)
        mask: (H, W) binary mask (255=watermark region) for local frames
        flow_mask: (H, W) binary flow_mask (255=watermark region, more dilated) for optical flow completion
        resize_to: Optional, short side size to resize to
        num_local_frames: Number of local frames, the rest are reference frames. None means all are local frames

    Returns:
        (frames_tensor, mask_tensor, flow_mask_tensor, original_size, process_size)
    """
    from PIL import Image
    from core.utils import to_tensors as to_tensors_fn

    if num_local_frames is None:
        num_local_frames = len(frame_pils)

    # Compute processing size
    orig_w, orig_h = frame_pils[0].size
    original_size = (orig_w, orig_h)

    if resize_to is not None:
        scale = resize_to / min(orig_h, orig_w)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        new_w = new_w - new_w % 8
        new_h = new_h - new_h % 8
        process_size = (new_w, new_h)
    else:
        new_w = orig_w - orig_w % 8
        new_h = orig_h - orig_h % 8
        process_size = (new_w, new_h)

    # Resize frames
    resized_frames = []
    for f in frame_pils:
        if process_size != (orig_w, orig_h):
            f = f.resize(process_size, Image.LANCZOS)
        resized_frames.append(f)

    # Resize mask and flow_mask
    mask_img = Image.fromarray(mask)
    flow_mask_img = Image.fromarray(flow_mask)
    if process_size != (orig_w, orig_h):
        mask_img = mask_img.resize(process_size, Image.NEAREST)
        flow_mask_img = flow_mask_img.resize(process_size, Image.NEAREST)
    mask_np = np.array(mask_img)
    flow_mask_np = np.array(flow_mask_img)

    # Process mask: binarize (mask already dilated in generate_mask)
    mask_binary = (mask_np > 128).astype(np.uint8) * 255
    flow_mask_binary = (flow_mask_np > 128).astype(np.uint8) * 255

    # Convert to tensor
    frames_t = to_tensors_fn()(resized_frames).unsqueeze(0) * 2.0 - 1.0  # (1,T,3,H,W)

    # Create mask tensor: local frames use watermark mask, reference frames use zero mask
    mask_imgs = []
    for i in range(len(resized_frames)):
        if i < num_local_frames:
            mask_imgs.append(Image.fromarray(mask_binary))
        else:
            # Reference frames: mask is all zeros (no watermark region)
            mask_imgs.append(Image.fromarray(np.zeros_like(mask_binary)))
    mask_t = to_tensors_fn()(mask_imgs).unsqueeze(0)  # (1,T,1,H,W)

    # Create flow_mask tensor: only used for optical flow completion of local frames
    flow_mask_imgs = [Image.fromarray(flow_mask_binary)] * num_local_frames
    flow_mask_t = to_tensors_fn()(flow_mask_imgs).unsqueeze(0)  # (1,T_local,1,H,W)

    return frames_t, mask_t, flow_mask_t, original_size, process_size


def precompute_all_flows(
    frame_paths: List[str],
    mask: np.ndarray,
    flow_mask: np.ndarray,
    resize_to: Optional[int],
    fix_raft,
    fix_flow_complete,
    raft_iter: int = 20,
    use_fp16: bool = True,
    device: str = "cuda",
    chunk_size: int = 5,
) -> List[Tuple[int, torch.Tensor, torch.Tensor]]:
    """
    Precompute all optical flows for the entire video and save to CPU memory.
    Returns [(start_idx, pred_flows_f_cpu, pred_flows_b_cpu), ...] list,
    each tuple corresponds to chunk_size flow pairs (last chunk may be smaller).

    Each chunk processes (chunk_size + 1) input frames to produce chunk_size flow pairs,
    chunks overlap by 1 frame to ensure cross-boundary flows are also available.

    Args:
        frame_paths: List of all frame paths
        mask: (H, W) binary mask (only for preprocessing interface compatibility, actually only uses flow_mask)
        flow_mask: (H, W) binary flow_mask
        resize_to: Optional downscale resolution
        fix_raft: RAFT optical flow model
        fix_flow_complete: Recurrent flow completion model
        raft_iter: RAFT iteration count
        use_fp16: Half precision
        device: Device
        chunk_size: Number of flow pairs per batch
    """
    from PIL import Image

    total_frames = len(frame_paths)
    device_obj = torch.device(device)
    dummy_mask = np.zeros_like(flow_mask)  # flow precomputation does not need mask

    print(f"\nPrecomputing optical flow: {total_frames} frames, {chunk_size} flow pairs per batch")
    flows_list: List[Tuple[int, torch.Tensor, torch.Tensor]] = []
    pbar = tqdm(total=max(0, total_frames - 1), desc="Flow precomputation", unit="flow")

    for start_idx in range(0, total_frames - 1, chunk_size):
        # Each chunk takes (chunk_size + 1) frames -> chunk_size flow pairs
        end_idx = min(start_idx + chunk_size + 1, total_frames)
        actual_num_flows = end_idx - start_idx - 1
        if actual_num_flows <= 0:
            break

        neighbor_ids = list(range(start_idx, end_idx))
        l_t = len(neighbor_ids)

        # Load frames
        local_pils = [Image.open(frame_paths[i]).convert("RGB") for i in neighbor_ids]

        # Preprocessing (only needs frames_tensor and flow_mask_tensor)
        frames_tensor, _, flow_mask_tensor, _, _ = \
            preprocess_frames_for_propainter(
                local_pils, dummy_mask, flow_mask, resize_to, num_local_frames=l_t
            )
        frames_tensor = frames_tensor.to(device_obj)
        flow_mask_tensor = flow_mask_tensor.to(device_obj)
        if use_fp16:
            frames_tensor = frames_tensor.half()
            flow_mask_tensor = flow_mask_tensor.half()

        local_frames = frames_tensor[:, :l_t, :, :, :]

        # ---- Compute optical flow + completion ----
        try:
            with torch.inference_mode():
                video_length = local_frames.size(1)
                frame_size = local_frames.size(-1)

                if frame_size <= 640:
                    short_clip_len = 12
                elif frame_size <= 720:
                    short_clip_len = 8
                elif frame_size <= 1280:
                    short_clip_len = 4
                else:
                    short_clip_len = 2

                raft_input = local_frames.float()

                if video_length > short_clip_len:
                    gt_flows_f_list, gt_flows_b_list = [], []
                    for f in range(0, video_length, short_clip_len):
                        end_f = min(video_length, f + short_clip_len)
                        if f == 0:
                            flows_f, flows_b = fix_raft(raft_input[:, f:end_f], iters=raft_iter)
                        else:
                            flows_f, flows_b = fix_raft(raft_input[:, f - 1:end_f], iters=raft_iter)
                        gt_flows_f_list.append(flows_f)
                        gt_flows_b_list.append(flows_b)
                        del flows_f, flows_b
                        torch.cuda.empty_cache()

                    gt_flows_f = torch.cat(gt_flows_f_list, dim=1)
                    gt_flows_b = torch.cat(gt_flows_b_list, dim=1)
                    del gt_flows_f_list, gt_flows_b_list
                    gt_flows_bi = (gt_flows_f, gt_flows_b)
                else:
                    gt_flows_bi = fix_raft(raft_input, iters=raft_iter)
                    torch.cuda.empty_cache()

                del raft_input
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

                if use_fp16:
                    gt_flows_bi = (gt_flows_bi[0].half(), gt_flows_bi[1].half())
                    torch.cuda.empty_cache()

                # Flow completion
                subvideo_length = 80
                flow_length = gt_flows_bi[0].size(1)

                if flow_length > subvideo_length:
                    pred_flows_f, pred_flows_b = [], []
                    pad_len = 5
                    for f in range(0, flow_length, subvideo_length):
                        s_f = max(0, f - pad_len)
                        e_f = min(flow_length, f + subvideo_length + pad_len)
                        pad_len_s = max(0, f) - s_f
                        pad_len_e = e_f - min(flow_length, f + subvideo_length)

                        pred_flows_bi_sub, _ = fix_flow_complete.forward_bidirect_flow(
                            (gt_flows_bi[0][:, s_f:e_f], gt_flows_bi[1][:, s_f:e_f]),
                            flow_mask_tensor[:, s_f:e_f + 1]
                        )
                        pred_flows_bi_sub = fix_flow_complete.combine_flow(
                            (gt_flows_bi[0][:, s_f:e_f], gt_flows_bi[1][:, s_f:e_f]),
                            pred_flows_bi_sub,
                            flow_mask_tensor[:, s_f:e_f + 1]
                        )
                        pred_flows_f.append(pred_flows_bi_sub[0][:, pad_len_s:e_f - s_f - pad_len_e])
                        pred_flows_b.append(pred_flows_bi_sub[1][:, pad_len_s:e_f - s_f - pad_len_e])
                        torch.cuda.empty_cache()

                    pred_flows_f = torch.cat(pred_flows_f, dim=1)
                    pred_flows_b = torch.cat(pred_flows_b, dim=1)
                else:
                    pred_flows_bi, _ = fix_flow_complete.forward_bidirect_flow(gt_flows_bi, flow_mask_tensor)
                    pred_flows_bi = fix_flow_complete.combine_flow(gt_flows_bi, pred_flows_bi, flow_mask_tensor)
                    pred_flows_f = pred_flows_bi[0]
                    pred_flows_b = pred_flows_bi[1]
                    torch.cuda.empty_cache()

                del gt_flows_bi
                torch.cuda.empty_cache()

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"\n  [OOM] Flow precomputation out of VRAM, cleaning up and retrying...")
                _cleanup_memory(verbose=True)
                # Single retry
                with torch.inference_mode():
                    raft_input = local_frames.float()
                    gt_flows_bi = fix_raft(raft_input, iters=raft_iter)
                    del raft_input
                    torch.cuda.empty_cache()
                    if use_fp16:
                        gt_flows_bi = (gt_flows_bi[0].half(), gt_flows_bi[1].half())
                    pred_flows_bi, _ = fix_flow_complete.forward_bidirect_flow(gt_flows_bi, flow_mask_tensor)
                    pred_flows_bi = fix_flow_complete.combine_flow(gt_flows_bi, pred_flows_bi, flow_mask_tensor)
                    pred_flows_f = pred_flows_bi[0]
                    pred_flows_b = pred_flows_bi[1]
                    del gt_flows_bi
                    torch.cuda.empty_cache()
            else:
                raise

        # Move to CPU and truncate to actual flow count
        pred_flows_f_cpu = pred_flows_f[:, :actual_num_flows].cpu()
        pred_flows_b_cpu = pred_flows_b[:, :actual_num_flows].cpu()
        del pred_flows_f, pred_flows_b
        torch.cuda.empty_cache()

        flows_list.append((start_idx, pred_flows_f_cpu, pred_flows_b_cpu))

        # Cleanup
        del frames_tensor, flow_mask_tensor, local_frames, local_pils
        _cleanup_memory(verbose=(start_idx % (chunk_size * 10) == 0))

        pbar.update(actual_num_flows)

    pbar.close()
    print(f"Flow precomputation complete: {len(flows_list)} chunks")
    return flows_list


def _get_flows_for_window(
    window_start: int,
    window_end: int,
    precomputed_flows: List[Tuple[int, torch.Tensor, torch.Tensor]],
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Extract flows for the specified window [window_start, window_end) from precomputed flow list.
    Returns (flows_f, flows_b) CPU tensor, or (None, None) if not found.

    A window with T frames needs T-1 flow pairs (indices from window_start to window_end-2).
    """
    need_start = window_start
    need_end = window_end - 1  # index of last flow pair (exclusive)
    if need_end <= need_start:
        return None, None

    flows_f_parts: List[torch.Tensor] = []
    flows_b_parts: List[torch.Tensor] = []

    for chunk_start, chunk_flows_f, chunk_flows_b in precomputed_flows:
        num_flows = chunk_flows_f.size(1)
        chunk_end = chunk_start + num_flows  # this chunk covers flow index range [start, end)

        # Check if there is an intersection
        if chunk_end <= need_start or chunk_start >= need_end:
            continue

        part_start = max(need_start, chunk_start)
        part_end = min(need_end, chunk_end)
        offset = part_start - chunk_start
        length = part_end - part_start

        flows_f_parts.append(chunk_flows_f[:, offset:offset + length])
        flows_b_parts.append(chunk_flows_b[:, offset:offset + length])

    if not flows_f_parts:
        return None, None

    return torch.cat(flows_f_parts, dim=1), torch.cat(flows_b_parts, dim=1)


def _cleanup_memory(verbose: bool = False):
    """Triple memory cleanup: del references + gc.collect() + CUDA cache clear"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    if verbose:
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"  [Memory cleanup] Allocated: {alloc:.2f}GB, Reserved: {reserved:.2f}GB")


def process_chunks(
    frame_paths: List[str],
    mask: np.ndarray,
    flow_mask: np.ndarray,
    output_dir: str,
    model,
    precomputed_flows: List[Tuple[int, torch.Tensor, torch.Tensor]],
    chunk_size: int = 5,
    resize_to: Optional[int] = None,
    use_fp16: bool = True,
    device: str = "cuda",
    neighbor_length: Optional[int] = None,
    ref_stride: int = 10,
    max_ref_frames: int = 0,
) -> None:
    """
    Sliding window inference — core processing function.

    Uses precomputed optical flow + sliding window + reference frame strategy:
    1. Flows are precomputed by precompute_all_flows() and stored in CPU memory
    2. Each window contains neighbor_length local frames + several reference frames
    3. Model output only uses inpainting results in the mask region, non-mask regions retain original pixels
    4. Overlap frames are averaged by immediate read+write to disk (no giant frame_accum array)

    Args:
        frame_paths: List of all frame paths
        mask: (H, W) binary mask (255=watermark region, dilated 5 times)
        flow_mask: (H, W) binary flow_mask (255=watermark region, dilated 8 times)
        output_dir: Output frame directory
        model: ProPainter inpainting model
        precomputed_flows: Precomputed flow list from precompute_all_flows()
        chunk_size: Frames per batch (default 5, reduce to 3 when VRAM is tight)
        resize_to: Downscale resolution (e.g. 480)
        use_fp16: Half precision
        device: Device
        neighbor_length: Sliding window size (default=chunk_size*2)
        ref_stride: Reference frame stride
        max_ref_frames: Maximum reference frames (0=disable ref frames)
    """
    from PIL import Image

    total_frames = len(frame_paths)
    device_obj = torch.device(device)
    use_cuda = device_obj.type == "cuda"
    accum_dtype = np.float32

    if neighbor_length is None:
        neighbor_length = min(chunk_size * 2, total_frames)
    neighbor_stride = max(1, neighbor_length // 2)

    # Reference frame count: max_ref_frames=0 means disable reference frames
    use_ref_frames = max_ref_frames > 0

    print(f"\n{'=' * 60}")
    print(f"Sliding window inference: {total_frames} total frames")
    print(f"  Window size: {neighbor_length}, Stride: {neighbor_stride}")
    if use_ref_frames:
        print(f"  Reference frames: max {max_ref_frames}, stride {ref_stride}")
    else:
        print(f"  Reference frames: disabled (saves VRAM)")
    if resize_to:
        print(f"  Downscale resolution: {resize_to}p (restored after inference)")
    print(f"  FP16: {use_fp16}, GPU: {use_cuda}")
    print(f"{'=' * 60}")

    os.makedirs(output_dir, exist_ok=True)

    # Preload reference frames (PIL Image) — only when reference frames are enabled
    ref_frames_pil = {}
    if use_ref_frames:
        ref_indices = get_ref_index(0, [], total_frames, ref_stride=ref_stride)
        if len(ref_indices) > max_ref_frames:
            step = len(ref_indices) / max_ref_frames
            ref_indices = [ref_indices[int(i * step)] for i in range(max_ref_frames)]

        for idx in ref_indices:
            try:
                ref_frames_pil[idx] = Image.open(frame_paths[idx]).convert("RGB")
            except Exception as e:
                print(f"  WARNING: Cannot load reference frame {idx}: {e}")

    # Prepare mask (original resolution, 3 channels, for blending)
    mask_3ch = np.stack([mask] * 3, axis=-1).astype(accum_dtype) / 255.0  # (H, W, 3)

    # frame_count dict tracks how many times each frame was written (for overlap averaging)
    frame_count: Dict[int, int] = {}

    pbar = tqdm(total=total_frames, desc="Inference progress", unit="frame")

    for window_start in range(0, total_frames, neighbor_stride):
        window_end = min(window_start + neighbor_length, total_frames)
        neighbor_ids = list(range(window_start, window_end))

        # Select reference frames
        ref_ids = []
        if use_ref_frames:
            mid_id = (window_start + window_end) // 2
            ref_ids = get_ref_index(mid_id, neighbor_ids, total_frames, ref_stride=ref_stride)
            ref_ids = [i for i in ref_ids if i in ref_frames_pil]

        # Local frames + reference frames
        local_pils = [Image.open(frame_paths[i]).convert("RGB") for i in neighbor_ids]
        ref_pils = [ref_frames_pil[i] for i in ref_ids] if use_ref_frames else []
        all_pils = local_pils + ref_pils
        l_t = len(local_pils)

        # ====== Step 1: Preprocessing ======
        frames_tensor, mask_tensor, _, original_size, process_size = \
            preprocess_frames_for_propainter(
                all_pils, mask, flow_mask, resize_to, num_local_frames=l_t
            )

        frames_tensor = frames_tensor.to(device_obj)
        mask_tensor = mask_tensor.to(device_obj)

        if use_fp16:
            frames_tensor = frames_tensor.half()
            mask_tensor = mask_tensor.half()

        local_frames_tensor = frames_tensor[:, :l_t, :, :, :]
        local_mask = mask_tensor[:, :l_t, :, :]

        # ====== Step 2: Use precomputed optical flow ======
        pred_flows_f_cpu, pred_flows_b_cpu = _get_flows_for_window(
            window_start, window_end, precomputed_flows
        )
        if pred_flows_f_cpu is None:
            print(f"\n  WARNING: Window [{window_start}, {window_end}) has no precomputed flow, skipping")
            for i, idx in enumerate(neighbor_ids):
                _write_or_average_frame(frame_paths, output_dir, idx, None, mask_3ch, frame_count)
            pbar.update(len(neighbor_ids))
            continue

        # ====== Step 3: Image propagation + Transformer inference ======
        try:
            pred_flows_f = pred_flows_f_cpu.to(device_obj)
            pred_flows_b = pred_flows_b_cpu.to(device_obj)
            if use_fp16:
                pred_flows_f = pred_flows_f.half()
                pred_flows_b = pred_flows_b.half()

            masked_frames = local_frames_tensor * (1 - local_mask)

            with torch.inference_mode():
                updated_frames, updated_masks = model.img_propagation(
                    masked_frames, (pred_flows_f, pred_flows_b), local_mask, "nearest"
                )
                torch.cuda.empty_cache()

                updated_frames = local_frames_tensor * (1 - local_mask) + \
                                 updated_frames.view(local_frames_tensor.shape) * local_mask

            # ====== Step 4: Transformer inference ======
            if use_ref_frames and len(ref_ids) > 0:
                ref_masks = mask_tensor[:, l_t:, :, :]
                all_updated_masks = torch.cat([updated_masks, ref_masks], dim=1)
                # Concatenate propagated local frames with original reference frames
                full_updated_frames = torch.cat([updated_frames, frames_tensor[:, l_t:]], dim=1)
            else:
                all_updated_masks = updated_masks
                full_updated_frames = updated_frames

            with torch.inference_mode():
                comp_frames = model(
                    masked_frames=full_updated_frames,
                    completed_flows=(pred_flows_f, pred_flows_b),
                    masks_in=mask_tensor,
                    masks_updated=all_updated_masks,
                    num_local_frames=l_t,
                )
                torch.cuda.empty_cache()

            # ====== Step 5: Post-processing ======
            result_np = comp_frames.squeeze(0).float().cpu().numpy()
            result_np = (result_np + 1.0) / 2.0
            result_np = np.clip(result_np * 255, 0, 255).astype(np.uint8)
            result_np = result_np.transpose(0, 2, 3, 1)

            if process_size != original_size:
                resized_results = []
                for i in range(result_np.shape[0]):
                    img = cv2.resize(result_np[i], original_size, interpolation=cv2.INTER_LANCZOS4)
                    resized_results.append(img)
                result_np = np.stack(resized_results)

            # Blend and write to disk
            for i, idx in enumerate(neighbor_ids):
                orig_frame = cv2.imread(frame_paths[idx])
                if orig_frame is None:
                    print(f"  WARNING: Cannot read frame {idx}")
                    continue
                orig_frame = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2RGB)

                blended = result_np[i].astype(accum_dtype) * mask_3ch + \
                          orig_frame.astype(accum_dtype) * (1 - mask_3ch)
                blended = blended.astype(np.uint8)

                _write_or_average_frame(frame_paths, output_dir, idx, blended, mask_3ch, frame_count)

            # Cleanup
            del pred_flows_f, pred_flows_b
            del comp_frames, result_np
            del updated_frames, updated_masks, all_updated_masks, masked_frames

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"\n  [OOM] Inference out of VRAM, cleaning up and retrying...")
                _cleanup_memory(verbose=True)

                try:
                    # Retry: re-fetch flows + re-run inference
                    pred_flows_f = pred_flows_f_cpu.to(device_obj)
                    pred_flows_b = pred_flows_b_cpu.to(device_obj)
                    if use_fp16:
                        pred_flows_f = pred_flows_f.half()
                        pred_flows_b = pred_flows_b.half()

                    masked_frames = local_frames_tensor * (1 - local_mask)
                    with torch.inference_mode():
                        updated_frames, updated_masks = model.img_propagation(
                            masked_frames, (pred_flows_f, pred_flows_b), local_mask, "nearest"
                        )
                        torch.cuda.empty_cache()
                        updated_frames = local_frames_tensor * (1 - local_mask) + \
                                         updated_frames.view(local_frames_tensor.shape) * local_mask

                    if use_ref_frames and len(ref_ids) > 0:
                        ref_masks = mask_tensor[:, l_t:, :, :]
                        all_updated_masks = torch.cat([updated_masks, ref_masks], dim=1)
                        # Concatenate propagated local frames with original reference frames
                        full_updated_frames = torch.cat([updated_frames, frames_tensor[:, l_t:]], dim=1)
                    else:
                        all_updated_masks = updated_masks
                        full_updated_frames = updated_frames

                    with torch.inference_mode():
                        comp_frames = model(
                            masked_frames=full_updated_frames,
                            completed_flows=(pred_flows_f, pred_flows_b),
                            masks_in=mask_tensor,
                            masks_updated=all_updated_masks,
                            num_local_frames=l_t,
                        )
                        torch.cuda.empty_cache()

                    result_np = comp_frames.squeeze(0).float().cpu().numpy()
                    result_np = (result_np + 1.0) / 2.0
                    result_np = np.clip(result_np * 255, 0, 255).astype(np.uint8)
                    result_np = result_np.transpose(0, 2, 3, 1)

                    if process_size != original_size:
                        resized_results = []
                        for i in range(result_np.shape[0]):
                            img = cv2.resize(result_np[i], original_size, interpolation=cv2.INTER_LANCZOS4)
                            resized_results.append(img)
                        result_np = np.stack(resized_results)

                    for i, idx in enumerate(neighbor_ids):
                        orig_frame = cv2.imread(frame_paths[idx])
                        if orig_frame is None:
                            continue
                        orig_frame = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2RGB)
                        blended = result_np[i].astype(accum_dtype) * mask_3ch + \
                                  orig_frame.astype(accum_dtype) * (1 - mask_3ch)
                        blended = blended.astype(np.uint8)
                        _write_or_average_frame(frame_paths, output_dir, idx, blended, mask_3ch, frame_count)

                    del pred_flows_f, pred_flows_b, comp_frames, result_np
                    del updated_frames, updated_masks, all_updated_masks, masked_frames

                except RuntimeError as e2:
                    if "out of memory" in str(e2).lower():
                        print(f"\n  [OOM] Retry still failed, using original frame fallback for window [{window_start}, {window_end})")
                        _cleanup_memory(verbose=True)
                        for i, idx in enumerate(neighbor_ids):
                            _write_or_average_frame(frame_paths, output_dir, idx, None, mask_3ch, frame_count)
                    else:
                        raise
            else:
                raise

        # ====== Memory cleanup (end of window) ======
        del frames_tensor, mask_tensor
        del local_frames_tensor, local_mask
        del local_pils, ref_pils, all_pils
        _cleanup_memory(verbose=(window_start % (neighbor_stride * 10) == 0))

        pbar.update(len(neighbor_ids))

    pbar.close()

    # Ensure all frames are written (unprocessed frames use originals)
    for idx in range(total_frames):
        if idx not in frame_count:
            orig = cv2.imread(frame_paths[idx])
            if orig is not None:
                out_path = os.path.join(output_dir, f"frame_{idx:06d}.jpg")
                cv2.imwrite(out_path, orig)

    print(f"\nInference complete! Output frames: {output_dir}")
    _cleanup_memory(verbose=True)


def _write_or_average_frame(
    frame_paths: List[str],
    output_dir: str,
    idx: int,
    blended: Optional[np.ndarray],
    mask_3ch: np.ndarray,
    frame_count: Dict[int, int],
) -> None:
    """
    Write blended frame to disk. If it is an overlap frame (already exists), read the existing frame and average before writing back.
    If blended is None (OOM fallback), use the original frame directly.
    """
    out_path = os.path.join(output_dir, f"frame_{idx:06d}.jpg")

    if blended is None:
        # OOM fallback: use original frame
        orig = cv2.imread(frame_paths[idx])
        if orig is None:
            return
        cv2.imwrite(out_path, orig)
        frame_count[idx] = frame_count.get(idx, 0) + 1
        return

    if idx not in frame_count:
        # First write
        cv2.imwrite(out_path, cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
        frame_count[idx] = 1
    else:
        # Overlap frame: read existing, weighted average, write back
        existing = cv2.imread(out_path)
        if existing is None:
            cv2.imwrite(out_path, cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
            frame_count[idx] = 1
            return
        existing_rgb = cv2.cvtColor(existing, cv2.COLOR_BGR2RGB)
        n = frame_count[idx]
        averaged = ((existing_rgb.astype(np.float32) * n + blended.astype(np.float32)) / (n + 1)).astype(np.uint8)
        cv2.imwrite(out_path, cv2.cvtColor(averaged, cv2.COLOR_RGB2BGR))
        frame_count[idx] = n + 1


def reconstruct_video(
    frame_dir: str,
    output_path: str,
    fps: float,
    crf: int = 18,
    input_video_path: Optional[str] = None,
) -> None:
    """
    Use FFmpeg to merge frame sequence into video, optionally copy audio track from original video.

    Args:
        frame_dir: Frame directory
        output_path: Output video path
        fps: Frame rate
        crf: Video quality (lower is better, recommended 18-23)
        input_video_path: Original video path (for copying audio track)
    """
    ffmpeg = get_ffmpeg_path()
    input_pattern = os.path.join(frame_dir, "frame_%06d.jpg")

    # First generate video without audio
    temp_output = output_path + ".tmp.mp4"
    cmd = [
        ffmpeg,
        "-framerate", str(fps),
        "-i", input_pattern,
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-hide_banner", "-loglevel", "error",
        "-y",
        temp_output
    ]

    print(f"Merging frames into video -> {output_path}")
    subprocess.run(cmd, check=True)

    # If original video exists, try to copy audio track
    if input_video_path and os.path.exists(input_video_path):
        # Check if original video has audio track
        probe_cmd = [ffmpeg, "-i", input_video_path, "-f", "null", "-"]
        try:
            probe_result = subprocess.run(
                probe_cmd, capture_output=True, text=True,
                encoding="utf-8", errors="replace"
            )
            # Check if there is an audio stream
            if "Audio:" in probe_result.stderr:
                print("Original video audio track detected, merging...")
                merge_cmd = [
                    ffmpeg,
                    "-i", temp_output,
                    "-i", input_video_path,
                    "-c:v", "copy",
                    "-c:a", "aac",
                    "-map", "0:v:0",
                    "-map", "1:a:0?",
                    "-shortest",
                    "-hide_banner", "-loglevel", "error",
                    "-y",
                    output_path
                ]
                subprocess.run(merge_cmd, check=True)
                os.remove(temp_output)
                print("Audio track merge complete.")
            else:
                # No audio track, just rename
                os.rename(temp_output, output_path)
                print("Original video has no audio track, video stream only.")
        except subprocess.CalledProcessError:
            # Merge failed, use video without audio
            os.rename(temp_output, output_path)
            print("Audio track merge failed, video stream only.")
    else:
        os.rename(temp_output, output_path)
        print("Video generation complete (no audio track).")


def cleanup_temp(*dirs: str) -> None:
    """Safely delete temp directories"""
    for d in dirs:
        if d and os.path.exists(d):
            try:
                shutil.rmtree(d)
                print(f"  Cleaned: {d}")
            except Exception as e:
                print(f"  Cleanup failed {d}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="DeWatermark — Video Watermark Removal with ProPainter (VRAM ≥ 4 GB)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:
  # Basic usage (auto downscale based on VRAM and resolution)
  python remove_watermark.py -i video.mp4 -o output.mp4

  # Manually specify downscale resolution
  python remove_watermark.py -i video.mp4 -o output.mp4 --resize 480

  # No downscale (when VRAM is sufficient)
  python remove_watermark.py -i video.mp4 -o output.mp4 --resize none

  # Limit VRAM budget to 4GB (auto more conservative)
  python remove_watermark.py -i video.mp4 -o output.mp4 --vram-budget 4

  # Further reduce frames per batch
  python remove_watermark.py -i video.mp4 -o output.mp4 --resize 480 --chunk-size 3

  # Use existing mask (skip selection)
  python remove_watermark.py -i video.mp4 -o output.mp4 --mask mask.png
        """
    )

    # Core parameters
    parser.add_argument("-i", "--input", required=True, help="Input video path")
    parser.add_argument("-o", "--output", default="output_clean.mp4", help="Output video path (default: output_clean.mp4)")
    parser.add_argument("-m", "--mask", default="mask.png", help="Mask path (default: mask.png, interactive selection if not exists)")
    parser.add_argument("--flow-mask", default="flow_mask.png", help="Flow Mask path (default: flow_mask.png)")

    # Precision control
    parser.add_argument("--fp32", action="store_true",
                        help="Use FP32 full precision (default FP16, saves VRAM)")
    parser.add_argument("--no-cuda", action="store_true",
                        help="Force CPU usage (very slow, last resort)")

    # --- Memory/downscale strategy (all configurable) ---
    parser.add_argument("--resize", type=str, default="auto",
                        help="Inference resolution: number (e.g. 480/720) = fixed downscale; 'auto' = auto based on VRAM; 'none' = original resolution (default: auto)")
    parser.add_argument("--chunk-size", type=int, default=5,
                        help="Frames per batch (default 5, reduce to 3 when VRAM is tight)")
    parser.add_argument("--raft-iter", type=int, default=20,
                        help="RAFT optical flow iterations (default 20, can reduce to 10 for speed)")
    parser.add_argument("--vram-budget", type=float, default=None,
                        help="VRAM budget (GB) for auto downscale decision (default: auto-detect total GPU VRAM)")
    parser.add_argument("--neighbor-length", type=int, default=None,
                        help="Sliding window size (default=chunk_size*2, affects temporal consistency)")
    parser.add_argument("--ref-stride", type=int, default=10,
                        help="Reference frame stride (default 10, smaller = more reference frames)")
    parser.add_argument("--max-ref-frames", type=int, default=None,
                        help="Maximum reference frames (default: auto based on VRAM)")

    # Video parameters
    parser.add_argument("--crf", type=int, default=18,
                        help="Output video quality CRF (lower is better, default 18)")
    parser.add_argument("--quality", type=int, default=95,
                        help="JPEG frame quality 1-100 (default 95)")

    # Other
    parser.add_argument("--keep-temp", action="store_true",
                        help="Keep temp files (debugging)")
    parser.add_argument("--force-mask", action="store_true",
                        help="Force regenerate Mask (ignore existing mask.png)")

    args = parser.parse_args()

    # Parse --resize parameter
    if args.resize == "none":
        args.resize = None
    elif args.resize == "auto":
        pass  # will auto-determine later
    else:
        try:
            args.resize = int(args.resize)
        except ValueError:
            print(f"ERROR: --resize invalid: {args.resize}, valid values: auto / none / number (e.g. 480)")
            sys.exit(1)

    # ====== Environment detection ======
    print("=" * 60)
    print("DeWatermark v2.2 — ProPainter Video Watermark Removal")
    print("=" * 60)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device_str = "cuda" if use_cuda else "cpu"
    use_fp16 = not args.fp32 and use_cuda

    print(f"FFmpeg: {get_ffmpeg_path()}")
    if use_cuda:
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {props.total_memory / 1024**3:.1f} GB")
        print(f"CUDA: {torch.version.cuda}")
        print(f"Precision: {'FP16' if use_fp16 else 'FP32'}")
    else:
        print("Mode: CPU (very slow)")
        print(f"Precision: FP32")

    # ====== Check input ======
    if not os.path.exists(args.input):
        print(f"ERROR: Input video does not exist: {args.input}")
        sys.exit(1)

    # ====== Validate VRAM, determine VRAM budget ======
    if use_cuda:
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if args.vram_budget is not None:
            vram_budget = min(args.vram_budget, total_vram)
            print(f"VRAM budget (user-specified): {vram_budget:.1f} GB / Total {total_vram:.1f} GB")
        else:
            vram_budget = total_vram
            print(f"VRAM budget (auto): {vram_budget:.1f} GB / Total {total_vram:.1f} GB")
    else:
        vram_budget = 0

    # ====== Auto-tune parameters based on VRAM ======
    # VRAM budget determines inference strategy
    if args.neighbor_length is None:
        if vram_budget <= 6:
            args.neighbor_length = args.chunk_size  # Low VRAM: no overlap, saves VRAM
        else:
            args.neighbor_length = min(args.chunk_size * 2, 10)  # High VRAM: has overlap

    if args.max_ref_frames is None:
        if vram_budget <= 6:
            args.max_ref_frames = 0  # Low VRAM: disable reference frames
        elif vram_budget <= 8:
            args.max_ref_frames = 2  # Medium VRAM: few reference frames
        else:
            args.max_ref_frames = 4  # High VRAM: more reference frames

    # Print current strategy
    resize_is_auto = (isinstance(args.resize, str) and args.resize == "auto")
    if resize_is_auto:
        print(f"Downscale: auto (based on video resolution and {vram_budget:.0f}GB VRAM budget)")
    elif args.resize is not None:
        print(f"Downscale: {args.resize}p (manual)")
    else:
        print(f"Downscale: original (--resize none)")
    print(f"Chunk size: {args.chunk_size} frames/batch")
    print(f"Window size: {args.neighbor_length}, Reference frames: {args.max_ref_frames}")

    # ====== Prepare working directory ======
    base_temp = tempfile.mkdtemp(prefix="propainter_")
    frames_input_dir = os.path.join(base_temp, "frames_input")
    frames_output_dir = os.path.join(base_temp, "frames_output")
    os.makedirs(frames_input_dir, exist_ok=True)
    os.makedirs(frames_output_dir, exist_ok=True)

    print(f"\nTemp directory: {base_temp}")

    try:
        # ====== Step 1: Extract frames ======
        print(f"\n[Step 1/5] Extracting video frames...")
        frame_paths, fps, (width, height), total_frames = extract_frames(
            args.input, frames_input_dir, quality=args.quality
        )

        if total_frames == 0:
            print("ERROR: Video has no frames")
            sys.exit(1)

        # Smart downscale based on resolution and VRAM budget
        if resize_is_auto and use_cuda:
            min_dim = min(width, height)

            # Safe resolution table: recommended max safe resolution based on VRAM budget
            if vram_budget <= 4:
                safe_max = 360
            elif vram_budget <= 6.5:
                safe_max = 480
            elif vram_budget <= 10:
                safe_max = 720
            else:
                safe_max = 1080

            if min_dim > safe_max:
                if min_dim > 720 and safe_max >= 540:
                    target = 480
                elif min_dim > 540 and safe_max >= 360:
                    target = max(360, safe_max)
                else:
                    target = safe_max

                print(f"\n*** Video {min_dim}p + {vram_budget:.0f}GB VRAM -> auto downscale to {target}p for inference ***")
                print(f"*** (use --resize 720 / --resize none to override) ***")
                args.resize = target
            else:
                print(f"\n*** Video {min_dim}p within {vram_budget:.0f}GB VRAM safe range, using original resolution ***")
                args.resize = None
        elif resize_is_auto:
            args.resize = None

        # ====== Step 2: Generate/load Mask and Flow Mask ======
        print(f"\n[Step 2/5] Preparing watermark Mask and Flow Mask...")
        mask, flow_mask = load_or_generate_mask(
            frame_paths[0], args.mask, args.flow_mask, width, height,
            force_regenerate=args.force_mask
        )

        if mask.max() == 0:
            print("WARNING: Mask is empty, no watermark region marked. Copying original frames directly...")
            # Copy frames directly to output directory
            import shutil as _shutil
            for i, src in enumerate(frame_paths):
                dst = os.path.join(frames_output_dir, f"frame_{i:06d}.jpg")
                _shutil.copy2(src, dst)
        else:
            # ====== Step 3: Load models ======
            print(f"\n[Step 3/5] Loading ProPainter models...")
            fix_raft, fix_flow_complete, model = load_propainter_models(
                use_fp16=use_fp16,
                device=device_str,
            )

            # ====== Step 3b: Precompute optical flow ======
            print(f"\n[Step 3b/5] Precomputing full video optical flow...")
            flows_list = precompute_all_flows(
                frame_paths=frame_paths,
                mask=mask,
                flow_mask=flow_mask,
                resize_to=args.resize if isinstance(args.resize, int) else None,
                fix_raft=fix_raft,
                fix_flow_complete=fix_flow_complete,
                raft_iter=args.raft_iter,
                use_fp16=use_fp16,
                device=device_str,
                chunk_size=args.chunk_size,
            )

            # Release RAFT model (no longer needed)
            print("  Releasing RAFT model...")
            del fix_raft
            _cleanup_memory(verbose=True)

            # ====== Step 4: Sliding window inference ======
            print(f"\n[Step 4/5] Starting watermark removal inference (using precomputed optical flow)...")
            process_chunks(
                frame_paths=frame_paths,
                mask=mask,
                flow_mask=flow_mask,
                output_dir=frames_output_dir,
                model=model,
                precomputed_flows=flows_list,
                chunk_size=args.chunk_size,
                resize_to=args.resize if isinstance(args.resize, int) else None,
                use_fp16=use_fp16,
                device=device_str,
                neighbor_length=args.neighbor_length,
                ref_stride=args.ref_stride,
                max_ref_frames=args.max_ref_frames,
            )

            # Release models
            del fix_flow_complete, model
            _cleanup_memory(verbose=True)
            print("Models released")

        # ====== Step 5: Reconstruct video ======
        print(f"\n[Step 5/5] Merging frames into output video...")
        reconstruct_video(
            frames_output_dir, args.output, fps, crf=args.crf,
            input_video_path=args.input,
        )

        print(f"\n{'=' * 60}")
        print(f"✓ Done! Output video: {args.output}")
        print(f"{'=' * 60}")

    except KeyboardInterrupt:
        print("\n\nUser interrupted. Cleaning up...")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # ====== Cleanup temp files ======
        if not args.keep_temp:
            print("\nCleaning up temp files...")
            cleanup_temp(base_temp)
        else:
            print(f"\nTemp files kept at: {base_temp}")


if __name__ == "__main__":
    main()