"""
DeWatermark — Video Watermark Removal with ProPainter
Extreme memory optimization for low-VRAM GPUs (tested on 6 GB).
"""

import os

# 必须在 import torch 之前设置，防止显存碎片化导致的 OOM
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
from typing import Tuple, List, Optional

warnings.filterwarnings("ignore")

import numpy as np
import cv2
import torch
from tqdm import tqdm


def get_ffmpeg_path() -> str:
    """获取 FFmpeg 可执行文件路径，优先系统PATH，其次imageio-ffmpeg内置"""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg

    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass

    raise RuntimeError(
        "未找到 FFmpeg。请执行以下操作之一:\n"
        "  1. 安装 FFmpeg 并添加到系统 PATH\n"
        "  2. pip install imageio-ffmpeg (install.bat 已包含)"
    )


def extract_frames(
    video_path: str,
    output_dir: str,
    quality: int = 95,
) -> Tuple[List[str], float, Tuple[int, int], int]:
    """
    使用 FFmpeg 将视频逐帧提取为 JPEG 存到磁盘。
    整个视频绝不加载到 Python 内存中。

    Args:
        video_path: 输入视频路径
        output_dir: 帧输出目录
        quality: JPEG 质量 (1-100)

    Returns:
        (frame_paths, fps, (width, height), total_frames)
    """
    ffmpeg = get_ffmpeg_path()

    # 先获取视频信息
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
        raise RuntimeError(f"无法读取视频文件: {video_path}")

    # 解析分辨率
    res_match = re.search(r'(\d{2,5})x(\d{2,5})', stderr)
    if not res_match:
        raise RuntimeError(f"无法解析视频分辨率: {video_path}")
    width, height = int(res_match.group(1)), int(res_match.group(2))

    # 解析 FPS
    fps_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:fps|FPS)', stderr)
    fps = float(fps_match.group(1)) if fps_match else 30.0

    # 解析总帧数 (从 nb_frames 或 duration * fps 估算)
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

    print(f"视频信息: {width}x{height}, {fps:.2f} FPS, ~{total_frames} 帧")

    # FFmpeg 逐帧提取 JPEG
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

    print("正在从视频提取帧到磁盘...")
    subprocess.run(extract_cmd, check=True)

    # 收集所有帧路径
    frame_paths = sorted([
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if f.endswith('.jpg')
    ])
    actual_count = len(frame_paths)
    print(f"提取完成: {actual_count} 帧 → {output_dir}")

    return frame_paths, fps, (width, height), actual_count


def generate_mask(
    first_frame_path: str,
    mask_path: str,
    flow_mask_path: str,
    width: int,
    height: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    弹出窗口让用户在第一帧上框选水印位置，自动生成 Mask 和 Flow Mask。

    Args:
        first_frame_path: 第一帧 JPEG 路径
        mask_path: 保存 mask 的路径
        flow_mask_path: 保存 flow_mask 的路径
        width: 视频宽度
        height: 视频高度

    Returns:
        (mask, flow_mask): (H, W) 二值图，水印区域=255，其它=0
    """
    import scipy.ndimage

    frame = cv2.imread(first_frame_path)
    if frame is None:
        raise RuntimeError(f"无法读取第一帧: {first_frame_path}")

    print("\n" + "=" * 60)
    print("请在弹出窗口中框选水印区域")
    print("操作提示:")
    print("  - 鼠标拖拽框选水印位置")
    print("  - 按 SPACE/ENTER 确认当前选框")
    print("  - 按 ESC 跳过（不处理）")
    print("  - 可多次框选多个水印区域")
    print("=" * 60 + "\n")

    mask = np.zeros((height, width), dtype=np.uint8)

    while True:
        roi = cv2.selectROI("Select Watermark - SPACE=confirm, ESC=finish", frame, showCrosshair=True, fromCenter=False)

        if roi[2] == 0 or roi[3] == 0:
            break

        x, y, w, h = roi
        mask[y:y + h, x:x + w] = 255
        print(f"  已标记水印区域: x={x}, y={y}, w={w}, h={h}")

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.destroyAllWindows()

    if mask.max() == 0:
        print("\nWARNING: 未选择任何水印区域，将跳过去水印处理")
        flow_mask = mask.copy()
    else:
        # 生成 mask (膨胀5次，用于图像修复)
        mask_dilated = scipy.ndimage.binary_dilation(mask > 128, iterations=5).astype(np.uint8) * 255
        cv2.imwrite(mask_path, mask_dilated)
        print(f"\nMask 已保存: {mask_path}")

        # 生成 flow_mask (膨胀8次，用于光流补全，更大的区域确保光流稳定)
        flow_mask = scipy.ndimage.binary_dilation(mask > 128, iterations=8).astype(np.uint8) * 255
        cv2.imwrite(flow_mask_path, flow_mask)
        print(f"Flow Mask 已保存: {flow_mask_path}")

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
    加载已有 Mask/Flow Mask 或通过交互框选生成新 Mask。

    Args:
        first_frame_path: 第一帧路径（生成新mask时需要）
        mask_path: Mask 保存/加载路径
        flow_mask_path: Flow Mask 保存/加载路径
        width: 视频宽度
        height: 视频高度
        force_regenerate: 强制重新生成

    Returns:
        (mask, flow_mask): (H, W) 二值图
    """
    if os.path.exists(mask_path) and os.path.exists(flow_mask_path) and not force_regenerate:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        flow_mask = cv2.imread(flow_mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is not None and flow_mask is not None:
            print(f"已加载现有 Mask: {mask_path}")
            print(f"已加载现有 Flow Mask: {flow_mask_path}")
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
    选择参考帧索引，为 ProPainter Transformer 提供全局上下文。

    参考帧是视频中不在当前邻域内的帧，用于帮助模型理解全局场景信息。

    Args:
        mid_neighbor_id: 当前邻域的中心帧索引
        neighbor_ids: 当前邻域的帧索引列表
        length: 视频总帧数
        ref_stride: 参考帧选取步长
        ref_num: 参考帧数量上限 (-1 表示不限制)

    Returns:
        参考帧索引列表
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
# ProPainter 模型加载模块
# ============================================================

PROPAINTER_REPO_URL = "https://github.com/sczhou/ProPainter.git"
PROPAINTER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ProPainter")
PRETRAIN_URL_BASE = "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/"

WEIGHT_FILES = {
    "raft-things.pth": "RAFT 光流模型",
    "recurrent_flow_completion.pth": "循环流补全模型",
    "ProPainter.pth": "ProPainter 修复模型",
}


def ensure_propainter() -> str:
    """
    确保 ProPainter 源码存在，如不存在则自动 git clone。
    返回 ProPainter 目录路径。
    """
    if os.path.exists(os.path.join(PROPAINTER_DIR, "inference_propainter.py")):
        print(f"ProPainter 源码已存在: {PROPAINTER_DIR}")
        return PROPAINTER_DIR

    # 目录存在但不完整（上次 clone 失败），删除后重试
    if os.path.exists(PROPAINTER_DIR):
        print(f"检测到不完整的 ProPainter 目录，正在清理: {PROPAINTER_DIR}")
        shutil.rmtree(PROPAINTER_DIR, ignore_errors=True)

    print(f"首次运行，正在下载 ProPainter 源码...")
    print(f"源: {PROPAINTER_REPO_URL}")

    git_path = shutil.which("git")
    if git_path:
        cmd = [git_path, "clone", "--depth", "1", PROPAINTER_REPO_URL, PROPAINTER_DIR]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True, encoding="utf-8", errors="replace")
            print("ProPainter 源码下载完成。")
            return PROPAINTER_DIR
        except subprocess.CalledProcessError as e:
            print(f"Git clone 失败: {e.stderr}")

    raise RuntimeError(
        "无法下载 ProPainter 源码。请手动执行:\n"
        f"  git clone --depth 1 {PROPAINTER_REPO_URL} \"{PROPAINTER_DIR}\""
    )


def download_weights(weights_dir: str, force: bool = False) -> dict:
    """
    下载 ProPainter 预训练权重文件。
    返回 {name: path} 字典。
    """
    import requests

    os.makedirs(weights_dir, exist_ok=True)
    weight_paths = {}

    for fname, desc in WEIGHT_FILES.items():
        local_path = os.path.join(weights_dir, fname)
        if os.path.exists(local_path) and not force:
            print(f"  [已存在] {fname} ({desc})")
            weight_paths[fname] = local_path
            continue

        url = PRETRAIN_URL_BASE + fname
        print(f"  下载 {fname} ({desc})...")
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
            print(f"    {fname} 下载完成。")
        except Exception as e:
            print(f"    下载 {fname} 失败: {e}")
            if os.path.exists(local_path):
                os.remove(local_path)
            raise RuntimeError(f"权重下载失败: {fname}\n请手动下载到 {weights_dir}/")

    return weight_paths


def load_propainter_models(
    weights_dir: str = "weights",
    use_fp16: bool = True,
    device: str = "cuda",
):
    """
    加载 ProPainter 三个核心模型，支持 FP16 半精度。

    Returns:
        (fix_raft, fix_flow_complete, model):
            - fix_raft: RAFT_bi 光流模型 (FP32, RAFT用FP32更稳定)
            - fix_flow_complete: RecurrentFlowCompleteNet
            - model: InpaintGenerator (ProPainter 修复网络)
    """
    # 确保 ProPainter 源码可用
    propainter_root = ensure_propainter()
    if propainter_root not in sys.path:
        sys.path.insert(0, propainter_root)

    # 下载权重
    weights_dir = os.path.join(PROPAINTER_DIR, "weights") if weights_dir == "weights" else weights_dir
    weight_paths = download_weights(weights_dir)

    device_obj = torch.device(device)
    print(f"\n加载 ProPainter 模型到 {device_obj}...")

    # 1. RAFT 光流模型 (保持 FP32，光流精度敏感)
    print("  [1/3] 加载 RAFT 光流模型...")
    from model.modules.flow_comp_raft import RAFT_bi
    fix_raft = RAFT_bi(weight_paths["raft-things.pth"], device_obj)
    fix_raft.eval()
    print(f"    RAFT 加载完成 (FP32)")

    # 2. 循环流补全网络
    print("  [2/3] 加载循环流补全网络...")
    from model.recurrent_flow_completion import RecurrentFlowCompleteNet
    fix_flow_complete = RecurrentFlowCompleteNet(weight_paths["recurrent_flow_completion.pth"])
    for p in fix_flow_complete.parameters():
        p.requires_grad = False
    fix_flow_complete.to(device_obj)
    fix_flow_complete.eval()

    # 3. ProPainter 修复网络
    print("  [3/3] 加载 ProPainter 修复网络...")
    from model.propainter import InpaintGenerator
    model = InpaintGenerator(model_path=weight_paths["ProPainter.pth"]).to(device_obj)
    model.eval()

    # FP16 转换（RAFT 除外）
    if use_fp16 and device != "cpu":
        print("  启用 FP16 半精度...")
        fix_flow_complete = fix_flow_complete.half()
        model = model.half()
    else:
        print("  使用 FP32 全精度")

    torch.cuda.empty_cache()
    print(f"  模型加载完成。")

    return fix_raft, fix_flow_complete, model


def preprocess_frames_for_propainter(
    frame_pils: list,
    mask: np.ndarray,
    flow_mask: np.ndarray,
    resize_to: Optional[int] = None,
    num_local_frames: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[int, int], Tuple[int, int]]:
    """
    从 PIL 图像列表预处理为 ProPainter 输入格式。

    支持参考帧：前 num_local_frames 帧为本地帧（带水印 mask），
    其余为参考帧（mask 为零，表示无水印）。

    Args:
        frame_pils: PIL Image 列表 (本地帧 + 参考帧)
        mask: (H, W) 二值 mask (255=水印区域)，用于本地帧
        flow_mask: (H, W) 二值 flow_mask (255=水印区域，更大膨胀)，用于光流补全
        resize_to: 可选，缩放到的短边尺寸
        num_local_frames: 本地帧数量，其余为参考帧。None 表示全部为本地帧

    Returns:
        (frames_tensor, mask_tensor, flow_mask_tensor, original_size, process_size)
        frames_tensor: (1, T, 3, H, W) 归一化到 [-1, 1]
        mask_tensor:  (1, T, 1, H, W) 二值 mask（本地帧有水印区域，参考帧为0）
        flow_mask_tensor: (1, T_local, 1, H, W) 二值 flow_mask
    """
    from PIL import Image
    import scipy.ndimage
    from core.utils import to_tensors as to_tensors_fn

    if num_local_frames is None:
        num_local_frames = len(frame_pils)

    # 计算处理尺寸
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

    # Resize 帧
    resized_frames = []
    for f in frame_pils:
        if process_size != (orig_w, orig_h):
            f = f.resize(process_size, Image.LANCZOS)
        resized_frames.append(f)

    # Resize mask 和 flow_mask
    mask_img = Image.fromarray(mask)
    flow_mask_img = Image.fromarray(flow_mask)
    if process_size != (orig_w, orig_h):
        mask_img = mask_img.resize(process_size, Image.NEAREST)
        flow_mask_img = flow_mask_img.resize(process_size, Image.NEAREST)
    mask_np = np.array(mask_img)
    flow_mask_np = np.array(flow_mask_img)

    # 处理 mask: 膨胀 + 二值化 (mask 已经在 generate_mask 中膨胀过了)
    mask_binary = (mask_np > 128).astype(np.uint8) * 255
    flow_mask_binary = (flow_mask_np > 128).astype(np.uint8) * 255

    # 转为 tensor
    frames_t = to_tensors_fn()(resized_frames).unsqueeze(0) * 2.0 - 1.0  # (1,T,3,H,W)

    # 创建 mask tensor: 本地帧使用水印 mask，参考帧 mask 为零
    mask_imgs = []
    for i in range(len(resized_frames)):
        if i < num_local_frames:
            mask_imgs.append(Image.fromarray(mask_binary))
        else:
            # 参考帧: mask 为全零（无水印区域）
            mask_imgs.append(Image.fromarray(np.zeros_like(mask_binary)))
    mask_t = to_tensors_fn()(mask_imgs).unsqueeze(0)  # (1,T,1,H,W)

    # 创建 flow_mask tensor: 仅用于本地帧的光流补全
    flow_mask_imgs = [Image.fromarray(flow_mask_binary)] * num_local_frames
    flow_mask_t = to_tensors_fn()(flow_mask_imgs).unsqueeze(0)  # (1,T_local,1,H,W)

    return frames_t, mask_t, flow_mask_t, original_size, process_size


def propainter_infer_chunk(
    frames_tensor: torch.Tensor,
    mask_tensor: torch.Tensor,
    flow_mask_tensor: torch.Tensor,
    fix_raft,
    fix_flow_complete,
    model,
    raft_iter: int = 20,
    subvideo_length: int = 80,
    use_fp16: bool = True,
    num_local_frames: Optional[int] = None,
) -> torch.Tensor:
    """
    对一个小 chunk 的视频帧执行 ProPainter 推理。

    支持参考帧：前 num_local_frames 帧为本地帧（需要修复），
    其余为参考帧（提供全局上下文）。

    关键修复：
    - 使用 flow_mask（更大膨胀）进行光流补全
    - img_propagation 结果与原始帧混合
    - 模型 forward 使用 num_local_frames 区分本地帧和参考帧
    - 最终输出仅返回本地帧的修复结果

    Args:
        frames_tensor: (1, T, 3, H, W) 归一化到 [-1, 1] (本地帧 + 参考帧)
        mask_tensor:  (1, T, 1, H, W) 二值 mask（本地帧有水印，参考帧为0）
        flow_mask_tensor: (1, T_local, 1, H, W) 二值 flow_mask（更大膨胀）
        fix_raft: RAFT 光流模型
        fix_flow_complete: 流补全模型
        model: ProPainter 修复模型
        raft_iter: RAFT 迭代次数
        subvideo_length: 子视频长度
        use_fp16: 半精度
        num_local_frames: 本地帧数量，其余为参考帧

    Returns:
        output: (1, l_t, 3, H, W) 修复后的本地帧（归一化到 [-1, 1]）
    """
    if num_local_frames is None:
        num_local_frames = frames_tensor.size(1)

    device = frames_tensor.device
    video_length = num_local_frames  # 光流仅计算本地帧

    with torch.no_grad():
        # ---- 1. 计算光流 (RAFT 用 FP32，仅对本地帧) ----
        local_frames = frames_tensor[:, :num_local_frames, :, :, :]

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

        # 释放 RAFT FP32 输入
        del raft_input
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # ---- FP16 转换 ----
        if use_fp16:
            frames_tensor = frames_tensor.half()
            mask_tensor = mask_tensor.half()
            flow_mask_tensor = flow_mask_tensor.half()
            flows_f_fp16 = gt_flows_bi[0].half()
            flows_b_fp16 = gt_flows_bi[1].half()
            del gt_flows_bi
            gt_flows_bi = (flows_f_fp16, flows_b_fp16)
            torch.cuda.empty_cache()

        # ---- 2. 流补全（使用 flow_mask，更大膨胀） ----
        # flow_mask_tensor 仅覆盖本地帧
        local_flow_mask = flow_mask_tensor  # (1, T_local, 1, H, W)

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
                    local_flow_mask[:, s_f:e_f + 1]
                )
                pred_flows_bi_sub = fix_flow_complete.combine_flow(
                    (gt_flows_bi[0][:, s_f:e_f], gt_flows_bi[1][:, s_f:e_f]),
                    pred_flows_bi_sub,
                    local_flow_mask[:, s_f:e_f + 1]
                )
                pred_flows_f.append(pred_flows_bi_sub[0][:, pad_len_s:e_f - s_f - pad_len_e])
                pred_flows_b.append(pred_flows_bi_sub[1][:, pad_len_s:e_f - s_f - pad_len_e])
                torch.cuda.empty_cache()

            pred_flows_f = torch.cat(pred_flows_f, dim=1)
            pred_flows_b = torch.cat(pred_flows_b, dim=1)
            pred_flows_bi = (pred_flows_f, pred_flows_b)
        else:
            pred_flows_bi, _ = fix_flow_complete.forward_bidirect_flow(gt_flows_bi, local_flow_mask)
            pred_flows_bi = fix_flow_complete.combine_flow(gt_flows_bi, pred_flows_bi, local_flow_mask)
            torch.cuda.empty_cache()

        # ---- 3. 图像传播（使用 mask，较小膨胀） ----
        # 仅对本地帧进行图像传播
        local_mask = mask_tensor[:, :num_local_frames, :, :]  # (1, T_local, 1, H, W)
        local_frames_tensor = frames_tensor[:, :num_local_frames, :, :, :]
        masked_frames = local_frames_tensor * (1 - local_mask)

        # 光流也仅用于本地帧
        local_pred_flows_bi = pred_flows_bi  # 已经是本地帧的光流

        updated_frames, updated_masks = model.img_propagation(
            masked_frames, local_pred_flows_bi, local_mask, "nearest"
        )
        torch.cuda.empty_cache()

        # 关键修复: 将传播结果与原始帧混合
        # 传播结果仅在 mask 区域有效，非 mask 区域保留原始帧
        updated_frames = local_frames_tensor * (1 - local_mask) + \
                         updated_frames.view(local_frames_tensor.shape) * local_mask

        # ---- 4. 构建完整输入（本地帧 + 参考帧） ----
        # 将 updated_frames 和 updated_masks 扩展到包含参考帧
        if frames_tensor.size(1) > num_local_frames:
            # 有参考帧
            ref_frames = frames_tensor[:, num_local_frames:, :, :]
            ref_masks = mask_tensor[:, num_local_frames:, :, :]  # 参考帧 mask 为零

            # 拼接: updated 本地帧 + 参考帧
            all_updated_frames = torch.cat([updated_frames, ref_frames], dim=1)
            all_masks = mask_tensor  # 本地帧有 mask，参考帧 mask 为零
            all_updated_masks = torch.cat([updated_masks, ref_masks], dim=1)
        else:
            # 没有参考帧
            all_updated_frames = updated_frames
            all_masks = mask_tensor
            all_updated_masks = updated_masks

        # ---- 5. Transformer 推理 ----
        masked_all_frames = frames_tensor * (1 - all_masks)

        comp_frames = model(
            masked_frames=masked_all_frames,
            completed_flows=local_pred_flows_bi,
            masks_in=all_masks,
            masks_updated=all_updated_masks,
            num_local_frames=num_local_frames,
        )
        torch.cuda.empty_cache()

    # 仅返回本地帧的结果
    return comp_frames


def _cleanup_memory(verbose: bool = False):
    """三重内存清理：del引用 + gc.collect() + CUDA缓存清空"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    if verbose:
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"  [内存清理] 已分配: {alloc:.2f}GB, 已保留: {reserved:.2f}GB")


def process_chunks(
    frame_paths: List[str],
    mask: np.ndarray,
    flow_mask: np.ndarray,
    output_dir: str,
    fix_raft,
    fix_flow_complete,
    model,
    chunk_size: int = 5,
    resize_to: Optional[int] = None,
    use_fp16: bool = True,
    raft_iter: int = 20,
    device: str = "cuda",
    neighbor_length: Optional[int] = None,
    ref_stride: int = 10,
) -> None:
    """
    滑动窗口推理 — 核心处理函数。

    使用滑动窗口 + 参考帧策略，与 ProPainter 官方推理逻辑对齐：
    1. 每个窗口包含 neighbor_length 帧本地帧 + 若干参考帧
    2. 参考帧提供全局上下文，mask 为零（无水印）
    3. 模型输出仅在 mask 区域使用修复结果，非 mask 区域保留原始像素
    4. 重叠区域取多次推理的平均值

    Args:
        frame_paths: 所有帧路径列表
        mask: (H, W) 二值 mask (255=水印区域，已膨胀5次)
        flow_mask: (H, W) 二值 flow_mask (255=水印区域，已膨胀8次)
        output_dir: 输出帧目录
        fix_raft: RAFT 光流模型
        fix_flow_complete: 流补全模型
        model: ProPainter 修复模型
        chunk_size: 每批处理帧数 (默认5，显存紧张可降到3)
        resize_to: 降级分辨率 (如 480)
        use_fp16: 半精度
        raft_iter: RAFT 迭代次数
        device: 设备
        neighbor_length: 滑动窗口大小 (默认=chunk_size*2)
        ref_stride: 参考帧选取步长
    """
    from PIL import Image

    total_frames = len(frame_paths)
    device_obj = torch.device(device)
    use_cuda = device_obj.type == "cuda"

    if neighbor_length is None:
        neighbor_length = min(chunk_size * 2, total_frames)
    neighbor_stride = max(1, neighbor_length // 2)

    print(f"\n{'=' * 60}")
    print(f"滑动窗口推理: 共 {total_frames} 帧")
    print(f"  窗口大小: {neighbor_length}, 步长: {neighbor_stride}")
    print(f"  参考帧步长: {ref_stride}")
    if resize_to:
        print(f"  降级分辨率: {resize_to}p (推理后还原)")
    print(f"  FP16: {use_fp16}, GPU: {use_cuda}")
    print(f"{'=' * 60}")

    os.makedirs(output_dir, exist_ok=True)

    # 预加载参考帧 (PIL Image)
    ref_indices = get_ref_index(0, [], total_frames, ref_stride=ref_stride)
    # 限制参考帧数量以节省显存
    max_ref = min(8, len(ref_indices))
    if len(ref_indices) > max_ref:
        # 均匀采样
        step = len(ref_indices) / max_ref
        ref_indices = [ref_indices[int(i * step)] for i in range(max_ref)]

    ref_frames_pil = {}
    for idx in ref_indices:
        try:
            ref_frames_pil[idx] = Image.open(frame_paths[idx]).convert("RGB")
        except Exception as e:
            print(f"  WARNING: 无法加载参考帧 {idx}: {e}")

    # 累积结果和计数（用于重叠区域平均）
    # 使用 float32 累积以避免精度损失
    first_frame = cv2.imread(frame_paths[0])
    if first_frame is None:
        raise RuntimeError(f"无法读取第一帧: {frame_paths[0]}")
    orig_h, orig_w = first_frame.shape[:2]
    accum_dtype = np.float32
    frame_accum = np.zeros((total_frames, orig_h, orig_w, 3), dtype=accum_dtype)
    frame_count = np.zeros(total_frames, dtype=accum_dtype)

    pbar = tqdm(total=total_frames, desc="推理进度", unit="帧")

    for window_start in range(0, total_frames, neighbor_stride):
        window_end = min(window_start + neighbor_length, total_frames)
        neighbor_ids = list(range(window_start, window_end))

        # 选择参考帧
        mid_id = (window_start + window_end) // 2
        ref_ids = get_ref_index(mid_id, neighbor_ids, total_frames, ref_stride=ref_stride)
        # 过滤掉无法加载的参考帧
        ref_ids = [i for i in ref_ids if i in ref_frames_pil]

        # 本地帧 + 参考帧
        local_pils = [Image.open(frame_paths[i]).convert("RGB") for i in neighbor_ids]
        ref_pils = [ref_frames_pil[i] for i in ref_ids]
        all_pils = local_pils + ref_pils
        l_t = len(local_pils)

        # ====== 步骤 1: 预处理 ======
        frames_tensor, mask_tensor, flow_mask_tensor, original_size, process_size = \
            preprocess_frames_for_propainter(
                all_pils, mask, flow_mask, resize_to, num_local_frames=l_t
            )

        frames_tensor = frames_tensor.to(device_obj)
        mask_tensor = mask_tensor.to(device_obj)
        flow_mask_tensor = flow_mask_tensor.to(device_obj)

        # ====== 步骤 2: ProPainter 推理 ======
        with torch.inference_mode():
            result_tensor = propainter_infer_chunk(
                frames_tensor, mask_tensor, flow_mask_tensor,
                fix_raft, fix_flow_complete, model,
                raft_iter=raft_iter,
                use_fp16=use_fp16,
                num_local_frames=l_t,
            )

        # ====== 步骤 3: 后处理 + 与原始帧混合 ======
        # result_tensor: (1, l_t, 3, H, W) in [-1, 1]
        result_np = result_tensor.squeeze(0).float().cpu().numpy()  # (l_t, 3, H, W)
        result_np = (result_np + 1.0) / 2.0  # [-1,1] → [0,1]
        result_np = np.clip(result_np * 255, 0, 255).astype(np.uint8)
        result_np = result_np.transpose(0, 2, 3, 1)  # (l_t, H, W, 3)

        # 如果推理时缩放了，还原回原始尺寸
        if process_size != original_size:
            resized_results = []
            for i in range(result_np.shape[0]):
                img = cv2.resize(result_np[i], original_size, interpolation=cv2.INTER_LANCZOS4)
                resized_results.append(img)
            result_np = np.stack(resized_results)

        # 准备 mask（原始分辨率，3通道，用于混合）
        mask_3ch = np.stack([mask] * 3, axis=-1).astype(accum_dtype) / 255.0  # (H, W, 3)

        # 加载原始帧并与模型输出混合
        # 关键修复: 仅在 mask 区域使用模型输出，非 mask 区域保留原始像素
        for i, idx in enumerate(neighbor_ids):
            # 读取原始帧
            orig_frame = cv2.imread(frame_paths[idx])
            if orig_frame is None:
                print(f"  WARNING: 无法读取帧 {idx}")
                continue
            orig_frame = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2RGB)

            # 混合: mask 区域使用模型输出，非 mask 区域保留原始像素
            blended = result_np[i].astype(accum_dtype) * mask_3ch + \
                      orig_frame.astype(accum_dtype) * (1 - mask_3ch)
            blended = blended.astype(np.uint8)

            # 累积结果（用于重叠区域平均）
            frame_accum[idx] += blended.astype(accum_dtype)
            frame_count[idx] += 1

        # ====== 步骤 4: 内存清理 ======
        del frames_tensor, mask_tensor, flow_mask_tensor, result_tensor, result_np
        del local_pils, ref_pils, all_pils
        _cleanup_memory(verbose=(window_start % (neighbor_stride * 10) == 0))

        pbar.update(len(neighbor_ids) - (len(neighbor_ids) - min(len(neighbor_ids), total_frames - window_start)))

    pbar.close()

    # ====== 步骤 5: 计算平均值并保存 ======
    print("\n保存输出帧...")
    for idx in range(total_frames):
        if frame_count[idx] > 0:
            avg_frame = (frame_accum[idx] / frame_count[idx]).astype(np.uint8)
        else:
            # 未处理的帧，使用原始帧
            orig = cv2.imread(frame_paths[idx])
            avg_frame = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)

        out_path = os.path.join(output_dir, f"frame_{idx:06d}.jpg")
        cv2.imwrite(out_path, cv2.cvtColor(avg_frame, cv2.COLOR_RGB2BGR))

    print(f"\n推理完成! 输出帧: {output_dir}")
    _cleanup_memory(verbose=True)


def reconstruct_video(
    frame_dir: str,
    output_path: str,
    fps: float,
    crf: int = 18,
    input_video_path: Optional[str] = None,
) -> None:
    """
    使用 FFmpeg 将帧序列合并为视频，并可选地从原始视频复制音轨。

    Args:
        frame_dir: 帧目录
        output_path: 输出视频路径
        fps: 帧率
        crf: 视频质量 (越小越好，推荐 18-23)
        input_video_path: 原始视频路径（用于复制音轨）
    """
    ffmpeg = get_ffmpeg_path()
    input_pattern = os.path.join(frame_dir, "frame_%06d.jpg")

    # 先生成无声视频
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

    print(f"正在合并帧为视频 → {output_path}")
    subprocess.run(cmd, check=True)

    # 如果有原始视频，尝试复制音轨
    if input_video_path and os.path.exists(input_video_path):
        # 检查原始视频是否有音轨
        probe_cmd = [ffmpeg, "-i", input_video_path, "-f", "null", "-"]
        try:
            probe_result = subprocess.run(
                probe_cmd, capture_output=True, text=True,
                encoding="utf-8", errors="replace"
            )
            # 检查是否有音频流
            if "Audio:" in probe_result.stderr:
                print("检测到原始视频音轨，正在合并...")
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
                print("音轨合并完成。")
            else:
                # 没有音轨，直接重命名
                os.rename(temp_output, output_path)
                print("原始视频无音轨，仅生成视频流。")
        except subprocess.CalledProcessError:
            # 合并失败，使用无声视频
            os.rename(temp_output, output_path)
            print("音轨合并失败，仅生成视频流。")
    else:
        os.rename(temp_output, output_path)
        print("视频生成完成（无音轨）。")


def cleanup_temp(*dirs: str) -> None:
    """安全删除临时目录"""
    for d in dirs:
        if d and os.path.exists(d):
            try:
                shutil.rmtree(d)
                print(f"  已清理: {d}")
            except Exception as e:
                print(f"  清理失败 {d}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="DeWatermark — Video Watermark Removal with ProPainter (VRAM ≥ 4 GB)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 基本用法 (自动根据显存和分辨率降级)
  python remove_watermark.py -i video.mp4 -o output.mp4

  # 手动指定降级分辨率
  python remove_watermark.py -i video.mp4 -o output.mp4 --resize 480

  # 不使用降级 (显存够大时)
  python remove_watermark.py -i video.mp4 -o output.mp4 --resize none

  # 限制显存预算为4GB (自动更保守)
  python remove_watermark.py -i video.mp4 -o output.mp4 --vram-budget 4

  # 进一步降低每批帧数
  python remove_watermark.py -i video.mp4 -o output.mp4 --resize 480 --chunk-size 3

  # 使用已有mask（跳过框选）
  python remove_watermark.py -i video.mp4 -o output.mp4 --mask mask.png
        """
    )

    # 核心参数
    parser.add_argument("-i", "--input", required=True, help="输入视频路径")
    parser.add_argument("-o", "--output", default="output_clean.mp4", help="输出视频路径 (默认: output_clean.mp4)")
    parser.add_argument("-m", "--mask", default="mask.png", help="Mask 路径 (默认: mask.png，不存在则交互框选)")
    parser.add_argument("--flow-mask", default="flow_mask.png", help="Flow Mask 路径 (默认: flow_mask.png)")

    # 精度控制
    parser.add_argument("--fp32", action="store_true",
                        help="使用 FP32 全精度 (默认FP16，更省显存)")
    parser.add_argument("--no-cuda", action="store_true",
                        help="强制使用 CPU (极慢，最后手段)")

    # --- 内存/降级策略 (全部可配置) ---
    parser.add_argument("--resize", type=str, default="auto",
                        help="推理分辨率: 数字(如480/720) = 固定降级; 'auto' = 根据显存自动判断; 'none' = 原分辨率 (默认: auto)")
    parser.add_argument("--chunk-size", type=int, default=5,
                        help="每批处理帧数 (默认5，显存紧张时降低到3)")
    parser.add_argument("--raft-iter", type=int, default=20,
                        help="RAFT 光流迭代次数 (默认20，可降至10加速)")
    parser.add_argument("--vram-budget", type=float, default=None,
                        help="显存预算(GB)，用于自动降级判断 (默认: 自动检测GPU总显存)")
    parser.add_argument("--neighbor-length", type=int, default=None,
                        help="滑动窗口大小 (默认=chunk_size*2，影响时序一致性)")
    parser.add_argument("--ref-stride", type=int, default=10,
                        help="参考帧选取步长 (默认10，越小参考帧越多)")

    # 视频参数
    parser.add_argument("--crf", type=int, default=18,
                        help="输出视频质量 CRF (越小越好，默认18)")
    parser.add_argument("--quality", type=int, default=95,
                        help="JPEG 帧质量 1-100 (默认95)")

    # 其他
    parser.add_argument("--keep-temp", action="store_true",
                        help="保留临时文件 (调试用)")
    parser.add_argument("--force-mask", action="store_true",
                        help="强制重新生成 Mask (忽略已有mask.png)")

    args = parser.parse_args()

    # 解析 --resize 参数
    if args.resize == "none":
        args.resize = None
    elif args.resize == "auto":
        pass  # 后面自动判断
    else:
        try:
            args.resize = int(args.resize)
        except ValueError:
            print(f"ERROR: --resize 参数无效: {args.resize}，可选值: auto / none / 数字(如480)")
            sys.exit(1)

    # ====== 环境检测 ======
    print("=" * 60)
    print("DeWatermark v2.0 — ProPainter Video Watermark Removal")
    print("=" * 60)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device_str = "cuda" if use_cuda else "cpu"
    use_fp16 = not args.fp32 and use_cuda

    print(f"FFmpeg: {get_ffmpeg_path()}")
    if use_cuda:
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {props.total_memory / 1024**3:.1f} GB")
        print(f"CUDA: {torch.version.cuda}")
        print(f"精度: {'FP16' if use_fp16 else 'FP32'}")
    else:
        print("模式: CPU (极慢)")
        print(f"精度: FP32")

    # ====== 检查输入 ======
    if not os.path.exists(args.input):
        print(f"ERROR: 输入视频不存在: {args.input}")
        sys.exit(1)

    # ====== 验证显存，确定显存预算 ======
    if use_cuda:
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if args.vram_budget is not None:
            vram_budget = min(args.vram_budget, total_vram)
            print(f"显存预算(用户指定): {vram_budget:.1f} GB / 总计 {total_vram:.1f} GB")
        else:
            vram_budget = total_vram
            print(f"显存预算(自动): {vram_budget:.1f} GB / 总计 {total_vram:.1f} GB")
    else:
        vram_budget = 0

    # 打印当前策略
    resize_is_auto = (isinstance(args.resize, str) and args.resize == "auto")
    if resize_is_auto:
        print(f"降级分辨率: 自动判断 (根据视频分辨率和{vram_budget:.0f}GB显存预算)")
    elif args.resize is not None:
        print(f"降级分辨率: {args.resize}p (手动指定)")
    else:
        print(f"降级分辨率: 原始 (--resize none)")
    print(f"Chunk 大小: {args.chunk_size} 帧/批")
    print(f"参考帧步长: {args.ref_stride}")

    # ====== 准备工作目录 ======
    base_temp = tempfile.mkdtemp(prefix="propainter_")
    frames_input_dir = os.path.join(base_temp, "frames_input")
    frames_output_dir = os.path.join(base_temp, "frames_output")
    os.makedirs(frames_input_dir, exist_ok=True)
    os.makedirs(frames_output_dir, exist_ok=True)

    print(f"\n临时目录: {base_temp}")

    try:
        # ====== 步骤 1: 抽帧 ======
        print(f"\n[步骤 1/5] 提取视频帧...")
        frame_paths, fps, (width, height), total_frames = extract_frames(
            args.input, frames_input_dir, quality=args.quality
        )

        if total_frames == 0:
            print("ERROR: 视频没有帧")
            sys.exit(1)

        # 基于分辨率和显存预算的智能降级
        if resize_is_auto and use_cuda:
            min_dim = min(width, height)

            # 安全分辨率表: 根据显存预算推荐最大安全分辨率
            if vram_budget <= 4:
                safe_max = 360
            elif vram_budget <= 6.5:
                safe_max = 540
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

                print(f"\n*** 视频 {min_dim}p + {vram_budget:.0f}GB显存 → 自动降级至 {target}p 推理 ***")
                print(f"*** (可手动指定 --resize 720 / --resize none 覆盖此行为) ***")
                args.resize = target
            else:
                print(f"\n*** 视频 {min_dim}p, 在 {vram_budget:.0f}GB 显存安全范围内, 原始分辨率推理 ***")
                args.resize = None
        elif resize_is_auto:
            args.resize = None

        # ====== 步骤 2: 生成/加载 Mask 和 Flow Mask ======
        print(f"\n[步骤 2/5] 准备水印 Mask 和 Flow Mask...")
        mask, flow_mask = load_or_generate_mask(
            frame_paths[0], args.mask, args.flow_mask, width, height,
            force_regenerate=args.force_mask
        )

        if mask.max() == 0:
            print("WARNING: Mask 为空，水印区域未标记。将直接复制原始帧...")
            # 直接复制帧到输出目录
            import shutil as _shutil
            for i, src in enumerate(frame_paths):
                dst = os.path.join(frames_output_dir, f"frame_{i:06d}.jpg")
                _shutil.copy2(src, dst)
        else:
            # ====== 步骤 3: 加载模型 ======
            print(f"\n[步骤 3/5] 加载 ProPainter 模型...")
            fix_raft, fix_flow_complete, model = load_propainter_models(
                use_fp16=use_fp16,
                device=device_str,
            )

            # ====== 步骤 4: 滑动窗口推理 ======
            print(f"\n[步骤 4/5] 开始去水印推理...")
            process_chunks(
                frame_paths=frame_paths,
                mask=mask,
                flow_mask=flow_mask,
                output_dir=frames_output_dir,
                fix_raft=fix_raft,
                fix_flow_complete=fix_flow_complete,
                model=model,
                chunk_size=args.chunk_size,
                resize_to=args.resize if isinstance(args.resize, int) else None,
                use_fp16=use_fp16,
                raft_iter=args.raft_iter,
                device=device_str,
                neighbor_length=args.neighbor_length,
                ref_stride=args.ref_stride,
            )

            # 释放模型
            del fix_raft, fix_flow_complete, model
            _cleanup_memory(verbose=True)
            print("模型已释放")

        # ====== 步骤 5: 合并视频 ======
        print(f"\n[步骤 5/5] 合并帧为输出视频...")
        reconstruct_video(
            frames_output_dir, args.output, fps, crf=args.crf,
            input_video_path=args.input,
        )

        print(f"\n{'=' * 60}")
        print(f"✓ 完成! 输出视频: {args.output}")
        print(f"{'=' * 60}")

    except KeyboardInterrupt:
        print("\n\n用户中断。正在清理...")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # ====== 清理临时文件 ======
        if not args.keep_temp:
            print("\n清理临时文件...")
            cleanup_temp(base_temp)
        else:
            print(f"\n临时文件保留在: {base_temp}")


if __name__ == "__main__":
    main()