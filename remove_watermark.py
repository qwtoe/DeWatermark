"""
ProPainter 视频去水印工具
针对 NVIDIA CMP 30HX (6GB VRAM) / 8GB 系统内存 极限优化
"""

import os
import sys
import gc
import subprocess
import tempfile
import shutil
import argparse
from pathlib import Path
from typing import Tuple, List, Optional

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
    import re
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


def reconstruct_video(
    frame_dir: str,
    output_path: str,
    fps: float,
    crf: int = 18,
) -> None:
    """
    使用 FFmpeg 将帧序列合并为视频。

    Args:
        frame_dir: 帧目录
        output_path: 输出视频路径
        fps: 帧率
        crf: 视频质量 (越小越好，推荐 18-23)
    """
    ffmpeg = get_ffmpeg_path()
    input_pattern = os.path.join(frame_dir, "frame_%06d.jpg")

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
        output_path
    ]

    print(f"正在合并帧为视频 → {output_path}")
    subprocess.run(cmd, check=True)
    print("视频生成完成。")


if __name__ == "__main__":
    print("=" * 60)
    print("ProPainter 视频去水印工具 v1.0")
    print("=" * 60)

    # 简单验证 FFmpeg
    ffmpeg = get_ffmpeg_path()
    print(f"FFmpeg: {ffmpeg}")

    # 验证 CUDA
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {props.total_mem / 1024**3:.1f} GB")
        print(f"CUDA: {torch.version.cuda}")
    else:
        print("WARNING: CUDA 不可用，将使用 CPU (极慢)")
