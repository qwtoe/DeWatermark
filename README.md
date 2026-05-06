# ClearFrame

**Production-grade video watermark removal with extreme memory optimization.**

ClearFrame removes watermarks, logos, and unwanted objects from videos using the state-of-the-art [ProPainter](https://github.com/sczhou/ProPainter) deep learning model — specifically engineered to run on **low-VRAM GPUs (4–6 GB)** without crashing.

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/CUDA-11.8%2B-76B900?logo=nvidia" alt="CUDA">
  <img src="https://img.shields.io/badge/VRAM-4GB%2B-important" alt="VRAM">
  <img src="https://img.shields.io/badge/license-MIT-blue" alt="License">
</p>

---

## Why ClearFrame?

Most video inpainting tools load an entire video into GPU memory — impossible on consumer GPUs. ClearFrame was built from the ground up with a **disk-first, chunked-inference** architecture:

| Challenge | Standard Approach | ClearFrame |
|-----------|------------------|------------|
| Video loading | Entire video → RAM/VRAM | FFmpeg subprocess → JPEG frames on disk |
| Model precision | FP32 (wastes VRAM) | FP16 throughout inference |
| Frame batching | All frames at once | 5-frame chunks with triple cleanup |
| Resolution | Original (OOM on 6 GB) | Auto‑scales to safe resolution for your VRAM budget |
| Memory leaks | Accumulates across batches | `del` + `gc.collect()` + `torch.cuda.empty_cache()` per chunk |

**Tested on:** NVIDIA CMP 30HX (6 GB VRAM) with 8 GB system RAM.

---

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/ClearFrame.git
cd ClearFrame

# Windows (recommended)
install.bat

# Linux / macOS
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

The installer automatically sets up PyTorch CUDA, all Python dependencies, and `imageio-ffmpeg`.

### 2. Run

```bash
# Auto-detect VRAM and scale resolution automatically
python remove_watermark.py -i video.mp4 -o clean.mp4
```

On first run, ClearFrame will:
1. Clone the ProPainter source code (one-time, ~100 MB)
2. Download pretrained weights (one-time, ~700 MB)
3. Open an interactive window to select the watermark region

### 3. Select Watermark

A window opens showing your video's first frame. **Click and drag** to draw rectangles around each watermark region. Press **SPACE** to confirm, **ESC** when done. The mask is saved as `mask.png` for reuse.

---

## CLI Reference

```
python remove_watermark.py -i INPUT [-o OUTPUT] [OPTIONS]
```

### Core Options

| Flag | Default | Description |
|------|---------|-------------|
| `-i`, `--input` | *(required)* | Input video path |
| `-o`, `--output` | `output_clean.mp4` | Output video path |
| `-m`, `--mask` | `mask.png` | Mask file (auto‑generated if missing) |

### Memory / Degradation Strategy

| Flag | Default | Description |
|------|---------|-------------|
| `--resize` | `auto` | Inference resolution: `auto` (smart detect), `none` (original), or a number like `480` / `720` |
| `--chunk-size` | `5` | Frames per GPU batch. Reduce to `3` if OOM persists |
| `--raft-iter` | `20` | RAFT optical flow iterations. Lower to `10` for speed |
| `--vram-budget` | *(auto)* | VRAM cap in GB for auto‑degradation logic (e.g., `--vram-budget 4`) |

### Precision Control

| Flag | Description |
|------|-------------|
| `--fp32` | Force FP32 precision (uses more VRAM) |
| `--no-cuda` | Force CPU inference (extremely slow, last resort) |

### Video Quality

| Flag | Default | Description |
|------|---------|-------------|
| `--crf` | `18` | Output video quality (lower = better, 18–23 recommended) |
| `--quality` | `95` | JPEG frame quality (1–100) |

### Utility

| Flag | Description |
|------|-------------|
| `--keep-temp` | Keep temporary files for debugging |
| `--force-mask` | Force re‑selection of watermark region |

---

## Usage Examples

```bash
# Basic usage — auto-scales for your VRAM
python remove_watermark.py -i input.mp4

# Force downscale to 480p for GPU inference (output restores original resolution)
python remove_watermark.py -i input.mp4 --resize 480

# 12 GB GPU — skip downscaling entirely
python remove_watermark.py -i input.mp4 --resize none

# 4 GB GPU budget — even more conservative auto-degradation
python remove_watermark.py -i input.mp4 --vram-budget 4

# Max survival mode for 4 GB cards
python remove_watermark.py -i input.mp4 --resize 480 --chunk-size 3 --vram-budget 4

# Reuse an existing mask
python remove_watermark.py -i input.mp4 -o output.mp4 --mask my_mask.png
```

---

## How It Works

### Architecture

```
Input Video
    │
    ▼
[FFmpeg subprocess]  ──►  frame_000000.jpg
                          frame_000001.jpg    ← on disk, never in RAM
                          ...
                          frame_001549.jpg
    │
    ▼
[OpenCV selectROI]   ──►  mask.png           ← interactive watermark selection
    │
    ▼
┌─── Chunk Loop (5 frames / batch) ──────────────────────┐
│                                                         │
│  Load JPEGs → Tensor(FP16) → GPU → ProPainter → CPU     │
│                                                         │
│  del tensor + gc.collect() + torch.cuda.empty_cache()   │
│                                                         │
└─────────────────────────────────────────────────────────┘
    │
    ▼
[FFmpeg subprocess]  ──►  output_clean.mp4
```

### Auto-Degradation Table

When `--resize auto` (default), ClearFrame picks a safe inference resolution based on your VRAM budget:

| VRAM Budget | Max Safe Resolution | Typical Use Case |
|-------------|---------------------|------------------|
| ≤ 4 GB | 360p | GTX 1050 Ti, MX series |
| ≤ 6.5 GB | 540p | GTX 1060, CMP 30HX, RTX 2060 |
| ≤ 10 GB | 720p | RTX 3060, RTX 3080 |
| > 10 GB | 1080p | RTX 3090, RTX 4090 |

**Output video is always reconstructed at the original resolution.** Only the GPU inference step is downscaled.

### Triple Memory Cleanup

After every chunk:

```python
del frames_tensor, mask_tensor, result_tensor, result_np  # drop references
gc.collect()                                               # reclaim Python heap
torch.cuda.empty_cache()                                   # release CUDA cache
torch.cuda.synchronize()                                   # flush pending ops
```

---

## Requirements

- **Python** 3.10+
- **NVIDIA GPU** with ≥ 4 GB VRAM (CPU mode supported but extremely slow)
- **CUDA** 11.8+ (driver)
- **Git** (for auto‑cloning ProPainter)
- **Disk space** ~2 GB (for weights + temporary frames)

Full dependency list: [`requirements.txt`](requirements.txt)

---

## Project Structure

```
ClearFrame/
├── remove_watermark.py    # Main script (single file, ~950 lines)
├── requirements.txt       # Python dependencies
├── install.bat            # One-click Windows installer
├── .gitignore
├── mask.png               # Generated watermark mask
├── ProPainter/            # Auto-cloned from GitHub (on first run)
│   └── weights/           # Downloaded pretrained models (~700 MB)
└── README.md
```

---

## FAQ

<details>
<summary><b>Q: CUDA out of memory — what now?</b></summary>

Try progressively more aggressive settings:

```bash
# Level 1: Let auto-degradation work (default)
python remove_watermark.py -i video.mp4

# Level 2: Force 480p
python remove_watermark.py -i video.mp4 --resize 480

# Level 3: 480p + smaller chunks
python remove_watermark.py -i video.mp4 --resize 480 --chunk-size 3

# Level 4: 480p + 3-frame chunks + 4 GB VRAM budget
python remove_watermark.py -i video.mp4 --resize 480 --chunk-size 3 --vram-budget 4

# Level 5: CPU mode (hours, not minutes)
python remove_watermark.py -i video.mp4 --no-cuda
```
</details>

<details>
<summary><b>Q: The OpenCV selection window shows garbled text?</b></summary>

This is a Windows encoding issue. All UI text has been switched to English in recent versions. If you still see garbled characters, the selection still works — just follow the on‑screen instructions (drag to select, SPACE to confirm, ESC to finish).
</details>

<details>
<summary><b>Q: Can I process very long videos (1 hour+)?</b></summary>

Yes. ClearFrame never loads the full video into memory. Processing time scales linearly with frame count. For a 1080p video downscaled to 480p on a 6 GB GPU, expect roughly 2–5 seconds per frame.
</details>

<details>
<summary><b>Q: Does it support audio?</b></summary>

Currently, output video is video‑only. If you need audio, you can merge it afterward with FFmpeg:

```bash
ffmpeg -i output_clean.mp4 -i original.mp4 -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 final.mp4
```
</details>

---

## Acknowledgments

ClearFrame is built on top of the excellent [ProPainter](https://github.com/sczhou/ProPainter) model by Zhou et al. All inpainting credit belongs to the original authors.

```
@inproceedings{zhou2023propainter,
  title={ProPainter: Improving Propagation and Transformer for Video Inpainting},
  author={Zhou, Shangchen and Li, Chongyi and Chan, Kelvin CK and Loy, Chen Change},
  booktitle={ICCV},
  year={2023}
}
```

---

## License

MIT License. See the [ProPainter license](https://github.com/sczhou/ProPainter/blob/main/LICENSE) for the underlying model.
