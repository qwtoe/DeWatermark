# AGENTS.md — DeWatermark

## What This Is

Single-file video watermark removal tool built on ProPainter. The entire application is `remove_watermark.py` (~1500 lines). ProPainter/ is an auto-cloned third-party dependency — do not modify it.

## Setup

```bash
# Windows
install.bat

# Linux/macOS
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

PyTorch CUDA must be installed separately (not in requirements.txt). First run auto-clones ProPainter source (~100 MB) and downloads weights (~700 MB) into `ProPainter/` and `weights/`.

## Running

```bash
python remove_watermark.py -i video.mp4 -o clean.mp4
```

First run opens an OpenCV GUI for mask selection (drag rectangles, SPACE confirm, ESC done). Mask saved as `mask.png`, flow mask as `flow_mask.png` — both reused on subsequent runs.

## Architecture (Critical for Editing)

The pipeline runs in this exact order — do not reorder:

1. **FFmpeg** extracts frames to disk as JPEGs (never loaded into RAM as a whole video)
2. **OpenCV selectROI** generates mask + flow_mask (dilation=5 and dilation=8 respectively)
3. **Model loading** — RAFT, flow_complete, ProPainter (RAFT freed after step 4)
4. **precompute_all_flows()** — RAFT optical flow + flow completion for ALL frames, stored on CPU
5. **RAFT freed** (`del fix_raft; gc.collect(); torch.cuda.empty_cache()`)
6. **process_chunks()** — sliding window inference using pre-computed flows + ProPainter only
7. **FFmpeg** reassembles frames into video, optionally merging audio from source

### Key Design Constraints

- **VRAM budget**: Target 4–6 GB. All inference must fit in this. RAFT is freed before ProPainter inference.
- **RAM budget**: Target 8 GB system RAM. No `frame_accum` numpy arrays — frames are written to disk immediately.
- **FP16**: Models run in FP16 except RAFT (FP32 for accuracy). Input tensors must be `.half()` when `use_fp16=True` or you get `Input type (float) and bias type (struct c10::Half)` error.
- **Mask vs flow_mask**: Two separate masks — `mask` (dilation=5, for inpainting) and `flow_mask` (dilation=8, for optical flow). Do not merge them.
- **Blend with original**: Model output only replaces the mask region. Non-mask pixels must use original frame: `blended = model_output * mask + original * (1 - mask)`. Forgetting this causes black screen.

## Common Pitfalls

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| Input tensors left as FP32 | `RuntimeError: Input type (float) and bias type (struct c10::Half)` | Convert to `.half()` before model input |
| Not blending with original | Black output video | `model_output * mask + original * (1 - mask)` |
| No reference frames | Poor quality, temporal inconsistency | `max_ref_frames > 0` (but 0 on ≤6GB VRAM) |
| RAFT not freed after flow computation | OOM during ProPainter inference | `del fix_raft` after `precompute_all_flows()` |
| Accumulating all frames in RAM | System swap → GPU OOM | Write frames to disk immediately, no numpy accumulation |
| Missing audio in output | Silent video | `reconstruct_video()` merges audio via FFmpeg `-map 1:a:0?` |
| Flow completion in FP16 produces NaN | Black watermark region in output | Force FP32 for flow completion network (`fix_flow_complete.float()`) |
| ProPainter model in FP16 produces NaN | Black watermark region in output | Force FP32 for ProPainter inference (`model.float()` during inference windows) |
| Transformer receives zero-masked frames | Black watermark region in output | Pass `updated_frames` (from img_propagation) to transformer, not `frames_tensor * (1 - mask)` |
| Stale flow cache with NaN data | NaN flows loaded from cache, black output | Cache loading validates for NaN and auto-invalidates |

## CLI Parameters (Non-Obvious Defaults)

- `--resize auto`: Downscale resolution based on VRAM (≤4GB→360p, ≤6.5GB→480p, ≤10GB→720p, >10GB→1080p)
- `--neighbor-length`: Default = `chunk_size` on ≤6GB VRAM (no overlap), `chunk_size*2` otherwise
- `--max-ref-frames`: Default = 0 on ≤6GB, 2 on ≤8GB, 4 on >8GB. Reference frames are disabled by default on low VRAM.
- `--flow-mask`: Separate from `--mask`. Flow mask uses larger dilation for stable optical flow.

## File Layout

```
remove_watermark.py   # Entire application (~1500 lines)
ProPainter/           # Auto-cloned at runtime — DO NOT EDIT
weights/              # Auto-downloaded model weights (~700 MB)
mask.png              # User-drawn watermark mask (auto-generated)
flow_mask.png         # Dilated mask for optical flow (auto-generated)
install.bat           # Windows one-click setup
requirements.txt      # Python deps (PyTorch installed separately)
```

## Testing

No automated test suite. Manual verification:

```bash
python remove_watermark.py -i test_video.mp4 -o test_output.mp4 --keep-temp
```

Check: output is not black, audio preserved, mask region inpainted, non-mask region unchanged.

## Git Commit Rules

- Every functional code change MUST be committed and pushed immediately after the change is verified.
- Commit messages MUST follow the Angular commit convention:
  - Format: `type(scope): subject`
  - Types: `feat` (new feature), `fix` (bug fix), `refactor` (code restructuring), `docs` (documentation), `chore` (maintenance), `perf` (performance), `test` (tests), `ci` (CI/CD)
  - Scope: short module/context name (e.g., `inference`, `mask`, `pipeline`)
  - Subject: concise English description, lowercase, no period at end
  - Body (optional): explain the "why" in more detail
  - Example: `fix(inference): pass updated_frames to transformer instead of zero-masked frames`
- All commits must be in English.
- Push to remote after each commit.