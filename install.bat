@echo off
echo ============================================================
echo   DeWatermark �?Environment Setup
echo   Target: NVIDIA CMP 30HX (6GB VRAM) / CUDA 13.1
echo ============================================================
echo.

REM --- Check Python ---
python --version >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.10+
    pause
    exit /b 1
)
echo [OK] Python found
echo.

REM --- Step 1: Install PyTorch CUDA ---
echo [1/3] Installing PyTorch CUDA 12.4 ...
echo.

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
if errorlevel 1 (
    echo.
    echo [WARN] CUDA 12.4 failed, trying CUDA 11.8 ...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
)

echo.
echo [2/3] Installing other dependencies (including imageio-ffmpeg) ...
pip install -r requirements.txt

if errorlevel 1 (
    echo [ERROR] Dependency installation failed
    pause
    exit /b 1
)

echo.
echo [3/3] Verifying installation ...
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
python -c "from remove_watermark import get_ffmpeg_path; print('FFmpeg:', get_ffmpeg_path())"

echo.
echo ============================================================
echo   Setup complete!
echo   Run: python remove_watermark.py -i your_video.mp4
echo ============================================================
pause
