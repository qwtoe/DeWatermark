@echo off
echo ============================================================
echo   ProPainter Watermark Remover - Environment Setup
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

REM --- Step 0: Install FFmpeg ---
echo [0/4] Installing FFmpeg ...

REM Option A: conda (if available)
where conda >nul 2>nul
if not errorlevel 1 (
    conda install -c conda-forge ffmpeg -y >nul 2>nul
    if not errorlevel 1 (
        echo [OK] FFmpeg installed via conda
        goto :ffmpeg_done
    )
)

REM Option B: winget (Windows 10/11)
where winget >nul 2>nul
if not errorlevel 1 (
    winget install Gyan.FFmpeg --accept-package-agreements --silent >nul 2>nul
    if not errorlevel 1 (
        echo [OK] FFmpeg installed via winget
        goto :ffmpeg_done
    )
)

echo [INFO] FFmpeg not found system-wide, will use imageio-ffmpeg (Python fallback)

:ffmpeg_done
echo.

REM --- Step 1: Install PyTorch CUDA ---
echo [1/4] Installing PyTorch CUDA 12.4 ...
echo.

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
if errorlevel 1 (
    echo.
    echo [WARN] CUDA 12.4 failed, trying CUDA 11.8 ...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
)

echo.
echo [2/4] Installing other dependencies ...
pip install -r requirements.txt

if errorlevel 1 (
    echo [ERROR] Dependency installation failed
    pause
    exit /b 1
)

echo.
echo [3/4] Verifying PyTorch ...
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

if errorlevel 1 (
    echo [ERROR] PyTorch verification failed
    pause
    exit /b 1
)

echo.
echo [4/4] Verifying FFmpeg ...
python -c "from remove_watermark import get_ffmpeg_path; print('FFmpeg:', get_ffmpeg_path())"

echo.
echo ============================================================
echo   Setup complete!
echo   Run: python remove_watermark.py -i your_video.mp4
echo ============================================================
pause
