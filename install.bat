@echo off
chcp 65001 >nul
echo ============================================================
echo   ProPainter 视频去水印 - 环境安装脚本
echo   适配: NVIDIA CMP 30HX (6GB VRAM) / CUDA 13.1
echo ============================================================
echo.

REM --- 检查 Python ---
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 未找到 Python，请先安装 Python 3.10+
    pause
    exit /b 1
)
echo [OK] Python 已找到
echo.

REM --- 步骤 1: 安装 PyTorch (CUDA 12.4 版本, 兼容 CUDA 13.1 驱动) ---
echo [1/3] 安装 PyTorch CUDA 版本...
echo 使用 CUDA 12.4 构建 (兼容你的 CUDA 13.1 驱动)
echo.

REM 方案 A: 使用 pip + PyTorch 官方 CUDA 12.4 wheel (推荐)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

if %errorlevel% neq 0 (
    echo.
    echo [警告] PyTorch CUDA 12.4 安装失败，尝试 CUDA 11.8...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
)

echo.
echo [2/3] 安装其余依赖...
pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo [错误] 依赖安装失败
    pause
    exit /b 1
)

echo.
echo [3/3] 验证安装...
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA 可用: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}'); print(f'显存: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB' if torch.cuda.is_available() else '')"

if %errorlevel% neq 0 (
    echo [错误] PyTorch 验证失败
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   安装完成！
echo   下一步: 将视频文件放入当前目录，运行:
echo     python remove_watermark.py --input 你的视频.mp4
echo ============================================================
pause
