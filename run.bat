@echo off
chcp 65001 > nul
title Drone Shadow Remover ^& Color Restorer

echo =========================================
echo   Drone Shadow Remover ^& Color Restorer
echo =========================================

:: Python 확인
python --version > nul 2>&1
if errorlevel 1 (
    echo ❌ Python이 설치되어 있지 않습니다.
    echo    https://www.python.org 에서 Python 3.10+ 를 설치하세요.
    pause
    exit /b 1
)

:: 의존성 설치 확인
python -c "import cv2, numpy, PIL, scipy, torch, tkinter" 2>nul
if errorlevel 1 (
    echo 📦 필수 패키지를 설치합니다. 잠시 기다려주세요...
    pip install opencv-python Pillow numpy scipy --quiet
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --quiet
    echo ✅ 설치 완료
)

echo 🚀 앱을 시작합니다...
python main.py

if errorlevel 1 (
    echo.
    echo ❌ 오류가 발생했습니다. 아래 오류를 확인하세요.
    pause
)
