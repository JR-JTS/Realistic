#!/bin/bash
# Drone Shadow Remover & Color Restorer
# Linux / macOS 실행 스크립트

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================="
echo "  Drone Shadow Remover & Color Restorer"
echo "========================================="

# Python 확인
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3이 설치되어 있지 않습니다."
    exit 1
fi

# 의존성 설치 확인
python3 -c "import cv2, numpy, PIL, scipy, torch, tkinter" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "📦 필수 패키지를 설치합니다..."
    pip3 install opencv-python Pillow numpy scipy torch torchvision --index-url https://download.pytorch.org/whl/cpu -q
fi

echo "🚀 앱을 시작합니다..."
python3 main.py
