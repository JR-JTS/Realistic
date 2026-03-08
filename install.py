"""
패키지 설치 스크립트
"""
import subprocess
import sys

packages = [
    "opencv-python",
    "Pillow",
    "numpy",
    "scipy",
]

torch_packages = [
    "--index-url", "https://download.pytorch.org/whl/cpu",
    "torch", "torchvision",
]

print("📦 기본 패키지 설치 중...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + packages)

print("📦 PyTorch (CPU) 설치 중...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + torch_packages)

print("✅ 모든 패키지 설치 완료!")
print("▶  python main.py  로 앱을 실행하세요.")
