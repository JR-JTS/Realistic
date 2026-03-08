"""
Processing Pipeline  –  드론 사진 그림자 제거 + 색상 복원
"""

import cv2
import numpy as np
import time
import os
from pathlib import Path
from typing import Optional, Tuple, Dict
import torch

from shadow_detection import (
    ShadowDetectorNet,
    detect_shadow_cv,
    detect_shadow_ai,
    get_soft_mask,
)
from color_restoration import (
    ColorRestorationNet,
    restore_shadow_color,
)


# ──────────────────────────────────────────────────────────
# 지원 확장자
# ──────────────────────────────────────────────────────────
SUPPORTED_EXT = {'.jpg', '.jpeg', '.png', '.tif', '.tiff',
                  '.bmp', '.webp', '.JPG', '.JPEG', '.PNG',
                  '.TIF', '.TIFF'}


# ──────────────────────────────────────────────────────────
# 전역 모델 (앱 시작 시 1회 로드)
# ──────────────────────────────────────────────────────────
_device        = 'cpu'
_shadow_model: Optional[ShadowDetectorNet]   = None
_color_model:  Optional[ColorRestorationNet] = None


def load_models(model_dir: str = "models") -> Dict[str, bool]:
    global _shadow_model, _color_model

    status = {}

    # Shadow detection model
    sp = os.path.join(model_dir, "shadow_detector.pth")
    if os.path.exists(sp):
        try:
            _shadow_model = ShadowDetectorNet()
            _shadow_model.load_state_dict(torch.load(sp, map_location=_device))
            _shadow_model.eval()
            status['shadow'] = True
        except Exception as e:
            print(f"Shadow model load failed: {e}")
            _shadow_model = None
            status['shadow'] = False
    else:
        status['shadow'] = False

    # Color restoration model
    cp = os.path.join(model_dir, "color_restore.pth")
    if os.path.exists(cp):
        try:
            _color_model = ColorRestorationNet()
            _color_model.load_state_dict(torch.load(cp, map_location=_device))
            _color_model.eval()
            status['color'] = True
        except Exception as e:
            print(f"Color model load failed: {e}")
            _color_model = None
            status['color'] = False
    else:
        status['color'] = False

    return status


# ──────────────────────────────────────────────────────────
# 단일 이미지 처리
# ──────────────────────────────────────────────────────────

def process_single(
    img_bgr: np.ndarray,
    # 탐지
    detection_mode: str = 'hybrid',      # 'cv' | 'hybrid'
    sensitivity: float  = 0.5,
    feather: int        = 25,
    # 복원
    radio_strength: float   = 0.80,
    color_strength: float   = 0.65,
    retinex_strength: float = 0.30,
    use_ai_color: bool      = True,
    # 선명화
    denoise_h: int         = 6,
    sharpen_amount: float  = 1.4,
    clahe_clip: float      = 2.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Returns: (result, binary_mask, soft_mask, stats_dict)
    """
    stats = {}
    h, w  = img_bgr.shape[:2]

    # 1. Shadow Detection
    t0 = time.perf_counter()
    if detection_mode == 'hybrid':
        binary = detect_shadow_ai(img_bgr, _shadow_model, _device)
    else:
        binary = detect_shadow_cv(img_bgr, sensitivity)
    soft = get_soft_mask(binary, feather)
    stats['detect_ms'] = round((time.perf_counter() - t0) * 1000, 1)
    stats['shadow_pct'] = round(float((binary > 0).sum()) / (h * w) * 100, 1)

    # 2. Color Restoration
    t0 = time.perf_counter()
    ai_model = _color_model if use_ai_color else None
    result = restore_shadow_color(
        img_bgr, soft,
        radio_strength=radio_strength,
        color_strength=color_strength,
        retinex_strength=retinex_strength,
        ai_model=ai_model,
        device=_device,
        denoise_h=denoise_h,
        sharpen_amount=sharpen_amount,
        clahe_clip=clahe_clip,
    )
    stats['restore_ms'] = round((time.perf_counter() - t0) * 1000, 1)
    stats['total_ms']   = stats['detect_ms'] + stats['restore_ms']

    return result, binary, soft, stats


# ──────────────────────────────────────────────────────────
# 폴더 스캔
# ──────────────────────────────────────────────────────────

def scan_folder(folder: str) -> list:
    """폴더 내 지원 이미지 파일 목록 반환 (재귀 없음)"""
    p = Path(folder)
    files = [str(f) for f in sorted(p.iterdir())
             if f.is_file() and f.suffix in SUPPORTED_EXT]
    return files


def scan_folder_recursive(folder: str) -> list:
    """하위 폴더까지 재귀 탐색"""
    p = Path(folder)
    files = [str(f) for f in sorted(p.rglob('*'))
             if f.is_file() and f.suffix in SUPPORTED_EXT]
    return files


# ──────────────────────────────────────────────────────────
# 비교 이미지 생성
# ──────────────────────────────────────────────────────────

def make_compare(orig: np.ndarray,
                  result: np.ndarray,
                  binary: np.ndarray,
                  soft: np.ndarray) -> np.ndarray:
    """4-panel: 원본 | 그림자 마스크(채색) | 복원 | diff"""
    h, w = orig.shape[:2]

    # 그림자 마스크 시각화
    mask_vis = orig.copy()
    overlay  = np.zeros_like(orig)
    overlay[binary > 0] = [0, 80, 255]
    mask_vis = cv2.addWeighted(mask_vis, 0.55, overlay, 0.45, 0)

    # Diff (복원 전후 차이 × 3 강조)
    diff = np.clip((result.astype(np.int32) - orig.astype(np.int32)) * 3 + 128, 0, 255).astype(np.uint8)

    font  = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.45, min(1.1, w / 900))
    thick = max(1, int(scale * 2))

    def label(img, text, col=(255,255,255)):
        out = img.copy()
        (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
        cv2.rectangle(out, (6, 6), (16+tw, 18+th), (0,0,0), -1)
        cv2.putText(out, text, (11, 11+th), font, scale, col, thick, cv2.LINE_AA)
        return out

    panels = [
        label(orig,     "Original",    (200,200,200)),
        label(mask_vis, "Shadow Mask", (0,160,255)),
        label(result,   "Restored",    (80,255,80)),
        label(diff,     "Diff x3",     (255,200,80)),
    ]
    return cv2.hconcat(panels)
