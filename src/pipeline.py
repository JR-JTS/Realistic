"""
Processing Pipeline  –  드론 사진 그림자 제거 + 색상 복원  v3.0

처리 경로:
  preview_mode=True  → 최대 800px 썸네일로 처리 (즉각 미리보기용, < 500ms 목표)
  preview_mode=False → 원본 해상도 풀 처리 (배치 저장용)
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
SUPPORTED_EXT = {
    '.jpg', '.jpeg', '.png', '.tif', '.tiff',
    '.bmp', '.webp',
    '.JPG', '.JPEG', '.PNG', '.TIF', '.TIFF',
}


# ──────────────────────────────────────────────────────────
# 전역 모델
# ──────────────────────────────────────────────────────────
_device        = 'cpu'
_shadow_model: Optional[ShadowDetectorNet]   = None
_color_model:  Optional[ColorRestorationNet] = None


def load_models(model_dir: str = "models") -> Dict[str, bool]:
    global _shadow_model, _color_model
    status = {}

    sp = os.path.join(model_dir, "shadow_detector.pth")
    if os.path.exists(sp):
        try:
            _shadow_model = ShadowDetectorNet()
            _shadow_model.load_state_dict(
                torch.load(sp, map_location=_device, weights_only=True))
            _shadow_model.eval()
            status['shadow'] = True
            print(f"Shadow model loaded: {sp}")
        except Exception as e:
            print(f"Shadow model load failed: {e}")
            _shadow_model = None
            status['shadow'] = False
    else:
        print(f"No shadow model at {sp} → using CV detection")
        status['shadow'] = False

    cp = os.path.join(model_dir, "color_restore.pth")
    if os.path.exists(cp):
        try:
            _color_model = ColorRestorationNet()
            _color_model.load_state_dict(
                torch.load(cp, map_location=_device, weights_only=True))
            _color_model.eval()
            status['color'] = True
            print(f"Color model loaded: {cp}")
        except Exception as e:
            print(f"Color model load failed: {e}")
            _color_model = None
            status['color'] = False
    else:
        print(f"No color model at {cp} → using CV restoration")
        status['color'] = False

    return status


# ──────────────────────────────────────────────────────────
# 단일 이미지 처리
# ──────────────────────────────────────────────────────────

PREVIEW_MAX_PX = 800   # 미리보기용 최대 픽셀 (한 변 기준)
BATCH_MAX_PX   = 0     # 배치 저장용 (0 = 무제한)


def _resize_for_processing(img: np.ndarray, max_px: int):
    """max_px 이하로 비율 축소. max_px=0이면 원본 반환."""
    if max_px <= 0:
        return img, 1.0
    h, w = img.shape[:2]
    long_side = max(h, w)
    if long_side <= max_px:
        return img, 1.0
    scale = max_px / long_side
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def process_single(
    img_bgr: np.ndarray,
    # 탐지
    detection_mode: str   = 'hybrid',
    sensitivity: float    = 0.45,
    feather: int          = 20,
    # Shadow/Highlight 복원 (v4.0)
    shadow_amount: float     = 0.70,    # 그림자 밝기 복원 강도
    highlight_amount: float  = 0.20,    # 하이라이트 압축 (과보정 방지)
    midtone_contrast: float  = 0.15,    # 중간톤 대비
    color_strength: float    = 0.35,    # 색상 편이 보정
    use_ai_color: bool       = True,
    # 품질 개선
    denoise_h: int           = 5,
    sharpen_amount: float    = 0.8,
    clahe_clip: float        = 2.0,
    # 하위 호환성
    radio_strength: float    = 0.70,
    retinex_strength: float  = 0.20,
    # 처리 모드
    preview_mode: bool       = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    단일 이미지 처리.
    Returns: (result_bgr, binary_mask, soft_mask, stats_dict)

    preview_mode=True  : PREVIEW_MAX_PX 이하로 처리 후 원본 크기로 upscale 반환
    preview_mode=False : 원본 해상도 그대로 처리
    """
    stats  = {}
    orig_h, orig_w = img_bgr.shape[:2]

    # ── 처리 해상도 결정
    max_px   = PREVIEW_MAX_PX if preview_mode else BATCH_MAX_PX
    work_img, scale = _resize_for_processing(img_bgr, max_px)
    wh, ww = work_img.shape[:2]

    # ── 1. 그림자 탐지 (work 해상도)
    t0 = time.perf_counter()
    try:
        if detection_mode == 'hybrid' and _shadow_model is not None:
            binary = detect_shadow_ai(work_img, _shadow_model, _device)
        else:
            binary = detect_shadow_cv(work_img, sensitivity)
    except Exception as e:
        print(f"Shadow detection error: {e}, fallback to CV")
        binary = detect_shadow_cv(work_img, sensitivity)

    soft = get_soft_mask(binary, feather)
    stats['detection_ms'] = round((time.perf_counter() - t0) * 1000, 1)
    stats['detect_ms']    = stats['detection_ms']
    stats['shadow_pct']   = round(float((binary > 0).sum()) / max(wh * ww, 1) * 100, 1)

    # ── 2. 색상 복원 (work 해상도에서 처리)
    t0 = time.perf_counter()
    ai_model = _color_model if use_ai_color else None
    try:
        result = restore_shadow_color(
            work_img, soft,
            shadow_amount     = shadow_amount,
            highlight_amount  = highlight_amount,
            midtone_contrast  = midtone_contrast,
            color_strength    = color_strength,
            ai_model          = ai_model,
            device            = _device,
            denoise_h         = denoise_h,
            sharpen_amount    = sharpen_amount,
            clahe_clip        = clahe_clip,
            radio_strength    = radio_strength,
            retinex_strength  = retinex_strength,
        )
    except Exception as e:
        print(f"Color restoration error: {e}")
        result = work_img.copy()

    stats['restoration_ms'] = round((time.perf_counter() - t0) * 1000, 1)
    stats['restore_ms']     = stats['restoration_ms']
    stats['total_ms']       = stats['detection_ms'] + stats['restoration_ms']
    stats['preview_mode']   = preview_mode
    stats['work_size']      = f"{ww}×{wh}"

    # ── 3. 미리보기 모드: 결과를 원본 크기로 upscale
    if scale < 1.0:
        result = cv2.resize(result, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        binary = cv2.resize(binary, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        soft   = cv2.resize(soft,   (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    return result, binary, soft, stats


# ──────────────────────────────────────────────────────────
# 폴더 스캔
# ──────────────────────────────────────────────────────────

def scan_folder(folder: str) -> list:
    p = Path(folder)
    if not p.is_dir():
        return []
    return [str(f) for f in sorted(p.iterdir())
            if f.is_file() and f.suffix in SUPPORTED_EXT]


def scan_folder_recursive(folder: str) -> list:
    p = Path(folder)
    if not p.is_dir():
        return []
    return [str(f) for f in sorted(p.rglob('*'))
            if f.is_file() and f.suffix in SUPPORTED_EXT]


# ──────────────────────────────────────────────────────────
# 비교 이미지 생성
# ──────────────────────────────────────────────────────────

def make_compare(orig: np.ndarray,
                  result: np.ndarray,
                  binary: np.ndarray,
                  soft: np.ndarray) -> np.ndarray:
    """4-panel: 원본 | 그림자 마스크 | 복원 | Diff×3"""
    h, w = orig.shape[:2]

    mask_vis = orig.copy()
    overlay  = np.zeros_like(orig)
    overlay[binary > 0] = [0, 80, 255]
    mask_vis = cv2.addWeighted(mask_vis, 0.55, overlay, 0.45, 0)

    diff = np.clip(
        (result.astype(np.int32) - orig.astype(np.int32)) * 3 + 128,
        0, 255).astype(np.uint8)

    font  = cv2.FONT_HERSHEY_SIMPLEX
    sc    = max(0.45, min(1.1, w / 900))
    thick = max(1, int(sc * 2))

    def label(img, text, col=(255, 255, 255)):
        out = img.copy()
        (tw, th), _ = cv2.getTextSize(text, font, sc, thick)
        cv2.rectangle(out, (6, 6), (16 + tw, 18 + th), (0, 0, 0), -1)
        cv2.putText(out, text, (11, 11 + th), font, sc, col, thick, cv2.LINE_AA)
        return out

    return cv2.hconcat([
        label(orig,     "Original",    (200, 200, 200)),
        label(mask_vis, "Shadow Mask", (0, 160, 255)),
        label(result,   "Restored",    (80, 255, 80)),
        label(diff,     "Diff x3",     (255, 200, 80)),
    ])
