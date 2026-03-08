"""
Main processing pipeline:
  1. Shadow Detection (CV / AI)
  2. Shadow Removal (illumination / color transfer)
  3. Shadow Region Enhancement (denoise + sharpen)
  4. AI Enhancement (CNN residual)
  5. Output + comparison
"""

import cv2
import numpy as np
import time
import os
from pathlib import Path
import torch
from typing import Optional, Tuple, Dict

from shadow_detection import (
    ShadowDetectorNet,
    detect_shadow_cv,
    detect_shadow_ai,
    get_soft_shadow_mask,
)
from shadow_removal import (
    remove_shadow_illumination,
    remove_shadow_color_transfer,
    enhance_shadow_region,
    enhance_with_ai,
    enhance_full_image,
    LightEnhanceNet,
)


# ────────────────────────────────────────────────
# Model Management
# ────────────────────────────────────────────────

_device = 'cpu'
_shadow_model: Optional[ShadowDetectorNet] = None
_enhance_model: Optional[LightEnhanceNet] = None


def load_models(model_dir: str = "models") -> Dict[str, bool]:
    """Load AI models if weight files exist, otherwise use CV fallback"""
    global _shadow_model, _enhance_model

    status = {}

    # Shadow detection model
    shadow_path = os.path.join(model_dir, "shadow_detector.pth")
    if os.path.exists(shadow_path):
        try:
            _shadow_model = ShadowDetectorNet()
            _shadow_model.load_state_dict(torch.load(shadow_path, map_location=_device))
            _shadow_model.eval()
            status["shadow_model"] = True
            print(f"✅ Shadow detection model loaded from {shadow_path}")
        except Exception as e:
            print(f"⚠️  Shadow model load failed: {e}, using CV fallback")
            _shadow_model = None
            status["shadow_model"] = False
    else:
        print("ℹ️  No pre-trained shadow model found → using Multi-Cue CV detection")
        status["shadow_model"] = False

    # Enhancement model
    enhance_path = os.path.join(model_dir, "enhancer.pth")
    if os.path.exists(enhance_path):
        try:
            _enhance_model = LightEnhanceNet()
            _enhance_model.load_state_dict(torch.load(enhance_path, map_location=_device))
            _enhance_model.eval()
            status["enhance_model"] = True
            print(f"✅ Enhancement model loaded from {enhance_path}")
        except Exception as e:
            print(f"⚠️  Enhance model load failed: {e}, using CV fallback")
            _enhance_model = None
            status["enhance_model"] = False
    else:
        print("ℹ️  No pre-trained enhance model found → using CLAHE + Unsharp Mask")
        status["enhance_model"] = False

    return status


# ────────────────────────────────────────────────
# Full Processing Pipeline
# ────────────────────────────────────────────────

def process_image(
    img_bgr: np.ndarray,
    # Detection params
    detection_mode: str = "ai_cv_hybrid",   # "cv_only" | "ai_cv_hybrid"
    shadow_sensitivity: float = 0.5,         # 0.0 ~ 1.0
    mask_feather: int = 25,                  # blur radius for soft mask
    # Removal params
    removal_method: str = "combined",        # "illumination" | "color_transfer" | "combined"
    removal_strength: float = 0.8,           # 0.0 ~ 1.0
    # Enhancement params
    enhance_mode: str = "ai_cv",             # "cv_only" | "ai_cv"
    sharpen_strength: float = 1.5,
    denoise_strength: int = 7,
    clahe_clip: float = 2.5,
    # Global adjustments
    brightness: float = 1.0,
    contrast: float = 1.05,
    saturation: float = 1.1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Full drone shadow processing pipeline.

    Returns:
        result_img:   Final processed image
        shadow_mask:  Binary shadow mask
        soft_mask:    Soft (feathered) shadow mask
        stats:        Timing and statistics dict
    """
    stats = {}
    h, w = img_bgr.shape[:2]

    # ── Step 1: Shadow Detection ─────────────────
    t0 = time.time()
    if detection_mode == "ai_cv_hybrid":
        shadow_mask = detect_shadow_ai(img_bgr, _shadow_model, _device)
    else:
        shadow_mask = detect_shadow_cv(img_bgr, shadow_sensitivity)

    soft_mask = get_soft_shadow_mask(shadow_mask, blur_radius=mask_feather)
    stats["detection_ms"] = round((time.time() - t0) * 1000, 1)
    stats["shadow_ratio"] = round(float((shadow_mask > 0).sum()) / (h * w) * 100, 1)

    # ── Step 2: Shadow Removal ───────────────────
    t0 = time.time()
    if removal_method == "illumination":
        removed = remove_shadow_illumination(img_bgr, soft_mask, removal_strength)
    elif removal_method == "color_transfer":
        removed = remove_shadow_color_transfer(img_bgr, soft_mask, removal_strength)
    else:  # combined (default)
        illum = remove_shadow_illumination(img_bgr, soft_mask, removal_strength * 0.6)
        color = remove_shadow_color_transfer(img_bgr, soft_mask, removal_strength * 0.5)
        removed = cv2.addWeighted(illum, 0.55, color, 0.45, 0)

    stats["removal_ms"] = round((time.time() - t0) * 1000, 1)

    # ── Step 3: Shadow Region Enhancement ───────
    t0 = time.time()
    if enhance_mode == "ai_cv":
        enhanced = enhance_with_ai(removed, _enhance_model, soft_mask, _device)
        enhanced = enhance_shadow_region(enhanced, soft_mask, sharpen_strength,
                                          denoise_strength, clahe_clip)
    else:
        enhanced = enhance_shadow_region(removed, soft_mask, sharpen_strength,
                                          denoise_strength, clahe_clip)
    stats["enhance_ms"] = round((time.time() - t0) * 1000, 1)

    # ── Step 4: Global Adjustments ───────────────
    result = enhance_full_image(enhanced, brightness, contrast, saturation)

    stats["total_ms"] = stats["detection_ms"] + stats["removal_ms"] + stats["enhance_ms"]

    return result, shadow_mask, soft_mask, stats


def create_comparison_image(original: np.ndarray, result: np.ndarray,
                              shadow_mask: np.ndarray) -> np.ndarray:
    """Create a 3-panel comparison: Original | Shadow Mask | Result"""
    h, w = original.shape[:2]

    # Shadow mask visualization (colorized)
    mask_color = np.zeros((h, w, 3), dtype=np.uint8)
    mask_color[shadow_mask > 0] = [0, 100, 255]  # orange for shadow
    mask_overlay = cv2.addWeighted(original, 0.6, mask_color, 0.4, 0)

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = max(0.5, min(1.2, w / 800))
    thick = max(1, int(fs * 2))

    def add_label(img, text, color=(255, 255, 255)):
        out = img.copy()
        (tw, th), _ = cv2.getTextSize(text, font, fs, thick)
        cv2.rectangle(out, (8, 8), (18 + tw, 20 + th), (0, 0, 0), -1)
        cv2.putText(out, text, (13, 13 + th), font, fs, color, thick, cv2.LINE_AA)
        return out

    orig_labeled   = add_label(original, "Original", (200, 200, 200))
    mask_labeled   = add_label(mask_overlay, "Shadow Mask", (0, 180, 255))
    result_labeled = add_label(result, "Processed", (100, 255, 100))

    return cv2.hconcat([orig_labeled, mask_labeled, result_labeled])


def save_results(original: np.ndarray, result: np.ndarray,
                  shadow_mask: np.ndarray, out_dir: str = "outputs",
                  prefix: str = "result") -> Dict[str, str]:
    """Save all output files"""
    os.makedirs(out_dir, exist_ok=True)

    paths = {}
    ts = int(time.time())

    result_path = os.path.join(out_dir, f"{prefix}_{ts}_processed.png")
    cv2.imwrite(result_path, result)
    paths["result"] = result_path

    mask_path = os.path.join(out_dir, f"{prefix}_{ts}_mask.png")
    cv2.imwrite(mask_path, shadow_mask)
    paths["mask"] = mask_path

    compare_path = os.path.join(out_dir, f"{prefix}_{ts}_compare.jpg")
    comparison = create_comparison_image(original, result, shadow_mask)
    cv2.imwrite(compare_path, comparison, [cv2.IMWRITE_JPEG_QUALITY, 92])
    paths["comparison"] = compare_path

    return paths
