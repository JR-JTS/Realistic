"""
Shadow Removal & Enhancement Module
Implements:
  1. Illumination-based shadow removal (Retinex + color correction)
  2. AI-driven shadow removal using inpainting-style CNN
  3. Shadow region enhancement (denoising + sharpening)
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from scipy.ndimage import gaussian_filter


# ────────────────────────────────────────────────
# Shadow Removal Methods
# ────────────────────────────────────────────────

def remove_shadow_illumination(img_bgr: np.ndarray, soft_mask: np.ndarray,
                                strength: float = 0.85) -> np.ndarray:
    """
    Multi-Scale Retinex + color transfer based shadow removal.
    Estimates illumination from shadow/non-shadow statistics.
    """
    img_f = img_bgr.astype(np.float32)
    h, w = img_f.shape[:2]

    shadow_bin = (soft_mask > 0.4).astype(bool)
    lit_bin = ~shadow_bin

    result = img_f.copy()

    if shadow_bin.sum() == 0 or lit_bin.sum() < 100:
        return img_bgr

    # ── Per-channel gain correction (shadow → lit statistics)
    for c in range(3):
        ch = img_f[:, :, c]
        shadow_vals = ch[shadow_bin]
        lit_vals = ch[lit_bin]

        # Robust mean (trim outliers)
        s_mean = np.percentile(shadow_vals, [20, 80])
        l_mean = np.percentile(lit_vals, [20, 80])

        s_avg = np.mean(shadow_vals[(shadow_vals >= s_mean[0]) & (shadow_vals <= s_mean[1])])
        l_avg = np.mean(lit_vals[(lit_vals >= l_mean[0]) & (lit_vals <= l_mean[1])])

        if s_avg > 0:
            gain = l_avg / (s_avg + 1e-6)
            gain = np.clip(gain, 1.0, 3.5)  # reasonable range
            result[:, :, c] = np.where(shadow_bin, ch * (1 + (gain - 1) * strength * soft_mask), ch)

    # ── Retinex illumination normalization on shadow region
    retinex = _multi_scale_retinex(result.astype(np.uint8))
    blend_w = soft_mask[:, :, np.newaxis] * 0.3 * strength
    result = result * (1 - blend_w) + retinex.astype(np.float32) * blend_w

    return np.clip(result, 0, 255).astype(np.uint8)


def _multi_scale_retinex(img: np.ndarray, sigmas=(15, 80, 250)) -> np.ndarray:
    """Multi-Scale Retinex for illumination normalization"""
    img_f = img.astype(np.float32) + 1.0
    log_img = np.log(img_f)

    msr = np.zeros_like(log_img)
    for sigma in sigmas:
        blurred = gaussian_filter(img_f, sigma=[sigma, sigma, 0])
        msr += log_img - np.log(blurred + 1.0)

    msr /= len(sigmas)

    # Normalize to [0, 255]
    for c in range(3):
        ch = msr[:, :, c]
        p2, p98 = np.percentile(ch, [2, 98])
        msr[:, :, c] = np.clip((ch - p2) / (p98 - p2 + 1e-8) * 255, 0, 255)

    return msr.astype(np.uint8)


def remove_shadow_color_transfer(img_bgr: np.ndarray, soft_mask: np.ndarray,
                                  strength: float = 0.85) -> np.ndarray:
    """
    Lab color space transfer: transfer color statistics from lit region to shadow region
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
    h, w = lab.shape[:2]

    shadow_bin = (soft_mask > 0.4).astype(bool)
    lit_bin = ~shadow_bin

    if shadow_bin.sum() == 0 or lit_bin.sum() < 100:
        return img_bgr

    result_lab = lab.copy()

    for c in range(3):
        ch = lab[:, :, c]
        s_mean, s_std = ch[shadow_bin].mean(), ch[shadow_bin].std() + 1e-8
        l_mean, l_std = ch[lit_bin].mean(), ch[lit_bin].std() + 1e-8

        # Color transfer in shadow region
        corrected = (ch - s_mean) * (l_std / s_std) + l_mean

        alpha = soft_mask * strength
        result_lab[:, :, c] = np.where(shadow_bin, ch * (1 - alpha) + corrected * alpha, ch)

    result_lab = np.clip(result_lab, 0, 255)
    result_bgr = cv2.cvtColor(result_lab.astype(np.uint8), cv2.COLOR_Lab2BGR)
    return result_bgr


# ────────────────────────────────────────────────
# Enhancement Module (after shadow removal)
# ────────────────────────────────────────────────

def enhance_shadow_region(img_bgr: np.ndarray, soft_mask: np.ndarray,
                           sharpen_strength: float = 1.5,
                           denoise_strength: int = 7,
                           clahe_clip: float = 2.5) -> np.ndarray:
    """
    Enhance shadow region quality:
    1. Denoising (Non-local means)
    2. CLAHE (adaptive histogram equalization)
    3. Unsharp masking (sharpening)
    """
    result = img_bgr.copy().astype(np.float32)

    # ── Step 1: Denoise shadow region
    h = max(1, min(denoise_strength, 10))
    denoised = cv2.fastNlMeansDenoisingColored(img_bgr, None, h, h, 7, 21)

    # ── Step 2: CLAHE on L channel
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2Lab)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    clahe_result = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)

    # ── Step 3: Unsharp masking for sharpening
    blurred = cv2.GaussianBlur(clahe_result, (0, 0), 2.5)
    sharpened = cv2.addWeighted(clahe_result, 1.0 + sharpen_strength * 0.5,
                                  blurred, -sharpen_strength * 0.5, 0)

    # ── Blend: only apply enhancement in shadow region
    mask_3ch = soft_mask[:, :, np.newaxis]
    blended = (img_bgr.astype(np.float32) * (1 - mask_3ch) +
               sharpened.astype(np.float32) * mask_3ch)

    return np.clip(blended, 0, 255).astype(np.uint8)


def enhance_full_image(img_bgr: np.ndarray, 
                        brightness: float = 1.0,
                        contrast: float = 1.0,
                        saturation: float = 1.0,
                        sharpness: float = 1.0) -> np.ndarray:
    """Apply global enhancement to full image"""
    img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

    if brightness != 1.0:
        img_pil = ImageEnhance.Brightness(img_pil).enhance(brightness)
    if contrast != 1.0:
        img_pil = ImageEnhance.Contrast(img_pil).enhance(contrast)
    if saturation != 1.0:
        img_pil = ImageEnhance.Color(img_pil).enhance(saturation)
    if sharpness != 1.0:
        img_pil = ImageEnhance.Sharpness(img_pil).enhance(sharpness)

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


# ────────────────────────────────────────────────
# AI Enhancement: Super-Resolution style (SRCNN-lite)
# ────────────────────────────────────────────────

class LightEnhanceNet(nn.Module):
    """
    Lightweight image enhancement CNN.
    Residual learning: predicts the correction to add to the image.
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.res_blocks = nn.Sequential(
            *[ResBlock(16) for _ in range(4)]
        )
        self.output = nn.Conv2d(16, 3, 3, padding=1)

    def forward(self, x):
        feat = self.features(x)
        feat = self.res_blocks(feat)
        residual = torch.tanh(self.output(feat)) * 0.3
        return torch.clamp(x + residual, 0, 1)


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)


def enhance_with_ai(img_bgr: np.ndarray, enhance_model: Optional[nn.Module],
                     soft_mask: np.ndarray, device: str = 'cpu') -> np.ndarray:
    """Apply AI enhancement selectively to shadow region"""
    if enhance_model is None:
        return enhance_shadow_region(img_bgr, soft_mask)

    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).unsqueeze(0).to(device)

    enhance_model.eval()
    with torch.no_grad():
        enhanced_t = enhance_model(tensor)

    enhanced_rgb = (enhanced_t[0].cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
    enhanced_bgr = cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2BGR)

    # Blend only in shadow region
    mask_3ch = soft_mask[:, :, np.newaxis]
    blended = (img_bgr.astype(np.float32) * (1 - mask_3ch) +
               enhanced_bgr.astype(np.float32) * mask_3ch)

    return np.clip(blended, 0, 255).astype(np.uint8)
