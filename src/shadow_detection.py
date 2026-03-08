"""
Shadow Detection Module for Drone Images
Combines traditional CV + AI-based detection
"""

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


# ────────────────────────────────────────────────
# Lightweight CNN-based Shadow Detector
# ────────────────────────────────────────────────
class ShadowDetectorNet(nn.Module):
    """Lightweight encoder-decoder for shadow mask prediction"""

    def __init__(self):
        super().__init__()
        # Encoder
        self.enc1 = self._block(3, 32)
        self.enc2 = self._block(32, 64)
        self.enc3 = self._block(64, 128)

        # Bottleneck
        self.bottleneck = self._block(128, 256)

        # Decoder
        self.dec3 = self._block(256 + 128, 128)
        self.dec2 = self._block(128 + 64, 64)
        self.dec1 = self._block(64 + 32, 32)

        self.out_conv = nn.Conv2d(32, 1, kernel_size=1)
        self.pool = nn.MaxPool2d(2, 2)

    def _block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        b = self.bottleneck(self.pool(e3))

        d3 = self.dec3(torch.cat([F.interpolate(b, e3.shape[2:], mode='bilinear', align_corners=False), e3], dim=1))
        d2 = self.dec2(torch.cat([F.interpolate(d3, e2.shape[2:], mode='bilinear', align_corners=False), e2], dim=1))
        d1 = self.dec1(torch.cat([F.interpolate(d2, e1.shape[2:], mode='bilinear', align_corners=False), e1], dim=1))

        return torch.sigmoid(self.out_conv(d1))


# ────────────────────────────────────────────────
# Traditional CV-based Shadow Detection
# ────────────────────────────────────────────────
def detect_shadow_cv(img_bgr: np.ndarray, sensitivity: float = 0.5) -> np.ndarray:
    """
    Multi-cue shadow detection using HSV, Lab color space.
    Returns binary mask (255 = shadow, 0 = non-shadow)
    """
    h, w = img_bgr.shape[:2]

    # ── Cue 1: HSV – low saturation + low value regions
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    v_thresh = np.percentile(V, 30 + sensitivity * 20)
    shadow_v = (V < v_thresh).astype(np.uint8)

    # ── Cue 2: Lab – low L channel
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
    L = lab[:, :, 0]
    l_thresh = np.percentile(L, 25 + sensitivity * 20)
    shadow_l = (L < l_thresh).astype(np.uint8)

    # ── Cue 3: Ratio of blue channel (shadows have bluish tint)
    img_f = img_bgr.astype(np.float32) + 1e-6
    b_ratio = img_f[:, :, 0] / (img_f[:, :, 2] + 1e-6)   # B/(R)
    b_thresh = np.percentile(b_ratio, 70)
    shadow_b = (b_ratio > b_thresh).astype(np.uint8)

    # ── Combine cues
    combined = (shadow_v.astype(int) + shadow_l.astype(int) + shadow_b.astype(int))
    threshold_votes = 2
    shadow_mask = (combined >= threshold_votes).astype(np.uint8) * 255

    # ── Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_CLOSE, kernel)
    shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    return shadow_mask


def detect_shadow_ai(img_bgr: np.ndarray, model: Optional[nn.Module], device: str = 'cpu') -> np.ndarray:
    """AI-based shadow detection using UNet-style CNN"""
    if model is None:
        return detect_shadow_cv(img_bgr)

    h, w = img_bgr.shape[:2]
    # Pad to multiple of 8
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_padded = np.pad(img_rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')

    tensor = torch.from_numpy(img_padded.transpose(2, 0, 1)).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        pred = model(tensor)

    mask = pred[0, 0].cpu().numpy()
    mask = mask[:h, :w]
    binary = (mask > 0.5).astype(np.uint8) * 255

    # Refine with CV cues
    cv_mask = detect_shadow_cv(img_bgr)
    refined = cv2.bitwise_or(binary, cv_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel)

    return refined


def get_soft_shadow_mask(binary_mask: np.ndarray, blur_radius: int = 25) -> np.ndarray:
    """Convert binary mask to soft (feathered) mask for smooth blending"""
    soft = cv2.GaussianBlur(binary_mask.astype(np.float32), (blur_radius | 1, blur_radius | 1), 0)
    soft = soft / (soft.max() + 1e-8)
    return soft
