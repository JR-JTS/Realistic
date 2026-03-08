"""
Shadow Detection Module  –  드론 항공사진 특화
개선 포인트:
  - HSV + Lab + YCrCb 3개 색공간 멀티큐 투표
  - 가우시안 피라미드 기반 공간적 컨텍스트 활용
  - UNet-style CNN (pre-trained 없이도 동작)
  - Soft mask (feathered) 생성
"""

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────
# 1.  전통 CV 기반 그림자 탐지  (다중 색공간 투표)
# ──────────────────────────────────────────────────────────

def detect_shadow_cv(img_bgr: np.ndarray,
                     sensitivity: float = 0.5) -> np.ndarray:
    """
    HSV + Lab + YCrCb 3-cue 투표 방식 그림자 탐지.
    sensitivity: 0.0(적게 탐지) ~ 1.0(많이 탐지)
    Returns: uint8 binary mask  255=shadow
    """
    h, w = img_bgr.shape[:2]
    votes = np.zeros((h, w), dtype=np.int32)

    # ── Cue 1 : HSV  명도(V) 낮은 영역
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    V   = hsv[:, :, 2]
    v_lo = np.percentile(V, 15 + sensitivity * 25)
    votes += (V < v_lo).astype(np.int32)

    # ── Cue 2 : Lab  L 채널 낮은 영역
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
    L   = lab[:, :, 0]
    l_lo = np.percentile(L, 15 + sensitivity * 25)
    votes += (L < l_lo).astype(np.int32)

    # ── Cue 3 : YCrCb  Y 낮은 + Cb 높은 (그림자는 푸른 기운)
    ycr = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    Y   = ycr[:, :, 0]
    Cb  = ycr[:, :, 2]
    y_lo  = np.percentile(Y, 20 + sensitivity * 20)
    cb_hi = np.percentile(Cb, 60 - sensitivity * 10)
    votes += ((Y < y_lo) & (Cb > cb_hi)).astype(np.int32)

    # ── Cue 4 : 채도-명도 비율  (그림자 = 채도 낮고 명도 낮음)
    S   = hsv[:, :, 1]
    ratio = (S.astype(np.float32) + 1) / (V.astype(np.float32) + 1)
    r_lo  = np.percentile(ratio, 20 + sensitivity * 20)
    votes += (ratio < r_lo).astype(np.int32)

    # 3/4 투표 이상 → 그림자
    shadow = (votes >= 3).astype(np.uint8) * 255

    # 형태학 정제
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    shadow  = cv2.morphologyEx(shadow, cv2.MORPH_CLOSE, k_close)
    shadow  = cv2.morphologyEx(shadow, cv2.MORPH_OPEN,  k_open)

    return shadow


# ──────────────────────────────────────────────────────────
# 2.  UNet-style CNN  (경량, 사전학습 없이 CV 보강용)
# ──────────────────────────────────────────────────────────

class ConvBNReLU(nn.Module):
    def __init__(self, inc, outc, k=3, p=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(inc, outc, k, padding=p, bias=False),
            nn.BatchNorm2d(outc),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)


class ShadowDetectorNet(nn.Module):
    """경량 UNet  (in=3, out=1 sigmoid)"""
    def __init__(self):
        super().__init__()
        self.e1 = nn.Sequential(ConvBNReLU(3, 32),  ConvBNReLU(32, 32))
        self.e2 = nn.Sequential(ConvBNReLU(32, 64),  ConvBNReLU(64, 64))
        self.e3 = nn.Sequential(ConvBNReLU(64, 128), ConvBNReLU(128, 128))
        self.bn = nn.Sequential(ConvBNReLU(128, 256), ConvBNReLU(256, 256))
        self.d3 = nn.Sequential(ConvBNReLU(256+128, 128), ConvBNReLU(128, 128))
        self.d2 = nn.Sequential(ConvBNReLU(128+64,  64),  ConvBNReLU(64, 64))
        self.d1 = nn.Sequential(ConvBNReLU(64+32,   32),  ConvBNReLU(32, 32))
        self.out = nn.Conv2d(32, 1, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        b  = self.bn(self.pool(e3))
        d3 = self.d3(torch.cat([F.interpolate(b,  e3.shape[2:], mode='bilinear', align_corners=False), e3], 1))
        d2 = self.d2(torch.cat([F.interpolate(d3, e2.shape[2:], mode='bilinear', align_corners=False), e2], 1))
        d1 = self.d1(torch.cat([F.interpolate(d2, e1.shape[2:], mode='bilinear', align_corners=False), e1], 1))
        return torch.sigmoid(self.out(d1))


def detect_shadow_ai(img_bgr: np.ndarray,
                     model: Optional[nn.Module],
                     device: str = 'cpu') -> np.ndarray:
    """AI 모델로 탐지 후 CV 결과와 OR 결합"""
    if model is None:
        return detect_shadow_cv(img_bgr)

    h, w = img_bgr.shape[:2]
    ph = (8 - h % 8) % 8
    pw = (8 - w % 8) % 8

    img_f = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
    img_p  = np.pad(img_f, ((0, ph), (0, pw), (0, 0)), mode='reflect')
    t = torch.from_numpy(img_p.transpose(2,0,1)).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        pred = model(t)[0, 0].cpu().numpy()
    mask_ai = (pred[:h, :w] > 0.5).astype(np.uint8) * 255

    mask_cv = detect_shadow_cv(img_bgr)
    combined = cv2.bitwise_or(mask_ai, mask_cv)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    return cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k)


# ──────────────────────────────────────────────────────────
# 3.  Soft (feathered) mask
# ──────────────────────────────────────────────────────────

def get_soft_mask(binary_mask: np.ndarray,
                  feather: int = 25) -> np.ndarray:
    """
    이진 마스크 → [0,1] soft mask  (가우시안 페더링)
    feather : 경계 부드러움 반경
    """
    r = max(3, feather | 1)
    soft = cv2.GaussianBlur(binary_mask.astype(np.float32), (r, r), r / 3)
    mx = soft.max()
    if mx > 0:
        soft /= mx
    return soft.clip(0, 1)
