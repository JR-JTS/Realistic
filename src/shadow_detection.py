"""
Shadow Detection Module  –  드론 항공사진 특화  v3.0
────────────────────────────────────────────────────
핵심 개선 (v3.0):
  - 어두운 물체(도로, 지붕, 아스팔트)와 그림자 엄격 구분
  - 그림자 판별 기준: 청색편이(Cb↑) + 밝기 상대적 저하 + 채도/밝기 비율
  - 과탐지 방지: 절대적으로 어두운 영역(V < 35) 중 색편이 없는 것 제외
  - 형태학 연산 최적화
"""

import cv2
import numpy as np
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────
# 1.  전통 CV 기반 그림자 탐지
# ──────────────────────────────────────────────────────────

def detect_shadow_cv(img_bgr: np.ndarray,
                     sensitivity: float = 0.5) -> np.ndarray:
    """
    드론 항공사진 특화 그림자 탐지.

    그림자 특징:
      1) 주변(비그림자)보다 상대적으로 어두움
      2) 청색편이 (Cb 증가, b채널 감소)
      3) 채도/밝기 비율(S/V)이 비그림자와 유사하게 유지됨

    어두운 물체(도로·지붕) 특징:
      1) 청색편이 없음 (Cb 낮음)
      2) S/V 비율도 낮음 (채도 자체가 낮음)
      3) 가장자리 없이 균일하게 어두움

    sensitivity: 0.1(엄격) ~ 1.0(관대)
    """
    h, w = img_bgr.shape[:2]

    # ── 색공간 변환
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
    ycr = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)

    V    = hsv[:, :, 2]    # 명도 0~255
    S    = hsv[:, :, 1]    # 채도 0~255
    L    = lab[:, :, 0]    # Lab L 0~255
    b_ch = lab[:, :, 2]    # Lab b (낮을수록 파랑)
    Y    = ycr[:, :, 0]    # 휘도
    Cb   = ycr[:, :, 2]    # 청색차 (그림자에서 증가)

    votes = np.zeros((h, w), dtype=np.float32)

    # ── Cue 1: 상대 밝기 낮음 (이미지 중앙값 기준)
    # percentile이 아닌 중앙값 대비 비율로 판단 → 전체가 어두운 이미지 오탐 방지
    v_median = float(np.median(V)) + 1.0
    # 중앙값의 (70 - sensitivity*20)% 이하인 픽셀
    rel_thresh = 0.70 - sensitivity * 0.20   # 0.5→0.60, 1.0→0.50
    dark_mask = V < (v_median * rel_thresh)
    votes += dark_mask.astype(np.float32)

    # ── Cue 2: 절대 밝기 낮음 (단, 극단적으로 어두운 물체 제외)
    # 아주 어두운 (V < 30) 물체는 그림자가 아닐 가능성 높음
    l_thresh = np.percentile(L, 10 + sensitivity * 20)
    abs_dark  = (L < l_thresh) & (V > 25)   # V > 25: 완전 검정 물체 제외
    votes += abs_dark.astype(np.float32)

    # ── Cue 3: 청색편이 (그림자 핵심 특징, 가중치 2.5)
    # Cb가 높고 Y가 낮아야 진짜 그림자
    cb_median = float(np.median(Cb))
    cb_std    = float(Cb.std()) + 1.0
    # 중앙값보다 0.5 sigma 이상 높은 Cb + 밝기도 낮음
    cb_thresh = cb_median + cb_std * (0.5 - sensitivity * 0.3)
    y_thresh  = np.percentile(Y, 20 + sensitivity * 20)
    shadow_chromatic = (Cb > cb_thresh) & (Y < y_thresh)
    votes += shadow_chromatic.astype(np.float32) * 2.5  # 강한 증거

    # ── Cue 4: Lab b채널 낮음 (파란 기운)
    b_median  = float(np.median(b_ch))
    b_thresh  = b_median - (float(b_ch.std()) * (0.3 + sensitivity * 0.2))
    votes    += (b_ch < b_thresh).astype(np.float32)

    # ── Cue 5: S/(V+eps) 비율 - 그림자는 이 값이 비그림자와 유사
    # 순수한 어두운 물체(도로): S도 낮고 V도 낮음 → 비율 작음
    # 그림자: V 낮지만 S는 유지 → 비율 큼
    eps       = 2.0
    sv_ratio  = S / (V + eps)
    sv_med    = float(np.median(sv_ratio))
    # sv_ratio가 중앙값 이상인 픽셀 (채도/밝기 비율 유지됨)
    votes    += (sv_ratio >= sv_med * 0.9).astype(np.float32)

    # ── Cue 6: 국소 대비 - 주변보다 어두운 영역
    # 가우시안 블러 대비 → 로컬하게 어두운 영역
    v_blur = cv2.GaussianBlur(V, (51, 51), 20)
    local_dark = V < (v_blur * (0.88 - sensitivity * 0.08))
    votes += local_dark.astype(np.float32)

    # ── 투표 임계값 결정
    # 최대 점수: 1 + 1 + 2.5 + 1 + 1 + 1 = 7.5
    # sensitivity=0.5 → 임계 3.5
    vote_thresh = 4.5 - sensitivity * 2.0   # 0.1→4.3, 0.5→3.5, 1.0→2.5
    shadow = (votes >= vote_thresh).astype(np.uint8) * 255

    # ── 형태학 정제
    # 1) 작은 잡음 제거
    k_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    shadow  = cv2.morphologyEx(shadow, cv2.MORPH_OPEN, k_small)
    # 2) 구멍 메우기
    k_fill  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    shadow  = cv2.morphologyEx(shadow, cv2.MORPH_CLOSE, k_fill)
    # 3) 가장자리 다듬기
    k_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    shadow  = cv2.morphologyEx(shadow, cv2.MORPH_OPEN, k_clean)

    return shadow


# ──────────────────────────────────────────────────────────
# 2.  UNet-style CNN  (경량)
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
    """AI + CV 조합 (AND 방식으로 과탐지 방지)"""
    if model is None:
        return detect_shadow_cv(img_bgr)

    h, w = img_bgr.shape[:2]
    ph = (8 - h % 8) % 8
    pw = (8 - w % 8) % 8

    img_f = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
    img_p = np.pad(img_f, ((0, ph), (0, pw), (0, 0)), mode='reflect')
    t = torch.from_numpy(img_p.transpose(2, 0, 1)).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        pred = model(t)[0, 0].cpu().numpy()

    # AI 마스크 (0.45 임계)
    mask_ai = (pred[:h, :w] > 0.45).astype(np.uint8) * 255

    # CV 마스크와 AND (두 방법 모두 탐지해야 최종 포함 → 과탐지 방지)
    mask_cv  = detect_shadow_cv(img_bgr)
    combined = cv2.bitwise_and(mask_ai, mask_cv)

    # 단, AI만 탐지한 고신뢰 영역도 일부 포함 (AI score > 0.7)
    high_conf = (pred[:h, :w] > 0.70).astype(np.uint8) * 255
    combined  = cv2.bitwise_or(combined, high_conf)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    return cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k)


# ──────────────────────────────────────────────────────────
# 3.  Soft (feathered) mask
# ──────────────────────────────────────────────────────────

def get_soft_mask(binary_mask: np.ndarray,
                  feather: int = 25) -> np.ndarray:
    """
    이진 마스크 → [0,1] soft mask (거리변환 + 가우시안 페더링)
    경계를 자연스럽게 처리
    """
    if binary_mask.max() == 0:
        return np.zeros_like(binary_mask, dtype=np.float32)

    feather = max(3, feather)

    # 거리 변환 기반 soft mask
    dist = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
    # feather 거리 내에서 선형 감쇠
    soft = np.clip(dist / max(feather * 0.7, 1.0), 0, 1)

    # 가우시안 블러로 부드럽게
    ks = min(feather | 1, 51)   # 홀수 보장
    if ks % 2 == 0:
        ks += 1
    soft = cv2.GaussianBlur(soft, (ks, ks), feather / 4.0)

    mx = soft.max()
    if mx > 1e-6:
        soft /= mx

    return soft.clip(0.0, 1.0).astype(np.float32)
