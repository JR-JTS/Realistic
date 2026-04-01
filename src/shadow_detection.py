"""
Shadow Detection Module  -  드론 항공사진 특화  v5.1
------------------------------------------------------
v5.0 완전 재설계 - 조명비율(Illumination Ratio) 물리 모델 기반

[핵심 물리 원리]
  태양광 아래 지표면 픽셀의 밝기:
    I(x) = R(x) * (L_direct + L_ambient)   <- 직사광 + 하늘 산란광
  그림자 픽셀의 밝기:
    I_shadow(x) = R(x) * L_ambient         <- 하늘 산란광만

  따라서:  I_shadow / I_lit = L_ambient / (L_direct + L_ambient) = 상수 k
           같은 재질이라면 shadow/lit 비율이 일정 (0.2~0.6)

  결론:
    1. 그림자 = 주변(같은 재질의 밝은 버전) 대비 ratio가 낮음
    2. 아스팔트 도로 = 주변도 똑같이 어두움 -> ratio ~ 1.0 -> 그림자 아님
    3. 퍼센타일 기반 절대 밝기 조건 완전 제거 (이미지마다 달라 오검출 원인)

[구현]
  ratio(x) = L(x) / L_blur_large(x)  (LAB L채널 사용)
  ratio 낮음  -> 주변보다 어두움 -> 그림자
  ratio ~ 1.0 -> 주변과 비슷    -> 아스팔트/검은지붕 -> 제외
  + LAB b* 채널 청색편이 보너스 (하늘 산란광만 받으면 b* 감소)
"""

import cv2
import numpy as np
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------
# 1.  CV 기반 그림자 탐지 v5.0 - 조명비율 물리 모델
# ----------------------------------------------------------

def _fast_large_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """
    큰 sigma GaussianBlur의 빠른 근사.
    sigma >= 20 이면 이미지를 1/8로 축소 후 blur → 원래 크기로 복원.
    sigma < 20 이면 일반 GaussianBlur 사용.
    속도: 기존 GaussianBlur(sigma=256) 대비 약 200~400배 빠름.
    """
    if sigma < 20:
        return cv2.GaussianBlur(img, (0, 0), sigma)
    h, w = img.shape[:2]
    scale = 8
    sw, sh = max(w // scale, 4), max(h // scale, 4)
    small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
    small_blur = cv2.GaussianBlur(small, (0, 0), sigma / scale)
    return cv2.resize(small_blur, (w, h), interpolation=cv2.INTER_LINEAR)


def detect_shadow_cv(img_bgr: np.ndarray,
                     sensitivity: float = 0.5) -> np.ndarray:
    """
    드론 항공사진 그림자 탐지 v5.0 - 조명비율(illumination ratio) 기반

    핵심 원리:
      ratio(x) = L(x) / L_blur_large(x)
      그림자 픽셀  -> ratio 낮음 (주변 평균보다 훨씬 어두움)
      아스팔트 도로 -> ratio ~ 1.0 (주변도 똑같이 어두움) -> 자동 제외

    sensitivity: 0.1(엄격) ~ 1.0(관대)
    """
    h, w = img_bgr.shape[:2]
    short = min(h, w)
    eps = 1.0

    # 색공간 변환
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

    L = lab[:, :, 0]   # 밝기 (지각적 균일)
    b = lab[:, :, 2]   # 청-황 축: 그림자는 b* 감소 (청색편이)
    V = hsv[:, :, 2]
    S = hsv[:, :, 1]

    score = np.zeros((h, w), dtype=np.float32)

    # ============================================================
    # [핵심] 다중 스케일 조명비율
    # sigma 작음: 건물 1개 크기 내 비교
    # sigma 큼:   블록/구역 크기 내 비교
    #
    # ratio = L(x) / L_blur(x)
    # ratio 낮을수록 -> 주변보다 어두움 -> 그림자 점수 부여
    # 아스팔트처럼 넓게 어두운 곳은 L_blur도 낮으므로 ratio~1 -> 점수 0
    # ============================================================
    ratio_thresh = 0.55 + sensitivity * 0.20  # 0.1->0.57  0.45->0.64  1.0->0.75

    scales = [
        (short * 0.05, 1.0),   # 소: 건물 1개
        (short * 0.12, 1.2),   # 중: 블록
        (short * 0.25, 1.0),   # 대: 구역
    ]
    for sigma, weight in scales:
        L_blur = _fast_large_blur(L, sigma)
        ratio = L / (L_blur + eps)
        # ratio 가 낮을수록 더 강한 그림자 신호
        shadow_cue = np.clip((ratio_thresh - ratio) / ratio_thresh, 0.0, 1.0)
        score += shadow_cue * weight

    # ============================================================
    # LAB b* 채널 - 청색편이 보너스
    # 직사광선 없이 하늘 산란광만 받으면 b* 감소 (파란쪽으로)
    # 아스팔트는 주변도 b* 낮음 -> 주변 대비 변화 없음 -> 보너스 없음
    # ============================================================
    ks_b = short * 0.08
    b_blur = _fast_large_blur(b, ks_b)
    b_diff = b_blur - b        # 양수 = 현재 픽셀이 주변보다 더 청색편이
    b_score = np.clip(b_diff / 10.0, 0.0, 1.0)
    score += b_score * 0.8

    # ============================================================
    # 자연 어두운 표면 페널티
    # 채도 낮음(S<30) + V<60 + v_ratio>0.80 = 아스팔트/검은지붕
    # ratio 기반에서 이미 대부분 걸러지지만 추가 안전망
    # ============================================================
    ks_v = short * 0.12
    V_blur = _fast_large_blur(V, ks_v)
    v_ratio = V / (V_blur + eps)
    natural_dark = (S < 30) & (V < 60) & (v_ratio > 0.80)
    score -= natural_dark.astype(np.float32) * 1.5

    # ============================================================
    # 이진화
    # score 최대 ~ 3.2 + 0.8 = 4.0
    # 0.1->2.08  0.45->1.66  1.0->1.0
    # ============================================================
    final_thresh = 2.2 - sensitivity * 1.2
    shadow = (score >= final_thresh).astype(np.uint8) * 255

    # 형태학 정제
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    shadow  = cv2.morphologyEx(shadow, cv2.MORPH_OPEN,  k_open)
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    shadow  = cv2.morphologyEx(shadow, cv2.MORPH_CLOSE, k_close)
    k_final = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    shadow  = cv2.morphologyEx(shadow, cv2.MORPH_OPEN,  k_final)

    return shadow


# ----------------------------------------------------------
# 2.  UNet-style CNN (경량) - AI 그림자 탐지용
# ----------------------------------------------------------

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
    """경량 UNet (in=3, out=1 sigmoid)"""
    def __init__(self):
        super().__init__()
        self.e1 = nn.Sequential(ConvBNReLU(3,   32),  ConvBNReLU(32,  32))
        self.e2 = nn.Sequential(ConvBNReLU(32,  64),  ConvBNReLU(64,  64))
        self.e3 = nn.Sequential(ConvBNReLU(64,  128), ConvBNReLU(128, 128))
        self.bn = nn.Sequential(ConvBNReLU(128, 256), ConvBNReLU(256, 256))
        self.d3 = nn.Sequential(ConvBNReLU(256+128, 128), ConvBNReLU(128, 128))
        self.d2 = nn.Sequential(ConvBNReLU(128+64,  64),  ConvBNReLU(64,  64))
        self.d1 = nn.Sequential(ConvBNReLU(64+32,   32),  ConvBNReLU(32,  32))
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
    """AI + CV 조합"""
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

    mask_ai = (pred[:h, :w] > 0.40).astype(np.uint8) * 255
    mask_cv  = detect_shadow_cv(img_bgr)

    # OR 방식: 둘 중 하나라도 탐지하면 포함
    combined = cv2.bitwise_or(mask_ai, mask_cv)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    return cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k)


# ──────────────────────────────────────────────────────────
# 3.  Soft (feathered) mask
# ──────────────────────────────────────────────────────────

def get_soft_mask(binary_mask: np.ndarray,
                  feather: int = 25) -> np.ndarray:
    """
    이진 마스크 → [0,1] soft mask (거리변환 + 가우시안 페더링)
    """
    if binary_mask.max() == 0:
        return np.zeros_like(binary_mask, dtype=np.float32)

    feather = max(3, feather)

    dist = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
    soft = np.clip(dist / max(feather * 0.7, 1.0), 0, 1)

    ks = min(feather | 1, 51)
    if ks % 2 == 0:
        ks += 1
    soft = cv2.GaussianBlur(soft, (ks, ks), feather / 4.0)

    mx = soft.max()
    if mx > 1e-6:
        soft /= mx

    return soft.clip(0.0, 1.0).astype(np.float32)
