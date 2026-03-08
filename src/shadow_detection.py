"""
Shadow Detection Module  –  드론 항공사진 특화  v4.0
────────────────────────────────────────────────────
v4.0 핵심 재설계:
  - 청색편이가 없는 도시 그림자도 탐지 (건물/구조물 그림자)
  - 절대 밝기 + 국소 대비 + 엣지 기반 경계 인식 통합
  - 그림자 내부 텍스처 보존 (과대 마스크 방지)
  - 두 가지 탐지 모드: 'conservative'(엄격) / 'aggressive'(관대)
  - sensitivity 파라미터로 연속 조정 가능
"""

import cv2
import numpy as np
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────
# 1.  전통 CV 기반 그림자 탐지 (v4.0 전면 재설계)
# ──────────────────────────────────────────────────────────

def detect_shadow_cv(img_bgr: np.ndarray,
                     sensitivity: float = 0.5) -> np.ndarray:
    """
    드론 항공사진 그림자 탐지 v4.0

    설계 원칙:
    1. 청색편이 여부와 무관하게 어두운 영역 탐지
    2. 국소 대비(가우시안 피라미드)로 상대적으로 어두운 영역 탐지
    3. 엣지 기반 그림자 경계 인식
    4. 형태학 연산으로 노이즈 제거 및 경계 정제

    sensitivity: 0.1(엄격, 확실한 그림자만) ~ 1.0(관대, 어두운 영역 전부)
    """
    h, w = img_bgr.shape[:2]

    # ── 색공간 변환
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
    ycr = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    V  = hsv[:, :, 2]
    S  = hsv[:, :, 1]
    L  = lab[:, :, 0]
    b_ch = lab[:, :, 2]
    Y  = ycr[:, :, 0]
    Cb = ycr[:, :, 2]

    votes = np.zeros((h, w), dtype=np.float32)

    # ────────────────────────────────────────────
    # Cue 1: 절대 밝기 낮음 (전체 이미지 기준)
    # 하위 percentile 임계값 → 확실하게 어두운 픽셀
    # sensitivity 높을수록 더 많은 어두운 영역 포함
    # ────────────────────────────────────────────
    pct_thresh = 20 + sensitivity * 25   # 0.5→32.5%, 1.0→45%
    v_thresh   = np.percentile(V, pct_thresh)
    votes     += (V < v_thresh).astype(np.float32) * 1.5

    # ────────────────────────────────────────────
    # Cue 2: 국소 대비 - 주변보다 어두운 영역
    # 두 가지 스케일 사용 (작은 스케일: 지역 그림자, 큰 스케일: 전체 분위기)
    # ────────────────────────────────────────────
    # 소규모 국소 (반경 약 5% 크기)
    blur_r1 = max(11, int(min(h, w) * 0.05) | 1)
    v_local1 = cv2.GaussianBlur(V, (blur_r1, blur_r1), blur_r1 / 3.0)
    rel_thresh1 = 0.82 - sensitivity * 0.12   # 0.5→0.76, 1.0→0.70
    votes += (V < v_local1 * rel_thresh1).astype(np.float32) * 1.5

    # 대규모 국소 (반경 약 15% 크기)
    blur_r2 = max(31, int(min(h, w) * 0.15) | 1)
    v_local2 = cv2.GaussianBlur(V, (blur_r2, blur_r2), blur_r2 / 3.0)
    rel_thresh2 = 0.75 - sensitivity * 0.10
    votes += (V < v_local2 * rel_thresh2).astype(np.float32) * 1.0

    # ────────────────────────────────────────────
    # Cue 3: Lab L 채널 낮음
    # ────────────────────────────────────────────
    l_thresh = np.percentile(L, pct_thresh)
    votes   += (L < l_thresh).astype(np.float32) * 1.0

    # ────────────────────────────────────────────
    # Cue 4: 청색편이 보너스 (있으면 가중치 추가, 없어도 탐지 가능)
    # ────────────────────────────────────────────
    cb_median = float(np.median(Cb))
    cb_std    = float(Cb.std()) + 0.5
    # Cb가 중앙값보다 높고, Y도 낮으면 → 전형적인 하늘 반사 그림자
    cb_score  = np.clip((Cb - cb_median) / (cb_std + 1e-6), 0, 2).astype(np.float32)
    y_thresh  = np.percentile(Y, 30 + sensitivity * 15)
    blue_shadow = ((Cb > cb_median) & (Y < y_thresh)).astype(np.float32)
    votes    += blue_shadow * cb_score * 0.8   # 보너스

    # ────────────────────────────────────────────
    # Cue 5: S/V 비율 - 그림자는 채도/밝기 비율 유지
    # (완전한 검은 물체는 S도 낮아서 비율이 낮음)
    # ────────────────────────────────────────────
    sv_ratio  = S.astype(np.float32) / (V + 2.0)
    sv_median = float(np.median(sv_ratio))
    # sv_ratio가 중앙값 이상인 픽셀 (채도 비율 유지)
    votes    += (sv_ratio >= sv_median * 0.8).astype(np.float32) * 0.5

    # ────────────────────────────────────────────
    # Cue 6: 엣지 근처 어두운 영역 (그림자 경계 패턴)
    # ────────────────────────────────────────────
    edges     = cv2.Canny(img_bgr, 30, 80).astype(np.float32) / 255.0
    # 엣지를 팽창시켜서 그림자 경계 근처 영역 표시
    edge_dil  = cv2.dilate(edges, np.ones((15, 15), np.uint8))
    # 엣지 근처이면서 어두운 픽셀 → 그림자 경계 후보
    near_edge_dark = (edge_dil > 0.5) & (V < v_thresh * 1.5)
    votes    += near_edge_dark.astype(np.float32) * 0.5

    # ────────────────────────────────────────────
    # 투표 임계값 결정
    # 최대 점수: 1.5+1.5+1.0+1.0+~2.0+0.5+0.5 ≈ 8.0
    # sensitivity=0.5 → 임계 3.5
    # ────────────────────────────────────────────
    vote_thresh = 4.5 - sensitivity * 2.5   # 0.1→4.25, 0.5→3.25, 1.0→2.0
    shadow = (votes >= vote_thresh).astype(np.uint8) * 255

    # ────────────────────────────────────────────
    # 형태학 정제
    # ────────────────────────────────────────────
    # 작은 잡음 제거
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    shadow  = cv2.morphologyEx(shadow, cv2.MORPH_OPEN, k_open)

    # 구멍 메우기 (그림자 내부 밝은 작은 반점 제거)
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    shadow  = cv2.morphologyEx(shadow, cv2.MORPH_CLOSE, k_close)

    # 최종 가장자리 다듬기
    k_final = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    shadow  = cv2.morphologyEx(shadow, cv2.MORPH_OPEN, k_final)

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
