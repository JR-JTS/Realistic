"""
Shadow Color Restoration Module
────────────────────────────────
핵심 목표: 그림자로 인해 어두워진 영역의 「색상(Color)」을 복원

접근법:
  1. [물리 기반]  조명 추정 → 그림자 제거
     - 그림자는 전방향 산란광(skylight)만 받고 직사광은 차단됨
     - Shadow-free region의 통계로 원래 조명을 역추정
     - 채널별 라디오메트릭 보정 (gain + offset)

  2. [색상 전달]  Reinhard Lab 색 전달
     - 비그림자 영역의 Lab 통계를 그림자 영역에 전달

  3. [AI 색상 복원]  경량 CNN  (색상 사전 학습 없이도 동작)
     - 잔차(residual) 학습: 그림자 이미지 → 색상 보정값 예측
     - 구조적으로 밝기를 올리면서 색상 왜곡(blue-shift) 제거

  4. [후처리]  CLAHE + Unsharp Mask → 복원된 영역 선명화
"""

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════
# STEP 1 : 물리 기반 조명 보정  (Radiometric Correction)
# ══════════════════════════════════════════════════════════

def radiometric_correction(img_bgr: np.ndarray,
                            soft_mask: np.ndarray,
                            strength: float = 0.85) -> np.ndarray:
    """
    그림자/비그림자 영역의 채널별 통계로 조명 게인(gain)과
    오프셋(offset)을 추정하여 그림자 픽셀을 보정.

    gain   = mean_lit / mean_shadow  (밝기 스케일)
    offset = 색상 편이 (blue-shift) 보정
    """
    shadow_bin = soft_mask > 0.45
    lit_bin    = soft_mask < 0.1

    if shadow_bin.sum() < 50 or lit_bin.sum() < 200:
        return img_bgr

    img_f  = img_bgr.astype(np.float64)
    result = img_f.copy()

    gains   = np.ones(3)
    offsets = np.zeros(3)

    for c in range(3):
        ch = img_f[:, :, c]
        sv = ch[shadow_bin]
        lv = ch[lit_bin]

        # 로버스트 통계 (5~95 퍼센타일)
        s_p5,  s_p95  = np.percentile(sv, [5, 95])
        l_p5,  l_p95  = np.percentile(lv, [5, 95])
        sv_r = sv[(sv >= s_p5) & (sv <= s_p95)]
        lv_r = lv[(lv >= l_p5) & (lv <= l_p95)]

        s_mean = sv_r.mean() if len(sv_r) else sv.mean()
        l_mean = lv_r.mean() if len(lv_r) else lv.mean()
        s_std  = sv_r.std()  if len(sv_r) else 1.0
        l_std  = lv_r.std()  if len(lv_r) else 1.0

        if s_mean > 1:
            gains[c]   = np.clip(l_mean / s_mean, 0.8, 4.0)
        if s_std > 0.5:
            # 그림자의 색상 왜곡(주로 파란쪽) 보정
            offsets[c] = (l_mean - s_mean * gains[c]) * 0.4

    # 그림자 영역에만 선택적 적용  (soft blend)
    for c in range(3):
        corrected = img_f[:, :, c] * gains[c] + offsets[c]
        alpha     = soft_mask * strength
        result[:, :, c] = img_f[:, :, c] * (1 - alpha) + corrected * alpha

    return np.clip(result, 0, 255).astype(np.uint8)


# ══════════════════════════════════════════════════════════
# STEP 2 : Lab 색 전달 (Reinhard Color Transfer)
# ══════════════════════════════════════════════════════════

def lab_color_transfer(img_bgr: np.ndarray,
                        soft_mask: np.ndarray,
                        strength: float = 0.7) -> np.ndarray:
    """
    그림자 영역 픽셀의 Lab 통계를 비그림자 영역 통계로 이동.
    색상(a, b 채널)과 밝기(L 채널) 모두 보정.
    """
    shadow_bin = soft_mask > 0.45
    lit_bin    = soft_mask < 0.1

    if shadow_bin.sum() < 50 or lit_bin.sum() < 200:
        return img_bgr

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
    result = lab.copy()

    for c in range(3):
        sv = lab[:, :, c][shadow_bin]
        lv = lab[:, :, c][lit_bin]

        s5, s95 = np.percentile(sv, [5, 95])
        l5, l95 = np.percentile(lv, [5, 95])
        sv = sv[(sv >= s5) & (sv <= s95)]
        lv = lv[(lv >= l5) & (lv <= l95)]

        s_mean, s_std = sv.mean(), max(sv.std(), 0.5)
        l_mean, l_std = lv.mean(), max(lv.std(), 0.5)

        # Reinhard transfer
        corrected = (lab[:, :, c] - s_mean) * (l_std / s_std) + l_mean
        alpha     = soft_mask * strength
        result[:, :, c] = lab[:, :, c] * (1 - alpha) + corrected * alpha

    result = np.clip(result, 0, 255).astype(np.uint8)
    return cv2.cvtColor(result, cv2.COLOR_Lab2BGR)


# ══════════════════════════════════════════════════════════
# STEP 3 : Multi-Scale Retinex (조명 성분 제거)
# ══════════════════════════════════════════════════════════

def retinex_shadow_lighten(img_bgr: np.ndarray,
                            soft_mask: np.ndarray,
                            strength: float = 0.4,
                            sigmas: Tuple = (15, 80, 200)) -> np.ndarray:
    """
    Multi-Scale Retinex로 조명 성분을 제거하여 반사율(색상) 복원.
    그림자 영역에만 선택적으로 적용.
    """
    img_f  = img_bgr.astype(np.float32) + 1.0
    msr    = np.zeros_like(img_f)

    for sigma in sigmas:
        blurred = gaussian_filter(img_f, sigma=[sigma, sigma, 0])
        msr    += np.log(img_f) - np.log(blurred + 1.0)

    msr /= len(sigmas)

    # 정규화  → [0,255]
    msr_norm = np.zeros_like(msr)
    for c in range(3):
        ch = msr[:, :, c]
        p2, p98 = np.percentile(ch, [1, 99])
        msr_norm[:, :, c] = np.clip((ch - p2) / (p98 - p2 + 1e-6) * 255, 0, 255)

    alpha  = soft_mask[:, :, np.newaxis] * strength
    result = img_bgr.astype(np.float32) * (1 - alpha) + msr_norm * alpha
    return np.clip(result, 0, 255).astype(np.uint8)


# ══════════════════════════════════════════════════════════
# STEP 4 : AI 색상 보정 CNN  (Residual Color Network)
# ══════════════════════════════════════════════════════════

class ColorRestorationNet(nn.Module):
    """
    그림자 이미지 입력 → 색상 보정 잔차(residual) 출력.
    사전 학습 없이도 초기 가중치로 약한 보정 제공;
    파인튜닝 시 정밀 복원 가능.
    """
    def __init__(self):
        super().__init__()

        # Encoder
        self.enc = nn.Sequential(
            nn.Conv2d(3,  64, 3, padding=1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.LeakyReLU(0.2, inplace=True),
        )
        # 색상 컨텍스트 (넓은 수용 영역)
        self.ctx = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=2,  dilation=2),  nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, 3, padding=4,  dilation=4),  nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, 3, padding=8,  dilation=8),  nn.LeakyReLU(0.2, inplace=True),
        )
        # Decoder
        self.dec = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64,  32, 3, padding=1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32,   3, 1),
        )
        # Xavier 초기화 (처음 실행 시도 의미있는 출력)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        e   = self.enc(x)
        c   = self.ctx(e)
        ec  = torch.cat([e, c], dim=1)
        res = torch.tanh(self.dec(ec)) * 0.25   # 작은 잔차 (-0.25 ~ +0.25)
        return torch.clamp(x + res, 0, 1)


def ai_color_restore(img_bgr: np.ndarray,
                      soft_mask: np.ndarray,
                      model: Optional[nn.Module],
                      device: str = 'cpu') -> np.ndarray:
    """AI 모델로 그림자 영역 색상 복원"""
    if model is None:
        return img_bgr

    h, w = img_bgr.shape[:2]
    # 패딩 (8의 배수)
    ph = (8 - h % 8) % 8
    pw = (8 - w % 8) % 8

    img_f = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
    img_p = np.pad(img_f, ((0, ph), (0, pw), (0, 0)), mode='reflect')
    t     = torch.from_numpy(img_p.transpose(2, 0, 1)).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        out = model(t)

    out_np  = out[0].cpu().numpy().transpose(1, 2, 0)[:h, :w]
    out_bgr = cv2.cvtColor((out_np * 255).clip(0, 255).astype(np.uint8),
                             cv2.COLOR_RGB2BGR)

    # 그림자 영역에만 블렌딩
    m = soft_mask[:, :, np.newaxis]
    blended = img_bgr.astype(np.float32) * (1 - m) + out_bgr.astype(np.float32) * m
    return blended.clip(0, 255).astype(np.uint8)


# ══════════════════════════════════════════════════════════
# STEP 5 : 선명화  (복원 후 디테일 강화)
# ══════════════════════════════════════════════════════════

def sharpen_shadow_region(img_bgr: np.ndarray,
                           soft_mask: np.ndarray,
                           denoise_h: int = 6,
                           sharpen_amount: float = 1.4,
                           clahe_clip: float = 2.0) -> np.ndarray:
    """
    그림자 복원 후 선명화:
      1. NL-Means 노이즈 제거
      2. CLAHE (L 채널)
      3. Unsharp Masking
    그림자 영역에만 soft blend 적용
    """
    h_val  = max(3, min(int(denoise_h), 12))
    denoised = cv2.fastNlMeansDenoisingColored(img_bgr, None, h_val, h_val, 7, 21)

    # CLAHE on L channel
    lab  = cv2.cvtColor(denoised, cv2.COLOR_BGR2Lab)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)

    # Unsharp mask
    blurred   = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
    sharpened = cv2.addWeighted(enhanced, 1 + sharpen_amount * 0.4,
                                 blurred, -sharpen_amount * 0.4, 0)

    m       = soft_mask[:, :, np.newaxis]
    result  = img_bgr.astype(np.float32) * (1 - m) + sharpened.astype(np.float32) * m
    return result.clip(0, 255).astype(np.uint8)


# ══════════════════════════════════════════════════════════
# STEP 6 : 통합  색상 복원 파이프라인
# ══════════════════════════════════════════════════════════

def restore_shadow_color(img_bgr: np.ndarray,
                          soft_mask: np.ndarray,
                          # 단계별 가중치
                          radio_strength: float = 0.8,
                          color_strength: float = 0.65,
                          retinex_strength: float = 0.3,
                          # AI
                          ai_model: Optional[nn.Module] = None,
                          device: str = 'cpu',
                          # 선명화
                          denoise_h: int = 6,
                          sharpen_amount: float = 1.4,
                          clahe_clip: float = 2.0) -> np.ndarray:
    """
    완전한 색상 복원 파이프라인.
    1) 라디오메트릭 보정 (조명 gain/offset)
    2) Lab 색 전달  (색상 통계 매칭)
    3) Multi-Scale Retinex (반사율 강조)
    4) AI 색상 보정 (있으면)
    5) 선명화
    """
    # 1. 라디오메트릭 보정
    step1 = radiometric_correction(img_bgr, soft_mask, radio_strength)

    # 2. Lab 색 전달
    step2 = lab_color_transfer(step1, soft_mask, color_strength)

    # 3. Retinex (반사율 강조)  - 그림자 영역만
    step3 = retinex_shadow_lighten(step2, soft_mask, retinex_strength)

    # 4. AI 색상 보정
    step4 = ai_color_restore(step3, soft_mask, ai_model, device)

    # 5. 선명화 (그림자 영역)
    step5 = sharpen_shadow_region(step4, soft_mask, denoise_h,
                                   sharpen_amount, clahe_clip)
    return step5
