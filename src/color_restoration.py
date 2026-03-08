"""
Shadow Color Restoration Module  v3.0
──────────────────────────────────────
핵심 수정 (v3.0):
  - 모든 보정을 오직 그림자 soft_mask > 임계값 영역에만 적용
  - 어두운 물체(도로, 지붕) 보호: 청색편이가 없는 어두운 픽셀은 보정 제외
  - gain 클리핑 더 엄격 (최대 2.0배)
  - Lab 색전달 강도 제한
  - Retinex 마스크 완전 적용
"""

import cv2
import numpy as np
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════
# 내부 유틸: 그림자 픽셀 신뢰도 마스크
# ══════════════════════════════════════════════════════════

def _shadow_confidence_mask(img_bgr: np.ndarray,
                              soft_mask: np.ndarray) -> np.ndarray:
    """
    soft_mask를 청색편이 확인으로 정제.
    청색편이(Cb↑) 없는 어두운 픽셀은 신뢰도를 낮춤 → 도로·지붕 보정 억제.
    반환: [0,1] float32 신뢰도 마스크 (soft_mask보다 더 선택적)
    """
    ycr  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    Y    = ycr[:, :, 0]
    Cb   = ycr[:, :, 2]

    # 이미지 전체 Cb 중앙값 대비 높은 픽셀 → 청색편이 있음
    cb_median = float(np.median(Cb))
    cb_std    = float(Cb.std()) + 1.0

    # 중앙값보다 얼마나 높은지 (0 ~ 1)
    blue_shift = np.clip((Cb - cb_median) / (cb_std * 1.5 + 1e-6), 0, 1).astype(np.float32)

    # 반대로, 중앙값보다 낮은 픽셀 (도로·지붕) → 신뢰도 페널티
    # Cb < 중앙값이면 보정 억제
    below_median = np.clip((cb_median - Cb) / (cb_std * 1.5 + 1e-6), 0, 1).astype(np.float32)
    penalty = 1.0 - below_median * 0.5   # 최대 50% 억제

    # 최종 신뢰도: soft_mask × 가중치 (0.25 ~ 1.0 사이)
    # 청색편이가 없으면 최소 25%만 보정
    confidence = soft_mask * (0.25 + 0.75 * blue_shift) * penalty
    return confidence.clip(0, 1).astype(np.float32)


# ══════════════════════════════════════════════════════════
# STEP 1: 물리 기반 조명 보정
# ══════════════════════════════════════════════════════════

def radiometric_correction(img_bgr: np.ndarray,
                            soft_mask: np.ndarray,
                            strength: float = 0.75) -> np.ndarray:
    """
    채널별 gain/offset을 shadow 픽셀에만 적용.
    lit 영역 통계를 기준으로 shadow 영역을 보정.
    청색편이 confidence로 어두운 물체 보정 억제.
    """
    # confidence 마스크로 실제 그림자 영역 추출
    confidence = _shadow_confidence_mask(img_bgr, soft_mask)

    # 통계용 마스크: confidence 높은 그림자 / 확실한 비그림자
    shadow_bin = confidence > 0.50   # confidence 기반 (청색편이 있는 그림자)
    lit_bin    = soft_mask < 0.06    # 확실한 비그림자

    ns = shadow_bin.sum()
    nl = lit_bin.sum()
    if ns < 100 or nl < 300:
        return img_bgr.copy()

    img_f = img_bgr.astype(np.float32)

    # 샘플링으로 통계 계산
    MAX_SAMP = 5000
    s_idx = np.where(shadow_bin.ravel())[0]
    l_idx = np.where(lit_bin.ravel())[0]
    rng = np.random.default_rng(42)
    if len(s_idx) > MAX_SAMP:
        s_idx = rng.choice(s_idx, MAX_SAMP, replace=False)
    if len(l_idx) > MAX_SAMP:
        l_idx = rng.choice(l_idx, MAX_SAMP, replace=False)

    flat = img_f.reshape(-1, 3)
    sv   = flat[s_idx]
    lv   = flat[l_idx]

    s_med = np.median(sv, axis=0).clip(5.0, None)
    l_med = np.median(lv, axis=0)

    # gain 엄격 클리핑 (1.0 ~ 1.9 배)
    gains   = np.clip(l_med / s_med, 1.0, 1.9)
    offsets = np.clip((l_med - s_med * gains) * 0.15, -15, 15)

    # 최종 alpha: confidence × strength (청색편이 없는 영역은 자동으로 작아짐)
    alpha3 = (confidence * strength).clip(0, 1)[:, :, np.newaxis]

    corrected = img_f * gains[np.newaxis, np.newaxis, :] + offsets[np.newaxis, np.newaxis, :]
    corrected = corrected.clip(0, 255)

    result = img_f * (1.0 - alpha3) + corrected * alpha3
    return np.clip(result, 0, 255).astype(np.uint8)


# ══════════════════════════════════════════════════════════
# STEP 2: Lab 색 전달 (그림자 영역 색상 편이 보정)
# ══════════════════════════════════════════════════════════

def lab_color_transfer(img_bgr: np.ndarray,
                        soft_mask: np.ndarray,
                        strength: float = 0.55) -> np.ndarray:
    """
    그림자의 청색 편이를 보정.
    shadow 영역의 Lab 통계를 lit 영역에 맞춤.
    """
    shadow_bin = soft_mask > 0.55
    lit_bin    = soft_mask < 0.08

    if shadow_bin.sum() < 100 or lit_bin.sum() < 300:
        return img_bgr.copy()

    lab_u8 = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
    lab    = lab_u8.astype(np.float32)

    MAX_SAMP = 5000
    flat  = lab.reshape(-1, 3)
    s_idx = np.where(shadow_bin.ravel())[0]
    l_idx = np.where(lit_bin.ravel())[0]
    rng = np.random.default_rng(42)
    if len(s_idx) > MAX_SAMP:
        s_idx = rng.choice(s_idx, MAX_SAMP, replace=False)
    if len(l_idx) > MAX_SAMP:
        l_idx = rng.choice(l_idx, MAX_SAMP, replace=False)

    sv = flat[s_idx]
    lv = flat[l_idx]

    s_mean = np.median(sv, axis=0)
    l_mean = np.median(lv, axis=0)
    s_std  = np.maximum(sv.std(axis=0), 1.0)
    l_std  = np.maximum(lv.std(axis=0), 1.0)

    # std 비율 클리핑 강화 (0.7 ~ 1.5)
    ratio = np.clip(l_std / s_std, 0.7, 1.5)
    ratio_3d = ratio[np.newaxis, np.newaxis, :]
    s_m      = s_mean[np.newaxis, np.newaxis, :]
    l_m      = l_mean[np.newaxis, np.newaxis, :]

    corrected = (lab - s_m) * ratio_3d + l_m
    corrected = np.clip(corrected, 0, 255)

    # 청색편이 신뢰도 적용
    confidence = _shadow_confidence_mask(img_bgr, soft_mask)
    alpha      = (confidence * strength).clip(0, 1)[:, :, np.newaxis]

    result    = lab * (1.0 - alpha) + corrected * alpha
    result_u8 = np.clip(result, 0, 255).astype(np.uint8)
    return cv2.cvtColor(result_u8, cv2.COLOR_Lab2BGR)


# ══════════════════════════════════════════════════════════
# STEP 3: 고속 Retinex (그림자 영역 밝기 보정)
# ══════════════════════════════════════════════════════════

def retinex_shadow_lighten(img_bgr: np.ndarray,
                            soft_mask: np.ndarray,
                            strength: float = 0.25) -> np.ndarray:
    """
    Multi-Scale Retinex를 그림자 영역에만 적용.
    strength=0 이면 즉시 반환.
    """
    if strength <= 0.01:
        return img_bgr.copy()

    h, w = img_bgr.shape[:2]

    # Retinex용 축소 (최대 400px)
    MAX_PX = 400
    scale  = min(1.0, MAX_PX / max(h, w))
    if scale < 1.0:
        th, tw = int(h * scale), int(w * scale)
        small  = cv2.resize(img_bgr, (tw, th), interpolation=cv2.INTER_AREA)
        smask  = cv2.resize(soft_mask, (tw, th), interpolation=cv2.INTER_LINEAR)
    else:
        small, smask = img_bgr.copy(), soft_mask.copy()

    img_f = small.astype(np.float32) + 1.0
    msr   = np.zeros_like(img_f)

    for sigma in (15, 80, 200):
        s  = max(3, int(sigma * scale))
        ks = min(s * 4 + 1, 201)
        ks = ks if ks % 2 == 1 else ks + 1
        blurred = cv2.GaussianBlur(img_f, (ks, ks), float(s))
        msr    += np.log1p(img_f) - np.log1p(blurred)

    msr /= 3.0

    # 정규화 (각 채널 독립)
    msr_norm = np.zeros_like(msr)
    for c in range(3):
        ch  = msr[:, :, c]
        p2  = float(np.percentile(ch, 2))
        p98 = float(np.percentile(ch, 98))
        msr_norm[:, :, c] = np.clip((ch - p2) / max(p98 - p2, 1e-6) * 255, 0, 255)

    # 그림자 마스크 영역에만 적용 (confidence 기반)
    confidence = _shadow_confidence_mask(small if scale < 1.0 else img_bgr,
                                          smask)
    alpha    = (confidence * strength).clip(0, 1)[:, :, np.newaxis]
    r_small  = small.astype(np.float32) * (1.0 - alpha) + msr_norm * alpha
    r_small  = np.clip(r_small, 0, 255).astype(np.uint8)

    if scale < 1.0:
        r_full = cv2.resize(r_small, (w, h), interpolation=cv2.INTER_LINEAR)
        conf_f = cv2.resize(
            _shadow_confidence_mask(img_bgr, soft_mask),
            (w, h), interpolation=cv2.INTER_LINEAR)
        m_full = (conf_f * strength).clip(0, 1)[:, :, np.newaxis]
        result = img_bgr.astype(np.float32) * (1.0 - m_full) + r_full.astype(np.float32) * m_full
        return np.clip(result, 0, 255).astype(np.uint8)

    return r_small


# ══════════════════════════════════════════════════════════
# STEP 4: AI 색상 보정 CNN
# ══════════════════════════════════════════════════════════

class ColorRestorationNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3,  64, 3, padding=1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.LeakyReLU(0.2, inplace=True),
        )
        self.ctx = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=2, dilation=2), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, 3, padding=4, dilation=4), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, 3, padding=8, dilation=8), nn.LeakyReLU(0.2, inplace=True),
        )
        self.dec = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64,  32, 3, padding=1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32,   3, 1),
        )
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
        res = torch.tanh(self.dec(ec)) * 0.12  # 잔차 제한 (과보정 방지)
        return torch.clamp(x + res, 0, 1)


def ai_color_restore(img_bgr: np.ndarray,
                      soft_mask: np.ndarray,
                      model,
                      device: str = 'cpu',
                      max_side: int = 800) -> np.ndarray:
    """AI 모델 그림자 영역 색상 복원 (그림자 영역만 적용)"""
    if model is None:
        return img_bgr.copy()

    h, w   = img_bgr.shape[:2]
    scale  = min(1.0, max_side / max(h, w))

    if scale < 1.0:
        th, tw = int(h * scale), int(w * scale)
        proc   = cv2.resize(img_bgr, (tw, th), interpolation=cv2.INTER_AREA)
        pmask  = cv2.resize(soft_mask, (tw, th), interpolation=cv2.INTER_LINEAR)
    else:
        proc, pmask = img_bgr.copy(), soft_mask.copy()

    ph   = (8 - proc.shape[0] % 8) % 8
    pw   = (8 - proc.shape[1] % 8) % 8
    img_f = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
    img_p = np.pad(img_f, ((0, ph), (0, pw), (0, 0)), mode='reflect')
    t     = torch.from_numpy(img_p.transpose(2, 0, 1)).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        out = model(t)

    oh, ow  = proc.shape[:2]
    out_np  = out[0].cpu().numpy().transpose(1, 2, 0)[:oh, :ow]
    out_bgr = cv2.cvtColor(
        (out_np * 255).clip(0, 255).astype(np.uint8),
        cv2.COLOR_RGB2BGR)

    # confidence 마스크로 블렌딩
    conf_s = _shadow_confidence_mask(proc, pmask)
    m      = conf_s[:, :, np.newaxis].clip(0, 1)
    blended = proc.astype(np.float32) * (1 - m) + out_bgr.astype(np.float32) * m
    result_s = blended.clip(0, 255).astype(np.uint8)

    if scale < 1.0:
        result_full = cv2.resize(result_s, (w, h), interpolation=cv2.INTER_LINEAR)
        conf_f      = _shadow_confidence_mask(img_bgr, soft_mask)
        m_full      = conf_f[:, :, np.newaxis].clip(0, 1)
        final       = img_bgr.astype(np.float32) * (1 - m_full) + result_full.astype(np.float32) * m_full
        return final.clip(0, 255).astype(np.uint8)

    return result_s


# ══════════════════════════════════════════════════════════
# STEP 5: 선명화 (그림자 ROI만)
# ══════════════════════════════════════════════════════════

def sharpen_shadow_region(img_bgr: np.ndarray,
                           soft_mask: np.ndarray,
                           denoise_h: int = 4,
                           sharpen_amount: float = 1.0,
                           clahe_clip: float = 1.8) -> np.ndarray:
    """
    그림자 ROI에만 bilateral filter + CLAHE + unsharp masking.
    비그림자 영역은 절대 건드리지 않음.
    """
    h, w = img_bgr.shape[:2]

    shadow_bin = soft_mask > 0.35
    if shadow_bin.sum() < 200:
        return img_bgr.copy()

    ys, xs = np.where(shadow_bin)
    pad    = 20
    y1 = max(0, int(ys.min()) - pad)
    y2 = min(h, int(ys.max()) + pad)
    x1 = max(0, int(xs.min()) - pad)
    x2 = min(w, int(xs.max()) + pad)

    roi      = img_bgr[y1:y2, x1:x2].copy()
    roi_mask = soft_mask[y1:y2, x1:x2]

    roi_h, roi_w = roi.shape[:2]
    MAX_ROI      = 800
    roi_scale    = min(1.0, MAX_ROI / max(roi_h, roi_w, 1))

    if roi_scale < 1.0:
        proc_roi  = cv2.resize(roi,
            (int(roi_w * roi_scale), int(roi_h * roi_scale)),
            interpolation=cv2.INTER_AREA)
        proc_mask = cv2.resize(roi_mask,
            (int(roi_w * roi_scale), int(roi_h * roi_scale)),
            interpolation=cv2.INTER_LINEAR)
    else:
        proc_roi, proc_mask = roi.copy(), roi_mask.copy()

    # bilateral filter (denoise)
    d_val    = max(3, min(int(denoise_h * 0.7), 7))
    sigma_c  = float(denoise_h * 5)
    sigma_s  = float(denoise_h * 2)
    denoised = cv2.bilateralFilter(proc_roi, d_val, sigma_c, sigma_s)

    # CLAHE (L채널만)
    lab_img = cv2.cvtColor(denoised, cv2.COLOR_BGR2Lab)
    clahe   = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    lab_img[:, :, 0] = clahe.apply(lab_img[:, :, 0])
    enhanced = cv2.cvtColor(lab_img, cv2.COLOR_Lab2BGR)

    # Unsharp mask
    blurred   = cv2.GaussianBlur(enhanced, (0, 0), 1.5)
    amt       = sharpen_amount * 0.30
    sharpened = cv2.addWeighted(enhanced, 1.0 + amt, blurred, -amt, 0)

    # 마스크 블렌딩
    m         = proc_mask[:, :, np.newaxis].clip(0, 1)
    processed = proc_roi.astype(np.float32) * (1.0 - m) + sharpened.astype(np.float32) * m
    processed = processed.clip(0, 255).astype(np.uint8)

    if roi_scale < 1.0:
        processed = cv2.resize(processed, (roi_w, roi_h), interpolation=cv2.INTER_LINEAR)

    result = img_bgr.copy()
    result[y1:y2, x1:x2] = processed
    return result


# ══════════════════════════════════════════════════════════
# 통합 파이프라인
# ══════════════════════════════════════════════════════════

def restore_shadow_color(img_bgr: np.ndarray,
                          soft_mask: np.ndarray,
                          radio_strength: float = 0.70,
                          color_strength: float = 0.55,
                          retinex_strength: float = 0.20,
                          ai_model=None,
                          device: str = 'cpu',
                          denoise_h: int = 4,
                          sharpen_amount: float = 1.0,
                          clahe_clip: float = 1.8) -> np.ndarray:
    """
    통합 파이프라인.
    모든 처리는 img_bgr/soft_mask 해상도에서 수행.
    """
    step1 = radiometric_correction(img_bgr, soft_mask, radio_strength)
    step2 = lab_color_transfer(step1, soft_mask, color_strength)
    step3 = retinex_shadow_lighten(step2, soft_mask, retinex_strength)
    step4 = ai_color_restore(step3, soft_mask, ai_model, device)
    step5 = sharpen_shadow_region(step4, soft_mask, denoise_h, sharpen_amount, clahe_clip)
    return step5
