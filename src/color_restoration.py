"""
Shadow Color Restoration Module  v7.0
══════════════════════════════════════════════════════════════════════════════
[v7.0 설계 철학]

Photoshop Shadow/Highlight와 동일한 개념 — 완전 분리된 두 연산:

  SHADOW  기능: 그림자 영역을 밝게 + 색상 복원
    - 어두운 픽셀(soft_mask > threshold)에만 적용
    - 채널별 독립 gain으로 색상 동시 복원
    - 밝은 영역은 절대 건드리지 않음

  HIGHLIGHT 기능: 밝은 영역의 날아간 텍스처 복원
    - 밝은 픽셀(V > bright_threshold)에만 적용
    - 로컬 대비 강화로 디테일 살리기
    - 어두운 영역은 절대 건드리지 않음

[v7.0 파이프라인]
  STEP 1: shadow_restore()     — 그림자 영역 색상+밝기 복원
  STEP 2: highlight_restore()  — 밝은 영역 텍스처 복원
  STEP 3: enhance_quality()    — 전체 품질 개선 (CLAHE, 노이즈, 선명도)
  STEP 4: suppress_fringing()  — 경계 아티팩트 억제

[실시간 미리보기 지원]
  - process_roi(): 지정 ROI 영역만 처리 → 빠른 응답
  - 각 스텝 독립 실행 가능 → 슬라이더 즉시 반영
"""

import cv2
import numpy as np
from typing import Optional, Tuple
import torch
import torch.nn as nn


# ══════════════════════════════════════════════════════════════════════
# 내부 유틸
# ══════════════════════════════════════════════════════════════════════

def _compute_channel_gain(img_bgr: np.ndarray,
                           shadow_mask: np.ndarray,
                           lit_mask: np.ndarray
                           ) -> np.ndarray:
    """
    그림자/밝은 영역의 채널별 중앙값으로 gain 계산.
    Returns: shape (3,) float32 — [B, G, R] gain
    """
    img_f = img_bgr.astype(np.float32)
    flat  = img_f.reshape(-1, 3)

    si = np.where(shadow_mask.ravel())[0]
    li = np.where(lit_mask.ravel())[0]

    if len(si) < 50 or len(li) < 50:
        return np.array([3.0, 3.0, 3.0], dtype=np.float32)

    rng = np.random.default_rng(42)
    if len(si) > 5000: si = rng.choice(si, 5000, replace=False)
    if len(li) > 5000: li = rng.choice(li, 5000, replace=False)

    s_med = np.maximum(np.percentile(flat[si], 50, axis=0), 4.0)
    l_med = np.maximum(np.percentile(flat[li], 50, axis=0), 4.0)

    return (l_med / s_med).astype(np.float32)


def _hue_consistent_gain(img_bgr: np.ndarray,
                           shadow_mask: np.ndarray,
                           lit_mask: np.ndarray,
                           global_gain: np.ndarray,
                           n_bins: int = 18) -> np.ndarray:
    """
    Hue 구간별 로컬 gain 계산.
    Returns: shape (H, W, 3) float32 — per-pixel per-channel gain map
    """
    hsv   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    H     = hsv[:, :, 0]  # 0~180
    img_f = img_bgr.astype(np.float32)

    gain_map  = np.ones(img_bgr.shape, dtype=np.float32) * global_gain
    bin_size  = 180.0 / n_bins

    for i in range(n_bins):
        h_lo = i * bin_size
        h_hi = (i + 1) * bin_size
        hue_px = (H >= h_lo) & (H < h_hi)

        s_hue = hue_px & shadow_mask
        l_hue = hue_px & lit_mask

        if s_hue.sum() < 20 or l_hue.sum() < 20:
            continue

        s_vals = img_f[s_hue]
        l_vals = img_f[l_hue]
        s_med  = np.maximum(np.median(s_vals, axis=0), 3.0)
        l_med  = np.maximum(np.median(l_vals, axis=0), 3.0)
        local  = l_med / s_med

        trust = min(min(s_hue.sum(), l_hue.sum()) / 150.0, 1.0)
        gain_map[hue_px] = local * trust + global_gain * (1.0 - trust)

    return gain_map.astype(np.float32)


def _saturation_weight(img_bgr: np.ndarray,
                        low_s: float = 20.0,
                        high_s: float = 55.0) -> np.ndarray:
    """채도 기반 색상 복원 가중치 [0,1]."""
    S = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)[:, :, 1].astype(np.float32)
    return np.clip((S - low_s) / (high_s - low_s), 0.0, 1.0)


# ══════════════════════════════════════════════════════════════════════
# STEP 1: Shadow 복원 — 그림자 영역만 처리
# ══════════════════════════════════════════════════════════════════════

def shadow_restore(img_bgr: np.ndarray,
                   soft_mask: np.ndarray,
                   strength: float = 0.85,
                   use_hue_consistent: bool = True) -> np.ndarray:
    """
    그림자 영역 색상+밝기 복원.

    ▸ 동작 원리:
      1. 그림자(soft_mask > 0.5) vs 밝은(soft_mask < 0.1) 영역으로 채널별 gain 계산
      2. Hue-consistent gain: 같은 색조 픽셀군의 실제 감쇠 비율 추정
      3. 저채도(검은 차량/도로) → 채널 평균 gain (색상 보호, 밝기만 올림)
         고채도(초록/붉은 지붕) → 채널별 독립 gain (색상 복원)
      4. soft_mask로 경계 부드럽게 블렌딩

    ▸ 밝은 영역(soft_mask < 0.1)은 절대 건드리지 않음.
    """
    img_f = img_bgr.astype(np.float32)

    shadow_mask = soft_mask > 0.50
    lit_mask    = soft_mask < 0.08

    if shadow_mask.sum() < 50:
        return img_bgr.copy()

    # ── gain 계산
    global_gain = _compute_channel_gain(img_bgr, shadow_mask, lit_mask)

    if use_hue_consistent:
        gain_map = _hue_consistent_gain(img_bgr, shadow_mask, lit_mask, global_gain)
    else:
        gain_map = np.ones(img_bgr.shape, dtype=np.float32) * global_gain

    # gain map 스무딩 (경계 아티팩트 방지)
    ksize = 25
    gain_map = cv2.GaussianBlur(gain_map, (ksize, ksize), 10.0)

    # ── 채도 기반 gain 혼합
    sat_w    = _saturation_weight(img_bgr)[:, :, np.newaxis]
    mean_gain = gain_map.mean(axis=2, keepdims=True)
    final_gain = gain_map * sat_w + mean_gain * (1.0 - sat_w)

    # ── 복원 이미지 계산
    restored = np.clip(img_f * final_gain, 0, 255)

    # ── soft_mask로 블렌딩 (strength 반영)
    alpha  = np.clip(soft_mask * strength, 0.0, 1.0)[:, :, np.newaxis]
    result = img_f * (1.0 - alpha) + restored * alpha

    return np.clip(result, 0, 255).astype(np.uint8)


# ══════════════════════════════════════════════════════════════════════
# STEP 2: Highlight 복원 — 밝은 영역만 처리
# ══════════════════════════════════════════════════════════════════════

def highlight_restore(img_bgr: np.ndarray,
                      soft_mask: np.ndarray,
                      strength: float = 0.30,
                      bright_thresh: float = 0.70) -> np.ndarray:
    """
    밝은 영역 텍스처/디테일 복원.

    ▸ Photoshop Highlights 기능과 동일한 원리:
      - 날아간(uniformly bright) 영역에서 로컬 대비를 증폭
      - 주변 픽셀과의 차이(디테일)를 강조 → 텍스처 살아남
      - CLAHE와 언샤프 마스킹 조합으로 구현

    ▸ 그림자 영역(soft_mask > 0.5)은 절대 건드리지 않음.
    ▸ shadow_restore와 독립 연산.
    """
    if strength < 0.01:
        return img_bgr.copy()

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
    L   = lab[:, :, 0]   # 0~255

    # 날아간 영역 식별: 로컬 평균 대비 편차가 작은 밝은 영역
    L_blur = cv2.GaussianBlur(L, (0, 0), 25.0)
    L_norm = L / 255.0

    # 밝고 uniform한 영역 (날아간 하이라이트)
    # 조건: L > 180 AND 로컬 분산 낮음
    local_var = (L - L_blur) ** 2
    local_var_blur = cv2.GaussianBlur(local_var, (0, 0), 15.0)
    flat_bright = (L_norm > bright_thresh).astype(np.float32)
    flat_weight = np.clip(1.0 - local_var_blur / (20.0 ** 2), 0, 1)
    hl_mask     = flat_bright * flat_weight

    # 그림자 영역 제외 (처리 안 함)
    shadow_exclude = np.clip(1.0 - soft_mask * 2.0, 0.0, 1.0)
    hl_mask = hl_mask * shadow_exclude

    if hl_mask.max() < 0.01:
        return img_bgr.copy()

    # 로컬 대비 강화: 언샤프 마스킹 (하이라이트 영역의 디테일)
    # unsharp = L + (L - L_blur) * amount → 로컬 분산 증폭
    amount  = strength * 1.5
    L_sharp = L + (L - L_blur) * amount

    # 클리핑: 너무 밝아지지 않도록 (max 원본 ×1.05)
    L_sharp = np.clip(L_sharp, 0, np.clip(L * 1.05, 0, 255))

    # hl_mask로 원본과 블렌딩
    alpha = hl_mask * strength
    L_result = L * (1.0 - alpha) + L_sharp * alpha
    lab[:, :, 0] = np.clip(L_result, 0, 255)

    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_Lab2BGR)


# ══════════════════════════════════════════════════════════════════════
# STEP 3: 품질 개선 — CLAHE + 노이즈 + 선명도
# ══════════════════════════════════════════════════════════════════════

def enhance_quality(img_bgr: np.ndarray,
                    soft_mask: np.ndarray,
                    clahe_clip: float = 2.0,
                    denoise_h: int    = 4,
                    sharpen_amount: float = 0.7) -> np.ndarray:
    """
    전체 품질 개선.
    CLAHE는 그림자 영역에 집중 적용.
    노이즈 제거는 그림자 ROI에만.
    언샤프 마스킹은 전체 (하이라이트 억제 포함).
    """
    h, w   = img_bgr.shape[:2]
    result = img_bgr.copy()

    # ── CLAHE (그림자 영역 중심)
    if clahe_clip > 0.1:
        lab   = cv2.cvtColor(result, cv2.COLOR_BGR2Lab)
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
        L_orig  = lab[:, :, 0].astype(np.float32)
        L_clahe = clahe.apply(lab[:, :, 0]).astype(np.float32)
        # 그림자 영역에만 강하게, 밝은 영역은 약하게
        alpha_c = np.clip(soft_mask * 1.3, 0, 1)
        lab[:, :, 0] = np.clip(
            L_orig * (1 - alpha_c) + L_clahe * alpha_c, 0, 255
        ).astype(np.uint8)
        result = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)

    # ── Bilateral 노이즈 억제 (그림자 ROI만)
    if denoise_h >= 2:
        shadow_bin = soft_mask > 0.25
        if shadow_bin.sum() > 200:
            ys, xs = np.where(shadow_bin)
            pad = 20
            y1 = max(0, int(ys.min()) - pad)
            y2 = min(h, int(ys.max()) + pad)
            x1 = max(0, int(xs.min()) - pad)
            x2 = min(w, int(xs.max()) + pad)
            roi      = result[y1:y2, x1:x2]
            roi_mask = soft_mask[y1:y2, x1:x2]
            d_val    = max(3, min(int(denoise_h * 0.7), 9))
            denoised = cv2.bilateralFilter(roi, d_val,
                                            float(denoise_h * 7),
                                            float(denoise_h * 3))
            m = roi_mask[:, :, np.newaxis].clip(0, 1)
            blended = roi.astype(np.float32) * (1-m) + denoised.astype(np.float32) * m
            result[y1:y2, x1:x2] = blended.clip(0, 255).astype(np.uint8)

    # ── 언샤프 마스킹 (Luminance)
    if sharpen_amount > 0.05:
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2Lab)
        L   = lab[:, :, 0].astype(np.float32)
        blur1   = cv2.GaussianBlur(L, (0, 0), 1.0)
        blur2   = cv2.GaussianBlur(L, (0, 0), 2.0)
        unsharp = L + (L - blur1) * sharpen_amount * 0.40 \
                    + (L - blur2) * sharpen_amount * 0.15
        # 밝은 픽셀 샤프닝 억제
        hl_w    = np.clip((L / 255.0 - 0.78) / 0.20, 0, 1)
        unsharp = L * hl_w + unsharp * (1.0 - hl_w)
        lab[:, :, 0] = np.clip(unsharp, 0, 255).astype(np.uint8)
        result = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)

    return result


# ══════════════════════════════════════════════════════════════════════
# STEP 4: Fringe 억제
# ══════════════════════════════════════════════════════════════════════

def suppress_fringing(img_bgr: np.ndarray,
                       soft_mask: np.ndarray,
                       strength: float = 0.30) -> np.ndarray:
    """그림자 경계 색상 아티팩트 억제."""
    edge = np.clip(1.0 - np.abs(soft_mask - 0.45) / 0.20, 0, 1).astype(np.float32)
    if edge.max() < 0.01:
        return img_bgr.copy()
    lab    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
    a_blur = cv2.GaussianBlur(lab[:, :, 1], (9, 9), 2.5)
    b_blur = cv2.GaussianBlur(lab[:, :, 2], (9, 9), 2.5)
    w = edge * strength
    lab[:, :, 1] = lab[:, :, 1] * (1-w) + a_blur * w
    lab[:, :, 2] = lab[:, :, 2] * (1-w) + b_blur * w
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_Lab2BGR)


# ══════════════════════════════════════════════════════════════════════
# AI CNN (미세 보정)
# ══════════════════════════════════════════════════════════════════════

class ColorRestorationNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3,  64, 3, padding=1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.LeakyReLU(0.2, inplace=True),
        )
        self.ctx = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=2,  dilation=2), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, 3, padding=4,  dilation=4), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, 3, padding=8,  dilation=8), nn.LeakyReLU(0.2, inplace=True),
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
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        e  = self.enc(x)
        c  = self.ctx(e)
        ec = torch.cat([e, c], dim=1)
        return torch.clamp(x + torch.tanh(self.dec(ec)) * 0.08, 0, 1)


def ai_refine(img_bgr: np.ndarray,
               soft_mask: np.ndarray,
               model: Optional[nn.Module],
               device: str = 'cpu',
               max_side: int = 900) -> np.ndarray:
    """AI 모델 미세 보정 (그림자 영역만)."""
    if model is None:
        return img_bgr.copy()

    h, w  = img_bgr.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        proc  = cv2.resize(img_bgr,  (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
        pmask = cv2.resize(soft_mask,(int(w*scale), int(h*scale)), interpolation=cv2.INTER_LINEAR)
    else:
        proc, pmask = img_bgr.copy(), soft_mask.copy()

    ph = (8 - proc.shape[0] % 8) % 8
    pw = (8 - proc.shape[1] % 8) % 8
    img_f = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
    img_p = np.pad(img_f, ((0,ph),(0,pw),(0,0)), mode='reflect')
    t     = torch.from_numpy(img_p.transpose(2,0,1)).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        out = model(t)

    oh, ow = proc.shape[:2]
    out_np  = out[0].cpu().numpy().transpose(1,2,0)[:oh,:ow]
    out_bgr = cv2.cvtColor((out_np*255).clip(0,255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    sat_w = _saturation_weight(proc)
    m     = (pmask * sat_w * 0.5).clip(0,0.5)[:,:,np.newaxis]
    blended = proc.astype(np.float32)*(1-m) + out_bgr.astype(np.float32)*m
    result_s = blended.clip(0,255).astype(np.uint8)

    if scale < 1.0:
        result_full = cv2.resize(result_s, (w,h), interpolation=cv2.INTER_LINEAR)
        sat_f  = _saturation_weight(img_bgr)
        m_full = (soft_mask * sat_f * 0.5).clip(0,0.5)[:,:,np.newaxis]
        final  = img_bgr.astype(np.float32)*(1-m_full) + result_full.astype(np.float32)*m_full
        return final.clip(0,255).astype(np.uint8)
    return result_s


# ══════════════════════════════════════════════════════════════════════
# 통합 파이프라인  v7.0
# ══════════════════════════════════════════════════════════════════════

def restore_shadow_color(
    img_bgr: np.ndarray,
    soft_mask: np.ndarray,
    # Shadow 파라미터
    shadow_strength: float     = 0.85,   # 그림자 밝기+색상 복원 강도
    use_hue_consistent: bool   = True,   # Hue 기반 정밀 색상 복원
    # Highlight 파라미터
    highlight_strength: float  = 0.40,   # 밝은 영역 텍스처 복원 강도
    # 품질 개선
    clahe_clip: float          = 2.0,
    denoise_h: int             = 4,
    sharpen_amount: float      = 0.7,
    # AI
    ai_model                   = None,
    device: str                = 'cpu',
    # ── 하위 호환성 파라미터 (pipeline.py 구버전 호환)
    color_restore_strength: float = 0.85,
    highlight_protect: float      = 0.25,
    shadow_amount: float          = 0.70,
    highlight_amount: float       = 0.20,
    midtone_contrast: float       = 0.15,
    color_strength: float         = 0.35,
    radio_strength: float         = 0.70,
    retinex_strength: float       = 0.0,
    shadow_lift: float            = 0.20,
    blur_radius: int              = 60,
) -> np.ndarray:
    """
    v7.0 통합 파이프라인.

    Shadow와 Highlight 완전 분리:
    - shadow_restore: 그림자만 처리, 밝은 영역 건드리지 않음
    - highlight_restore: 밝은 영역만 처리, 그림자 건드리지 않음
    """
    # 하위 호환성: 구버전 파라미터 매핑
    eff_shadow    = max(shadow_strength, color_restore_strength, shadow_amount * 0.9)
    eff_highlight = max(highlight_strength, highlight_protect * 1.5)

    # 1. 그림자 복원 (어두운 영역만)
    step1 = shadow_restore(img_bgr, soft_mask,
                           strength=min(eff_shadow, 1.0),
                           use_hue_consistent=use_hue_consistent)

    # 2. 하이라이트 복원 (밝은 영역만)
    step2 = highlight_restore(step1, soft_mask,
                               strength=eff_highlight)

    # 3. 품질 개선
    step3 = enhance_quality(step2, soft_mask,
                             clahe_clip=clahe_clip,
                             denoise_h=denoise_h,
                             sharpen_amount=sharpen_amount)

    # 4. 경계 Fringe 억제
    step4 = suppress_fringing(step3, soft_mask)

    # 5. AI 미세 보정 (있으면)
    step5 = ai_refine(step4, soft_mask, ai_model, device)

    return step5


# ══════════════════════════════════════════════════════════════════════
# ROI 전용 처리 (실시간 미리보기용)
# ══════════════════════════════════════════════════════════════════════

def process_roi(img_bgr: np.ndarray,
                soft_mask: np.ndarray,
                roi_rect: Optional[Tuple[int,int,int,int]] = None,
                **kwargs) -> np.ndarray:
    """
    지정된 ROI 영역만 처리하여 결과 반환.
    
    roi_rect: (x1, y1, x2, y2) — None이면 전체 이미지 처리
    
    실시간 미리보기에서 보이는 영역만 처리할 때 사용.
    처리 속도를 대폭 향상 (전체 이미지 대비 4~16배 빠름).
    """
    if roi_rect is None:
        return restore_shadow_color(img_bgr, soft_mask, **kwargs)

    x1, y1, x2, y2 = roi_rect
    h, w = img_bgr.shape[:2]
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w, x2); y2 = min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return img_bgr.copy()

    roi_img  = img_bgr[y1:y2, x1:x2]
    roi_mask = soft_mask[y1:y2, x1:x2]

    roi_result = restore_shadow_color(roi_img, roi_mask, **kwargs)

    result = img_bgr.copy()
    result[y1:y2, x1:x2] = roi_result
    return result


# ══════════════════════════════════════════════════════════════════════
# 하위 호환성
# ══════════════════════════════════════════════════════════════════════

def per_channel_gain_restore(img_bgr, soft_mask, strength=0.85, use_hue_consistent=True):
    return shadow_restore(img_bgr, soft_mask, strength=strength,
                          use_hue_consistent=use_hue_consistent)

def highlight_compress(img_bgr, soft_mask, compress_strength=0.25):
    return highlight_restore(img_bgr, soft_mask, strength=compress_strength * 1.5)

def shadow_detail_enhance(img_bgr, soft_mask, clahe_clip=2.0, denoise_h=4, sharpen_amount=0.7):
    return enhance_quality(img_bgr, soft_mask, clahe_clip, denoise_h, sharpen_amount)

def illumination_aware_color_restore(img_bgr, soft_mask, strength=0.85, blur_radius=60):
    return shadow_restore(img_bgr, soft_mask, strength=strength)

def shadow_lift_highlight_protect(img_bgr, soft_mask, shadow_lift=0.20, highlight_protect=0.25):
    return img_bgr.copy()

def enhance_image_quality(img_bgr, soft_mask, clahe_clip=2.0, denoise_h=4, sharpen_amount=0.7):
    return enhance_quality(img_bgr, soft_mask, clahe_clip, denoise_h, sharpen_amount)
