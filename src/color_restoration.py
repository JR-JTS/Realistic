"""
Shadow Color Restoration Module  v4.0
──────────────────────────────────────
v4.0 완전 재설계:
  ■ Photoshop Shadow/Highlight 방식 채택
    - shadow_lift : 어두운 영역만 선택적으로 밝힘 (tone curve 기반)
    - highlight_protect : 밝은 영역은 건드리지 않거나 살짝 내림
    - midtone_contrast : 중간톤 대비 강화
  ■ 색상 채도(Saturation) 보호: 검은 차량처럼 채도 낮은 물체는 밝기 복원만, 색상 변경 금지
  ■ Tone-mapping 기반 복원: gain 방식 대신 tone-curve 적용 → 하이라이트 클리핑 방지
  ■ 영상 품질 개선 파이프라인:
    - 어두운 영역 CLAHE (로컬 대비)
    - 전체 노이즈 억제 (fastNlMeansDenoisingColored)
    - 적응형 언샤프 마스킹
    - 색상 진동(fringing) 억제
"""

import cv2
import numpy as np
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════
# 내부 유틸
# ══════════════════════════════════════════════════════════

def _build_tone_curve(shadow_lift: float,
                      highlight_compress: float,
                      midtone_contrast: float) -> np.ndarray:
    """
    Photoshop Shadow/Highlight 방식의 톤 커브 생성 (0~255 LUT).

    shadow_lift        : 0~1, 어두운 영역 밝기 상승량 (0.5 = Photoshop 기본)
    highlight_compress : 0~1, 밝은 영역 압축량 (0 = 그대로, 0.3 = 살짝 내림)
    midtone_contrast   : -1~1, 중간톤 대비 (양수=강화, 음수=완화)
    """
    x = np.linspace(0.0, 1.0, 256)

    # ── Shadow Lift: 어두운 영역 선택적 상승 (부드러운 S-커브 아랫부분)
    # Photoshop은 0~128 범위에 집중, 그 이상은 서서히 감소
    shadow_weight = np.clip(1.0 - (x / 0.5) ** 1.5, 0, 1)  # 0에서 1, 0.5에서 0
    y = x + shadow_lift * shadow_weight * (1.0 - x)          # 밝은 곳은 덜 올림

    # ── Highlight Compress: 밝은 영역 살짝 압축
    highlight_weight = np.clip((x - 0.7) / 0.3, 0, 1) ** 2
    y = y - highlight_compress * highlight_weight * x

    # ── Midtone Contrast: S-커브 중간 부분
    if abs(midtone_contrast) > 0.01:
        # 시그모이드 기반 S-커브
        k = midtone_contrast * 5.0
        sig = 1.0 / (1.0 + np.exp(-k * (x - 0.5)))
        sig = (sig - sig.min()) / (sig.max() - sig.min())  # 0~1 정규화
        blend = np.clip(4 * x * (1 - x), 0, 1)             # 중간톤 가중치
        y = y * (1 - blend * abs(midtone_contrast)) + \
            sig * blend * abs(midtone_contrast) + \
            y * (1 - blend * abs(midtone_contrast))
        y = x + (y - x) * abs(midtone_contrast)

    lut = np.clip(y * 255.0, 0, 255).astype(np.uint8)
    return lut


def _apply_lut_masked(img_bgr: np.ndarray,
                      lut: np.ndarray,
                      mask: np.ndarray) -> np.ndarray:
    """
    LUT를 마스크 영역에만 블렌딩 적용.
    mask : float32 [0,1]
    """
    # LUT 적용 (전체)
    corrected = cv2.LUT(img_bgr, lut)
    # 마스크 블렌딩
    m = mask[:, :, np.newaxis].clip(0, 1)
    result = img_bgr.astype(np.float32) * (1.0 - m) + corrected.astype(np.float32) * m
    return result.clip(0, 255).astype(np.uint8)


def _saturation_protection_mask(img_bgr: np.ndarray) -> np.ndarray:
    """
    채도가 낮은 픽셀(검은 차량, 콘크리트 등)의 색상 변환 억제 마스크.
    S < 40 이면 색상 보정 금지 (밝기만 복원)
    반환: float32 [0,1], 높을수록 색상 보정 허용
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    S   = hsv[:, :, 1].astype(np.float32)
    # S 20 이하: 완전 억제, S 60 이상: 완전 허용
    sat_mask = np.clip((S - 20.0) / 40.0, 0, 1)
    return sat_mask.astype(np.float32)


# ══════════════════════════════════════════════════════════
# STEP 1: Photoshop Shadow/Highlight 복원
# ══════════════════════════════════════════════════════════

def shadow_highlight_restore(img_bgr: np.ndarray,
                              soft_mask: np.ndarray,
                              shadow_amount: float = 0.70,
                              highlight_amount: float = 0.20,
                              midtone_contrast: float = 0.15,
                              radius: int = 40) -> np.ndarray:
    """
    Photoshop Shadow/Highlight 알고리즘.

    shadow_amount     : 그림자 영역 밝기 복원 강도 (0~1)
    highlight_amount  : 하이라이트 압축 강도 (0~1, 과보정 방지)
    midtone_contrast  : 중간톤 대비 강화 (-1~1)
    radius            : 로컬 평균 계산 반경 (픽셀)

    핵심 아이디어:
    1) 각 픽셀의 로컬 평균 밝기로 shadow/midtone/highlight 영역 분류
    2) 분류된 영역별로 다른 tone curve 적용
    3) 채도 낮은 픽셀(검은 물체)은 색상 변경 없이 밝기만 복원
    """
    h, w = img_bgr.shape[:2]

    # ── 로컬 평균 밝기 계산 (Photoshop의 "Tonal Width" 개념)
    # Lab L 채널 사용 (인지 균일 밝기)
    lab  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
    L    = lab[:, :, 0] / 255.0  # 0~1 정규화

    ks   = min(radius * 2 + 1, 151)
    ks   = ks if ks % 2 == 1 else ks + 1
    L_blur = cv2.GaussianBlur(L, (ks, ks), float(radius / 2))

    # ── 영역 분류 (soft mask 고려)
    # shadow 영역: L_blur 낮고 soft_mask 높음
    shadow_tonal = np.clip(1.0 - L_blur / 0.45, 0, 1) ** 1.5   # 밝기 0~45% 구간
    highlight_tonal = np.clip((L_blur - 0.65) / 0.35, 0, 1) ** 2  # 밝기 65~100% 구간

    shadow_lift_mask     = shadow_tonal * soft_mask              # 그림자 탐지 × 톤
    highlight_press_mask = highlight_tonal * (1.0 - soft_mask * 0.5)  # 밝은 영역

    # ── 채도 보호 마스크 (검은 차량 등 색상 변경 방지)
    sat_prot = _saturation_protection_mask(img_bgr)

    # ── Lab L 채널에 Shadow Lift 적용
    # Shadow: L 값을 tone curve로 올림
    shadow_gain  = shadow_amount * shadow_lift_mask
    # 밝기 상승: 어두울수록 더 많이 (tone curve 효과)
    L_lifted = L + shadow_gain * (1.0 - L) * 0.7   # 이미 밝은 부분은 덜 올림
    L_lifted = np.clip(L_lifted, 0, 1)

    # Highlight Compress: 과도하게 밝은 영역 살짝 내림
    highlight_reduce = highlight_amount * highlight_press_mask * 0.4
    L_result = L_lifted - highlight_reduce * L_lifted
    L_result = np.clip(L_result, 0, 1)

    # Midtone Contrast: 중간톤 S-커브
    if abs(midtone_contrast) > 0.01:
        midtone_w = 4.0 * L_result * (1.0 - L_result)  # 0.5에서 최대
        k = midtone_contrast * 3.0
        sig = 1.0 / (1.0 + np.exp(-k * (L_result - 0.5)))
        sig = (sig - 0.5) * midtone_w * abs(midtone_contrast) * 0.3
        L_result = np.clip(L_result + sig, 0, 1)

    # ── L 채널 적용 (a, b 채널은 채도 보호 마스크에 따라 선택)
    lab_result = lab.copy()
    lab_result[:, :, 0] = L_result * 255.0

    # a, b 채널 색상 이동: 채도 낮은 물체는 색상 유지
    # 그림자 특유의 청색편이 제거 (a, b 채널을 밝은 영역 평균으로 이동)
    lit_mask = soft_mask < 0.05
    if lit_mask.sum() > 200:
        lit_a_mean = float(np.median(lab[:, :, 1][lit_mask]))
        lit_b_mean = float(np.median(lab[:, :, 2][lit_mask]))
        shd_mask_bin = soft_mask > 0.4

        if shd_mask_bin.sum() > 100:
            # 색상 보정: 채도 높은 그림자만 색상 이동 (채도 낮으면 유지)
            color_alpha = (shadow_lift_mask * sat_prot * 0.4).clip(0, 0.4)[:, :, np.newaxis]
            target_ab   = np.stack([
                np.full_like(lab[:, :, 1], lit_a_mean),
                np.full_like(lab[:, :, 2], lit_b_mean)
            ], axis=2)
            lab_result[:, :, 1:] = (lab[:, :, 1:] * (1.0 - color_alpha) +
                                     target_ab * color_alpha).clip(0, 255)

    result_bgr = cv2.cvtColor(
        np.clip(lab_result, 0, 255).astype(np.uint8),
        cv2.COLOR_Lab2BGR)
    return result_bgr


# ══════════════════════════════════════════════════════════
# STEP 2: 색상 채도 복원 (그림자 영역 색상 이동 보정)
# ══════════════════════════════════════════════════════════

def color_cast_correction(img_bgr: np.ndarray,
                           soft_mask: np.ndarray,
                           strength: float = 0.40) -> np.ndarray:
    """
    그림자의 청색 편이(color cast) 보정.
    채도 낮은 픽셀(검은 차량)은 건너뜀.
    """
    if strength <= 0.01:
        return img_bgr.copy()

    shadow_bin = soft_mask > 0.50
    lit_bin    = soft_mask < 0.06

    if shadow_bin.sum() < 100 or lit_bin.sum() < 200:
        return img_bgr.copy()

    sat_prot   = _saturation_protection_mask(img_bgr)

    lab    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
    flat   = lab.reshape(-1, 3)

    MAX_SAMP = 4000
    rng  = np.random.default_rng(0)
    s_idx = np.where(shadow_bin.ravel())[0]
    l_idx = np.where(lit_bin.ravel())[0]
    if len(s_idx) > MAX_SAMP: s_idx = rng.choice(s_idx, MAX_SAMP, replace=False)
    if len(l_idx) > MAX_SAMP: l_idx = rng.choice(l_idx, MAX_SAMP, replace=False)

    sv = flat[s_idx]
    lv = flat[l_idx]

    # a, b 채널 색상 이동만 (L은 shadow_highlight_restore에서 처리)
    diff_a = float(np.median(lv[:, 1]) - np.median(sv[:, 1]))
    diff_b = float(np.median(lv[:, 2]) - np.median(sv[:, 2]))

    # 색상 이동 제한 (최대 ±15)
    diff_a = np.clip(diff_a, -15, 15)
    diff_b = np.clip(diff_b, -15, 15)

    # 채도 보호 × soft mask × strength
    alpha = (soft_mask * sat_prot * strength).clip(0, 1)

    lab_result = lab.copy()
    lab_result[:, :, 1] = np.clip(lab[:, :, 1] + diff_a * alpha, 0, 255)
    lab_result[:, :, 2] = np.clip(lab[:, :, 2] + diff_b * alpha, 0, 255)

    return cv2.cvtColor(lab_result.astype(np.uint8), cv2.COLOR_Lab2BGR)


# ══════════════════════════════════════════════════════════
# STEP 3: 영상 품질 개선 (전체 이미지)
# ══════════════════════════════════════════════════════════

def enhance_image_quality(img_bgr: np.ndarray,
                           soft_mask: np.ndarray,
                           clahe_clip: float = 2.0,
                           denoise_h: int = 5,
                           sharpen_amount: float = 1.0) -> np.ndarray:
    """
    영상 품질 개선 3단계:
    1) 그림자 영역 CLAHE (로컬 대비 복원)
    2) 선택적 노이즈 억제 (그림자 영역 위주)
    3) 전체 적응형 언샤프 마스킹
    """
    h, w = img_bgr.shape[:2]
    result = img_bgr.copy()

    # ── 1. 그림자 영역 CLAHE
    if clahe_clip > 0.1:
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2Lab)
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
        L_orig  = lab[:, :, 0].astype(np.float32)
        L_clahe = clahe.apply(lab[:, :, 0]).astype(np.float32)

        # 그림자 영역에만 적용 (비그림자는 원본 유지)
        alpha_clahe = np.clip(soft_mask * 1.5, 0, 1)
        lab[:, :, 0] = np.clip(
            L_orig * (1 - alpha_clahe) + L_clahe * alpha_clahe,
            0, 255).astype(np.uint8)
        result = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)

    # ── 2. 선택적 노이즈 억제
    if denoise_h >= 2:
        # 그림자 ROI에만 bilateral filter (속도 최적화)
        shadow_bin = soft_mask > 0.30
        if shadow_bin.sum() > 400:
            ys, xs = np.where(shadow_bin)
            pad = 15
            y1 = max(0, int(ys.min()) - pad)
            y2 = min(h, int(ys.max()) + pad)
            x1 = max(0, int(xs.min()) - pad)
            x2 = min(w, int(xs.max()) + pad)

            roi = result[y1:y2, x1:x2]
            roi_mask = soft_mask[y1:y2, x1:x2]

            d_val   = max(3, min(int(denoise_h * 0.6), 7))
            sigma_c = float(denoise_h * 8)
            sigma_s = float(denoise_h * 3)
            denoised = cv2.bilateralFilter(roi, d_val, sigma_c, sigma_s)

            m = roi_mask[:, :, np.newaxis].clip(0, 1)
            blended = roi.astype(np.float32) * (1.0 - m) + denoised.astype(np.float32) * m
            result[y1:y2, x1:x2] = blended.clip(0, 255).astype(np.uint8)

    # ── 3. 적응형 언샤프 마스킹 (Luminance 채널만)
    if sharpen_amount > 0.05:
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2Lab)
        L   = lab[:, :, 0].astype(np.float32)

        # 다중 스케일 언샤프 (엣지 보존)
        blur1 = cv2.GaussianBlur(L, (0, 0), 1.0)
        blur2 = cv2.GaussianBlur(L, (0, 0), 2.5)
        unsharp = L + (L - blur1) * sharpen_amount * 0.5 \
                    + (L - blur2) * sharpen_amount * 0.2

        # 하이라이트에서 언샤프 억제 (과보정 방지)
        highlight_w = np.clip((L / 255.0 - 0.75) / 0.25, 0, 1)
        unsharp = L * highlight_w + unsharp * (1.0 - highlight_w)

        lab[:, :, 0] = np.clip(unsharp, 0, 255).astype(np.uint8)
        result = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)

    return result


# ══════════════════════════════════════════════════════════
# STEP 4: 색상 진동(Fringing) 억제 및 최종 다듬기
# ══════════════════════════════════════════════════════════

def suppress_fringing(img_bgr: np.ndarray,
                      soft_mask: np.ndarray) -> np.ndarray:
    """
    그림자 경계에서 발생하는 색상 진동(purple/green fringe) 억제.
    경계 영역의 채도를 원본에 가깝게 부드럽게 만듦.
    """
    # 경계 = soft_mask 0.2~0.6 구간
    edge_mask = np.clip(
        1.0 - np.abs(soft_mask - 0.4) / 0.2, 0, 1
    ).astype(np.float32)

    if edge_mask.max() < 0.01:
        return img_bgr.copy()

    lab   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
    # a, b 채널을 경계에서 약간 부드럽게
    ks    = 7
    a_blur = cv2.GaussianBlur(lab[:, :, 1], (ks, ks), 2.0)
    b_blur = cv2.GaussianBlur(lab[:, :, 2], (ks, ks), 2.0)

    alpha = edge_mask[:, :, np.newaxis] * 0.3  # 최대 30%만 부드럽게
    lab[:, :, 1] = lab[:, :, 1] * (1 - edge_mask * 0.3) + a_blur * (edge_mask * 0.3)
    lab[:, :, 2] = lab[:, :, 2] * (1 - edge_mask * 0.3) + b_blur * (edge_mask * 0.3)

    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_Lab2BGR)


# ══════════════════════════════════════════════════════════
# STEP 5: AI 색상 복원 CNN (구조 유지)
# ══════════════════════════════════════════════════════════

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
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        e  = self.enc(x)
        c  = self.ctx(e)
        ec = torch.cat([e, c], dim=1)
        # 잔차 제한 축소 (0.08): 과보정 방지
        res = torch.tanh(self.dec(ec)) * 0.08
        return torch.clamp(x + res, 0, 1)


def ai_color_restore(img_bgr: np.ndarray,
                     soft_mask: np.ndarray,
                     model,
                     device: str = 'cpu',
                     max_side: int = 800) -> np.ndarray:
    """AI 모델 그림자 영역 색상 복원 (채도 보호 적용)"""
    if model is None:
        return img_bgr.copy()

    h, w  = img_bgr.shape[:2]
    scale = min(1.0, max_side / max(h, w))

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

    # 채도 보호 × soft_mask 블렌딩
    sat_prot = _saturation_protection_mask(proc)
    m = (pmask * sat_prot * 0.7).clip(0, 1)[:, :, np.newaxis]
    blended  = proc.astype(np.float32) * (1 - m) + out_bgr.astype(np.float32) * m
    result_s = blended.clip(0, 255).astype(np.uint8)

    if scale < 1.0:
        result_full = cv2.resize(result_s, (w, h), interpolation=cv2.INTER_LINEAR)
        sat_f  = _saturation_protection_mask(img_bgr)
        m_full = (soft_mask * sat_f * 0.7).clip(0, 1)[:, :, np.newaxis]
        final  = img_bgr.astype(np.float32) * (1 - m_full) + result_full.astype(np.float32) * m_full
        return final.clip(0, 255).astype(np.uint8)

    return result_s


# ══════════════════════════════════════════════════════════
# 통합 파이프라인  v4.0
# ══════════════════════════════════════════════════════════

def restore_shadow_color(img_bgr: np.ndarray,
                          soft_mask: np.ndarray,
                          # Shadow/Highlight 파라미터
                          shadow_amount: float    = 0.70,   # 그림자 밝기 복원
                          highlight_amount: float = 0.20,   # 하이라이트 압축 (과보정 방지)
                          midtone_contrast: float = 0.15,   # 중간톤 대비
                          # 색상 보정
                          color_strength: float   = 0.35,   # 색상 편이 보정
                          # 품질 개선
                          clahe_clip: float       = 2.0,    # CLAHE 강도
                          denoise_h: int          = 5,      # 노이즈 억제
                          sharpen_amount: float   = 0.8,    # 선명도
                          # AI 모델
                          ai_model                = None,
                          device: str             = 'cpu',
                          # 하위 호환성 (구 파라미터 무시)
                          radio_strength: float   = 0.70,
                          color_strength_compat: float = 0.55,
                          retinex_strength: float = 0.20,
                          ) -> np.ndarray:
    """
    v4.0 통합 파이프라인:
    1) Shadow/Highlight 복원  (Photoshop 방식)
    2) 색상 편이 보정          (채도 보호 포함)
    3) AI 색상 복원            (있을 경우)
    4) 영상 품질 개선          (CLAHE + 노이즈 + 선명도)
    5) 경계 Fringe 억제

    ※ radio_strength, retinex_strength 파라미터는 하위 호환성을 위해 유지되지만
       내부에서 shadow_amount / highlight_amount 로 매핑됩니다.
    """
    # ── 하위 호환성 매핑
    # 구버전 radio_strength → shadow_amount 로 사용 (호출부가 그대로 넘길 때)
    effective_shadow = max(shadow_amount, radio_strength * 0.9)
    effective_shadow = min(effective_shadow, 0.95)

    # 1. Shadow/Highlight 복원
    step1 = shadow_highlight_restore(
        img_bgr, soft_mask,
        shadow_amount     = effective_shadow,
        highlight_amount  = highlight_amount,
        midtone_contrast  = midtone_contrast,
        radius            = 40)

    # 2. 색상 편이 보정 (채도 낮은 물체는 건너뜀)
    step2 = color_cast_correction(step1, soft_mask, color_strength)

    # 3. AI 색상 복원 (있을 경우, 잔차 방식이라 과보정 없음)
    step3 = ai_color_restore(step2, soft_mask, ai_model, device)

    # 4. 영상 품질 개선
    step4 = enhance_image_quality(
        step3, soft_mask,
        clahe_clip      = clahe_clip,
        denoise_h       = denoise_h,
        sharpen_amount  = sharpen_amount)

    # 5. 경계 Fringe 억제
    step5 = suppress_fringing(step4, soft_mask)

    return step5
