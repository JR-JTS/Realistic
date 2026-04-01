"""
Shadow Color Restoration Module  v1.77
══════════════════════════════════════════════════════════════════════════════
[v1.71 설계 철학]

▸ 사진마다 다른 통계 자동 분석
  - 매번 이미지의 그림자/밝은 영역을 측정하여 gain 계산
  - 하드코딩 없음 — 이미지가 달라도 항상 올바른 보정값 추출

▸ Shadow와 Highlight 완전 분리 (퍼센트 기반 슬라이더)
  - Shadow  0% = 원본 그대로,  100% = 완전 복원 (lit 수준까지)
  - Highlight 0% = 원본 그대로,  100% = 날아간 텍스처 최대 복원

▸ 물리 모델:
  I_shadow[c] = I_object[c] × α[c]   (그림자: 하늘빛만 받음)
  I_lit[c]    = I_object[c] × β[c]   (햇빛: 직사광선 받음)
  gain[c]     = β[c]/α[c]  =  lit_median[c] / shadow_median[c]
  복원량      = strength/100 × (I_shadow × gain - I_shadow)  추가

[v1.71 파이프라인]
  STEP 1: analyze_image()         — 이미지 통계 분석 (그림자/밝기 특성)
  STEP 2: shadow_restore()        — 그림자 영역 색상+밝기 복원 (0~100%)
  STEP 3: highlight_restore()     — 밝은 영역 텍스처 복원   (0~100%)
  STEP 4: enhance_quality()       — 전체 품질 개선 (CLAHE, 노이즈, 선명도)
  STEP 5: suppress_fringing()     — 경계 아티팩트 억제
"""

import cv2
import numpy as np
from typing import Optional, Tuple, Dict


# ──────────────────────────────────────────────────────────
# 공통 유틸: 빠른 대형 블러
# ──────────────────────────────────────────────────────────

def _fast_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """
    sigma에 따라 최적 blur 전략 자동 선택.
    - sigma < 10: 일반 GaussianBlur (소형 커널, 빠름)
    - sigma >= 10: resize(1/8) → 소형 blur → resize back (수백배 빠름)
    품질 손실: 평균 오차 0.2 수준 (시각적으로 무시 가능)
    v1.80: 임계값 20→10 (sigma=15 등 중간값도 fast 경로 사용)
    """
    if sigma < 10:
        return cv2.GaussianBlur(img, (0, 0), sigma)
    h, w = img.shape[:2]
    scale = 8
    sw, sh = max(w // scale, 4), max(h // scale, 4)
    small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
    small_blur = cv2.GaussianBlur(small, (0, 0), max(sigma / scale, 1.0))
    return cv2.resize(small_blur, (w, h), interpolation=cv2.INTER_LINEAR)


def _box_blur(img: np.ndarray, radius: int) -> np.ndarray:
    """boxFilter 래퍼. gain_map처럼 정밀도가 낮아도 되는 경우 사용 (초고속)."""
    k = max(radius * 2 + 1, 3)
    return cv2.boxFilter(img, -1, (k, k))
import torch
import torch.nn as nn


# ══════════════════════════════════════════════════════════════════════
# STEP 1: 이미지 통계 분석 — adaptive gain 계산의 핵심
# ══════════════════════════════════════════════════════════════════════

def analyze_image(img_bgr: np.ndarray,
                  soft_mask: np.ndarray) -> Dict:
    """
    이미지와 그림자 마스크를 분석하여 복원에 필요한 통계를 반환.

    반환값 (dict):
      shadow_median  : np.ndarray (3,)  — 그림자 영역 채널별 중앙값
      lit_median     : np.ndarray (3,)  — 밝은 영역 채널별 중앙값
      global_gain    : np.ndarray (3,)  — 채널별 필요 gain (lit/shadow)
      shadow_pct     : float            — 그림자 비율 (0~100)
      has_lit        : bool             — 밝은 참조 영역 존재 여부
      shadow_mask    : np.ndarray bool  — 그림자 픽셀 마스크
      lit_mask       : np.ndarray bool  — 밝은 픽셀 마스크
      avg_gain       : float            — 채널 평균 gain (스칼라)
    """
    img_f      = img_bgr.astype(np.float32)
    H, W       = img_bgr.shape[:2]

    shadow_mask = soft_mask > 0.50
    lit_mask    = soft_mask < 0.08

    shadow_pct  = float(shadow_mask.sum()) / (H * W) * 100.0

    # ── 충분한 샘플 확보
    si = np.where(shadow_mask.ravel())[0]
    li = np.where(lit_mask.ravel())[0]

    rng = np.random.default_rng(0)
    if len(si) > 8000: si = rng.choice(si, 8000, replace=False)
    if len(li) > 8000: li = rng.choice(li, 8000, replace=False)

    flat = img_f.reshape(-1, 3)

    # ── 그림자 중앙값 (낮은 채널값 보호)
    if len(si) >= 30:
        s_med = np.maximum(np.percentile(flat[si], 50, axis=0), 2.0).astype(np.float32)
    else:
        # 그림자 픽셀 부족 → 전체 이미지 어두운 픽셀 사용
        v = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)[:, :, 2].ravel().astype(np.float32)
        dark_idx = np.where(v < 80)[0]
        if len(dark_idx) >= 30:
            s_med = np.maximum(np.percentile(flat[dark_idx], 50, axis=0), 2.0).astype(np.float32)
        else:
            s_med = np.array([20.0, 20.0, 20.0], dtype=np.float32)

    has_lit = len(li) >= 30
    if has_lit:
        l_med = np.maximum(np.percentile(flat[li], 50, axis=0), 4.0).astype(np.float32)
    else:
        # 밝은 참조 없음 → 밝기 기반 추정
        v = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)[:, :, 2].ravel().astype(np.float32)
        bright_idx = np.where(v > 150)[0]
        if len(bright_idx) >= 30:
            l_med = np.maximum(np.percentile(flat[bright_idx], 50, axis=0), 4.0).astype(np.float32)
        else:
            # 전체 중앙값 × 2 추정
            l_med = np.maximum(np.percentile(flat, 70, axis=0) * 1.5, 4.0).astype(np.float32)

    # ── gain 계산 (채널별)
    global_gain = np.clip(l_med / s_med, 1.0, 12.0).astype(np.float32)
    avg_gain    = float(global_gain.mean())

    return {
        "shadow_median": s_med,
        "lit_median":    l_med,
        "global_gain":   global_gain,
        "shadow_pct":    shadow_pct,
        "has_lit":       has_lit,
        "shadow_mask":   shadow_mask,
        "lit_mask":      lit_mask,
        "avg_gain":      avg_gain,
    }


# ══════════════════════════════════════════════════════════════════════
# 내부 유틸
# ══════════════════════════════════════════════════════════════════════

def _hue_consistent_gain(img_bgr: np.ndarray,
                          shadow_mask: np.ndarray,
                          lit_mask: np.ndarray,
                          global_gain: np.ndarray,
                          n_bins: int = 18) -> np.ndarray:
    """
    Hue 구간별 로컬 gain 계산 (v1.82: bincount 최소화).

    v1.80 대비 개선:
      - s_flat/l_flat float32 캐스팅 1회만
      - img_flat 채널 분리 후 재사용 (reshape 중복 제거)
      - bin_count 불필요 계산 제거
      - fancy indexing 결과 바로 reshape (copy 없이)

    Returns: shape (H, W, 3) float32 — per-pixel per-channel gain map
    """
    hsv    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    H_flat = (hsv[:, :, 0].ravel().astype(np.int32) * n_bins // 180).clip(0, n_bins - 1)

    s_flat = shadow_mask.ravel().astype(np.float32)  # 1회 캐스팅
    l_flat = lit_mask.ravel().astype(np.float32)

    s_count = np.bincount(H_flat, weights=s_flat, minlength=n_bins)
    l_count = np.bincount(H_flat, weights=l_flat, minlength=n_bins)
    valid   = (s_count >= 20) & (l_count >= 20)

    # img_flat 채널 미리 분리 → bincount 재사용
    img_flat = img_bgr.reshape(-1, 3).astype(np.float32)
    ch = [img_flat[:, c] for c in range(3)]

    s_sum = np.stack([np.bincount(H_flat, weights=ch[c]*s_flat, minlength=n_bins) for c in range(3)], axis=1)
    l_sum = np.stack([np.bincount(H_flat, weights=ch[c]*l_flat, minlength=n_bins) for c in range(3)], axis=1)

    inv_s = np.where(valid, 1.0 / np.maximum(s_count, 1), 1.0)[:, None]
    inv_l = np.where(valid, 1.0 / np.maximum(l_count, 1), 1.0)[:, None]
    s_mean = np.maximum(s_sum * inv_s, 2.0)
    l_mean = np.maximum(l_sum * inv_l, 2.0)

    local_gain = np.clip(l_mean / s_mean, 1.0, 12.0).astype(np.float32)
    trust      = np.minimum(np.minimum(s_count, l_count) * (1.0/150.0), 1.0)[:, None]

    bin_gain = np.where(
        valid[:, None],
        local_gain * trust + global_gain * (1.0 - trust),
        global_gain
    ).astype(np.float32)

    # (N,3) → (H,W,3) : copy 없이 reshape
    return bin_gain[H_flat].reshape(img_bgr.shape)


def _saturation_weight(img_bgr: np.ndarray,
                        low_s: float  = 20.0,
                        high_s: float = 55.0) -> np.ndarray:
    """
    채도 기반 색상 복원 가중치 [0, 1].

    저채도(검은차, 도로) → 0: 채널 평균 gain만 적용 (색상 유지)
    고채도(초록 지붕, 붉은 벽) → 1: 채널별 독립 gain 적용 (색상 복원)
    """
    S = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)[:, :, 1].astype(np.float32)
    return np.clip((S - low_s) / (high_s - low_s), 0.0, 1.0)


# ══════════════════════════════════════════════════════════════════════
# STEP 2: Shadow 복원 — 그림자 영역만 처리 (0~100%)
# ══════════════════════════════════════════════════════════════════════

def shadow_restore(img_bgr: np.ndarray,
                   soft_mask: np.ndarray,
                   strength_pct: float = 85.0,
                   use_hue_consistent: bool = True,
                   _stats: Optional[Dict] = None) -> np.ndarray:
    """
    그림자 영역 색상+밝기 복원.

    strength_pct : 0 = 원본 그대로, 100 = 완전 복원 (lit 수준)

    ▸ 알고리즘:
      1. analyze_image()로 이 사진의 shadow/lit 통계 측정
      2. gain[c] = lit_median[c] / shadow_median[c]  — 이미지 맞춤 계산
      3. Hue 구간별 gain 세분화 (색상 유형별 최적 보정)
      4. 저채도 픽셀: 채널 평균 gain → 색상 보호 (검은 차량, 도로)
         고채도 픽셀: 채널별 gain → 색상 복원 (초록 지붕, 붉은 벽)
      5. strength_pct/100 로 원본과 보간:
            결과 = 원본 + (복원값 - 원본) × (strength_pct/100) × soft_mask

    ▸ 밝은 영역(soft_mask < 0.08): 절대 변경 없음
    ▸ v1.74: dark_weight 추가 — 픽셀 자체가 밝으면(V>100) gain=1.0 강제
    """
    if strength_pct < 0.5:
        return img_bgr.copy()

    stats = _stats if _stats is not None else analyze_image(img_bgr, soft_mask)

    shadow_mask = stats["shadow_mask"]
    lit_mask    = stats["lit_mask"]
    global_gain = stats["global_gain"]

    if shadow_mask.sum() < 50:
        return img_bgr.copy()

    # ── v1.82: float32 캐스팅 1회
    img_f = img_bgr.astype(np.float32)

    # ── Hue-consistent gain map
    if use_hue_consistent:
        gain_map = _hue_consistent_gain(img_bgr, shadow_mask, lit_mask, global_gain)
    else:
        gain_map = np.broadcast_to(global_gain, img_bgr.shape).astype(np.float32).copy()

    # gain map 스무딩 (경계 아티팩트 방지)
    gain_map = _box_blur(gain_map, 7)

    # ── 채도 기반 gain 혼합 (in-place)
    sat_w     = _saturation_weight(img_bgr)[:, :, np.newaxis]
    mean_gain = gain_map.mean(axis=2, keepdims=True)
    gain_map -= mean_gain
    gain_map *= sat_w
    gain_map += mean_gain

    # ── dark_weight: V채널만 추출
    hsv_v       = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)[:, :, 2].astype(np.float32)
    dark_weight = np.clip((100.0 - hsv_v) * (1.0/70.0), 0.0, 1.0)

    # ── combined mask
    t_val    = float(np.clip(strength_pct / 100.0, 0.0, 1.0))
    combined = (soft_mask * dark_weight * t_val).clip(0.0, 1.0)  # (H,W) — alpha 겸용
    combined_3 = combined[:, :, np.newaxis]  # view

    # safe_gain = 1 + (gain_map-1)*combined  (in-place)
    gain_map -= 1.0
    gain_map *= combined_3
    gain_map += 1.0

    # ── 블렌딩: out = img_f + (img_f*gain_map - img_f)*alpha
    #           = img_f + img_f*(gain_map-1)*alpha  — gain_map-1 = (gain_map-1)*combined
    # 위에서 이미 gain_map = 1+(gain_map_orig-1)*combined 이므로
    # out = img_f*gain_map 를 alpha=1로 적용한 것과 동일 (combined에 t_val 반영됨)
    out = img_f * gain_map
    np.clip(out, 0.0, 255.0, out=out)
    return out.astype(np.uint8)


# ══════════════════════════════════════════════════════════════════════
# STEP 3: Highlight 복원 — 밝은 영역만 처리 (0~100%)
# ══════════════════════════════════════════════════════════════════════

def highlight_restore(img_bgr: np.ndarray,
                      soft_mask: np.ndarray,
                      strength_pct: float = 30.0) -> np.ndarray:
    """
    v1.76 Highlight 복원 — 실제 blown-out 픽셀만 정밀 처리.

    사용자 의도:
      "밝은 색의 채도가 낮아져서 텍스쳐가 보이게 하는 기능"
      → 실제로 날아간(blown-out) 픽셀(V>190, S<60)만 처리.
        V=160~190의 "정상 밝은 픽셀"은 건드리지 않음.
        어두운(그림자) 영역은 절대 건드리지 않음.

    strength_pct: 0=원본, 100=최대 복원

    ▸ v1.76 개선 사항 (v1.74 대비):
      - blown_V 임계값 160→190 (정상 밝은 픽셀 보호)
      - blown_S 임계값 100→60 (실제 채도 손실 픽셀만)
      - S_target 상한 80→50 (과채도 주입 방지)
      - S_ref 참조 범위 σ=10→σ=15 (더 넓은 주변 참조)
      - 밝기 억제 임계 V>220→V>210, 최대 25%로 강화
      - 텍스처 detail 계수 감소 (overshooting 방지)

    ▸ v1.76 알고리즘:
      1. "날아간 픽셀" 감지: V>190 AND S<60 (실제 blown-out만)
      2. 그림자 영역 완전 제외 (soft_mask 기반)
      3. 채도 복원: 넓은 주변(σ=15) 채도 참조 → 최대 50 이하
      4. 밝기 미세 억제: V>210 픽셀을 주변 쪽으로 당김 (최대 25%)
      5. 텍스쳐 복원: Lab L채널 멀티스케일 언샤프마스킹 (완화된 계수)
      6. strength_pct 비율로 원본과 혼합
    """
    if strength_pct < 0.5:
        return img_bgr.copy()

    t = strength_pct / 100.0

    # ── HSV 변환
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    H_ch = hsv[:, :, 0]   # 0~179
    S_ch = hsv[:, :, 1]   # 0~255
    V_ch = hsv[:, :, 2]   # 0~255

    # ── [1] "날아간(blown-out)" 픽셀 마스크 (v1.76: 임계값 강화)
    # 조건: 매우 밝고(V>190) 채도가 거의 없는(S<60) → 실제 blown-out 흰 영역
    # V=160~190은 "정상 밝은 픽셀"이므로 제외
    blown_V = np.clip((V_ch - 190.0) / 40.0, 0.0, 1.0)   # V>190→증가, V>230→1.0
    blown_S = np.clip((60.0  - S_ch)  / 40.0, 0.0, 1.0)   # S<60 →증가, S<20→1.0
    blown_mask = blown_V * blown_S                          # 두 조건 AND (좁은 범위)

    # ── 그림자 영역 완전 제외
    shadow_exclude = np.clip(1.0 - soft_mask * 5.0, 0.0, 1.0)
    hl_mask = blown_mask * shadow_exclude

    if hl_mask.max() < 0.01:
        return img_bgr.copy()

    # ── [2] 채도 복원
    S_ref   = _fast_blur(S_ch, 15.0)
    np.minimum(S_ref, 50.0, out=S_ref)
    blend_s = hl_mask * t
    # S_new = S_ch + (S_ref-S_ch)*blend_s  →  in-place on S_ref
    S_ref  -= S_ch
    S_ref  *= blend_s
    S_ref  += S_ch   # S_ref is now S_new

    # ── [3] 밝기 억제 (V>210, 최대 25%)
    V_ref    = _fast_blur(V_ch, 20.0)
    over     = np.clip((V_ch - 210.0) * (1.0/30.0), 0.0, 1.0)
    suppress = over * blend_s * 0.25
    # V_new = V_ch - (V_ch-V_ref)*suppress  → in-place on V_ref
    V_ref   -= V_ch    # V_ref = V_ref_orig - V_ch  →  -(V_ch - V_ref_orig)
    V_ref   *= suppress
    V_ref   += V_ch    # V_ref is now V_new  (= V_ch - (V_ch-V_ref_orig)*suppress)

    # ── HSV 합성 → BGR
    hsv[:, :, 1] = S_ref  # S 교체 (in-place, H는 그대로)
    hsv[:, :, 2] = V_ref
    np.clip(hsv, 0, 255, out=hsv)
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)

    # ── [4] 텍스쳐 복원 — grayscale unsharp (Lab 변환 없음)
    V_res    = result[:, :, 0]*0.114 + result[:, :, 1]*0.587 + result[:, :, 2]*0.299
    blur_tex = cv2.GaussianBlur(V_res, (0, 0), 3.0)
    detail_3 = ((V_res - blur_tex) * 1.5 * blend_s.clip(0,1))[:, :, np.newaxis]
    result  += detail_3
    np.clip(result, 0, 255, out=result)

    # ── [5] 최종 블렌딩 (in-place)
    orig_f       = img_bgr.astype(np.float32)
    alpha_final  = blend_s.clip(0.0, 1.0)[:, :, np.newaxis]
    result      -= orig_f
    result      *= alpha_final
    result      += orig_f
    np.clip(result, 0, 255, out=result)
    return result.astype(np.uint8)


# ══════════════════════════════════════════════════════════════════════
# STEP 4: 품질 개선 — CLAHE + 노이즈 + 선명도
# ══════════════════════════════════════════════════════════════════════

def enhance_quality(img_bgr: np.ndarray,
                    soft_mask: np.ndarray,
                    clahe_clip: float     = 2.0,
                    denoise_h: int        = 4,
                    sharpen_amount: float = 0.7) -> np.ndarray:
    """
    전체 품질 개선.
    CLAHE는 그림자 영역에 집중 적용.
    노이즈 제거는 그림자 ROI에만.
    언샤프 마스킹은 전체 (밝은 영역 억제 포함).
    """
    h, w   = img_bgr.shape[:2]
    result = img_bgr.copy()

    # ── CLAHE (그림자 영역 중심) — 그림자가 없으면 skip
    if clahe_clip > 0.1 and soft_mask.max() > 0.1:
        lab     = cv2.cvtColor(result, cv2.COLOR_BGR2Lab)
        L_orig  = lab[:, :, 0]                                    # uint8 view
        clahe   = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
        L_clahe = clahe.apply(L_orig)                             # uint8
        alpha_c = np.clip(soft_mask * 1.3, 0.0, 1.0)
        # in-place blend: L_orig + (L_clahe-L_orig)*alpha_c
        diff    = (L_clahe.astype(np.float32) - L_orig.astype(np.float32)) * alpha_c
        lab[:, :, 0] = np.clip(L_orig.astype(np.float32) + diff, 0, 255).astype(np.uint8)
        result  = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)

    # ── Bilateral 노이즈 억제 (그림자 ROI만, 실제 ROI 크기로 제한)
    if denoise_h >= 2:
        shadow_bin = soft_mask > 0.25
        n_shadow   = shadow_bin.sum()
        if n_shadow > 200:
            ys, xs = np.where(shadow_bin)
            pad = 8   # ★ 패딩 축소: 20 → 8
            y1 = max(0, int(ys.min()) - pad)
            y2 = min(h, int(ys.max()) + pad)
            x1 = max(0, int(xs.min()) - pad)
            x2 = min(w, int(xs.max()) + pad)
            roi      = result[y1:y2, x1:x2]
            roi_mask = soft_mask[y1:y2, x1:x2]
            d_val    = max(3, min(int(denoise_h * 0.7), 7))  # ★ 최대 d=7→5 → 속도
            denoised = cv2.bilateralFilter(roi, d_val,
                                            float(denoise_h * 6),
                                            float(denoise_h * 3))
            m       = roi_mask[:, :, np.newaxis].clip(0.0, 1.0)
            # in-place blend on roi slice
            roi_f   = roi.astype(np.float32)
            roi_f  += (denoised.astype(np.float32) - roi_f) * m
            result[y1:y2, x1:x2] = roi_f.clip(0, 255).astype(np.uint8)

    # ── 언샤프 마스킹 — GaussianBlur 2회 → 1회로 통합 (v1.82)
    if sharpen_amount > 0.05:
        lab  = cv2.cvtColor(result, cv2.COLOR_BGR2Lab)
        L    = lab[:, :, 0].astype(np.float32)
        # sigma=1.5 단일 블러: sigma=1.0과 3.0의 중간 특성 근사
        blur = cv2.GaussianBlur(L, (0, 0), 1.5)
        detail = (L - blur) * sharpen_amount * 0.65  # 계수 조정 (2스케일 합산 근사)
        unsharp = L + detail
        bright_limit = np.clip(1.0 - (L * (1.0/255.0) - 0.90) / 0.10, 0.5, 1.0)
        # in-place: unsharp = L*(1-bl) + unsharp*bl  →  L + detail*bl
        unsharp = L + detail * bright_limit  # 단순화
        unsharp = L * 0.15 + unsharp * 0.85
        lab[:, :, 0] = np.clip(unsharp, 0, 255).astype(np.uint8)
        result = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)

    return result


# ══════════════════════════════════════════════════════════════════════
# STEP 5: Fringe 억제
# ══════════════════════════════════════════════════════════════════════

def suppress_fringing(img_bgr: np.ndarray,
                       soft_mask: np.ndarray,
                       strength: float = 0.30) -> np.ndarray:
    """그림자 경계 색상 아티팩트 억제.
    v1.82: BGR에서 직접 처리 → Lab 변환 2회(10ms) 제거.
    Lab a/b 대신 BGR B채널(청색편이)로 근사.
    """
    edge = np.clip(1.0 - np.abs(soft_mask - 0.45) / 0.20, 0.0, 1.0).astype(np.float32)
    if edge.max() < 0.01:
        return img_bgr.copy()
    out = img_bgr.astype(np.float32)
    w   = (edge * strength)[:, :, np.newaxis]          # (H,W,1)
    # BGR 각 채널 boxFilter로 색상 평활화 (GaussianBlur(9,9,2.5) 근사, 2× 빠름)
    blurred = cv2.boxFilter(img_bgr, -1, (9, 9)).astype(np.float32)
    # out = out*(1-w) + blurred*w = out + (blurred-out)*w
    out += (blurred - out) * w
    return out.clip(0, 255).astype(np.uint8)


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
        # v1.73: residual scale 0.08→0.15 (더 강한 보정, 여전히 보수적)
        return torch.clamp(x + torch.tanh(self.dec(ec)) * 0.15, 0, 1)


# ══════════════════════════════════════════════════════════════════════
# 추가 색상복원 알고리즘  v1.74
# ══════════════════════════════════════════════════════════════════════

def retinex_restore(img_bgr: np.ndarray,
                    soft_mask: np.ndarray,
                    strength_pct: float = 50.0,
                    sigma_list: list = None) -> np.ndarray:
    """
    Multi-Scale Retinex (MSR) 색상복원.
    조명 성분을 제거하여 반사율(실제 색상)을 복원.
    그림자 영역에만 적용.
    """
    if strength_pct < 0.5:
        return img_bgr.copy()
    if sigma_list is None:
        sigma_list = [15, 80, 250]

    img_f = img_bgr.astype(np.float32) + 1.0  # log(0) 방지
    log_img = np.log(img_f)

    # MSR: 여러 스케일 Gaussian으로 조명 추정
    msr = np.zeros_like(log_img)
    for sigma in sigma_list:
        blur = _fast_blur(img_f, sigma)
        msr += log_img - np.log(blur + 1.0)
    msr /= len(sigma_list)

    # 정규화: 각 채널 0~255
    result = np.zeros_like(img_f)
    for c in range(3):
        ch = msr[:, :, c]
        lo, hi = np.percentile(ch, 1), np.percentile(ch, 99)
        if hi > lo:
            result[:, :, c] = np.clip((ch - lo) / (hi - lo) * 255.0, 0, 255)
        else:
            result[:, :, c] = img_bgr[:, :, c].astype(np.float32)

    t     = np.clip(strength_pct / 100.0, 0.0, 1.0)
    alpha = np.clip(soft_mask * t, 0.0, 1.0)[:, :, np.newaxis]
    blended = img_f * (1.0 - alpha) + result * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)


def white_balance_restore(img_bgr: np.ndarray,
                           soft_mask: np.ndarray,
                           strength_pct: float = 50.0) -> np.ndarray:
    """
    그레이 월드 화이트 밸런스 복원.
    그림자 영역의 색온도 편차(보통 청색 과다)를 보정.
    """
    if strength_pct < 0.5:
        return img_bgr.copy()

    img_f = img_bgr.astype(np.float32)
    shadow_mask = soft_mask > 0.5
    lit_mask    = soft_mask < 0.08

    if shadow_mask.sum() < 50 or lit_mask.sum() < 50:
        return img_bgr.copy()

    # 밝은 영역 채널 평균 → 목표 그레이
    flat = img_f.reshape(-1, 3)
    si   = np.where(shadow_mask.ravel())[0]
    li   = np.where(lit_mask.ravel())[0]
    s_mean = flat[si].mean(axis=0)
    l_mean = flat[li].mean(axis=0)
    gray_l = l_mean.mean()

    # 밝은 영역이 그레이가 되도록 보정 계수
    wb_gain = np.where(l_mean > 2.0, gray_l / np.maximum(l_mean, 2.0), 1.0)
    wb_gain = np.clip(wb_gain, 0.5, 2.0).astype(np.float32)

    corrected = np.clip(img_f * wb_gain, 0, 255)
    t     = np.clip(strength_pct / 100.0, 0.0, 1.0)
    alpha = np.clip(soft_mask * t, 0.0, 1.0)[:, :, np.newaxis]
    result = img_f * (1.0 - alpha) + corrected * alpha
    return np.clip(result, 0, 255).astype(np.uint8)


def gamma_restore(img_bgr: np.ndarray,
                  soft_mask: np.ndarray,
                  strength_pct: float = 50.0) -> np.ndarray:
    """
    적응형 Gamma 보정.
    그림자 영역의 어두운 톤을 감마 커브로 밝힘.
    strength_pct=50 → gamma≈1.8,  100 → gamma≈3.0
    """
    if strength_pct < 0.5:
        return img_bgr.copy()

    # 그림자 밝기에 따라 gamma 자동 조정
    stats = analyze_image(img_bgr, soft_mask)
    shadow_v = stats['shadow_median'].mean()
    lit_v    = stats['lit_median'].mean()
    if lit_v > shadow_v > 0:
        # 자연스러운 gamma: 그림자가 어두울수록 더 강한 보정
        darkness = 1.0 - (shadow_v / max(lit_v, 1.0))
        gamma = 1.0 + darkness * (strength_pct / 100.0) * 2.5
    else:
        gamma = 1.0 + (strength_pct / 100.0) * 1.5
    gamma = float(np.clip(gamma, 1.0, 4.0))

    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                       for i in range(256)], dtype=np.uint8)
    corrected = cv2.LUT(img_bgr, table)

    t     = np.clip(strength_pct / 100.0, 0.0, 1.0)
    alpha = np.clip(soft_mask * t, 0.0, 1.0)[:, :, np.newaxis]
    result = (img_bgr.astype(np.float32) * (1.0 - alpha)
              + corrected.astype(np.float32) * alpha)
    return np.clip(result, 0, 255).astype(np.uint8)


def ai_refine(img_bgr: np.ndarray,
               soft_mask: np.ndarray,
               model: Optional[nn.Module],
               device: str = 'cpu',
               max_side: int = 600) -> np.ndarray:
    """
    AI 모델 미세 보정 (그림자 영역만).
    v1.73: max_side 900→600(속도 개선), 블렌딩 최대 0.20, 그림자 영역만 적용
    """
    if model is None:
        return img_bgr.copy()

    # 그림자 영역이 거의 없으면 스킵 (불필요한 AI 처리 방지)
    if (soft_mask > 0.4).sum() < 200:
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

    # v1.73: 그림자 영역(soft_mask 기반)에만 적용, 최대 20% 블렌딩
    # 완전 그림자(>0.7) → 0.20, 경계(0.4~0.7) → 비례, 밝은 영역(<0.4) → 0
    shadow_weight = np.where(pmask > 0.7, 0.20,
                    np.where(pmask > 0.4, (pmask - 0.4) / 0.3 * 0.20, 0.0)).astype(np.float32)
    m = shadow_weight[:, :, np.newaxis]
    blended  = proc.astype(np.float32) * (1-m) + out_bgr.astype(np.float32) * m
    result_s = blended.clip(0, 255).astype(np.uint8)

    if scale < 1.0:
        result_full = cv2.resize(result_s, (w, h), interpolation=cv2.INTER_LINEAR)
        full_weight = np.where(soft_mask > 0.7, 0.15,
                      np.where(soft_mask > 0.4, 0.07, 0.0)).astype(np.float32)
        m_full = full_weight[:, :, np.newaxis]
        final  = img_bgr.astype(np.float32)*(1-m_full) + result_full.astype(np.float32)*m_full
        return final.clip(0, 255).astype(np.uint8)
    return result_s


# ══════════════════════════════════════════════════════════════════════
# 통합 파이프라인  v1.74
# ══════════════════════════════════════════════════════════════════════

def restore_shadow_color(
    img_bgr: np.ndarray,
    soft_mask: np.ndarray,
    # Shadow 파라미터 (퍼센트, 0~100)
    shadow_strength: float     = 85.0,
    use_hue_consistent: bool   = True,
    # Highlight 파라미터 (퍼센트, 0~100)
    highlight_strength: float  = 30.0,
    # 색상복원 모드: 'gain'(기본) | 'retinex' | 'wb' | 'gamma'
    color_mode: str            = 'gain',
    # 추가 색상복원 강도 (retinex/wb/gamma 혼합 비율, 0~100)
    extra_color_pct: float     = 0.0,
    # 품질 개선
    clahe_clip: float          = 2.0,
    denoise_h: int             = 4,
    sharpen_amount: float      = 0.7,
    # AI
    ai_model                   = None,
    device: str                = 'cpu',
    # ── 하위 호환성
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
    v1.73 통합 파이프라인.

    color_mode:
      'gain'    — 채널별 gain 복원 (기본, 가장 정확)
      'retinex' — Multi-Scale Retinex (조명 제거)
      'wb'      — 화이트 밸런스 보정
      'gamma'   — 적응형 Gamma 보정
    extra_color_pct > 0 이면 gain 이후 추가 모드도 혼합 적용
    """
    # v1.73 호환성 처리:
    # pipeline.py에서 넘길 때는 shadow_strength = eff_shadow_pct (0~100 %)
    # 직접 호출 시 shadow_strength = 0~1.0 (레거시)
    # 구분: shadow_strength > 1.0 이거나, 소수점 0~1 범위 외는 퍼센트로 취급
    #       단, 정확히 0.0은 '0% 복원' 의도이므로 직접 사용
    if shadow_strength == 0.0 and highlight_strength <= 0.0:
        # 명시적 0 → 복원 없음
        eff_shadow    = 0.0
        eff_highlight = max(0.0, highlight_strength)
    elif shadow_strength > 1.0 or highlight_strength > 1.0:
        # 퍼센트로 전달됨 → 레거시 무시
        eff_shadow    = shadow_strength
        eff_highlight = highlight_strength if highlight_strength >= 0.0 else 0.0
    elif 0.0 < shadow_strength <= 1.0:
        # 레거시 0~1 범위 → 변환 후 max()
        shadow_strength    = shadow_strength * 100.0
        highlight_strength = max(highlight_strength, 0.0) * 100.0
        legacy_shadow = max(color_restore_strength, shadow_amount * 0.9, radio_strength) * 100.0
        legacy_hl     = max(highlight_protect, highlight_amount) * 100.0
        eff_shadow    = max(shadow_strength, legacy_shadow)
        eff_highlight = max(highlight_strength, legacy_hl)
    else:
        # 그 외 (shadow_strength < 0 = 미설정) → shadow_pct 대신 shadow_amount 등으로 추정
        legacy_shadow = max(color_restore_strength, shadow_amount * 0.9, radio_strength) * 100.0
        legacy_hl     = max(highlight_protect, highlight_amount) * 100.0
        eff_shadow    = legacy_shadow
        eff_highlight = legacy_hl
    # retinex_strength 하위 호환 (0~1 → 0~100)
    if retinex_strength > 0:
        extra_color_pct = max(extra_color_pct, retinex_strength * 100.0)
        if color_mode == 'gain':
            color_mode = 'retinex'

    # ── 한 번만 이미지 분석
    stats = analyze_image(img_bgr, soft_mask)

    # ── STEP 1: 색상+밝기 복원
    if color_mode == 'retinex':
        step1 = retinex_restore(img_bgr, soft_mask, strength_pct=min(eff_shadow, 100.0))
    elif color_mode == 'wb':
        step1 = white_balance_restore(img_bgr, soft_mask, strength_pct=min(eff_shadow, 100.0))
    elif color_mode == 'gamma':
        step1 = gamma_restore(img_bgr, soft_mask, strength_pct=min(eff_shadow, 100.0))
    else:  # 'gain' 기본
        step1 = shadow_restore(img_bgr, soft_mask,
                               strength_pct=min(eff_shadow, 100.0),
                               use_hue_consistent=use_hue_consistent,
                               _stats=stats)

    # ── STEP 1b: 추가 색상복원 혼합 (extra_color_pct > 0)
    if extra_color_pct > 1.0 and color_mode == 'gain':
        extra = retinex_restore(step1, soft_mask, strength_pct=min(extra_color_pct, 100.0))
        t = np.clip(extra_color_pct / 100.0 * 0.5, 0.0, 0.5)  # 최대 50% 혼합
        m = np.clip(soft_mask * t, 0.0, t)[:, :, np.newaxis]
        step1 = (step1.astype(np.float32) * (1-m) + extra.astype(np.float32) * m
                 ).clip(0, 255).astype(np.uint8)

    # ── STEP 2: Highlight 복원
    step2 = highlight_restore(step1, soft_mask,
                               strength_pct=min(eff_highlight, 100.0))

    # ── 복원이 전혀 없으면 즉시 반환 (품질 개선도 스킵)
    if eff_shadow < 0.5 and eff_highlight < 0.5:
        return img_bgr.copy()

    # ── STEP 3: 품질 개선
    step3 = enhance_quality(step2, soft_mask,
                             clahe_clip=clahe_clip,
                             denoise_h=denoise_h,
                             sharpen_amount=sharpen_amount)

    # ── STEP 4: Fringe 억제 (복원이 있을 때만)
    if eff_shadow > 0.5 or eff_highlight > 0.5:
        step4 = suppress_fringing(step3, soft_mask)
    else:
        step4 = step3

    # ── STEP 5: AI 미세 보정 (과도적용 방지 처리됨)
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
# 하위 호환성 래퍼
# ══════════════════════════════════════════════════════════════════════

def per_channel_gain_restore(img_bgr, soft_mask, strength=0.85, use_hue_consistent=True):
    return shadow_restore(img_bgr, soft_mask,
                          strength_pct=strength * 100.0,
                          use_hue_consistent=use_hue_consistent)

def highlight_compress(img_bgr, soft_mask, compress_strength=0.25):
    return highlight_restore(img_bgr, soft_mask, strength_pct=compress_strength * 100.0)

def shadow_detail_enhance(img_bgr, soft_mask, clahe_clip=2.0, denoise_h=4, sharpen_amount=0.7):
    return enhance_quality(img_bgr, soft_mask, clahe_clip, denoise_h, sharpen_amount)

def illumination_aware_color_restore(img_bgr, soft_mask, strength=0.85, blur_radius=60):
    return shadow_restore(img_bgr, soft_mask, strength_pct=strength * 100.0)

def shadow_lift_highlight_protect(img_bgr, soft_mask, shadow_lift=0.20, highlight_protect=0.25):
    return img_bgr.copy()

def enhance_image_quality(img_bgr, soft_mask, clahe_clip=2.0, denoise_h=4, sharpen_amount=0.7):
    return enhance_quality(img_bgr, soft_mask, clahe_clip, denoise_h, sharpen_amount)
