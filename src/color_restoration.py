"""
Shadow Color Restoration Module  v6.0
══════════════════════════════════════════════════════════════════════════════
[물리 모델]
그림자 = 직사광선 차단 + 하늘 산란광만 남은 상태

    I_shadow[c] = I_object[c] × α[c]   (채널별 감쇠 계수)
    I_lit[c]    = I_object[c] × β[c]

복원 수식:
    I_restored[c] = I_shadow[c] × (β[c] / α[c])
                  = I_shadow[c] × gain[c]
                  여기서 gain[c] = lit_median[c] / shadow_median[c]

[v6.0 핵심 변화]
이전 문제:
  - soft_mask 블렌딩이 gain의 절반만 적용 (alpha=0.85 × mask→0.5 = 0.43배)
  - v_adj로 또 gain 감소
  - 결과: 목표 gain 8×에서 실제 3.5× 달성 (43%)

v6.0 해결책:
  1. gain_map을 먼저 계산, soft_mask는 "어느 픽셀에 적용할지"만 결정
  2. 채도(S)에 따른 "색상 복원 vs 밝기 복원" 분리
     - 저채도(검은 차량, 도로) → 채널 평균 gain 적용 (회색 유지)
     - 고채도(초록 지붕, 빨간 지붕) → 채널별 독립 gain 적용 (색상 복원)
  3. gain 클리핑 상한 제거 (실제 필요 gain 8~10× 허용)
  4. 글로벌 통계 → 로컬 통계 보완 (동일 재질 패치 탐색)
  5. CLAHE/블러를 통한 gain 맵 스무딩 (아티팩트 방지)

[v6.0 파이프라인]
  STEP 1: per_channel_gain_restore   ← 핵심: 채널별 gain으로 색상+밝기 동시 복원
  STEP 2: highlight_compress         ← 밝은 영역 과보정 방지
  STEP 3: shadow_detail_enhance      ← 그림자 내부 디테일 복원 (CLAHE)
  STEP 4: suppress_fringing          ← 경계 색상 진동 억제
  STEP 5: ai_color_restore (선택)    ← AI 미세 보정
"""

import cv2
import numpy as np
from typing import Optional, Tuple
import torch
import torch.nn as nn


# ══════════════════════════════════════════════════════════════════════
# 내부 유틸
# ══════════════════════════════════════════════════════════════════════

def _compute_global_gain(img_bgr: np.ndarray,
                          soft_mask: np.ndarray,
                          shadow_thresh: float = 0.55,
                          lit_thresh: float = 0.08
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """
    그림자/밝은 영역의 채널별 중앙값으로 글로벌 gain 계산.

    Returns:
        global_gain: shape (3,) float32 — 채널별 gain [B, G, R]
        sky_color  : shape (3,) float32 — 채널 편이 벡터 (G=1.0 기준)
    """
    shadow_px = soft_mask > shadow_thresh
    lit_px    = soft_mask < lit_thresh

    img_f = img_bgr.astype(np.float32)
    flat  = img_f.reshape(-1, 3)

    if shadow_px.sum() < 100 or lit_px.sum() < 100:
        return np.array([4.0, 4.0, 4.0], dtype=np.float32), \
               np.array([1.0, 1.0, 1.0], dtype=np.float32)

    rng = np.random.default_rng(42)
    si  = np.where(shadow_px.ravel())[0]
    li  = np.where(lit_px.ravel())[0]
    if len(si) > 8000: si = rng.choice(si, 8000, replace=False)
    if len(li) > 8000: li = rng.choice(li, 8000, replace=False)

    # 10~90 percentile 기반 강건한 중앙값
    s_med = np.maximum(np.percentile(flat[si], 50, axis=0), 4.0)   # [B, G, R]
    l_med = np.maximum(np.percentile(flat[li], 50, axis=0), 4.0)

    # 채널별 필요 gain (상한 없음 — 실제로 6~10× 필요)
    global_gain = l_med / s_med   # e.g. [6.7, 8.9, 8.2]

    # 하늘 산란광 색상 편이 (G=1.0 기준)
    ratio = s_med / l_med          # shadow/lit 비율
    g     = ratio[1] + 1e-6
    sky_color = np.array([ratio[0]/g, 1.0, ratio[2]/g], dtype=np.float32)

    return global_gain.astype(np.float32), sky_color


def _saturation_weight(img_bgr: np.ndarray,
                        low_s: float = 20.0,
                        high_s: float = 60.0) -> np.ndarray:
    """
    채도 기반 가중치 (색상 복원 강도 결정).
    S < low_s  → 0.0: 회색/검정 → 채널 평균 gain만 (색상 보호)
    S > high_s → 1.0: 유채색   → 채널별 독립 gain (색상 복원)
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    S   = hsv[:, :, 1].astype(np.float32)
    return np.clip((S - low_s) / (high_s - low_s), 0.0, 1.0)


def _smooth_gain_map(gain_map: np.ndarray,
                      soft_mask: np.ndarray,
                      sigma: float = 15.0) -> np.ndarray:
    """
    gain 맵을 그림자 마스크 기반으로 스무딩.
    경계에서의 급격한 전환 방지.
    """
    # gain이 적용될 영역에서만 블러
    ksize = int(sigma * 3) | 1
    blurred = cv2.GaussianBlur(gain_map, (ksize, ksize), sigma)

    # soft_mask > 0.2인 영역에서는 블러된 gain 사용
    mask_3d = (soft_mask > 0.2).astype(np.float32)[:, :, np.newaxis]
    return gain_map * (1 - mask_3d) + blurred * mask_3d


def _build_hue_consistent_gain(img_bgr: np.ndarray,
                                 soft_mask: np.ndarray,
                                 global_gain: np.ndarray) -> np.ndarray:
    """
    동일 색상(Hue) 픽셀의 밝기 비율로 per-pixel gain 추정.

    원리:
    - 이미지 내 같은 색상(Hue)을 가진 밝은 픽셀과 어두운 픽셀을 찾아
    - 두 그룹의 BGR 중앙값 비율 = 해당 색상의 실제 감쇠 비율

    이 방법은 global gain보다 더 정확하지만 계산 비용이 높음.
    Hue를 16개 구간으로 나누어 각 구간별 gain 계산.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    H   = hsv[:, :, 0]   # 0~180
    img_f = img_bgr.astype(np.float32)

    # gain_map 초기화: 전역 gain으로 시작
    gain_map = np.ones(img_bgr.shape, dtype=np.float32) * global_gain[np.newaxis, np.newaxis, :]

    # Hue 구간별 처리
    n_bins = 18
    bin_size = 180.0 / n_bins

    for i in range(n_bins):
        h_lo = i * bin_size
        h_hi = (i + 1) * bin_size
        hue_mask = (H >= h_lo) & (H < h_hi)

        shadow_hue = hue_mask & (soft_mask > 0.50)
        lit_hue    = hue_mask & (soft_mask < 0.10)

        if shadow_hue.sum() < 30 or lit_hue.sum() < 30:
            continue  # 해당 Hue 구간 샘플 부족 → 글로벌 gain 유지

        s_vals = img_f[shadow_hue]   # N×3
        l_vals = img_f[lit_hue]

        s_med = np.maximum(np.median(s_vals, axis=0), 3.0)
        l_med = np.maximum(np.median(l_vals, axis=0), 3.0)
        local_gain = l_med / s_med   # 이 Hue 구간의 실제 gain

        # 글로벌 gain과 혼합 (신뢰도: 샘플 수 기반)
        n_samples = min(shadow_hue.sum(), lit_hue.sum())
        trust = min(n_samples / 200.0, 1.0)  # 200개 이상이면 완전 신뢰

        blended_gain = local_gain * trust + global_gain * (1.0 - trust)

        gain_map[hue_mask] = blended_gain

    return gain_map.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════
# STEP 1: 핵심 — per-channel gain 색상 복원
# ══════════════════════════════════════════════════════════════════════

def per_channel_gain_restore(img_bgr: np.ndarray,
                               soft_mask: np.ndarray,
                               strength: float = 1.0,
                               use_hue_consistent: bool = True) -> np.ndarray:
    """
    그림자 색상+밝기 동시 복원의 핵심 함수.

    설계 원칙:
    ─────────
    1) 글로벌 gain 계산: lit/shadow 채널 중앙값 비율
       (상한 없음 — 실제 6~10× 허용)

    2) Hue-consistent gain (선택):
       동일 색상 픽셀군의 실제 감쇠 비율 추정
       → 더 정확한 채널별 gain

    3) 채도 기반 gain 혼합:
       저채도(검정/회색) → 채널 평균 gain (흰색 차량이 노란색 안 됨)
       고채도(녹색 지붕) → 채널별 독립 gain (녹색 복원)

    4) soft_mask를 gain에 직접 통합:
       최종 픽셀 = 원본 × (1-mask) + (원본×gain) × mask
       → 블렌딩이 gain을 감소시키지 않음!

    5) 과보정 방지: 결과가 lit 통계를 크게 벗어나면 억제
    """
    img_f = img_bgr.astype(np.float32)

    # ── 1. 글로벌 gain 계산
    global_gain, sky_color = _compute_global_gain(img_bgr, soft_mask)

    # ── 2. per-pixel gain map
    if use_hue_consistent:
        gain_map = _build_hue_consistent_gain(img_bgr, soft_mask, global_gain)
    else:
        gain_map = np.ones(img_bgr.shape, dtype=np.float32) * global_gain

    # gain map 스무딩 (아티팩트 방지)
    gain_map = _smooth_gain_map(gain_map, soft_mask, sigma=12.0)

    # ── 3. 채도 기반 gain 혼합
    sat_w   = _saturation_weight(img_bgr, low_s=20.0, high_s=60.0)  # (H,W)
    sat_3d  = sat_w[:, :, np.newaxis]

    # 채도 낮은 픽셀: 채널 평균 gain 사용 (채널 비율 보존 = 색상 안 바뀜)
    mean_gain = gain_map.mean(axis=2, keepdims=True)  # 채널 평균
    final_gain = gain_map * sat_3d + mean_gain * (1.0 - sat_3d)

    # ── 4. 복원 이미지
    restored = np.clip(img_f * final_gain, 0, 255)

    # ── 5. soft_mask 기반 블렌딩
    # strength=1.0일 때 그림자 내부(mask=1.0)는 완전히 복원
    # strength=0.7이면 70% 복원
    alpha = np.clip(soft_mask * strength, 0.0, 1.0)[:, :, np.newaxis]
    result = img_f * (1.0 - alpha) + restored * alpha

    # ── 6. 자연스러운 밝기 범위 클리핑
    # 너무 밝아지면 억제 (255 이하)
    return np.clip(result, 0, 255).astype(np.uint8)


# ══════════════════════════════════════════════════════════════════════
# STEP 2: Highlight 압축 (과보정 방지)
# ══════════════════════════════════════════════════════════════════════

def highlight_compress(img_bgr: np.ndarray,
                        soft_mask: np.ndarray,
                        compress_strength: float = 0.25) -> np.ndarray:
    """
    밝은 영역이 과보정되어 하얗게 날아가는 것 방지.
    Photoshop Highlight 기능과 동일한 원리.

    compress_strength:
      0.0 = 압축 없음
      0.5 = 강한 압축 (하이라이트 디테일 강조)
    """
    if compress_strength < 0.01:
        return img_bgr.copy()

    lab   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
    L     = lab[:, :, 0] / 255.0   # 0~1

    # 로컬 평균 밝기
    L_blur = cv2.GaussianBlur(L, (61, 61), 20.0)

    # 과보정 여부: 로컬 평균이 0.75 이상인 영역
    hl_mask = np.clip((L_blur - 0.72) / 0.25, 0.0, 1.0) ** 1.5

    # 그림자 마스크 영역 강조 (그림자였던 곳 위주로)
    hl_alpha = hl_mask * soft_mask * compress_strength * 0.5

    # L 채널 압축: 밝은 부분을 살짝 내림
    L_compressed = L - hl_alpha * (L - 0.80)
    L_compressed = np.clip(L_compressed, 0, 1)

    lab[:, :, 0] = L_compressed * 255.0
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_Lab2BGR)


# ══════════════════════════════════════════════════════════════════════
# STEP 3: 그림자 디테일 향상
# ══════════════════════════════════════════════════════════════════════

def shadow_detail_enhance(img_bgr: np.ndarray,
                           soft_mask: np.ndarray,
                           clahe_clip: float = 2.0,
                           denoise_h: int    = 4,
                           sharpen_amount: float = 0.8) -> np.ndarray:
    """
    그림자 영역 디테일 향상:
    1) CLAHE — 로컬 대비 강화 (텍스처 살리기)
    2) Bilateral 노이즈 억제 — 게인 증폭된 노이즈 제거
    3) 언샤프 마스킹 — 경계 선명도 향상

    그림자 마스크 영역에만 적용.
    """
    h, w   = img_bgr.shape[:2]
    result = img_bgr.copy()

    # ── 1. CLAHE (Lab L 채널)
    if clahe_clip > 0.1:
        lab   = cv2.cvtColor(result, cv2.COLOR_BGR2Lab)
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
        L_orig  = lab[:, :, 0].astype(np.float32)
        L_clahe = clahe.apply(lab[:, :, 0]).astype(np.float32)
        # 그림자 영역에만 CLAHE 적용
        alpha_c = np.clip(soft_mask * 1.2, 0, 1)
        lab[:, :, 0] = np.clip(
            L_orig * (1 - alpha_c) + L_clahe * alpha_c, 0, 255
        ).astype(np.uint8)
        result = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)

    # ── 2. 그림자 ROI bilateral 노이즈 억제
    if denoise_h >= 2:
        shadow_bin = soft_mask > 0.25
        if shadow_bin.sum() > 400:
            ys, xs = np.where(shadow_bin)
            pad = 20
            y1 = max(0, int(ys.min()) - pad)
            y2 = min(h, int(ys.max()) + pad)
            x1 = max(0, int(xs.min()) - pad)
            x2 = min(w, int(xs.max()) + pad)

            roi      = result[y1:y2, x1:x2]
            roi_mask = soft_mask[y1:y2, x1:x2]

            d_val    = max(3, min(int(denoise_h * 0.7), 9))
            denoised = cv2.bilateralFilter(
                roi, d_val,
                float(denoise_h * 7), float(denoise_h * 3))

            m = roi_mask[:, :, np.newaxis].clip(0, 1)
            blended = (roi.astype(np.float32) * (1 - m) +
                       denoised.astype(np.float32) * m)
            result[y1:y2, x1:x2] = blended.clip(0, 255).astype(np.uint8)

    # ── 3. 적응형 언샤프 마스킹
    if sharpen_amount > 0.05:
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2Lab)
        L   = lab[:, :, 0].astype(np.float32)

        # 두 스케일 언샤프
        blur1   = cv2.GaussianBlur(L, (0, 0), 1.0)
        blur2   = cv2.GaussianBlur(L, (0, 0), 2.0)
        detail1 = (L - blur1) * sharpen_amount * 0.40
        detail2 = (L - blur2) * sharpen_amount * 0.15
        unsharp = L + detail1 + detail2

        # 하이라이트 보호: 밝은 픽셀은 샤프닝 억제
        hl_w    = np.clip((L / 255.0 - 0.78) / 0.20, 0, 1)
        unsharp = L * hl_w + unsharp * (1.0 - hl_w)

        lab[:, :, 0] = np.clip(unsharp, 0, 255).astype(np.uint8)
        result = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)

    return result


# ══════════════════════════════════════════════════════════════════════
# STEP 4: 경계 Fringe 억제
# ══════════════════════════════════════════════════════════════════════

def suppress_fringing(img_bgr: np.ndarray,
                       soft_mask: np.ndarray,
                       strength: float = 0.35) -> np.ndarray:
    """
    그림자 경계부의 색상 아티팩트(프린징) 억제.
    경계 픽셀의 색상을 양쪽 영역의 평균으로 부드럽게 혼합.
    """
    # 경계 마스크: soft_mask 0.3~0.6 구간
    edge_mask = np.clip(
        1.0 - np.abs(soft_mask - 0.45) / 0.20, 0, 1
    ).astype(np.float32)

    if edge_mask.max() < 0.01:
        return img_bgr.copy()

    lab    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
    a_blur = cv2.GaussianBlur(lab[:, :, 1], (9, 9), 2.5)
    b_blur = cv2.GaussianBlur(lab[:, :, 2], (9, 9), 2.5)

    w = edge_mask * strength
    lab[:, :, 1] = lab[:, :, 1] * (1 - w) + a_blur * w
    lab[:, :, 2] = lab[:, :, 2] * (1 - w) + b_blur * w

    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_Lab2BGR)


# ══════════════════════════════════════════════════════════════════════
# STEP 5: AI 색상 복원 CNN (미세 보정)
# ══════════════════════════════════════════════════════════════════════

class ColorRestorationNet(nn.Module):
    """
    잔차 CNN.
    illumination_aware 복원 후 남은 미세한 색상 부정확도 보정.
    """
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
        e   = self.enc(x)
        c   = self.ctx(e)
        ec  = torch.cat([e, c], dim=1)
        # 잔차 매우 작게 — 미세 보정만
        res = torch.tanh(self.dec(ec)) * 0.08
        return torch.clamp(x + res, 0, 1)


def ai_color_restore(img_bgr: np.ndarray,
                      soft_mask: np.ndarray,
                      model: Optional[nn.Module],
                      device: str = 'cpu',
                      max_side: int = 900) -> np.ndarray:
    """AI 모델 미세 보정 (채도 가중치 적용)."""
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
        (out_np * 255).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    # 그림자 영역에만 AI 결과 적용 (채도 보호)
    sat_w = _saturation_weight(proc, 20.0, 65.0)
    m     = (pmask * sat_w * 0.6).clip(0, 0.6)[:, :, np.newaxis]
    blended  = proc.astype(np.float32) * (1 - m) + out_bgr.astype(np.float32) * m
    result_s = blended.clip(0, 255).astype(np.uint8)

    if scale < 1.0:
        result_full = cv2.resize(result_s, (w, h), interpolation=cv2.INTER_LINEAR)
        sat_f  = _saturation_weight(img_bgr, 20.0, 65.0)
        m_full = (soft_mask * sat_f * 0.6).clip(0, 0.6)[:, :, np.newaxis]
        final  = (img_bgr.astype(np.float32) * (1 - m_full) +
                  result_full.astype(np.float32) * m_full)
        return final.clip(0, 255).astype(np.uint8)

    return result_s


# ══════════════════════════════════════════════════════════════════════
# 메인 파이프라인  v6.0
# ══════════════════════════════════════════════════════════════════════

def restore_shadow_color(
    img_bgr: np.ndarray,
    soft_mask: np.ndarray,
    # ── 핵심: 색상 복원
    color_restore_strength: float = 0.90,  # per-channel gain 적용 강도 (0~1)
    use_hue_consistent: bool       = True,  # Hue 기반 정밀 gain 사용
    # ── 하이라이트 보호
    highlight_protect: float       = 0.25,  # 밝은 영역 과보정 억제
    # ── 품질 개선
    clahe_clip: float              = 2.0,
    denoise_h: int                 = 4,
    sharpen_amount: float          = 0.8,
    # ── AI
    ai_model                       = None,
    device: str                    = 'cpu',
    # ── 하위 호환성 파라미터 (pipeline.py 구버전 호환)
    shadow_amount: float           = 0.70,
    highlight_amount: float        = 0.20,
    midtone_contrast: float        = 0.15,
    color_strength: float          = 0.35,
    radio_strength: float          = 0.70,
    retinex_strength: float        = 0.0,
    shadow_lift: float             = 0.30,
    blur_radius: int               = 60,
) -> np.ndarray:
    """
    v6.0 통합 파이프라인.

    핵심 변화:
    - per_channel_gain_restore가 gain을 직접 soft_mask에 통합
    - 블렌딩 alpha가 gain을 감소시키지 않음
    - Hue-consistent gain으로 채널별 색상 정확 복원
    - 저채도 픽셀(검정 차량)은 채널 평균 gain → 색상 유지

    처리 순서:
    1. per_channel_gain_restore  ← 핵심
    2. highlight_compress        ← 하이라이트 보호
    3. shadow_detail_enhance     ← CLAHE + 노이즈 제거 + 선명도
    4. suppress_fringing         ← 경계 정리
    5. ai_color_restore          ← AI 미세 보정 (있으면)
    """
    # 하위 호환성: shadow_amount가 color_restore_strength보다 크면 사용
    eff_strength   = max(color_restore_strength, shadow_amount * 0.9)
    eff_hl_protect = max(highlight_protect, highlight_amount)

    # 1. 핵심 색상 복원
    step1 = per_channel_gain_restore(
        img_bgr, soft_mask,
        strength          = min(eff_strength, 1.0),
        use_hue_consistent = use_hue_consistent)

    # 2. 하이라이트 압축
    step2 = highlight_compress(
        step1, soft_mask,
        compress_strength = eff_hl_protect)

    # 3. 품질 개선
    step3 = shadow_detail_enhance(
        step2, soft_mask,
        clahe_clip     = clahe_clip,
        denoise_h      = denoise_h,
        sharpen_amount = sharpen_amount)

    # 4. 경계 Fringe 억제
    step4 = suppress_fringing(step3, soft_mask)

    # 5. AI 미세 보정
    step5 = ai_color_restore(step4, soft_mask, ai_model, device)

    return step5


# ══════════════════════════════════════════════════════════════════════
# 하위 호환성 — 이전 버전 함수명 유지
# ══════════════════════════════════════════════════════════════════════

def illumination_aware_color_restore(img_bgr, soft_mask, strength=0.85, blur_radius=60):
    """v5.0 호환성 — per_channel_gain_restore로 위임."""
    return per_channel_gain_restore(img_bgr, soft_mask, strength=strength)


def shadow_lift_highlight_protect(img_bgr, soft_mask,
                                   shadow_lift=0.30, highlight_protect=0.25):
    """v5.0 호환성 — highlight_compress로 위임."""
    return highlight_compress(img_bgr, soft_mask, compress_strength=highlight_protect)


def enhance_image_quality(img_bgr, soft_mask,
                           clahe_clip=2.0, denoise_h=5, sharpen_amount=0.8):
    """v5.0 호환성 — shadow_detail_enhance로 위임."""
    return shadow_detail_enhance(img_bgr, soft_mask, clahe_clip, denoise_h, sharpen_amount)
