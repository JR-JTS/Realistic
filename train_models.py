"""
Drone Shadow Model Trainer
──────────────────────────
합성 드론 항공사진 데이터로 두 모델을 학습시켜 .pth 파일 생성.

ShadowDetectorNet  → models/shadow_detector.pth  (~7.4 MB)
ColorRestorationNet → models/color_restore.pth   (~0.9 MB)

실행: python train_models.py
      (CPU 기준 약 5~10분 소요)
"""

import os, sys, time, random
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from shadow_detection import ShadowDetectorNet
from color_restoration import ColorRestorationNet

DEVICE     = "cpu"
MODEL_DIR  = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────
# 합성 데이터 생성기 (드론 항공사진 특성 모델링)
# ──────────────────────────────────────────────────────────────

def _make_drone_base(h=256, w=256):
    """드론 항공사진 느낌의 합성 배경 (풀밭, 도로, 건물 지붕)"""
    img = np.zeros((h, w, 3), dtype=np.float32)

    # 1) 그라데이션 베이스 (지형 조명)
    gx = np.linspace(0.55, 0.80, w, dtype=np.float32)
    gy = np.linspace(0.50, 0.75, h, dtype=np.float32)
    base = (np.outer(gy, gx)[:, :, None] * 255).astype(np.float32)
    img += np.repeat(base, 3, axis=2)

    n_patches = random.randint(4, 10)
    for _ in range(n_patches):
        x0 = random.randint(0, w-1)
        y0 = random.randint(0, h-1)
        pw = random.randint(20, w//2)
        ph = random.randint(20, h//2)
        col = np.array([
            random.randint(50, 200),
            random.randint(60, 220),
            random.randint(40, 180)
        ], dtype=np.float32)
        img[y0:y0+ph, x0:x0+pw] = col

    return np.clip(img, 0, 255).astype(np.uint8)


def _make_shadow_mask(h=256, w=256):
    """불규칙한 그림자 마스크 생성 (건물, 나무 그림자 모양)"""
    mask = np.zeros((h, w), dtype=np.uint8)
    n = random.randint(1, 4)
    for _ in range(n):
        shape = random.choice(["poly", "ellipse", "blob"])
        if shape == "ellipse":
            cx = random.randint(w//4, 3*w//4)
            cy = random.randint(h//4, 3*h//4)
            rx = random.randint(w//8, w//3)
            ry = random.randint(h//12, h//4)
            angle = random.randint(0, 180)
            cv2.ellipse(mask, (cx,cy), (rx,ry), angle, 0, 360, 255, -1)
        elif shape == "poly":
            n_pts = random.randint(4, 8)
            cx    = random.randint(w//4, 3*w//4)
            cy    = random.randint(h//4, 3*h//4)
            r     = random.randint(w//10, w//3)
            angles= sorted(random.uniform(0, 360) for _ in range(n_pts))
            pts   = []
            for a in angles:
                rad = np.radians(a)
                jitter = random.uniform(0.6, 1.0)
                pts.append([int(cx + r*jitter*np.cos(rad)),
                             int(cy + r*jitter*np.sin(rad))])
            cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], 255)
        else:  # blob
            n_b = random.randint(3, 6)
            for _ in range(n_b):
                bx = random.randint(0, w-1)
                by = random.randint(0, h-1)
                br = random.randint(15, 60)
                cv2.circle(mask, (bx, by), br, 255, -1)

    # 부드럽게
    k = random.choice([11, 21, 31])
    mask = cv2.GaussianBlur(mask, (k,k), k//3)
    _, mask = cv2.threshold(mask, 80, 255, cv2.THRESH_BINARY)
    return mask


def _apply_shadow(img, mask):
    """
    그림자 적용:
    - 밝기 감소 (랜덤 계수)
    - 색상 편이 (파란 기운)
    """
    gain  = np.random.uniform(0.35, 0.65)
    b_shift = np.random.uniform(5, 20)   # Blue 채널 약간 증가
    r_shift = np.random.uniform(-10, 0)  # Red 채널 약간 감소

    shadow_f  = img.astype(np.float32)
    alpha     = (mask.astype(np.float32) / 255.0)[:, :, None]

    # 채널별 변환
    shaded = shadow_f.copy()
    shaded[:,:,0] = shadow_f[:,:,0] * gain + r_shift  # B 채널 (OpenCV BGR)
    shaded[:,:,1] = shadow_f[:,:,1] * gain
    shaded[:,:,2] = shadow_f[:,:,2] * gain + r_shift  # R 채널

    # Blue shift (채널 0 = B in BGR)
    shaded[:,:,0] = shaded[:,:,0] + b_shift

    result = shadow_f * (1 - alpha) + shaded * alpha
    return np.clip(result, 0, 255).astype(np.uint8)


# ──────────────────────────────────────────────────────────────
# 데이터셋
# ──────────────────────────────────────────────────────────────

class ShadowDetectDataset(Dataset):
    def __init__(self, n=1200, size=128):
        self.n    = n
        self.size = size
        random.seed(42)
        np.random.seed(42)

    def __len__(self): return self.n

    def __getitem__(self, _):
        h = w = self.size
        base = _make_drone_base(h, w)
        mask = _make_shadow_mask(h, w)
        shad = _apply_shadow(base, mask)

        # [0,1] 정규화
        img_t  = torch.from_numpy(
            cv2.cvtColor(shad, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
        ).permute(2,0,1)
        mask_t = torch.from_numpy(mask.astype(np.float32) / 255.).unsqueeze(0)
        return img_t, mask_t


class ColorRestoreDataset(Dataset):
    def __init__(self, n=1200, size=128):
        self.n    = n
        self.size = size
        random.seed(99)
        np.random.seed(99)

    def __len__(self): return self.n

    def __getitem__(self, _):
        h = w = self.size
        clean = _make_drone_base(h, w)
        mask  = _make_shadow_mask(h, w)
        shad  = _apply_shadow(clean, mask)

        to_t = lambda img: torch.from_numpy(
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
        ).permute(2,0,1)
        return to_t(shad), to_t(clean)


# ──────────────────────────────────────────────────────────────
# 학습 루프
# ──────────────────────────────────────────────────────────────

def train_shadow_detector(epochs=8, batch=16):
    print("\n━━━ ShadowDetectorNet 학습 시작 ━━━")
    model   = ShadowDetectorNet().to(DEVICE)
    ds      = ShadowDetectDataset(n=1200)
    loader  = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=0)
    opt     = optim.Adam(model.parameters(), lr=3e-4)
    sched   = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    bce     = nn.BCELoss()

    for ep in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        for imgs, masks in loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            opt.zero_grad()
            pred = model(imgs)
            loss = bce(pred, masks)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        sched.step()
        print(f"  Epoch {ep}/{epochs}  loss={total_loss/len(loader):.4f}  lr={sched.get_last_lr()[0]:.6f}")

    out = os.path.join(MODEL_DIR, "shadow_detector.pth")
    torch.save(model.state_dict(), out)
    sz = os.path.getsize(out) / 1024 / 1024
    print(f"  ✅ 저장: {out}  ({sz:.1f} MB)")


def train_color_restore(epochs=8, batch=16):
    print("\n━━━ ColorRestorationNet 학습 시작 ━━━")
    model   = ColorRestorationNet().to(DEVICE)
    ds      = ColorRestoreDataset(n=1200)
    loader  = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=0)
    opt     = optim.Adam(model.parameters(), lr=3e-4)
    sched   = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    l1_loss = nn.L1Loss()

    for ep in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        for shad, clean in loader:
            shad, clean = shad.to(DEVICE), clean.to(DEVICE)
            opt.zero_grad()
            # 마스크 없이 전체 이미지 색상 복원 학습
            pred = model(shad)
            loss = l1_loss(pred, clean)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        sched.step()
        print(f"  Epoch {ep}/{epochs}  loss={total_loss/len(loader):.4f}  lr={sched.get_last_lr()[0]:.6f}")

    out = os.path.join(MODEL_DIR, "color_restore.pth")
    torch.save(model.state_dict(), out)
    sz = os.path.getsize(out) / 1024 / 1024
    print(f"  ✅ 저장: {out}  ({sz:.1f} MB)")


if __name__ == "__main__":
    t_start = time.time()
    print("🛸  Drone Shadow Model Trainer")
    print(f"   Device: {DEVICE}")
    print(f"   출력 폴더: {MODEL_DIR}")

    train_shadow_detector(epochs=8, batch=16)
    train_color_restore(epochs=8, batch=16)

    elapsed = time.time() - t_start
    m, s = divmod(int(elapsed), 60)
    print(f"\n🎉 완료! 총 소요 시간: {m}분 {s}초")
    print(f"   {MODEL_DIR}/shadow_detector.pth")
    print(f"   {MODEL_DIR}/color_restore.pth")
