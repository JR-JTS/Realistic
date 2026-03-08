"""
Drone Shadow Remover & Color Restorer
Desktop Application  –  Tkinter GUI  v3.0
────────────────────────────────────────
실행: python main.py

v3.0 수정 사항:
  • ZoomCanvas: 드래그·줌 이벤트 충돌 완전 수정, 왼쪽 패널 휠 간섭 제거
  • 비교 탭: 레이아웃 완전 수정 (행/열 weight 정확하게 재설정)
  • 미리보기 플래그: finally 블록으로 항상 해제 보장
  • 파일 선택 → 원본 표시 → 더블클릭 → 비교 탭으로 자동 전환
  • 재처리: 슬라이더 조정 후 버튼 클릭 시 즉시 재처리 가능
  • 그림자 탐지 과보정 방지 (청색편이 기반 신뢰도 가중치)
  • 처리 오류 상세 로그
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import queue
import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc

import cv2
import numpy as np
from PIL import Image, ImageTk

# ── src 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from pipeline import (
    load_models, process_single, scan_folder,
    scan_folder_recursive, make_compare, SUPPORTED_EXT
)

# ──────────────────────────────────────────────────────────
# AI 모델 다운로드 정보
# ──────────────────────────────────────────────────────────
MODEL_SERVER_BASE = "https://8080-i65w3eivwijb8fftrewca-2b54fc91.sandbox.novita.ai"

MODEL_INFO = [
    {
        "name":    "shadow_detector.pth",
        "label":   "그림자 탐지 AI  (ShadowDetectorNet)",
        "desc":    "UNet 스타일 경량 CNN – 드론 항공사진 그림자 영역 탐지\n"
                   "크기: ~7.5 MB  |  아키텍처: UNet (1.9M 파라미터)",
        "url":     MODEL_SERVER_BASE + "/models/shadow_detector.pth",
        "size_mb": 7.5,
        "dest":    "shadow_detector.pth",
    },
    {
        "name":    "color_restore.pth",
        "label":   "색상 복원 AI  (ColorRestorationNet)",
        "desc":    "Dilated-Conv 잔차학습 CNN – 그림자 영역 색상/밝기 복원\n"
                   "크기: ~0.9 MB  |  아키텍처: Residual (241K 파라미터)",
        "url":     MODEL_SERVER_BASE + "/models/color_restore.pth",
        "size_mb": 0.9,
        "dest":    "color_restore.pth",
    },
]

# ──────────────────────────────────────────────────────────
# 색상 테마
# ──────────────────────────────────────────────────────────
DARK   = "#1e1e2e"
DARK2  = "#2a2a3e"
DARK3  = "#313155"
ACCENT = "#7c6af7"
ACC2   = "#00c9ff"
GREEN  = "#3ddc84"
RED    = "#ff5f57"
WARN   = "#ffcb6b"
TEXT   = "#e0e0f0"
TEXT2  = "#9090b0"
WHITE  = "#ffffff"
CARD   = "#252538"

# ──────────────────────────────────────────────────────────
# 유틸 함수
# ──────────────────────────────────────────────────────────

def cv2pil(img_bgr: np.ndarray, max_side: int = 0) -> Image.Image:
    """BGR numpy → PIL RGB (선택적 축소)"""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    if max_side > 0:
        w, h = pil.size
        if max(w, h) > max_side:
            r = max_side / max(w, h)
            pil = pil.resize((max(1, int(w * r)), max(1, int(h * r))),
                             Image.BILINEAR)
    return pil


def safe_imread(path: str) -> "np.ndarray | None":
    """한글/유니코드 경로 대응 imread"""
    try:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is not None:
            return img
        with open(path, 'rb') as f:
            arr = np.frombuffer(f.read(), dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


# ──────────────────────────────────────────────────────────
# 커스텀 위젯: 플랫 버튼
# ──────────────────────────────────────────────────────────

class FlatButton(tk.Label):
    def __init__(self, parent, text="", command=None,
                 bg=ACCENT, fg=WHITE, hover=DARK3,
                 width=160, height=36, font_size=11, **kw):
        kw.pop("radius", None)
        super().__init__(
            parent, text=text, bg=bg, fg=fg,
            font=("Segoe UI", font_size, "bold"),
            cursor="hand2", relief="flat",
            padx=max(4, (width - len(text) * (font_size + 2)) // 2),
            pady=max(2, (height - font_size - 6) // 2),
            **kw
        )
        self._bg = bg; self._hover = hover; self._fg = fg
        self._enabled = True; self._cmd = command
        self.bind("<Enter>",         self._on_enter)
        self.bind("<Leave>",         self._on_leave)
        self.bind("<ButtonPress-1>", self._on_press)

    def _on_enter(self, _):
        if self._enabled: self.config(bg=self._hover)
    def _on_leave(self, _):
        if self._enabled: self.config(bg=self._bg)
    def _on_press(self, _):
        if self._enabled and self._cmd: self._cmd()

    def set_enabled(self, v: bool):
        self._enabled = v
        self.config(bg=self._bg if v else DARK3,
                    fg=self._fg if v else TEXT2,
                    cursor="hand2" if v else "")


# ──────────────────────────────────────────────────────────
# 라벨 슬라이더
# ──────────────────────────────────────────────────────────

class LabeledSlider(tk.Frame):
    def __init__(self, parent, label, from_, to, init, fmt=".2f", **kw):
        super().__init__(parent, bg=CARD, **kw)
        self._fmt = fmt
        self.var  = tk.DoubleVar(value=init)

        top = tk.Frame(self, bg=CARD)
        top.pack(fill="x")
        tk.Label(top, text=label, bg=CARD, fg=TEXT,
                  font=("Segoe UI", 9)).pack(side="left")
        self.val_lbl = tk.Label(top, text=format(init, fmt),
                                 bg=CARD, fg=ACC2,
                                 font=("Segoe UI", 9, "bold"), width=6, anchor="e")
        self.val_lbl.pack(side="right")
        ttk.Scale(self, from_=from_, to=to, variable=self.var,
                   orient="horizontal", command=self._upd).pack(fill="x", pady=(2, 0))

    def _upd(self, v):
        self.val_lbl.config(text=format(float(v), self._fmt))

    def get(self):
        return self.var.get()


# ──────────────────────────────────────────────────────────
# AI 모델 다운로드 다이얼로그
# ──────────────────────────────────────────────────────────

class ModelDownloadDialog(tk.Toplevel):
    def __init__(self, parent, model_dir: str, reload_callback=None):
        super().__init__(parent)
        self.title("AI 모델 다운로드")
        self.geometry("680x520"); self.minsize(600, 440)
        self.configure(bg=DARK); self.resizable(True, True)
        self.transient(parent); self.grab_set()
        self._model_dir = model_dir
        self._reload_cb = reload_callback
        self._dl_threads = {}; self._cancel_flags = {}
        self._queue = queue.Queue()
        self._build_ui(); self._refresh_status(); self._poll()

    def _build_ui(self):
        hdr = tk.Frame(self, bg=DARK3, height=52)
        hdr.pack(fill="x"); hdr.pack_propagate(False)
        tk.Label(hdr, text="AI 모델 다운로드 관리",
                  bg=DARK3, fg=WHITE,
                  font=("Segoe UI", 13, "bold")).pack(side="left", padx=16, pady=10)
        tk.Label(hdr, text="모델 설치 시 그림자 탐지·색상 복원 정확도 향상",
                  bg=DARK3, fg=TEXT2, font=("Segoe UI", 9)).pack(side="left", padx=4)

        url_f = tk.Frame(self, bg=DARK2, pady=6)
        url_f.pack(fill="x", padx=12, pady=(8, 0))
        tk.Label(url_f, text="서버 URL:", bg=DARK2, fg=TEXT2,
                  font=("Segoe UI", 9)).pack(side="left", padx=8)
        self._url_var = tk.StringVar(value=MODEL_SERVER_BASE)
        tk.Entry(url_f, textvariable=self._url_var, bg="#1a1a30", fg=ACC2,
                  insertbackground=WHITE, relief="flat",
                  font=("Consolas", 9), width=50).pack(
            side="left", fill="x", expand=True, ipady=3, padx=4)
        FlatButton(url_f, "적용", command=self._apply_url,
                    bg=DARK3, hover=DARK2, fg=TEXT2,
                    width=60, height=28, font_size=9).pack(side="left", padx=6)

        tk.Label(self,
                  text="직접 복사한 경우 [models] 폴더에 넣으면 자동 인식",
                  bg=DARK, fg=TEXT2, font=("Segoe UI", 8)).pack(
            anchor="w", padx=20, pady=(2, 4))

        self._cards = {}
        for info in MODEL_INFO:
            self._cards[info["name"]] = self._make_card(info)

        btn_f = tk.Frame(self, bg=DARK, pady=8)
        btn_f.pack(fill="x", padx=12)
        FlatButton(btn_f, "전체 다운로드", command=self._download_all,
                    bg=ACCENT, hover="#5b4dd6", width=140, height=36, font_size=10
                    ).pack(side="left", padx=(0, 8))
        FlatButton(btn_f, "상태 새로고침", command=self._refresh_status,
                    bg=DARK3, hover=DARK2, fg=TEXT2, width=130, height=36, font_size=10
                    ).pack(side="left", padx=(0, 8))
        FlatButton(btn_f, "적용 후 닫기", command=self._apply_and_close,
                    bg="#1a6b3a", hover="#0f4a28", width=120, height=36, font_size=10
                    ).pack(side="left")

        log_f = tk.Frame(self, bg=CARD)
        log_f.pack(fill="both", expand=True, padx=12, pady=(4, 8))
        tk.Label(log_f, text="다운로드 로그", bg=CARD, fg=TEXT2,
                  font=("Segoe UI", 9, "bold"), padx=8).pack(anchor="w", pady=(4, 2))
        sb = ttk.Scrollbar(log_f, orient="vertical")
        self._log_text = tk.Text(log_f, bg=CARD, fg=TEXT, font=("Consolas", 8),
                                  relief="flat", wrap="word", state="disabled",
                                  height=6, yscrollcommand=sb.set)
        sb.config(command=self._log_text.yview)
        sb.pack(side="right", fill="y")
        self._log_text.pack(fill="both", expand=True, padx=4, pady=(0, 4))
        for t, c in [("ok", GREEN), ("warn", WARN), ("error", RED), ("info", TEXT2)]:
            self._log_text.tag_config(t, foreground=c)

    def _make_card(self, info):
        name = info["name"]
        card = tk.Frame(self, bg=CARD, padx=12, pady=10)
        card.pack(fill="x", padx=12, pady=4)
        left = tk.Frame(card, bg=CARD)
        left.pack(side="left", fill="both", expand=True)
        title_f = tk.Frame(left, bg=CARD)
        title_f.pack(fill="x")
        tk.Label(title_f, text=info["label"], bg=CARD, fg=WHITE,
                  font=("Segoe UI", 10, "bold")).pack(side="left")
        status_lbl = tk.Label(title_f, text="…", bg=CARD, fg=TEXT2,
                               font=("Segoe UI", 8), padx=8)
        status_lbl.pack(side="left", padx=8)
        tk.Label(left, text=info["desc"], bg=CARD, fg=TEXT2,
                  font=("Segoe UI", 8), justify="left", anchor="w").pack(fill="x", pady=(2, 4))
        prog = ttk.Progressbar(left, mode="determinate", length=400)
        prog.pack(fill="x", pady=(2, 0))
        prog_lbl = tk.Label(left, text="", bg=CARD, fg=TEXT2, font=("Consolas", 8))
        prog_lbl.pack(anchor="e")
        right = tk.Frame(card, bg=CARD)
        right.pack(side="right", padx=(12, 0))
        dl_btn = FlatButton(right, "다운로드",
                             command=lambda n=name: self._download_one(n),
                             bg="#2a6496", hover="#1d4f75", width=100, height=32, font_size=9)
        dl_btn.pack(pady=(0, 6))
        cancel_btn = FlatButton(right, "중단",
                                 command=lambda n=name: self._cancel_one(n),
                                 bg="#6b2222", hover="#4a1515", width=100, height=32, font_size=9)
        cancel_btn.pack()
        cancel_btn.set_enabled(False)
        return {"info": info, "status_lbl": status_lbl, "prog": prog,
                "prog_lbl": prog_lbl, "dl_btn": dl_btn, "cancel_btn": cancel_btn}

    def _refresh_status(self):
        for name, card in self._cards.items():
            path = os.path.join(self._model_dir, name)
            if os.path.exists(path):
                sz = os.path.getsize(path) / 1024 / 1024
                card["status_lbl"].config(text=f"설치됨 ({sz:.1f} MB)", fg=GREEN)
                card["prog"]["value"] = 100
                card["prog_lbl"].config(text="완료")
            else:
                card["status_lbl"].config(text="미설치", fg=RED)
                card["prog"]["value"] = 0; card["prog_lbl"].config(text="")

    def _apply_url(self):
        base = self._url_var.get().rstrip("/")
        for info in MODEL_INFO:
            info["url"] = base + "/models/" + info["dest"]
        self._log("URL 변경: " + base, "info")

    def _download_one(self, name: str):
        if name in self._dl_threads and self._dl_threads[name].is_alive():
            return
        info = next(i for i in MODEL_INFO if i["name"] == name)
        card = self._cards[name]
        flag = threading.Event()
        self._cancel_flags[name] = flag
        card["dl_btn"].set_enabled(False); card["cancel_btn"].set_enabled(True)
        card["status_lbl"].config(text="다운로드 중…", fg=WARN)

        def _work():
            dest = os.path.join(self._model_dir, info["dest"])
            os.makedirs(self._model_dir, exist_ok=True)
            self._queue.put(("dl_log", name, f"시작: {info['url']}", "info"))
            try:
                req = urllib.request.Request(
                    info["url"], headers={"User-Agent": "DroneShadowRemover/3.0"})
                with urllib.request.urlopen(req, timeout=60) as resp:
                    total = int(resp.headers.get("Content-Length", 0))
                    done  = 0; t0 = time.time()
                    with open(dest, "wb") as f:
                        while True:
                            if flag.is_set():
                                self._queue.put(("dl_cancel", name)); return
                            data = resp.read(65536)
                            if not data: break
                            f.write(data); done += len(data)
                            elapsed = max(time.time() - t0, 0.001)
                            spd = done / elapsed / 1024
                            pct = int(done / total * 100) if total else 0
                            self._queue.put(("dl_progress", name, pct,
                                f"{done/1048576:.1f}/{total/1048576:.1f} MB ({spd:.0f} KB/s)"))
                sz = os.path.getsize(dest) / 1048576
                self._queue.put(("dl_done", name, f"완료: {info['dest']} ({sz:.1f} MB)"))
            except Exception as e:
                if os.path.exists(dest):
                    try: os.remove(dest)
                    except: pass
                self._queue.put(("dl_error", name, str(e)))

        t = threading.Thread(target=_work, daemon=True)
        self._dl_threads[name] = t; t.start()

    def _download_all(self):
        for info in MODEL_INFO:
            if not os.path.exists(os.path.join(self._model_dir, info["dest"])):
                self._download_one(info["name"])

    def _cancel_one(self, name):
        if name in self._cancel_flags: self._cancel_flags[name].set()

    def _apply_and_close(self):
        if self._reload_cb: self._reload_cb()
        self.destroy()

    def _poll(self):
        try:
            while True:
                msg  = self._queue.get_nowait()
                kind = msg[0]
                if kind == "dl_progress":
                    _, name, pct, txt = msg
                    self._cards[name]["prog"]["value"] = pct
                    self._cards[name]["prog_lbl"].config(text=txt)
                elif kind == "dl_done":
                    _, name, log_msg = msg
                    c = self._cards[name]
                    c["prog"]["value"] = 100; c["prog_lbl"].config(text="완료")
                    c["status_lbl"].config(text="설치됨", fg=GREEN)
                    c["dl_btn"].set_enabled(True); c["cancel_btn"].set_enabled(False)
                    self._log(log_msg, "ok")
                elif kind == "dl_cancel":
                    _, name = msg
                    c = self._cards[name]
                    c["status_lbl"].config(text="취소됨", fg=WARN)
                    c["prog"]["value"] = 0; c["prog_lbl"].config(text="")
                    c["dl_btn"].set_enabled(True); c["cancel_btn"].set_enabled(False)
                    self._log(f"취소: {name}", "warn")
                elif kind == "dl_error":
                    _, name, err = msg
                    c = self._cards[name]
                    c["status_lbl"].config(text="오류", fg=RED)
                    c["prog"]["value"] = 0; c["prog_lbl"].config(text="")
                    c["dl_btn"].set_enabled(True); c["cancel_btn"].set_enabled(False)
                    self._log(f"오류 ({name}): {err}", "error")
                elif kind == "dl_log":
                    _, name, msg_txt, tag = msg
                    self._log(msg_txt, tag)
        except queue.Empty:
            pass
        if self.winfo_exists():
            self.after(100, self._poll)

    def _log(self, text, tag="info"):
        self._log_text.config(state="normal")
        self._log_text.insert("end", f"[{time.strftime('%H:%M:%S')}] {text}\n", tag)
        self._log_text.see("end"); self._log_text.config(state="disabled")


# ──────────────────────────────────────────────────────────
# ZoomCanvas: 고속 줌/패닝 캔버스  (v3.0 완전 재작성)
# ──────────────────────────────────────────────────────────

class ZoomCanvas(tk.Canvas):
    """
    고속 줌+패닝 캔버스.

    v3.0 수정사항:
    - 렌더링: 가시 영역만 크롭 후 리사이즈 (대용량 이미지 대응)
    - 리샘플링: 축소=BILINEAR, 확대=NEAREST (약 5배 빠름)
    - _schedule_render 10ms 딜레이로 연속 이벤트 병합
    - 드래그·더블클릭 이벤트 독립 처리 (충돌 없음)
    - bind_all 사용 금지 (다른 위젯 간섭 방지)
    """
    MIN_SCALE = 0.03
    MAX_SCALE = 15.0

    def __init__(self, parent, bg=CARD, hint="이미지 없음",
                 label="", label_color=WHITE, **kw):
        super().__init__(parent, bg=bg, highlightthickness=0, **kw)
        self._pil_orig       = None   # PIL 이미지 (원본 크기)
        self._scale          = 1.0
        self._offset_x       = 0.0
        self._offset_y       = 0.0
        self._drag_x         = 0
        self._drag_y         = 0
        self._tk_img         = None   # GC 방지용
        self._hint           = hint
        self._label_text     = label
        self._label_color    = label_color
        self._render_pending = False
        self._dragging       = False

        self.bind("<Configure>",      self._on_configure)
        self.bind("<MouseWheel>",     self._on_wheel)
        self.bind("<Button-4>",       self._on_wheel)
        self.bind("<Button-5>",       self._on_wheel)
        self.bind("<ButtonPress-1>",  self._on_drag_start)
        self.bind("<B1-Motion>",      self._on_drag_move)
        self.bind("<ButtonRelease-1>",self._on_drag_end)
        # 더블클릭: 뷰 초기화 (드래그와 독립)
        self.bind("<Double-Button-1>",self._reset_view)
        self.bind("<Button-3>",       self._reset_view)

    # ──────────────────────────────────────────────────────
    # 공개 API

    def load_image(self, pil_img: Image.Image):
        """PIL 이미지를 로드하고 캔버스에 맞게 표시"""
        self._pil_orig = pil_img
        self._fit_to_canvas()

    def clear(self):
        """이미지 제거 및 힌트 표시"""
        self._pil_orig = None
        self._tk_img   = None
        self.delete("all")
        self._draw_hint()

    # ──────────────────────────────────────────────────────
    # 내부 메서드

    def _draw_hint(self):
        cw = self.winfo_width()
        ch = self.winfo_height()
        if cw > 30 and ch > 30:
            self.delete("all")
            self.create_text(cw // 2, ch // 2, text=self._hint,
                              fill=TEXT2, font=("Segoe UI", 11),
                              justify="center", tags="hint")

    def _fit_to_canvas(self):
        """이미지를 캔버스에 딱 맞게 초기 배치"""
        if self._pil_orig is None:
            return
        cw = self.winfo_width()
        ch = self.winfo_height()
        if cw < 10 or ch < 10:
            # 캔버스가 아직 배치 안 됨 → 나중에 재시도
            self.after(80, self._fit_to_canvas)
            return
        iw, ih = self._pil_orig.size
        scale  = min(cw / iw, ch / ih, 1.0)
        self._scale    = scale
        self._offset_x = (cw - iw * scale) / 2.0
        self._offset_y = (ch - ih * scale) / 2.0
        self._schedule_render()

    def _schedule_render(self):
        """중복 렌더 방지: 10ms 후 한 번만 렌더"""
        if not self._render_pending:
            self._render_pending = True
            self.after(10, self._do_render)

    def _do_render(self):
        self._render_pending = False
        if self._pil_orig is None:
            return
        self._render()

    def _render(self):
        if self._pil_orig is None:
            return
        cw = self.winfo_width()
        ch = self.winfo_height()
        if cw < 10 or ch < 10:
            return

        iw, ih = self._pil_orig.size
        ox = self._offset_x
        oy = self._offset_y
        sc = self._scale

        # 가시 영역: 이미지 좌표계로 변환
        img_x0 = max(0, int(-ox / sc))
        img_y0 = max(0, int(-oy / sc))
        img_x1 = min(iw, int((cw - ox) / sc) + 2)
        img_y1 = min(ih, int((ch - oy) / sc) + 2)

        if img_x1 <= img_x0 or img_y1 <= img_y0:
            self.delete("all")
            return

        # 크롭 후 리사이즈 (필요한 부분만)
        crop   = self._pil_orig.crop((img_x0, img_y0, img_x1, img_y1))
        dst_w  = max(1, int((img_x1 - img_x0) * sc))
        dst_h  = max(1, int((img_y1 - img_y0) * sc))

        # 리샘플링 선택: 축소=BILINEAR (품질), 확대=NEAREST (속도)
        resample = Image.BILINEAR if sc < 1.0 else Image.NEAREST
        resized  = crop.resize((dst_w, dst_h), resample)

        # 캔버스에서의 그리기 위치
        draw_x = max(0, int(ox + img_x0 * sc))
        draw_y = max(0, int(oy + img_y0 * sc))

        tk_img = ImageTk.PhotoImage(resized)
        self.delete("all")
        self.create_image(draw_x, draw_y, anchor="nw", image=tk_img, tags="img")
        self._tk_img = tk_img  # GC 방지

        # 레이블 (좌상단 반투명 배경)
        if self._label_text:
            lw = len(self._label_text) * 9 + 18
            self.create_rectangle(0, 0, lw, 26, fill="#000000", outline="",
                                   stipple="gray50")
            self.create_text(9, 13, text=self._label_text, anchor="w",
                              fill=self._label_color,
                              font=("Segoe UI", 10, "bold"), tags="lbl")

        # 줌 레벨 (우하단)
        zt = f" ×{sc:.2f}  [우클릭=초기화] "
        zw = len(zt) * 7 + 6
        self.create_rectangle(cw - zw, ch - 20, cw, ch,
                               fill="#000000", outline="", stipple="gray50")
        self.create_text(cw - 4, ch - 4, text=zt, anchor="se",
                          fill=TEXT2, font=("Consolas", 8), tags="zoom_lbl")

    def _on_configure(self, event):
        if self._pil_orig is None:
            self._draw_hint()
        else:
            self._schedule_render()

    def _on_wheel(self, event):
        if self._pil_orig is None:
            return
        # 줌 중심: 마우스 커서 위치
        mx = float(event.x)
        my = float(event.y)

        if event.num == 4:
            factor = 1.15
        elif event.num == 5:
            factor = 1.0 / 1.15
        elif event.delta != 0:
            factor = 1.15 if event.delta > 0 else 1.0 / 1.15
        else:
            return

        new_scale = max(self.MIN_SCALE, min(self.MAX_SCALE, self._scale * factor))
        if abs(new_scale - self._scale) < 1e-6:
            return
        ratio = new_scale / self._scale
        self._offset_x = mx - ratio * (mx - self._offset_x)
        self._offset_y = my - ratio * (my - self._offset_y)
        self._scale    = new_scale
        self._schedule_render()

    def _on_drag_start(self, event):
        self._drag_x  = event.x
        self._drag_y  = event.y
        self._dragging = True

    def _on_drag_move(self, event):
        if not self._dragging or self._pil_orig is None:
            return
        dx = event.x - self._drag_x
        dy = event.y - self._drag_y
        self._offset_x += dx
        self._offset_y += dy
        self._drag_x    = event.x
        self._drag_y    = event.y
        # 드래그 중에는 즉시 렌더 (딜레이 없음)
        if not self._render_pending:
            self._render_pending = True
            self.after(0, self._do_render)

    def _on_drag_end(self, event):
        self._dragging = False

    def _reset_view(self, event=None):
        self._fit_to_canvas()


# ──────────────────────────────────────────────────────────
# 메인 앱
# ──────────────────────────────────────────────────────────

class DroneApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Drone Shadow Remover & Color Restorer  v3.0")
        self.geometry("1560x960")
        self.minsize(1200, 720)
        self.configure(bg=DARK)
        self._set_icon()

        # ── 상태 변수
        self.input_folder  = tk.StringVar(value="")
        self.output_folder = tk.StringVar(value="")
        self.recursive_var = tk.BooleanVar(value=False)
        self.save_compare  = tk.BooleanVar(value=True)
        self.overwrite_var = tk.BooleanVar(value=False)
        self.worker_var    = tk.IntVar(value=1)

        self._file_list      = []
        self._selected_idx   = -1
        self._selected_path  = ""
        self._is_running     = False
        self._is_previewing  = False   # 미리보기 진행 중 플래그
        self._cancel_flag    = threading.Event()
        self._queue          = queue.Queue()
        self._preview_lock   = threading.Lock()

        # 현재 표시 이미지 (BGR)
        self._cur_orig   = None
        self._cur_result = None
        self._cur_binary = None
        self._cur_soft   = None

        self._model_status = {}

        self._build_ui()
        self._load_models_async()
        self._poll_queue()

    def _set_icon(self):
        try: self.iconbitmap("")
        except: pass

    # ──────────────────────────────────────────────────────
    # 모델 비동기 로드

    def _load_models_async(self):
        def _load():
            gc.collect()
            base   = os.path.dirname(os.path.abspath(__file__))
            status = load_models(os.path.join(base, "models"))
            self._model_status = status
            self._queue.put(("model_loaded", status))
        threading.Thread(target=_load, daemon=True).start()
        self._set_status("모델 로딩 중...", WARN)

    # ──────────────────────────────────────────────────────
    # UI 빌드

    def _build_ui(self):
        self._build_header()
        body = tk.Frame(self, bg=DARK)
        body.pack(fill="both", expand=True, padx=10, pady=(0, 8))
        body.columnconfigure(0, weight=0, minsize=280)
        body.columnconfigure(1, weight=1, minsize=400)
        body.columnconfigure(2, weight=0, minsize=300)
        body.rowconfigure(0, weight=1)
        self._build_left_panel(body)
        self._build_center_panel(body)
        self._build_right_panel(body)
        self._build_footer()

    # ── 헤더

    def _build_header(self):
        hdr = tk.Frame(self, bg=DARK3, height=60)
        hdr.pack(fill="x"); hdr.pack_propagate(False)

        left = tk.Frame(hdr, bg=DARK3)
        left.pack(side="left", padx=16)
        tk.Label(left, text="Drone Shadow Remover", bg=DARK3, fg=WHITE,
                  font=("Segoe UI", 14, "bold")).pack(anchor="w")
        tk.Label(left,
                  text="v3.0  |  클릭:원본  |  더블클릭:즉시처리  |  슬라이더 조정 후 미리보기 버튼으로 재처리",
                  bg=DARK3, fg=TEXT2, font=("Segoe UI", 9)).pack(anchor="w")

        FlatButton(hdr, "AI 모델 다운로드", command=self._open_model_download,
                    bg="#2a6496", hover="#1d4f75", width=150, height=32, font_size=9
                    ).pack(side="right", padx=8, pady=14)
        self.model_badge = tk.Label(hdr, text="모델 로딩 중", bg=DARK3, fg=WARN,
                                     font=("Segoe UI", 9))
        self.model_badge.pack(side="right", padx=8)

    # ── 왼쪽 패널 (스크롤 가능)

    def _build_left_panel(self, parent):
        outer = tk.Frame(parent, bg=DARK, width=280)
        outer.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        outer.pack_propagate(False)

        # 스크롤 캔버스 (왼쪽 패널 내용이 길기 때문)
        scroll_canvas = tk.Canvas(outer, bg=DARK, highlightthickness=0)
        sb     = ttk.Scrollbar(outer, orient="vertical", command=scroll_canvas.yview)
        scroll_canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        scroll_canvas.pack(side="left", fill="both", expand=True)

        inner  = tk.Frame(scroll_canvas, bg=DARK)
        win_id = scroll_canvas.create_window((0, 0), window=inner, anchor="nw")

        scroll_canvas.bind(
            "<Configure>",
            lambda e: scroll_canvas.itemconfig(win_id, width=e.width))
        inner.bind(
            "<Configure>",
            lambda e: scroll_canvas.configure(scrollregion=scroll_canvas.bbox("all")))

        # 왼쪽 패널 전용 휠 핸들러 (bind_all 사용 안 함)
        def _panel_wheel(e):
            delta = -1 * (e.delta // 120) if e.delta else (-1 if e.num == 5 else 1)
            scroll_canvas.yview_scroll(delta, "units")

        # 왼쪽 패널 내부 모든 자식에 휠 등록 (propagate 방식)
        self._left_scroll_canvas = scroll_canvas
        self._left_wheel_handler = _panel_wheel

        p = inner

        # ── 폴더 설정
        self._section(p, "폴더 설정")
        self._build_folder_section(p)

        # ── 동시 처리 수
        tw = tk.Frame(p, bg=DARK)
        tw.pack(fill="x", padx=10, pady=(4, 2))
        tk.Label(tw, text="동시 처리 수:", bg=DARK, fg=TEXT,
                  font=("Segoe UI", 9)).pack(side="left")
        for v in (1, 2, 4):
            rb = tk.Radiobutton(tw, text=str(v), variable=self.worker_var, value=v,
                                 bg=DARK, fg=TEXT2, selectcolor=DARK3,
                                 activebackground=DARK, font=("Segoe UI", 9))
            rb.pack(side="left", padx=3)
        tk.Label(tw, text="(메모리 부족 시 1)", bg=DARK, fg=TEXT2,
                  font=("Segoe UI", 7)).pack(side="left", padx=2)

        # ── 탐지 설정
        self._section(p, "그림자 탐지")
        self.detect_mode = tk.StringVar(value="hybrid")
        dm_f = tk.Frame(p, bg=DARK)
        dm_f.pack(fill="x", padx=10, pady=4)
        for txt, val in [("AI+CV 복합", "hybrid"), ("CV 방법만", "cv")]:
            tk.Radiobutton(dm_f, text=txt, variable=self.detect_mode, value=val,
                            bg=DARK, fg=TEXT, selectcolor=DARK3, activebackground=DARK,
                            font=("Segoe UI", 9)).pack(side="left", padx=5)

        self.sl_sensitivity = self._slider(p, "탐지 민감도",   0.1, 1.0, 0.45)
        self.sl_feather     = self._slider(p, "마스크 페더링", 3,   50,  20, fmt=".0f")

        # ── 색상 복원 설정 (v6.0)
        self._section(p, "그림자 색상+밝기 복원")
        self.use_ai_color = tk.BooleanVar(value=True)
        tk.Checkbutton(p, text="AI 미세 보정 (모델 로드 시 활성)",
                        variable=self.use_ai_color,
                        bg=DARK, fg=TEXT, selectcolor=DARK3, activebackground=DARK,
                        font=("Segoe UI", 9)).pack(anchor="w", padx=12)
        self.use_hue_consist = tk.BooleanVar(value=True)
        tk.Checkbutton(p, text="Hue 기반 정밀 색상 복원 (권장)",
                        variable=self.use_hue_consist,
                        bg=DARK, fg=TEXT, selectcolor=DARK3, activebackground=DARK,
                        font=("Segoe UI", 9)).pack(anchor="w", padx=12)

        self.sl_radio    = self._slider(p, "색상+밝기 복원 강도", 0.3, 1.0, 0.90)
        self.sl_highlight = self._slider(p, "Highlight 보호",     0.0, 0.6, 0.25)

        # ── 영상 품질 개선
        self._section(p, "영상 품질 개선")
        self.sl_denoise = self._slider(p, "노이즈 제거",     1,   12,  5,   fmt=".0f")
        self.sl_sharpen = self._slider(p, "선명도",          0.0, 2.0, 0.8)
        self.sl_clahe   = self._slider(p, "CLAHE 로컈대비",  0.5, 4.0, 2.0)

        # ── 실행 버튼
        self._section(p, "")
        self.btn_preview = FlatButton(
            p, "선택 이미지 미리보기 (재처리)",
            command=self._preview_selected,
            bg="#1a6b3a", hover="#0f4a28",
            width=250, height=36, font_size=10)
        self.btn_preview.pack(pady=(4, 2), padx=10)

        tk.Label(p,
                  text="더블클릭=즉시처리  |  슬라이더 조정 후 이 버튼으로 재처리",
                  bg=DARK, fg=WARN, font=("Segoe UI", 7)).pack(padx=12, anchor="w")

        ttk.Separator(p, orient="horizontal").pack(fill="x", padx=8, pady=6)

        btn_row = tk.Frame(p, bg=DARK)
        btn_row.pack(fill="x", padx=10, pady=2)
        self.btn_scan = FlatButton(btn_row, "파일 스캔",
                                    command=self._scan_files,
                                    bg="#2a6496", hover="#1d4f75",
                                    width=115, height=34, font_size=10)
        self.btn_scan.pack(side="left", padx=(0, 6))
        self.btn_start = FlatButton(btn_row, "전체 처리",
                                     command=self._start_processing,
                                     bg=ACCENT, hover="#5b4dd6",
                                     width=115, height=34, font_size=10)
        self.btn_start.pack(side="left")

        self.btn_cancel = FlatButton(
            p, "처리 중단",
            command=self._cancel_processing,
            bg="#8b3a3a", hover="#6b2222",
            width=250, height=32, font_size=10)
        self.btn_cancel.pack(pady=(4, 8), padx=10)
        self.btn_cancel.set_enabled(False)

        # 왼쪽 패널 내부 위젯들에 휠 이벤트 등록
        self._bind_wheel_to_children(inner, _panel_wheel)

    def _bind_wheel_to_children(self, widget, handler):
        """위젯과 모든 자식에 휠 이벤트 바인딩 (ZoomCanvas 제외)"""
        if isinstance(widget, ZoomCanvas):
            return   # ZoomCanvas는 자체 휠 핸들러 사용
        try:
            widget.bind("<MouseWheel>", handler, add="+")
            widget.bind("<Button-4>",   handler, add="+")
            widget.bind("<Button-5>",   handler, add="+")
        except Exception:
            pass
        for child in widget.winfo_children():
            self._bind_wheel_to_children(child, handler)

    def _section(self, parent, title):
        f = tk.Frame(parent, bg=DARK)
        f.pack(fill="x", padx=8, pady=(8, 2))
        if title:
            tk.Label(f, text=title, bg=DARK, fg=ACC2,
                      font=("Segoe UI", 10, "bold")).pack(side="left")
            ttk.Separator(f, orient="horizontal").pack(
                side="left", fill="x", expand=True, padx=6)

    def _build_folder_section(self, parent):
        # ── 입력 폴더
        tk.Label(parent, text="드론 사진 폴더", bg=DARK, fg=TEXT,
                  font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=12, pady=(4, 2))
        in_card = tk.Frame(parent, bg=DARK2, padx=8, pady=6)
        in_card.pack(fill="x", padx=10, pady=(0, 4))

        self._in_path_lbl = tk.Label(in_card, textvariable=self.input_folder,
                                      bg=DARK2, fg=ACC2, font=("Consolas", 8),
                                      wraplength=230, justify="left", anchor="w")
        self._in_path_lbl.pack(fill="x")
        self.input_folder.trace_add("write", lambda *_: self._upd_path_lbl(
            self._in_path_lbl, self.input_folder, "버튼으로 폴더 선택"))
        self._upd_path_lbl(self._in_path_lbl, self.input_folder, "버튼으로 폴더 선택")

        br = tk.Frame(in_card, bg=DARK2)
        br.pack(fill="x", pady=(4, 0))
        FlatButton(br, "폴더 선택", command=self._browse_input,
                    bg=ACCENT, hover="#5b4dd6", width=130, height=28, font_size=9
                    ).pack(side="left", padx=(0, 4))
        FlatButton(br, "직접 입력", command=lambda: self._toggle_entry("in"),
                    bg=DARK3, hover=DARK2, fg=TEXT2, width=80, height=28, font_size=9
                    ).pack(side="left")
        self._in_entry = tk.Frame(in_card, bg=DARK2)
        tk.Entry(self._in_entry, textvariable=self.input_folder,
                  bg="#1a1a30", fg=ACC2, insertbackground=WHITE,
                  relief="flat", font=("Consolas", 9)
                  ).pack(fill="x", pady=(4, 0), ipady=4)

        # ── 출력 폴더
        tk.Label(parent, text="저장 폴더", bg=DARK, fg=TEXT,
                  font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=12, pady=(6, 2))
        out_card = tk.Frame(parent, bg=DARK2, padx=8, pady=6)
        out_card.pack(fill="x", padx=10, pady=(0, 4))

        self._out_path_lbl = tk.Label(out_card, textvariable=self.output_folder,
                                       bg=DARK2, fg=GREEN, font=("Consolas", 8),
                                       wraplength=230, justify="left", anchor="w")
        self._out_path_lbl.pack(fill="x")
        self.output_folder.trace_add("write", lambda *_: self._upd_path_lbl(
            self._out_path_lbl, self.output_folder, "입력 폴더 선택 시 자동 설정"))
        self._upd_path_lbl(self._out_path_lbl, self.output_folder, "입력 폴더 선택 시 자동 설정")

        or_ = tk.Frame(out_card, bg=DARK2)
        or_.pack(fill="x", pady=(4, 0))
        FlatButton(or_, "폴더 선택", command=self._browse_output,
                    bg="#2a6496", hover="#1d4f75", width=130, height=28, font_size=9
                    ).pack(side="left", padx=(0, 4))
        FlatButton(or_, "직접 입력", command=lambda: self._toggle_entry("out"),
                    bg=DARK3, hover=DARK2, fg=TEXT2, width=80, height=28, font_size=9
                    ).pack(side="left")
        self._out_entry = tk.Frame(out_card, bg=DARK2)
        tk.Entry(self._out_entry, textvariable=self.output_folder,
                  bg="#1a1a30", fg=GREEN, insertbackground=WHITE,
                  relief="flat", font=("Consolas", 9)
                  ).pack(fill="x", pady=(4, 0), ipady=4)

        # ── 옵션 체크
        opts = tk.Frame(parent, bg=DARK)
        opts.pack(fill="x", padx=10, pady=(4, 2))
        self._chk(opts, "하위 폴더 포함",    self.recursive_var)
        self._chk(opts, "비교 이미지 저장",  self.save_compare)
        self._chk(opts, "기존 파일 덮어쓰기", self.overwrite_var)

    def _upd_path_lbl(self, lbl, var, placeholder):
        if not var.get():
            lbl.config(text=placeholder, fg=TEXT2)

    def _toggle_entry(self, which):
        f = self._in_entry if which == "in" else self._out_entry
        if f.winfo_ismapped(): f.pack_forget()
        else:                  f.pack(fill="x", pady=(4, 0))

    def _slider(self, parent, label, lo, hi, init, fmt=".2f"):
        s = LabeledSlider(parent, label, lo, hi, init, fmt=fmt, padx=10, pady=2)
        s.pack(fill="x", padx=10, pady=1)
        return s

    def _chk(self, parent, text, var):
        tk.Checkbutton(parent, text=text, variable=var, bg=DARK, fg=TEXT,
                        selectcolor=DARK3, activebackground=DARK,
                        font=("Segoe UI", 9)).pack(anchor="w")

    # ── 가운데 패널 (이미지 비교)

    def _build_center_panel(self, parent):
        frame = tk.Frame(parent, bg=DARK)
        frame.grid(row=0, column=1, sticky="nsew", padx=4)
        frame.rowconfigure(0, weight=0)   # 탭 바 (고정)
        frame.rowconfigure(1, weight=1)   # 이미지 영역 (확장)
        frame.rowconfigure(2, weight=0)   # 파일 목록 (고정)
        frame.columnconfigure(0, weight=1)

        # ── 탭 바
        tab_bar = tk.Frame(frame, bg=DARK2, height=40)
        tab_bar.grid(row=0, column=0, sticky="ew")
        tab_bar.pack_propagate(False)

        self._tab_btns   = {}
        self._active_tab = "compare"

        tabs = [("비교 보기", "compare"), ("원본", "orig"), ("복원 결과", "result")]
        for txt, key in tabs:
            b = tk.Label(tab_bar, text=txt, bg=DARK2, fg=TEXT2,
                          font=("Segoe UI", 10), padx=14, pady=9, cursor="hand2")
            b.pack(side="left")
            b.bind("<Button-1>", lambda e, k=key: self._switch_tab(k))
            self._tab_btns[key] = b

        tk.Label(tab_bar,
                  text="휠:줌  드래그:이동  우클릭/더블클릭:초기화",
                  bg=DARK2, fg=TEXT2, font=("Segoe UI", 7)).pack(side="right", padx=10)

        # ── 이미지 컨테이너
        # compare 탭: [왼쪽 캔버스 | 분할선 | 오른쪽 캔버스]
        # orig/result 탭: [단일 캔버스 (전체)]
        self._img_outer = tk.Frame(frame, bg=CARD)
        self._img_outer.grid(row=1, column=0, sticky="nsew")

        # 비교 탭용 ZoomCanvas (좌/우)
        self.zoom_left = ZoomCanvas(
            self._img_outer, bg="#0e0e20",
            hint="원본 이미지",
            label="📷 원본 (Original)", label_color=ACC2)
        self.zoom_right = ZoomCanvas(
            self._img_outer, bg="#0e1a0e",
            hint="미리보기 후 표시됩니다",
            label="✨ 복원 (Restored)", label_color=GREEN)

        # 단일 탭용 ZoomCanvas
        self.zoom_single = ZoomCanvas(
            self._img_outer, bg=CARD,
            hint="폴더 선택 → 파일 스캔\n→ 파일 더블클릭으로 즉시 처리\n\n결과 확인 후 [전체 처리] 버튼")

        # 분할선
        self._divider = tk.Frame(self._img_outer, bg=ACCENT, width=3)

        # 헤더 바 (비교 탭용)
        self._bar_left = tk.Frame(self._img_outer, bg="#161628", height=24)
        tk.Label(self._bar_left, text="  📷 원본 (Original) ",
                  bg="#161628", fg=ACC2,
                  font=("Segoe UI", 9, "bold")).pack(side="left", padx=4)

        self._bar_right = tk.Frame(self._img_outer, bg="#0f1f0f", height=24)
        tk.Label(self._bar_right, text="  ✨ 복원 (Restored) ",
                  bg="#0f1f0f", fg=GREEN,
                  font=("Segoe UI", 9, "bold")).pack(side="left", padx=4)

        # 초기 탭 설정
        self._apply_tab_layout("compare")

        # ── 파일 목록
        list_outer = tk.Frame(frame, bg=DARK2, height=125)
        list_outer.grid(row=2, column=0, sticky="ew", pady=(3, 0))
        list_outer.pack_propagate(False)

        hdr = tk.Frame(list_outer, bg=DARK2)
        hdr.pack(fill="x", padx=4, pady=(4, 0))
        tk.Label(hdr, text="파일 목록", bg=DARK2, fg=TEXT2,
                  font=("Segoe UI", 9, "bold")).pack(side="left", padx=4)
        self._cnt_lbl = tk.Label(hdr, text="(0개)", bg=DARK2, fg=TEXT2,
                                  font=("Segoe UI", 8))
        self._cnt_lbl.pack(side="left")
        tk.Label(hdr,
                  text="클릭:원본 표시  |  더블클릭:즉시 그림자제거+복원 처리",
                  bg=DARK2, fg=WARN, font=("Segoe UI", 8)).pack(side="right", padx=8)

        list_sb = ttk.Scrollbar(list_outer, orient="vertical")
        self.file_listbox = tk.Listbox(
            list_outer, bg=DARK2, fg=TEXT, selectbackground=ACCENT,
            relief="flat", borderwidth=0, font=("Consolas", 9),
            yscrollcommand=list_sb.set, activestyle="none")
        list_sb.config(command=self.file_listbox.yview)
        list_sb.pack(side="right", fill="y")
        self.file_listbox.pack(fill="both", expand=True, padx=4, pady=(2, 4))
        self.file_listbox.bind("<<ListboxSelect>>", self._on_file_select)
        self.file_listbox.bind("<Double-Button-1>",  self._on_file_double)

    def _switch_tab(self, key: str):
        self._active_tab = key
        for k, b in self._tab_btns.items():
            b.config(bg=DARK3 if k == key else DARK2,
                      fg=WHITE if k == key else TEXT2)
        self._apply_tab_layout(key)

    def _apply_tab_layout(self, key: str):
        """
        탭 레이아웃 완전 재구성.
        place() 사용하여 grid weight 버그 회피.
        """
        # 모든 위젯 숨기기
        for w in (self.zoom_single, self.zoom_left, self.zoom_right,
                  self._bar_left, self._bar_right, self._divider):
            w.place_forget()

        if key == "compare":
            # 비교 탭: 헤더바(24px) + 좌우 캔버스 (분할선 포함)
            # place_forget → place 방식으로 정밀 배치
            self._img_outer.update_idletasks()
            W = self._img_outer.winfo_width()
            H = self._img_outer.winfo_height()
            if W < 10 or H < 10:
                # 아직 배치 안 됨 → 나중에 재호출
                self._img_outer.after(100, lambda: self._apply_tab_layout("compare"))
                return

            HDR_H  = 24
            DIV_W  = 3
            half_w = (W - DIV_W) // 2

            # 헤더 바
            self._bar_left.place(x=0, y=0, width=half_w, height=HDR_H)
            self._bar_right.place(x=half_w + DIV_W, y=0, width=W - half_w - DIV_W, height=HDR_H)
            # 분할선
            self._divider.place(x=half_w, y=0, width=DIV_W, height=H)
            # 캔버스
            self.zoom_left.place(x=0, y=HDR_H, width=half_w, height=H - HDR_H)
            self.zoom_right.place(x=half_w + DIV_W, y=HDR_H,
                                   width=W - half_w - DIV_W, height=H - HDR_H)

            # 이미지 업데이트
            if self._cur_orig is not None:
                self.zoom_left.load_image(cv2pil(self._cur_orig))
            else:
                self.zoom_left.clear()
            if self._cur_result is not None:
                self.zoom_right.load_image(cv2pil(self._cur_result))
            else:
                self.zoom_right.clear()

        else:
            # 단일 탭: 전체 크기로 단일 캔버스
            self.zoom_single.place(x=0, y=0, relwidth=1.0, relheight=1.0)

            if key == "orig":
                self.zoom_single._label_text  = "📷 원본 (Original)"
                self.zoom_single._label_color = ACC2
                if self._cur_orig is not None:
                    self.zoom_single.load_image(cv2pil(self._cur_orig))
                else:
                    self.zoom_single._hint = "파일을 클릭하면 원본이 표시됩니다"
                    self.zoom_single.clear()
            else:  # result
                self.zoom_single._label_text  = "✨ 복원 (Restored)"
                self.zoom_single._label_color = GREEN
                if self._cur_result is not None:
                    self.zoom_single.load_image(cv2pil(self._cur_result))
                else:
                    self.zoom_single._hint = "미리보기 후 복원 결과가 표시됩니다"
                    self.zoom_single.clear()

    def _on_img_outer_resize(self, event):
        """창 크기 변경 시 compare 탭 재배치"""
        if self._active_tab == "compare":
            self._apply_tab_layout("compare")

    # ── 오른쪽 패널

    def _build_right_panel(self, parent):
        frame = tk.Frame(parent, bg=DARK, width=300)
        frame.grid(row=0, column=2, sticky="nsew", padx=(4, 0))
        frame.pack_propagate(False)

        # ── 단일 미리보기 결과
        self._sec_plain(frame, "선택 이미지 처리 결과")
        info_f = tk.Frame(frame, bg=CARD, pady=6)
        info_f.pack(fill="x", padx=8, pady=4)

        self._prev_vars = {}
        for key, label, init in [
            ("shadow_pct", "그림자 비율",    "-"),
            ("detect_ms",  "탐지 시간",      "-"),
            ("restore_ms", "복원 시간",      "-"),
            ("total_ms",   "전체 처리 시간", "-"),
        ]:
            row = tk.Frame(info_f, bg=CARD)
            row.pack(fill="x", padx=8, pady=2)
            tk.Label(row, text=label, bg=CARD, fg=TEXT2,
                      font=("Segoe UI", 9)).pack(side="left")
            v = tk.StringVar(value=init)
            tk.Label(row, textvariable=v, bg=CARD, fg=ACC2,
                      font=("Segoe UI", 9, "bold")).pack(side="right")
            self._prev_vars[key] = v

        self._prev_status = tk.Label(
            info_f,
            text="파일을 더블클릭하면\n즉시 그림자 제거+복원 처리됩니다",
            bg=CARD, fg=TEXT2, font=("Segoe UI", 8),
            wraplength=260, justify="left")
        self._prev_status.pack(fill="x", padx=8, pady=(4, 0))

        # ── 배치 처리 진행
        self._sec_plain(frame, "전체 배치 처리")
        pg_f = tk.Frame(frame, bg=CARD, pady=8)
        pg_f.pack(fill="x", padx=8, pady=4)
        tk.Label(pg_f, text="전체 진행", bg=CARD, fg=TEXT2,
                  font=("Segoe UI", 9)).pack(anchor="w", padx=8)
        self.prog_bar = ttk.Progressbar(pg_f, mode="determinate", length=270)
        self.prog_bar.pack(padx=8, pady=4)
        self.prog_lbl = tk.Label(pg_f, text="0 / 0", bg=CARD, fg=TEXT2,
                                  font=("Segoe UI", 9))
        self.prog_lbl.pack(anchor="e", padx=8)

        cur_f = tk.Frame(frame, bg=CARD, pady=6)
        cur_f.pack(fill="x", padx=8, pady=4)
        tk.Label(cur_f, text="현재 처리 중", bg=CARD, fg=TEXT2,
                  font=("Segoe UI", 9)).pack(anchor="w", padx=8)
        self.cur_lbl = tk.Label(cur_f, text="-", bg=CARD, fg=WARN,
                                 font=("Consolas", 8), wraplength=260, anchor="w")
        self.cur_lbl.pack(fill="x", padx=8)

        # ── 통계
        self._sec_plain(frame, "처리 통계")
        stats_f = tk.Frame(frame, bg=CARD)
        stats_f.pack(fill="x", padx=8, pady=4)
        self._stat_vars = {}
        for key, label, init in [
            ("total_files", "총 파일 수",    "0"),
            ("processed",   "처리 완료",     "0"),
            ("failed",      "실패",          "0"),
            ("shadow_avg",  "평균 그림자",    "0.0 %"),
            ("speed_avg",   "평균 처리 속도", "0 ms/장"),
            ("elapsed",     "경과 시간",      "00:00"),
        ]:
            row = tk.Frame(stats_f, bg=CARD)
            row.pack(fill="x", padx=8, pady=2)
            tk.Label(row, text=label, bg=CARD, fg=TEXT2,
                      font=("Segoe UI", 9)).pack(side="left")
            v = tk.StringVar(value=init)
            tk.Label(row, textvariable=v, bg=CARD, fg=ACC2,
                      font=("Segoe UI", 9, "bold")).pack(side="right")
            self._stat_vars[key] = v

        # ── 로그
        self._sec_plain(frame, "처리 로그")
        log_f = tk.Frame(frame, bg=CARD)
        log_f.pack(fill="both", expand=True, padx=8, pady=4)
        log_sb = ttk.Scrollbar(log_f, orient="vertical")
        self.log_text = tk.Text(log_f, bg=CARD, fg=TEXT, font=("Consolas", 8),
                                 relief="flat", wrap="word", state="disabled",
                                 yscrollcommand=log_sb.set)
        log_sb.config(command=self.log_text.yview)
        log_sb.pack(side="right", fill="y")
        self.log_text.pack(fill="both", expand=True)
        for t, c in [("info", TEXT2), ("ok", GREEN), ("warn", WARN), ("error", RED)]:
            self.log_text.tag_config(t, foreground=c)

    def _sec_plain(self, parent, title):
        f = tk.Frame(parent, bg=DARK)
        f.pack(fill="x", padx=8, pady=(8, 2))
        tk.Label(f, text=title, bg=DARK, fg=ACC2,
                  font=("Segoe UI", 10, "bold")).pack(side="left")
        ttk.Separator(f, orient="horizontal").pack(
            side="left", fill="x", expand=True, padx=6)

    # ── 푸터

    def _build_footer(self):
        ft = tk.Frame(self, bg=DARK3, height=26)
        ft.pack(fill="x", side="bottom"); ft.pack_propagate(False)
        self.status_lbl = tk.Label(ft, text="준비", bg=DARK3, fg=TEXT2,
                                    font=("Segoe UI", 9), padx=10)
        self.status_lbl.pack(side="left", pady=3)
        tk.Label(ft, text="v3.0  |  더블클릭: 즉시 처리  |  슬라이더 조정 → 미리보기 버튼으로 재처리",
                  bg=DARK3, fg=TEXT2, font=("Segoe UI", 8), padx=10
                  ).pack(side="right", pady=3)
        # 창 크기 변경 → compare 탭 재배치
        self.bind("<Configure>", self._on_window_resize)

    def _on_window_resize(self, event):
        if event.widget is self and self._active_tab == "compare":
            self.after(50, lambda: self._apply_tab_layout("compare"))

    # ──────────────────────────────────────────────────────
    # 이벤트 핸들러

    def _open_model_download(self):
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        ModelDownloadDialog(self, model_dir=model_dir,
                             reload_callback=self._reload_models).focus()

    def _reload_models(self):
        self._set_status("모델 재로드 중...", WARN)
        self.model_badge.config(text="재로딩...", fg=WARN)
        self._load_models_async()
        self._log("모델 재로드 요청됨", "warn")

    def _browse_input(self):
        init = self.input_folder.get().strip() or os.path.expanduser("~")
        if not os.path.isdir(init): init = os.path.expanduser("~")
        d = filedialog.askdirectory(title="드론 사진 폴더 선택", initialdir=init)
        if d:
            d = os.path.normpath(d)
            self.input_folder.set(d)
            if not self.output_folder.get():
                self.output_folder.set(os.path.join(d, "output_restored"))
            self._log(f"입력 폴더: {d}", "info")

    def _browse_output(self):
        init = self.output_folder.get().strip() or \
               self.input_folder.get().strip() or os.path.expanduser("~")
        if not os.path.isdir(init): init = os.path.expanduser("~")
        d = filedialog.askdirectory(title="저장 폴더 선택", initialdir=init)
        if d:
            self.output_folder.set(os.path.normpath(d))
            self._log(f"출력 폴더: {d}", "info")

    def _scan_files(self):
        folder = self.input_folder.get().strip()
        if not folder or not os.path.isdir(folder):
            messagebox.showwarning("경고", "유효한 입력 폴더를 선택하세요."); return

        files = scan_folder_recursive(folder) if self.recursive_var.get() \
                else scan_folder(folder)
        self._file_list = files

        self.file_listbox.delete(0, "end")
        for f in files:
            self.file_listbox.insert("end", os.path.basename(f))

        self._stat_vars["total_files"].set(str(len(files)))
        self._cnt_lbl.config(text=f"({len(files)}개)")
        self._set_status(f"{len(files)}개 파일 발견", GREEN if files else WARN)
        self._log(f"스캔: {folder} → {len(files)}개 발견", "info")
        if files:
            self._log("파일 더블클릭 → 즉시처리  |  미리보기 확인 후 [전체 처리]", "warn")

    def _on_file_select(self, event):
        """단일 클릭: 원본 이미지 표시 (처리 없음)"""
        sel = self.file_listbox.curselection()
        if not sel: return
        idx = sel[0]
        if idx >= len(self._file_list): return
        path = self._file_list[idx]

        self._selected_idx  = idx
        self._selected_path = path

        img = safe_imread(path)
        if img is None:
            self._log(f"❌ 이미지 로드 실패: {os.path.basename(path)}", "error")
            self._set_status(f"로드 실패: {os.path.basename(path)}", RED)
            return

        self._cur_orig   = img
        self._cur_result = None
        self._cur_binary = None
        self._cur_soft   = None

        # 원본 탭으로 전환
        self._switch_tab("orig")
        h, w = img.shape[:2]
        self._set_status(
            f"선택: {os.path.basename(path)}  ({w}×{h})  |  더블클릭=즉시처리", ACC2)
        self._prev_status.config(
            text=f"선택: {os.path.basename(path)}\n크기: {w}×{h}\n\n"
                 "더블클릭 또는 [미리보기] 버튼으로 처리", fg=WARN)

    def _on_file_double(self, event):
        """더블클릭: 즉시 미리보기 처리"""
        # 더블클릭 시 ListboxSelect도 먼저 발생하므로 _selected_path가 이미 설정됨
        if self._is_previewing:
            self._log("⚠ 처리 중입니다. 완료 후 다시 시도하세요.", "warn")
            return
        if not self._selected_path:
            return
        # 이미지가 로드 안 됐을 경우 다시 로드
        if self._cur_orig is None:
            img = safe_imread(self._selected_path)
            if img is None:
                self._log(f"❌ 이미지 로드 실패: {os.path.basename(self._selected_path)}", "error")
                return
            self._cur_orig = img
        self._preview_selected()

    # ── 단일 미리보기 처리

    def _preview_selected(self):
        """선택된 파일 1장 미리보기"""
        if self._is_previewing:
            self._log("⚠ 이미 처리 중입니다.", "warn"); return
        if not self._selected_path or not os.path.isfile(self._selected_path):
            messagebox.showwarning("경고",
                "파일 목록에서 이미지를 선택하거나 더블클릭하세요."); return

        self._is_previewing = True
        self.btn_preview.set_enabled(False)
        fname = os.path.basename(self._selected_path)
        self._prev_status.config(
            text=f"처리 중: {fname}\n미리보기 모드 (빠른 처리)...", fg=WARN)
        self._set_status(f"처리 중: {fname}", WARN)

        params = self._collect_params()
        path   = self._selected_path

        def _work():
            try:
                img = safe_imread(path)
                if img is None:
                    self._queue.put(("preview_error",
                                     f"이미지 로드 실패\n경로: {path}"))
                    return
                result, binary, soft, stats = process_single(
                    img, **params, preview_mode=True)
                self._queue.put(("preview_done",
                                  img, result, binary, soft, stats,
                                  os.path.basename(path)))
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                self._queue.put(("preview_error",
                                  f"{type(e).__name__}: {e}\n{tb[:600]}"))
            finally:
                # finally 보장: 어떤 경우에도 플래그 해제
                # (queue 처리에서도 해제하지만 예외 시 안전망)
                pass

        threading.Thread(target=_work, daemon=True).start()

    # ── 전체 배치 처리

    def _start_processing(self):
        if self._is_running: return
        if not self._file_list:
            self._scan_files()
            if not self._file_list:
                messagebox.showwarning("경고", "처리할 파일이 없습니다."); return

        out_dir = self.output_folder.get().strip()
        if not out_dir:
            messagebox.showwarning("경고", "출력 폴더를 지정하세요."); return

        if not messagebox.askyesno("전체 처리 확인",
                f"총 {len(self._file_list)}개 이미지를 처리합니다.\n"
                f"출력: {out_dir}\n\n현재 파라미터로 시작하시겠습니까?"):
            return

        os.makedirs(out_dir, exist_ok=True)
        self._is_running = True
        self._cancel_flag.clear()
        self.btn_start.set_enabled(False)
        self.btn_cancel.set_enabled(True)
        self.btn_preview.set_enabled(False)
        for k in ("processed", "failed"):
            self._stat_vars[k].set("0")
        self._stat_vars["shadow_avg"].set("0.0 %")
        self._stat_vars["speed_avg"].set("0 ms/장")
        self._stat_vars["elapsed"].set("00:00")
        self.prog_bar["value"] = 0

        params = self._collect_params()
        threading.Thread(target=self._run_batch,
                          args=(self._file_list, out_dir, params),
                          daemon=True).start()

    def _cancel_processing(self):
        self._cancel_flag.set()
        self._set_status("중단 요청 중...", WARN)

    def _collect_params(self):
        return {
            "detection_mode":         self.detect_mode.get(),
            "sensitivity":            float(self.sl_sensitivity.get()),
            "feather":                int(self.sl_feather.get()),
            # v6.0 색상 복원 파라미터
            "color_restore_strength": float(self.sl_radio.get()),
            "use_hue_consistent":     bool(self.use_hue_consist.get()),
            "highlight_protect":      float(self.sl_highlight.get()),
            "use_ai_color":           bool(self.use_ai_color.get()),
            # 품질 개선
            "denoise_h":              int(self.sl_denoise.get()),
            "sharpen_amount":         float(self.sl_sharpen.get()),
            "clahe_clip":             float(self.sl_clahe.get()),
            # 하위 호환성
            "shadow_amount":          float(self.sl_radio.get()),
            "highlight_amount":       float(self.sl_highlight.get()),
            "radio_strength":         float(self.sl_radio.get()),
            "retinex_strength":       0.0,
            "shadow_lift":            0.20,
            "blur_radius":            60,
        }

    # ── 배치 처리 스레드

    def _run_batch(self, files, out_dir, params):
        total       = len(files)
        done        = 0
        failed      = 0
        shadow_list = []
        speed_list  = []
        t_start     = time.time()
        n_workers   = self.worker_var.get()

        self._queue.put(("progress", 0, total, "", ""))

        def _one(path):
            if self._cancel_flag.is_set(): return None
            fname = os.path.basename(path)
            stem  = Path(path).stem
            ext   = Path(path).suffix
            out_p = os.path.join(out_dir, f"{stem}_restored{ext}")
            if os.path.exists(out_p) and not self.overwrite_var.get():
                return {"status": "skip", "fname": fname}

            img = safe_imread(path)
            if img is None:
                return {"status": "fail", "fname": fname, "msg": "이미지 로드 실패"}
            try:
                result, binary, soft, stats = process_single(
                    img, **params, preview_mode=False)
                cv2.imwrite(out_p, result)
                if self.save_compare.get():
                    cmp = make_compare(img, result, binary, soft)
                    cv2.imwrite(
                        os.path.join(out_dir, f"{stem}_compare.jpg"),
                        cmp, [cv2.IMWRITE_JPEG_QUALITY, 90])
                del img; gc.collect()
                return {"status": "ok", "fname": fname,
                        "stats": stats, "result": result}
            except Exception as e:
                return {"status": "fail", "fname": fname, "msg": str(e)}

        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futs = {ex.submit(_one, f): f for f in files}
            for fut in as_completed(futs):
                if self._cancel_flag.is_set(): break
                r = fut.result()
                if r is None: continue
                done += 1
                elapsed = time.time() - t_start
                if r["status"] == "ok":
                    st = r["stats"]
                    shadow_list.append(st["shadow_pct"])
                    speed_list.append(st["total_ms"])
                    self._queue.put(("progress", done, total, r["fname"],
                        f"완료 {r['fname']} | {st['total_ms']}ms | 그림자 {st['shadow_pct']}%"))
                    self._queue.put(("stats_update", done, failed,
                        sum(shadow_list) / len(shadow_list),
                        sum(speed_list)  / len(speed_list), elapsed))
                    self._queue.put(("batch_preview", None, r.get("result")))
                elif r["status"] == "skip":
                    self._queue.put(("progress", done, total, r["fname"],
                        f"건너뜀 {r['fname']}"))
                else:
                    failed += 1
                    self._queue.put(("progress", done, total, r["fname"],
                        f"실패 {r['fname']}: {r.get('msg', '')}"))
                    self._queue.put(("stats_update", done, failed,
                        sum(shadow_list) / max(len(shadow_list), 1),
                        sum(speed_list)  / max(len(speed_list),  1), elapsed))

        elapsed_final = time.time() - t_start
        self._queue.put(("done", done, failed, elapsed_final,
                          self._cancel_flag.is_set()))

    # ── 큐 폴링 (메인 스레드)

    def _poll_queue(self):
        try:
            while True:
                msg  = self._queue.get_nowait()
                kind = msg[0]

                if kind == "model_loaded":
                    st = msg[1]
                    parts = []
                    parts.append("탐지AI " + ("OK" if st.get("shadow") else "CV모드"))
                    parts.append("복원AI " + ("OK" if st.get("color")  else "CV모드"))
                    txt = "  |  ".join(parts)
                    self.model_badge.config(text=txt, fg=GREEN)
                    # 모델 로드 성공 시 자동으로 AI 모드 활성화
                    if st.get("shadow"):
                        self.detect_mode.set("hybrid")
                    if st.get("color"):
                        self.use_ai_color.set(True)
                    self._set_status("준비 완료 — 폴더 선택 후 파일 더블클릭으로 즉시 처리", GREEN)
                    self._log(f"모델 로드: {txt}", "ok")

                elif kind == "preview_done":
                    _, orig, result, binary, soft, stats, fname = msg
                    # 플래그 해제 (연속 처리 가능)
                    self._is_previewing = False
                    self.btn_preview.set_enabled(True)

                    self._cur_orig   = orig
                    self._cur_result = result
                    self._cur_binary = binary
                    self._cur_soft   = soft

                    # 비교 탭으로 전환 후 이미지 표시
                    self._switch_tab("compare")

                    # 통계 업데이트
                    self._prev_vars["shadow_pct"].set(f"{stats.get('shadow_pct', 0):.1f} %")
                    self._prev_vars["detect_ms"].set(f"{stats.get('detect_ms', 0):.0f} ms")
                    self._prev_vars["restore_ms"].set(f"{stats.get('restore_ms', 0):.0f} ms")
                    self._prev_vars["total_ms"].set(f"{stats.get('total_ms', 0):.0f} ms")
                    self._prev_status.config(
                        text=f"✅ 완료: {fname}\n"
                             f"그림자: {stats.get('shadow_pct', 0):.1f}%  "
                             f"처리: {stats.get('total_ms', 0):.0f}ms\n\n"
                             "결과 확인 후 슬라이더 조정→재처리\n또는 [전체 처리] 버튼 클릭",
                        fg=GREEN)
                    self._set_status(
                        f"미리보기 완료: {fname}  "
                        f"| 그림자 {stats.get('shadow_pct', 0):.1f}%  "
                        f"| {stats.get('total_ms', 0):.0f}ms",
                        GREEN)
                    self._log(
                        f"✅ 미리보기: {fname} | 그림자 {stats.get('shadow_pct', 0):.1f}% | "
                        f"탐지 {stats.get('detect_ms', 0):.0f}ms | "
                        f"복원 {stats.get('restore_ms', 0):.0f}ms | "
                        f"전체 {stats.get('total_ms', 0):.0f}ms", "ok")

                elif kind == "preview_error":
                    _, err = msg
                    self._is_previewing = False
                    self.btn_preview.set_enabled(True)
                    self._prev_status.config(
                        text=f"❌ 오류: {err[:120]}", fg=RED)
                    self._set_status("미리보기 실패", RED)
                    self._log(f"❌ 미리보기 오류:\n{err[:400]}", "error")

                elif kind == "progress":
                    _, done, total, fname, log_msg = msg
                    pct = int(done / total * 100) if total else 0
                    self.prog_bar["value"] = pct
                    self.prog_lbl.config(text=f"{done} / {total}  ({pct}%)")
                    self.cur_lbl.config(text=fname or "-")
                    if log_msg:
                        tag = "ok"   if "완료" in log_msg else \
                              "warn" if "건너뜀" in log_msg else "error"
                        self._log(log_msg, tag)

                elif kind == "stats_update":
                    _, done, failed, avg_s, avg_sp, elapsed = msg
                    self._stat_vars["processed"].set(str(done))
                    self._stat_vars["failed"].set(str(failed))
                    self._stat_vars["shadow_avg"].set(f"{avg_s:.1f} %")
                    self._stat_vars["speed_avg"].set(f"{avg_sp:.0f} ms/장")
                    m, s = divmod(int(elapsed), 60)
                    self._stat_vars["elapsed"].set(f"{m:02d}:{s:02d}")

                elif kind == "batch_preview":
                    _, orig, result = msg
                    if orig   is not None: self._cur_orig   = orig
                    if result is not None: self._cur_result = result
                    # 현재 탭에 맞게 업데이트
                    if self._active_tab == "compare":
                        if result is not None:
                            self.zoom_right.load_image(cv2pil(result))
                    elif self._active_tab == "result":
                        if result is not None:
                            self.zoom_single.load_image(cv2pil(result))

                elif kind == "done":
                    _, done, failed, elapsed, cancelled = msg
                    self._is_running = False
                    self.btn_start.set_enabled(True)
                    self.btn_cancel.set_enabled(False)
                    self.btn_preview.set_enabled(True)
                    m, s = divmod(int(elapsed), 60)
                    if cancelled:
                        self._set_status(f"중단됨  |  완료 {done}장", WARN)
                        self._log(f"중단됨: {done}장 처리  ({m:02d}:{s:02d})", "warn")
                    else:
                        self._set_status(
                            f"완료!  {done}장 처리  |  실패 {failed}장  |  {m:02d}:{s:02d}",
                            GREEN)
                        self._log(
                            f"✅ 완료! 총 {done}장 | 실패 {failed}장 | {m:02d}:{s:02d}",
                            "ok")
                    self.cur_lbl.config(text="-")

        except queue.Empty:
            pass
        self.after(80, self._poll_queue)

    # ── 헬퍼

    def _set_status(self, text, color=TEXT2):
        self.status_lbl.config(text=text, fg=color)

    def _log(self, text, tag="info"):
        self.log_text.config(state="normal")
        self.log_text.insert("end", f"[{time.strftime('%H:%M:%S')}] {text}\n", tag)
        self.log_text.see("end")
        self.log_text.config(state="disabled")


# ──────────────────────────────────────────────────────────
# 엔트리포인트
# ──────────────────────────────────────────────────────────

def main():
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

    app = DroneApp()

    style = ttk.Style(app)
    try:
        style.theme_use("clam")
    except Exception:
        pass
    style.configure("TProgressbar",
                     troughcolor=DARK2, background=ACCENT, thickness=14)
    style.configure("Vertical.TScrollbar",
                     background=DARK3, troughcolor=DARK2, arrowcolor=TEXT2)
    style.configure("Horizontal.TScrollbar",
                     background=DARK3, troughcolor=DARK2, arrowcolor=TEXT2)

    app.mainloop()


if __name__ == "__main__":
    main()
