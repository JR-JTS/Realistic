"""
Drone Shadow Remover & Color Restorer
Desktop Application  –  Tkinter GUI  v2.0
────────────────────────────────────────
실행: python main.py

새 기능 (v2.0):
  • 단일 이미지 사전 처리 & 미리보기 (파일 선택 → "이 이미지 처리" 버튼)
  • 원본/복원 이미지 줌(마우스 휠), 패닝(드래그)
  • 마음에 들면 "전체 배치 처리 시작" 버튼으로 일괄 처리
  • 처리 결과 마음에 안 들면 파라미터 조정 후 재처리 가능
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import queue
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

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
# 색상 테마
# ──────────────────────────────────────────────────────────
DARK  = "#1e1e2e"
DARK2 = "#2a2a3e"
DARK3 = "#313155"
ACCENT= "#7c6af7"
ACC2  = "#00c9ff"
GREEN = "#3ddc84"
RED   = "#ff5f57"
WARN  = "#ffcb6b"
TEXT  = "#e0e0f0"
TEXT2 = "#9090b0"
WHITE = "#ffffff"
CARD  = "#252538"

# ──────────────────────────────────────────────────────────
# 유틸
# ──────────────────────────────────────────────────────────

def cv2pil(img_bgr: np.ndarray, max_w: int = 0, max_h: int = 0) -> Image.Image:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    if max_w and max_h:
        pil.thumbnail((max_w, max_h), Image.LANCZOS)
    return pil

def pil2tk(pil_img: Image.Image) -> ImageTk.PhotoImage:
    return ImageTk.PhotoImage(pil_img)


# ──────────────────────────────────────────────────────────
# 커스텀 위젯: 플랫 버튼 (tk.Label 기반)
# ──────────────────────────────────────────────────────────

class FlatButton(tk.Label):
    def __init__(self, parent, text="", command=None,
                 bg=ACCENT, fg=WHITE, hover=DARK3,
                 width=160, height=36, radius=8,
                 font_size=11, **kw):
        kw.pop("radius", None)
        super().__init__(
            parent,
            text=text,
            bg=bg,
            fg=fg,
            font=("Segoe UI", font_size, "bold"),
            cursor="hand2",
            relief="flat",
            padx=max(4, (width - len(text) * (font_size + 2)) // 2),
            pady=max(2, (height - font_size - 6) // 2),
            **kw
        )
        self._bg      = bg
        self._hover   = hover
        self._fg      = fg
        self._enabled = True
        self._cmd     = command

        self.bind("<Enter>",         self._on_enter)
        self.bind("<Leave>",         self._on_leave)
        self.bind("<ButtonPress-1>", self._on_press)

    def _on_enter(self, _):
        if self._enabled:
            self.config(bg=self._hover)

    def _on_leave(self, _):
        if self._enabled:
            self.config(bg=self._bg)

    def _on_press(self, _):
        if self._enabled and self._cmd:
            self._cmd()

    def set_enabled(self, v: bool):
        self._enabled = v
        if v:
            self.config(bg=self._bg, fg=self._fg, cursor="hand2")
        else:
            self.config(bg=DARK3, fg=TEXT2, cursor="")


# ──────────────────────────────────────────────────────────
# 라벨 슬라이더 컴포넌트
# ──────────────────────────────────────────────────────────

class LabeledSlider(tk.Frame):
    def __init__(self, parent, label, from_, to, init,
                 resolution=0.05, fmt=".2f", **kw):
        super().__init__(parent, bg=CARD, **kw)
        self._fmt = fmt
        self.var  = tk.DoubleVar(value=init)

        top = tk.Frame(self, bg=CARD)
        top.pack(fill="x")
        tk.Label(top, text=label, bg=CARD, fg=TEXT,
                  font=("Segoe UI", 9)).pack(side="left")
        self.val_lbl = tk.Label(top, text=format(init, fmt),
                                 bg=CARD, fg=ACC2,
                                 font=("Segoe UI", 9, "bold"),
                                 width=6, anchor="e")
        self.val_lbl.pack(side="right")

        self.slider = ttk.Scale(self, from_=from_, to=to,
                                 variable=self.var, orient="horizontal",
                                 command=self._update)
        self.slider.pack(fill="x", pady=(2, 0))

    def _update(self, v):
        self.val_lbl.config(text=format(float(v), self._fmt))

    def get(self): return self.var.get()


# ──────────────────────────────────────────────────────────
# 줌 가능 캔버스 위젯
# ──────────────────────────────────────────────────────────

class ZoomCanvas(tk.Canvas):
    """
    마우스 휠 줌 + 드래그 패닝을 지원하는 이미지 캔버스.
    load_image(pil_img) 로 이미지를 로드.
    """
    MIN_SCALE = 0.1
    MAX_SCALE = 10.0

    def __init__(self, parent, bg=CARD, **kw):
        super().__init__(parent, bg=bg, highlightthickness=0, **kw)
        self._pil_orig  = None    # 원본 PIL 이미지
        self._scale     = 1.0
        self._offset_x  = 0
        self._offset_y  = 0
        self._drag_x    = 0
        self._drag_y    = 0
        self._tk_img    = None

        # 이벤트 바인딩
        self.bind("<Configure>",       self._on_configure)
        self.bind("<MouseWheel>",      self._on_wheel)
        self.bind("<Button-4>",        self._on_wheel)   # Linux 위로
        self.bind("<Button-5>",        self._on_wheel)   # Linux 아래로
        self.bind("<ButtonPress-1>",   self._on_drag_start)
        self.bind("<B1-Motion>",       self._on_drag_move)
        self.bind("<Double-Button-1>", self._reset_view)
        self.bind("<ButtonPress-3>",   self._reset_view)  # 우클릭: 초기화

    def load_image(self, pil_img: Image.Image):
        """PIL 이미지를 로드하고 캔버스에 맞게 초기 배치"""
        self._pil_orig = pil_img
        self._fit_to_canvas()

    def clear(self):
        self._pil_orig = None
        self.delete("all")
        self._tk_img = None

    def _fit_to_canvas(self):
        """이미지를 캔버스 크기에 맞게 자동 스케일"""
        if self._pil_orig is None:
            return
        cw = self.winfo_width()
        ch = self.winfo_height()
        if cw < 10 or ch < 10:
            self.after(100, self._fit_to_canvas)
            return

        iw, ih = self._pil_orig.size
        scale = min(cw / iw, ch / ih, 1.0)
        self._scale    = scale
        self._offset_x = (cw - iw * scale) / 2
        self._offset_y = (ch - ih * scale) / 2
        self._render()

    def _render(self):
        if self._pil_orig is None:
            return
        cw = self.winfo_width()
        ch = self.winfo_height()
        if cw < 10 or ch < 10:
            return

        iw, ih = self._pil_orig.size
        nw = max(1, int(iw * self._scale))
        nh = max(1, int(ih * self._scale))

        # 리사이즈 (캔버스 크기를 넘지 않는 범위에서 LANCZOS)
        resized = self._pil_orig.resize((nw, nh), Image.LANCZOS)
        tk_img  = ImageTk.PhotoImage(resized)

        self.delete("all")
        self.create_image(
            int(self._offset_x), int(self._offset_y),
            anchor="nw", image=tk_img, tags="img"
        )
        self._tk_img = tk_img  # 참조 유지 (GC 방지)

        # 줌 레벨 표시
        self.create_text(
            cw - 4, ch - 4,
            text=f"×{self._scale:.2f}  [더블클릭: 원래 크기]",
            anchor="se", fill=TEXT2,
            font=("Consolas", 8), tags="zoom_lbl"
        )

    def _on_configure(self, event):
        self._render()

    def _on_wheel(self, event):
        if self._pil_orig is None:
            return
        # 마우스 포인터 위치
        mx = self.canvasx(event.x)
        my = self.canvasy(event.y)

        # 스케일 변화율
        if event.num == 4:
            factor = 1.15
        elif event.num == 5:
            factor = 1 / 1.15
        else:
            factor = 1.15 if event.delta > 0 else 1 / 1.15

        new_scale = max(self.MIN_SCALE,
                        min(self.MAX_SCALE, self._scale * factor))
        ratio = new_scale / self._scale

        # 마우스 위치 기준으로 오프셋 조정 (줌 중심)
        self._offset_x = mx - ratio * (mx - self._offset_x)
        self._offset_y = my - ratio * (my - self._offset_y)
        self._scale    = new_scale
        self._render()

    def _on_drag_start(self, event):
        self._drag_x = event.x
        self._drag_y = event.y

    def _on_drag_move(self, event):
        if self._pil_orig is None:
            return
        dx = event.x - self._drag_x
        dy = event.y - self._drag_y
        self._offset_x += dx
        self._offset_y += dy
        self._drag_x    = event.x
        self._drag_y    = event.y
        self._render()

    def _reset_view(self, event=None):
        """더블클릭 또는 우클릭으로 초기 뷰 복원"""
        self._fit_to_canvas()


# ──────────────────────────────────────────────────────────
# 메인 앱 윈도우
# ──────────────────────────────────────────────────────────

class DroneApp(tk.Tk):

    # ── 초기화 ────────────────────────────────────────────

    def __init__(self):
        super().__init__()
        self.title("🛸  Drone Shadow Remover & Color Restorer  v2.0")
        self.geometry("1480x900")
        self.minsize(1200, 720)
        self.configure(bg=DARK)
        self._set_icon()

        # 상태 변수
        self.input_folder   = tk.StringVar(value="")
        self.output_folder  = tk.StringVar(value="")
        self.recursive_var  = tk.BooleanVar(value=False)
        self.save_compare   = tk.BooleanVar(value=True)
        self.overwrite_var  = tk.BooleanVar(value=False)
        self.worker_var     = tk.IntVar(value=2)

        self._file_list         = []
        self._selected_index    = -1
        self._selected_path     = ""
        self._is_running        = False
        self._is_previewing     = False
        self._cancel_flag       = threading.Event()
        self._queue             = queue.Queue()

        # 현재 미리보기용 BGR 이미지
        self._cur_orig   = None   # 원본 BGR
        self._cur_result = None   # 복원 BGR
        self._cur_binary = None   # 이진 마스크
        self._cur_soft   = None   # 소프트 마스크

        # 모델 상태
        self._model_status = {}

        self._build_ui()
        self._load_models_async()
        self._poll_queue()

    def _set_icon(self):
        try:
            self.iconbitmap("")
        except Exception:
            pass

    # ── 모델 비동기 로드 ──────────────────────────────────

    def _load_models_async(self):
        def _load():
            base   = os.path.dirname(__file__)
            status = load_models(os.path.join(base, "models"))
            self._model_status = status
            self._queue.put(("model_loaded", status))
        threading.Thread(target=_load, daemon=True).start()
        self._set_status("모델 로딩 중...", WARN)

    # ──────────────────────────────────────────────────────
    # UI 빌드
    # ──────────────────────────────────────────────────────

    def _build_ui(self):
        self._build_header()
        body = tk.Frame(self, bg=DARK)
        body.pack(fill="both", expand=True, padx=12, pady=(0, 10))
        body.columnconfigure(0, weight=0, minsize=290)
        body.columnconfigure(1, weight=1)
        body.columnconfigure(2, weight=0, minsize=320)
        body.rowconfigure(0, weight=1)

        self._build_left_panel(body)
        self._build_center_panel(body)
        self._build_right_panel(body)
        self._build_footer()

    # ── 헤더 ─────────────────────────────────────────────

    def _build_header(self):
        hdr = tk.Frame(self, bg=DARK3, height=64)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        left = tk.Frame(hdr, bg=DARK3)
        left.pack(side="left", padx=20)
        tk.Label(left, text="🛸", bg=DARK3, fg=WHITE,
                  font=("Segoe UI Emoji", 22)).pack(side="left", padx=(0,10))
        title_f = tk.Frame(left, bg=DARK3)
        title_f.pack(side="left")
        tk.Label(title_f, text="Drone Shadow Remover & Color Restorer",
                  bg=DARK3, fg=WHITE,
                  font=("Segoe UI", 14, "bold")).pack(anchor="w")
        tk.Label(title_f, text="v2.0  —  단일 미리보기 + 줌/패닝 + 배치 처리",
                  bg=DARK3, fg=TEXT2,
                  font=("Segoe UI", 9)).pack(anchor="w")

        self.model_badge = tk.Label(hdr, text="⏳ 모델 로딩 중",
                                     bg=DARK3, fg=WARN,
                                     font=("Segoe UI", 9))
        self.model_badge.pack(side="right", padx=20)

    # ── 왼쪽 패널 : 폴더 선택 + 설정 ─────────────────────

    def _build_left_panel(self, parent):
        frame = tk.Frame(parent, bg=DARK, width=290)
        frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        frame.pack_propagate(False)

        canvas = tk.Canvas(frame, bg=DARK, highlightthickness=0)
        sb     = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        inner = tk.Frame(canvas, bg=DARK)
        win_id = canvas.create_window((0,0), window=inner, anchor="nw")

        def _resize(e):
            canvas.itemconfig(win_id, width=e.width)
        canvas.bind("<Configure>", _resize)
        inner.bind("<Configure>", lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")))

        def _wheel(e):
            canvas.yview_scroll(-1*(e.delta//120 if e.delta else (-1 if e.num==5 else 1)), "units")
        canvas.bind_all("<MouseWheel>", _wheel)
        canvas.bind_all("<Button-4>",  _wheel)
        canvas.bind_all("<Button-5>",  _wheel)

        p = inner

        # ── 폴더 선택 ─────────────────
        self._section(p, "📁  폴더 설정")
        self._build_folder_section(p)

        # 처리 쓰레드 수
        tw = tk.Frame(p, bg=DARK)
        tw.pack(fill="x", padx=10, pady=2)
        tk.Label(tw, text="동시 처리 수", bg=DARK, fg=TEXT,
                  font=("Segoe UI", 9)).pack(side="left")
        for v in (1,2,4):
            tk.Radiobutton(tw, text=str(v), variable=self.worker_var,
                            value=v, bg=DARK, fg=TEXT2,
                            selectcolor=DARK3, activebackground=DARK,
                            font=("Segoe UI",9)).pack(side="left", padx=4)

        # ── 탐지 설정 ─────────────────
        self._section(p, "🔍  그림자 탐지")

        self.detect_mode = tk.StringVar(value="hybrid")
        dm_f = tk.Frame(p, bg=DARK)
        dm_f.pack(fill="x", padx=10, pady=4)
        for txt, val in [("AI + CV 복합", "hybrid"), ("CV 방법만", "cv")]:
            tk.Radiobutton(dm_f, text=txt, variable=self.detect_mode, value=val,
                            bg=DARK, fg=TEXT, selectcolor=DARK3,
                            activebackground=DARK,
                            font=("Segoe UI",9)).pack(side="left", padx=6)

        self.sl_sensitivity = self._slider(p, "탐지 민감도", 0.1, 1.0, 0.5)
        self.sl_feather     = self._slider(p, "마스크 페더링", 5, 60, 25, fmt=".0f")

        # ── 색상 복원 설정 ─────────────
        self._section(p, "🎨  색상 복원")

        self.use_ai_color = tk.BooleanVar(value=True)
        tk.Checkbutton(p, text="AI 색상 보정 사용", variable=self.use_ai_color,
                        bg=DARK, fg=TEXT, selectcolor=DARK3,
                        activebackground=DARK,
                        font=("Segoe UI",9)).pack(anchor="w", padx=12)

        self.sl_radio    = self._slider(p, "조명 보정 강도",   0.0, 1.0, 0.80)
        self.sl_color    = self._slider(p, "색 전달 강도",     0.0, 1.0, 0.65)
        self.sl_retinex  = self._slider(p, "Retinex 강도",    0.0, 0.6, 0.30)

        # ── 선명화 설정 ──────────────
        self._section(p, "✨  선명화 (복원 후)")

        self.sl_denoise  = self._slider(p, "노이즈 제거",      1, 15,  6, fmt=".0f")
        self.sl_sharpen  = self._slider(p, "선명도",          0.5, 3.0, 1.4)
        self.sl_clahe    = self._slider(p, "CLAHE 대비",      1.0, 5.0, 2.0)

        # ── 실행 버튼 ────────────────
        self._section(p, "")

        # 1) 선택한 이미지 1장 미리 처리
        self.btn_preview = FlatButton(p, "🔬  선택 이미지 미리보기",
                                       command=self._preview_selected,
                                       bg="#1a6b3a", hover="#0f4a28",
                                       width=260, height=36, font_size=10)
        self.btn_preview.pack(pady=(4,2), padx=10)

        # 도움말
        tk.Label(p, text="↑ 파일 목록에서 이미지를 선택 후 클릭",
                  bg=DARK, fg=TEXT2, font=("Segoe UI",7)).pack(padx=12, anchor="w")

        ttk.Separator(p, orient="horizontal").pack(fill="x", padx=10, pady=6)

        btn_f = tk.Frame(p, bg=DARK)
        btn_f.pack(fill="x", padx=10, pady=4)

        self.btn_scan = FlatButton(btn_f, "📂  파일 스캔",
                                    command=self._scan_files,
                                    bg="#2a6496", hover="#1d4f75",
                                    width=120, height=34, font_size=10)
        self.btn_scan.pack(side="left", padx=(0,6))

        # 2) 전체 배치 처리
        self.btn_start = FlatButton(btn_f, "▶  전체 처리",
                                     command=self._start_processing,
                                     bg=ACCENT, hover="#5b4dd6",
                                     width=120, height=34, font_size=10)
        self.btn_start.pack(side="left")

        self.btn_cancel = FlatButton(p, "⏹  중단",
                                      command=self._cancel_processing,
                                      bg="#8b3a3a", hover="#6b2222",
                                      width=260, height=34, font_size=10)
        self.btn_cancel.pack(pady=(4,8), padx=10)
        self.btn_cancel.set_enabled(False)

    def _section(self, parent, title):
        f = tk.Frame(parent, bg=DARK)
        f.pack(fill="x", padx=8, pady=(10,2))
        if title:
            tk.Label(f, text=title, bg=DARK, fg=ACC2,
                      font=("Segoe UI", 10, "bold")).pack(side="left")
            ttk.Separator(f, orient="horizontal").pack(
                side="left", fill="x", expand=True, padx=6)

    def _build_folder_section(self, parent):
        """폴더 설정 UI"""

        # ── 입력 폴더 ──────────────────────────────────────
        tk.Label(parent, text="📂  드론 사진 폴더", bg=DARK, fg=TEXT,
                  font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=12, pady=(6,2))

        in_card = tk.Frame(parent, bg=DARK2, padx=8, pady=8)
        in_card.pack(fill="x", padx=10, pady=(0,4))

        self._in_path_lbl = tk.Label(
            in_card, textvariable=self.input_folder,
            bg=DARK2, fg=ACC2,
            font=("Consolas", 8), wraplength=230, justify="left", anchor="w"
        )
        self._in_path_lbl.pack(fill="x")
        self.input_folder.trace_add("write", lambda *_: self._update_path_label(
            self._in_path_lbl, self.input_folder, "← 아래 버튼으로 폴더를 선택하세요"))
        self._update_path_label(self._in_path_lbl, self.input_folder, "← 아래 버튼으로 폴더를 선택하세요")

        btn_row_in = tk.Frame(in_card, bg=DARK2)
        btn_row_in.pack(fill="x", pady=(6,0))
        FlatButton(btn_row_in, "📂  폴더 선택",
                    command=self._browse_input,
                    bg=ACCENT, hover="#5b4dd6",
                    width=140, height=30, font_size=9
        ).pack(side="left", padx=(0,6))
        FlatButton(btn_row_in, "✏ 직접 입력",
                    command=lambda: self._toggle_direct_input("input"),
                    bg=DARK3, hover=DARK2, fg=TEXT2,
                    width=90, height=30, font_size=9
        ).pack(side="left")

        self._in_entry_frame = tk.Frame(in_card, bg=DARK2)
        tk.Entry(self._in_entry_frame,
                  textvariable=self.input_folder,
                  bg="#1a1a30", fg=ACC2,
                  insertbackground=WHITE, relief="flat",
                  font=("Consolas", 9)
        ).pack(fill="x", pady=(4,0), ipady=4)
        tk.Label(self._in_entry_frame,
                  text="예) C:\\Users\\이름\\드론사진  또는  /home/user/photos",
                  bg=DARK2, fg=TEXT2, font=("Segoe UI",7)).pack(anchor="w")

        # ── 출력 폴더 ──────────────────────────────────────
        tk.Label(parent, text="💾  저장 폴더", bg=DARK, fg=TEXT,
                  font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=12, pady=(8,2))

        out_card = tk.Frame(parent, bg=DARK2, padx=8, pady=8)
        out_card.pack(fill="x", padx=10, pady=(0,4))

        self._out_path_lbl = tk.Label(
            out_card, textvariable=self.output_folder,
            bg=DARK2, fg=GREEN,
            font=("Consolas", 8), wraplength=230, justify="left", anchor="w"
        )
        self._out_path_lbl.pack(fill="x")
        self.output_folder.trace_add("write", lambda *_: self._update_path_label(
            self._out_path_lbl, self.output_folder, "← 입력 폴더 선택 시 자동 설정"))
        self._update_path_label(self._out_path_lbl, self.output_folder, "← 입력 폴더 선택 시 자동 설정")

        btn_row_out = tk.Frame(out_card, bg=DARK2)
        btn_row_out.pack(fill="x", pady=(6,0))
        FlatButton(btn_row_out, "💾  폴더 선택",
                    command=self._browse_output,
                    bg="#2a6496", hover="#1d4f75",
                    width=140, height=30, font_size=9
        ).pack(side="left", padx=(0,6))
        FlatButton(btn_row_out, "✏ 직접 입력",
                    command=lambda: self._toggle_direct_input("output"),
                    bg=DARK3, hover=DARK2, fg=TEXT2,
                    width=90, height=30, font_size=9
        ).pack(side="left")

        self._out_entry_frame = tk.Frame(out_card, bg=DARK2)
        tk.Entry(self._out_entry_frame,
                  textvariable=self.output_folder,
                  bg="#1a1a30", fg=GREEN,
                  insertbackground=WHITE, relief="flat",
                  font=("Consolas", 9)
        ).pack(fill="x", pady=(4,0), ipady=4)
        tk.Label(self._out_entry_frame,
                  text="예) C:\\Users\\이름\\출력폴더  (없으면 자동 생성)",
                  bg=DARK2, fg=TEXT2, font=("Segoe UI",7)).pack(anchor="w")

        # ── 옵션 ───────────────────────────────────────────
        opts = tk.Frame(parent, bg=DARK)
        opts.pack(fill="x", padx=10, pady=(6,4))
        self._check(opts, "하위 폴더 포함 (재귀)",  self.recursive_var)
        self._check(opts, "비교 이미지 저장",       self.save_compare)
        self._check(opts, "기존 파일 덮어쓰기",     self.overwrite_var)

    def _update_path_label(self, lbl, var, placeholder):
        v = var.get()
        if not v:
            lbl.config(text=placeholder, fg=TEXT2)

    def _toggle_direct_input(self, which):
        frame = self._in_entry_frame if which == "input" else self._out_entry_frame
        if frame.winfo_ismapped():
            frame.pack_forget()
        else:
            frame.pack(fill="x", pady=(4,0))

    def _slider(self, parent, label, lo, hi, init, fmt=".2f"):
        s = LabeledSlider(parent, label, lo, hi, init,
                           fmt=fmt, padx=10, pady=3)
        s.pack(fill="x", padx=10, pady=2)
        return s

    def _check(self, parent, text, var):
        tk.Checkbutton(parent, text=text, variable=var,
                        bg=DARK, fg=TEXT, selectcolor=DARK3,
                        activebackground=DARK,
                        font=("Segoe UI",9)).pack(anchor="w")

    # ── 가운데 패널 : 줌 가능 미리보기 ──────────────────

    def _build_center_panel(self, parent):
        frame = tk.Frame(parent, bg=DARK)
        frame.grid(row=0, column=1, sticky="nsew")
        frame.rowconfigure(1, weight=1)
        frame.columnconfigure(0, weight=1)

        # ── 탭 헤더 ────────────────────────────────────────
        tab_f = tk.Frame(frame, bg=DARK2, height=40)
        tab_f.grid(row=0, column=0, sticky="ew")
        tab_f.pack_propagate(False)

        self._tab_btns  = {}
        self._active_tab = tk.StringVar(value="compare")

        for txt, key in [("비교 보기", "compare"), ("원본", "orig"), ("복원", "result")]:
            b = tk.Label(tab_f, text=txt, bg=DARK2, fg=TEXT2,
                          font=("Segoe UI",10), padx=18, pady=8, cursor="hand2")
            b.pack(side="left")
            b.bind("<Button-1>", lambda e, k=key: self._switch_tab(k))
            self._tab_btns[key] = b

        # 줌 조작 힌트 (오른쪽)
        tk.Label(tab_f, text="🔍 휠: 줌  |  드래그: 이동  |  더블클릭/우클릭: 원래크기",
                  bg=DARK2, fg=TEXT2, font=("Segoe UI",8)).pack(side="right", padx=12)

        # ── 이미지 표시 영역 (탭 전환용) ──────────────────
        # "compare" 탭: 나란히(원본|복원) 단일 ZoomCanvas
        # "orig"    탭: ZoomCanvas (원본만)
        # "result"  탭: ZoomCanvas (복원만)

        self._img_container = tk.Frame(frame, bg=CARD)
        self._img_container.grid(row=1, column=0, sticky="nsew", pady=(0,4))
        self._img_container.rowconfigure(0, weight=1)
        self._img_container.columnconfigure(0, weight=1)
        self._img_container.columnconfigure(1, weight=1)

        # 단일 이미지용 ZoomCanvas (orig / result 탭)
        self.zoom_canvas = ZoomCanvas(self._img_container, bg=CARD)
        self.zoom_canvas.grid(row=0, column=0, columnspan=2, sticky="nsew")

        # 비교용: 왼쪽(원본) + 오른쪽(복원)
        self.zoom_left  = ZoomCanvas(self._img_container, bg="#1a1a2e")
        self.zoom_right = ZoomCanvas(self._img_container, bg="#1a2e1a")

        # 라벨 (비교 탭용)
        self._lbl_orig_bar   = tk.Label(self._img_container, text="  원본  ",
                                         bg="#1a1a2e", fg=ACC2,
                                         font=("Segoe UI",9,"bold"))
        self._lbl_result_bar = tk.Label(self._img_container, text="  복원  ",
                                         bg="#1a2e1a", fg=GREEN,
                                         font=("Segoe UI",9,"bold"))

        # 초기 안내 텍스트
        self.zoom_canvas.create_text(
            400, 250,
            text="📂  폴더 선택 → 파일 스캔 → 파일 클릭\n"
                 "→  🔬 선택 이미지 미리보기  버튼으로 결과 확인\n\n"
                 "결과가 마음에 들면  ▶ 전체 처리  버튼으로\n"
                 "전체 이미지를 일괄 처리합니다",
            fill=TEXT2, font=("Segoe UI", 12), justify="center",
            tags="hint"
        )

        # 탭 초기화 (preview_canvas → zoom_canvas 로 변경 후 호출)
        self._switch_tab("compare")

        # ── 파일 리스트 (하단) ────────────────────────────
        list_f = tk.Frame(frame, bg=DARK2, height=130)
        list_f.grid(row=2, column=0, sticky="ew", pady=(4,0))
        list_f.pack_propagate(False)

        list_hdr = tk.Frame(list_f, bg=DARK2)
        list_hdr.pack(fill="x", pady=(4,0), padx=4)
        tk.Label(list_hdr, text="📋  파일 목록", bg=DARK2, fg=TEXT2,
                  font=("Segoe UI",9,"bold")).pack(side="left", padx=4)
        self._file_count_lbl = tk.Label(list_hdr, text="(0개)",
                                         bg=DARK2, fg=TEXT2,
                                         font=("Segoe UI",8))
        self._file_count_lbl.pack(side="left")
        tk.Label(list_hdr,
                  text="← 파일 클릭 후 [선택 이미지 미리보기] 버튼을 누르세요",
                  bg=DARK2, fg=WARN, font=("Segoe UI",8)).pack(side="right", padx=8)

        list_scroll = ttk.Scrollbar(list_f, orient="vertical")
        self.file_listbox = tk.Listbox(
            list_f, bg=DARK2, fg=TEXT, selectbackground=ACCENT,
            relief="flat", borderwidth=0, font=("Consolas",9),
            yscrollcommand=list_scroll.set, activestyle="none",
        )
        list_scroll.config(command=self.file_listbox.yview)
        list_scroll.pack(side="right", fill="y")
        self.file_listbox.pack(fill="both", expand=True, padx=4)
        self.file_listbox.bind("<<ListboxSelect>>", self._on_file_select)

    def _switch_tab(self, key):
        self._active_tab.set(key)
        for k, btn in self._tab_btns.items():
            btn.config(bg=DARK3 if k == key else DARK2,
                        fg=WHITE if k == key else TEXT2)
        self._apply_tab_layout(key)

    def _apply_tab_layout(self, key):
        """탭 전환 시 캔버스 레이아웃 변경"""
        # 모든 위젯 숨기기
        self.zoom_canvas.grid_remove()
        self.zoom_left.grid_remove()
        self.zoom_right.grid_remove()
        self._lbl_orig_bar.grid_remove()
        self._lbl_result_bar.grid_remove()

        if key == "compare":
            # 라벨 + 좌우 캔버스
            self._lbl_orig_bar.grid(row=0, column=0, sticky="ew")
            self._lbl_result_bar.grid(row=0, column=1, sticky="ew")
            self.zoom_left.grid(row=1, column=0, sticky="nsew")
            self.zoom_right.grid(row=1, column=1, sticky="nsew")
            self._img_container.rowconfigure(0, weight=0)
            self._img_container.rowconfigure(1, weight=1)
            # 이미지 적용
            if self._cur_orig   is not None: self.zoom_left.load_image(cv2pil(self._cur_orig))
            if self._cur_result is not None: self.zoom_right.load_image(cv2pil(self._cur_result))
        else:
            self._img_container.rowconfigure(0, weight=1)
            self._img_container.rowconfigure(1, weight=0)
            self.zoom_canvas.grid(row=0, column=0, columnspan=2, sticky="nsew")
            if key == "orig"   and self._cur_orig   is not None:
                self.zoom_canvas.load_image(cv2pil(self._cur_orig))
            elif key == "result" and self._cur_result is not None:
                self.zoom_canvas.load_image(cv2pil(self._cur_result))

    # ── 오른쪽 패널 : 진행상황 + 통계 ───────────────────

    def _build_right_panel(self, parent):
        frame = tk.Frame(parent, bg=DARK, width=320)
        frame.grid(row=0, column=2, sticky="nsew", padx=(8,0))
        frame.pack_propagate(False)

        # ── 단일 처리 결과 정보 ───────────────────────────
        self._section_plain(frame, "🔬  선택 이미지 처리 결과")

        self._preview_info_frame = tk.Frame(frame, bg=CARD, pady=8)
        self._preview_info_frame.pack(fill="x", padx=8, pady=4)

        self._preview_stat_vars = {}
        preview_items = [
            ("shadow_pct",   "그림자 비율",    "-"),
            ("detect_ms",    "탐지 시간",      "-"),
            ("restore_ms",   "복원 시간",      "-"),
            ("total_ms",     "전체 처리 시간", "-"),
        ]
        for key, label, init in preview_items:
            row = tk.Frame(self._preview_info_frame, bg=CARD)
            row.pack(fill="x", padx=10, pady=2)
            tk.Label(row, text=label, bg=CARD, fg=TEXT2,
                      font=("Segoe UI",9)).pack(side="left")
            v = tk.StringVar(value=init)
            tk.Label(row, textvariable=v, bg=CARD, fg=ACC2,
                      font=("Segoe UI",9,"bold")).pack(side="right")
            self._preview_stat_vars[key] = v

        # 처리 상태
        self._preview_status_lbl = tk.Label(
            self._preview_info_frame,
            text="파일 선택 후 [선택 이미지 미리보기] 클릭",
            bg=CARD, fg=TEXT2, font=("Segoe UI",8),
            wraplength=280, justify="left"
        )
        self._preview_status_lbl.pack(fill="x", padx=10, pady=(4,0))

        # ── 배치 진행 상황 ───────────────────────────────
        self._section_plain(frame, "📊  전체 배치 처리 진행")

        pg_f = tk.Frame(frame, bg=CARD, pady=10)
        pg_f.pack(fill="x", padx=8, pady=4)
        tk.Label(pg_f, text="전체 진행", bg=CARD, fg=TEXT2,
                  font=("Segoe UI",9)).pack(anchor="w", padx=10)
        self.prog_bar = ttk.Progressbar(pg_f, mode="determinate", length=290)
        self.prog_bar.pack(padx=10, pady=4)
        self.prog_lbl = tk.Label(pg_f, text="0 / 0", bg=CARD, fg=TEXT2,
                                   font=("Segoe UI",9))
        self.prog_lbl.pack(anchor="e", padx=10)

        cur_f = tk.Frame(frame, bg=CARD, pady=8)
        cur_f.pack(fill="x", padx=8, pady=4)
        tk.Label(cur_f, text="현재 처리 중", bg=CARD, fg=TEXT2,
                  font=("Segoe UI",9)).pack(anchor="w", padx=10)
        self.cur_lbl = tk.Label(cur_f, text="-", bg=CARD, fg=WARN,
                                 font=("Consolas",8), wraplength=270, anchor="w")
        self.cur_lbl.pack(fill="x", padx=10)

        # 통계
        self._section_plain(frame, "📈  처리 통계")
        stats_f = tk.Frame(frame, bg=CARD)
        stats_f.pack(fill="x", padx=8, pady=4)

        self._stat_vars = {}
        stat_items = [
            ("total_files",    "총 파일 수",      "0"),
            ("processed",      "처리 완료",        "0"),
            ("failed",         "실패",             "0"),
            ("shadow_avg",     "평균 그림자 비율", "0.0 %"),
            ("speed_avg",      "평균 처리 속도",   "0 ms/장"),
            ("elapsed",        "경과 시간",        "00:00"),
        ]
        for key, label, init in stat_items:
            row = tk.Frame(stats_f, bg=CARD)
            row.pack(fill="x", padx=10, pady=3)
            tk.Label(row, text=label, bg=CARD, fg=TEXT2,
                      font=("Segoe UI",9)).pack(side="left")
            v = tk.StringVar(value=init)
            tk.Label(row, textvariable=v, bg=CARD, fg=ACC2,
                      font=("Segoe UI",9,"bold")).pack(side="right")
            self._stat_vars[key] = v

        # 로그
        self._section_plain(frame, "📋  처리 로그")
        log_f = tk.Frame(frame, bg=CARD)
        log_f.pack(fill="both", expand=True, padx=8, pady=4)

        log_sb = ttk.Scrollbar(log_f, orient="vertical")
        self.log_text = tk.Text(
            log_f, bg=CARD, fg=TEXT, font=("Consolas",8),
            relief="flat", wrap="word", state="disabled",
            yscrollcommand=log_sb.set,
        )
        log_sb.config(command=self.log_text.yview)
        log_sb.pack(side="right", fill="y")
        self.log_text.pack(fill="both", expand=True)

        self.log_text.tag_config("info",  foreground=TEXT2)
        self.log_text.tag_config("ok",    foreground=GREEN)
        self.log_text.tag_config("warn",  foreground=WARN)
        self.log_text.tag_config("error", foreground=RED)

    def _section_plain(self, parent, title):
        f = tk.Frame(parent, bg=DARK)
        f.pack(fill="x", padx=8, pady=(10,2))
        tk.Label(f, text=title, bg=DARK, fg=ACC2,
                  font=("Segoe UI",10,"bold")).pack(side="left")
        ttk.Separator(f, orient="horizontal").pack(
            side="left", fill="x", expand=True, padx=6)

    # ── 푸터 ───────────────────────────────────────────

    def _build_footer(self):
        ft = tk.Frame(self, bg=DARK3, height=28)
        ft.pack(fill="x", side="bottom")
        ft.pack_propagate(False)
        self.status_lbl = tk.Label(ft, text="준비", bg=DARK3, fg=TEXT2,
                                    font=("Segoe UI",9), padx=12)
        self.status_lbl.pack(side="left", pady=4)
        tk.Label(ft, text="Drone Shadow Remover v2.0  |  🔬 선택 미리보기 → ▶ 전체 처리",
                  bg=DARK3, fg=TEXT2, font=("Segoe UI",8), padx=12).pack(side="right", pady=4)

    # ──────────────────────────────────────────────────────
    # 이벤트 핸들러
    # ──────────────────────────────────────────────────────

    def _browse_input(self):
        init = self.input_folder.get().strip()
        if not init or not os.path.isdir(init):
            init = os.path.expanduser("~")
        d = filedialog.askdirectory(
            title="드론 사진이 있는 폴더를 선택하세요",
            initialdir=init,
        )
        if d:
            d = os.path.normpath(d)
            self.input_folder.set(d)
            if not self.output_folder.get():
                self.output_folder.set(os.path.join(d, "output_restored"))
            self._log(f"📂 입력 폴더: {d}", "info")

    def _browse_output(self):
        init = self.output_folder.get().strip()
        if not init or not os.path.isdir(init):
            init = self.input_folder.get().strip() or os.path.expanduser("~")
        d = filedialog.askdirectory(
            title="결과를 저장할 폴더를 선택하세요",
            initialdir=init,
        )
        if d:
            d = os.path.normpath(d)
            self.output_folder.set(d)
            self._log(f"💾 출력 폴더: {d}", "info")

    def _scan_files(self):
        folder = self.input_folder.get().strip()
        if not folder or not os.path.isdir(folder):
            messagebox.showwarning("경고", "유효한 입력 폴더를 선택하세요.")
            return
        if self.recursive_var.get():
            files = scan_folder_recursive(folder)
        else:
            files = scan_folder(folder)
        self._file_list = files

        self.file_listbox.delete(0, "end")
        for f in files:
            self.file_listbox.insert("end", os.path.basename(f))

        self._stat_vars["total_files"].set(str(len(files)))
        self._file_count_lbl.config(text=f"({len(files)}개)")
        self._set_status(f"{len(files)}개 파일 발견", GREEN if files else WARN)
        self._log(f"📂 {folder}\n   → {len(files)}개 이미지 발견", "info")

        if files:
            self._log("💡 파일 클릭 후 [🔬 선택 이미지 미리보기] 버튼을 눌러 결과를 확인하세요", "warn")

    def _on_file_select(self, event):
        """파일 목록에서 선택 시 원본만 로드"""
        sel = self.file_listbox.curselection()
        if not sel:
            return
        idx  = sel[0]
        path = self._file_list[idx]
        self._selected_index = idx
        self._selected_path  = path

        try:
            img = cv2.imread(path)
            if img is not None:
                self._cur_orig   = img
                self._cur_result = None
                self._cur_binary = None
                self._cur_soft   = None
                # 탭 갱신
                self._apply_tab_layout(self._active_tab.get())
                # 원본 탭으로 자동 전환
                self._switch_tab("orig")
                h, w = img.shape[:2]
                self._set_status(f"선택: {os.path.basename(path)}  ({w}×{h})", ACC2)
                self._preview_status_lbl.config(
                    text=f"선택: {os.path.basename(path)}\n[🔬 미리보기] 버튼으로 처리하세요",
                    fg=WARN
                )
        except Exception as e:
            self._log(f"이미지 로드 실패: {e}", "error")

    # ──────────────────────────────────────────────────────
    # 단일 이미지 미리보기 처리
    # ──────────────────────────────────────────────────────

    def _preview_selected(self):
        """선택한 이미지 1장을 처리하고 결과를 미리보기"""
        if self._is_previewing:
            self._log("이미 처리 중입니다...", "warn")
            return
        if not self._selected_path or not os.path.isfile(self._selected_path):
            messagebox.showwarning("경고",
                "파일 목록에서 이미지를 먼저 선택하세요.\n"
                "파일 스캔이 안 된 경우 [📂 파일 스캔] 버튼을 먼저 클릭하세요.")
            return

        self._is_previewing = True
        self.btn_preview.set_enabled(False)
        self._preview_status_lbl.config(text="⏳ 처리 중...", fg=WARN)
        self._set_status(f"처리 중: {os.path.basename(self._selected_path)}", WARN)

        params = self._collect_params()
        path   = self._selected_path

        def _work():
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                self._queue.put(("preview_error", "이미지를 읽을 수 없습니다."))
                return
            try:
                result, binary, soft, stats = process_single(
                    img,
                    detection_mode=params["detection_mode"],
                    sensitivity=params["sensitivity"],
                    feather=params["feather"],
                    radio_strength=params["radio_strength"],
                    color_strength=params["color_strength"],
                    retinex_strength=params["retinex_strength"],
                    use_ai_color=params["use_ai_color"],
                    denoise_h=params["denoise_h"],
                    sharpen_amount=params["sharpen_amount"],
                    clahe_clip=params["clahe_clip"],
                )
                self._queue.put(("preview_done", img, result, binary, soft, stats,
                                 os.path.basename(path)))
            except Exception as e:
                import traceback
                self._queue.put(("preview_error", f"{e}\n{traceback.format_exc()}"))

        threading.Thread(target=_work, daemon=True).start()

    # ──────────────────────────────────────────────────────
    # 전체 배치 처리 시작 / 취소
    # ──────────────────────────────────────────────────────

    def _start_processing(self):
        if self._is_running:
            return
        if not self._file_list:
            self._scan_files()
            if not self._file_list:
                messagebox.showwarning("경고", "처리할 파일이 없습니다.")
                return

        out_dir = self.output_folder.get().strip()
        if not out_dir:
            messagebox.showwarning("경고", "출력 폴더를 지정하세요.")
            return

        # 전체 처리 전 확인
        ans = messagebox.askyesno(
            "전체 처리 확인",
            f"총 {len(self._file_list)}개 이미지를 처리합니다.\n"
            f"출력 폴더: {out_dir}\n\n"
            "현재 설정된 파라미터로 전체 처리를 시작하시겠습니까?",
            default="yes"
        )
        if not ans:
            return

        os.makedirs(out_dir, exist_ok=True)

        self._is_running  = True
        self._cancel_flag.clear()
        self.btn_start.set_enabled(False)
        self.btn_cancel.set_enabled(True)
        self.btn_preview.set_enabled(False)

        # 통계 초기화
        self._stat_vars["processed"].set("0")
        self._stat_vars["failed"].set("0")
        self._stat_vars["shadow_avg"].set("0.0 %")
        self._stat_vars["speed_avg"].set("0 ms/장")
        self._stat_vars["elapsed"].set("00:00")
        self.prog_bar["value"] = 0

        params = self._collect_params()
        t = threading.Thread(target=self._run_batch,
                              args=(self._file_list, out_dir, params),
                              daemon=True)
        t.start()

    def _cancel_processing(self):
        self._cancel_flag.set()
        self._set_status("중단 요청 중...", WARN)

    def _collect_params(self):
        return {
            "detection_mode":  self.detect_mode.get(),
            "sensitivity":     float(self.sl_sensitivity.get()),
            "feather":         int(self.sl_feather.get()),
            "radio_strength":  float(self.sl_radio.get()),
            "color_strength":  float(self.sl_color.get()),
            "retinex_strength":float(self.sl_retinex.get()),
            "use_ai_color":    self.use_ai_color.get(),
            "denoise_h":       int(self.sl_denoise.get()),
            "sharpen_amount":  float(self.sl_sharpen.get()),
            "clahe_clip":      float(self.sl_clahe.get()),
        }

    # ──────────────────────────────────────────────────────
    # 배치 처리 스레드
    # ──────────────────────────────────────────────────────

    def _run_batch(self, files, out_dir, params):
        total      = len(files)
        done       = 0
        failed     = 0
        shadow_list = []
        speed_list  = []
        t_start    = time.time()
        n_workers  = self.worker_var.get()

        self._queue.put(("progress", 0, total, "", ""))

        def _process_one(path):
            if self._cancel_flag.is_set():
                return None

            fname = os.path.basename(path)
            stem  = Path(path).stem
            ext   = Path(path).suffix

            out_path = os.path.join(out_dir, f"{stem}_restored{ext}")
            if os.path.exists(out_path) and not self.overwrite_var.get():
                return {"status": "skip", "path": path, "fname": fname}

            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                return {"status": "fail", "path": path, "fname": fname,
                        "msg": "이미지 읽기 실패"}
            try:
                result, binary, soft, stats = process_single(
                    img,
                    detection_mode=params["detection_mode"],
                    sensitivity=params["sensitivity"],
                    feather=params["feather"],
                    radio_strength=params["radio_strength"],
                    color_strength=params["color_strength"],
                    retinex_strength=params["retinex_strength"],
                    use_ai_color=params["use_ai_color"],
                    denoise_h=params["denoise_h"],
                    sharpen_amount=params["sharpen_amount"],
                    clahe_clip=params["clahe_clip"],
                )
                cv2.imwrite(out_path, result)

                if self.save_compare.get():
                    cmp_path = os.path.join(out_dir, f"{stem}_compare.jpg")
                    cmp = make_compare(img, result, binary, soft)
                    cv2.imwrite(cmp_path, cmp,
                                [cv2.IMWRITE_JPEG_QUALITY, 90])

                return {
                    "status":   "ok",
                    "path":     path,
                    "fname":    fname,
                    "out_path": out_path,
                    "stats":    stats,
                    "orig":     img,
                    "result":   result,
                }
            except Exception as e:
                import traceback
                return {"status": "fail", "path": path, "fname": fname,
                        "msg": str(e), "tb": traceback.format_exc()}

        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futures = {ex.submit(_process_one, f): f for f in files}
            for fut in as_completed(futures):
                if self._cancel_flag.is_set():
                    break

                r = fut.result()
                if r is None:
                    continue

                done    += 1
                elapsed  = time.time() - t_start

                if r["status"] == "ok":
                    stats = r["stats"]
                    shadow_list.append(stats["shadow_pct"])
                    speed_list.append(stats["total_ms"])
                    avg_shadow = sum(shadow_list) / len(shadow_list)
                    avg_speed  = sum(speed_list)  / len(speed_list)
                    self._queue.put((
                        "progress", done, total,
                        r["fname"],
                        f"✅ {r['fname']}  |  {stats['total_ms']}ms  |  그림자 {stats['shadow_pct']}%"
                    ))
                    self._queue.put(("stats_update",
                                      done, failed, avg_shadow, avg_speed, elapsed))
                    self._queue.put(("batch_preview", r["orig"], r["result"]))

                elif r["status"] == "skip":
                    self._queue.put(("progress", done, total, r["fname"],
                                      f"⏭️ {r['fname']}  (이미 처리됨, 건너뜀)"))
                else:
                    failed += 1
                    self._queue.put(("progress", done, total, r["fname"],
                                      f"❌ {r['fname']}  오류: {r.get('msg','')}"))
                    self._queue.put(("stats_update",
                                      done, failed,
                                      sum(shadow_list)/max(len(shadow_list),1),
                                      sum(speed_list)/max(len(speed_list),1),
                                      elapsed))

        elapsed_final = time.time() - t_start
        cancelled = self._cancel_flag.is_set()
        self._queue.put(("done", done, failed, elapsed_final, cancelled))

    # ──────────────────────────────────────────────────────
    # 큐 폴링
    # ──────────────────────────────────────────────────────

    def _poll_queue(self):
        try:
            while True:
                msg  = self._queue.get_nowait()
                kind = msg[0]

                if kind == "model_loaded":
                    status = msg[1]
                    parts  = []
                    if status.get("shadow"): parts.append("탐지AI ✅")
                    else:                    parts.append("탐지AI ❌(CV)")
                    if status.get("color"):  parts.append("색상AI ✅")
                    else:                    parts.append("색상AI ❌(CV)")
                    txt = "  |  ".join(parts)
                    self.model_badge.config(text=txt, fg=GREEN)
                    self._set_status("준비 완료 — 파일 스캔 후 이미지를 선택하세요", GREEN)
                    self._log(f"모델 로드 완료: {txt}", "ok")

                # ── 단일 미리보기 결과 ─────────────────────
                elif kind == "preview_done":
                    _, orig, result, binary, soft, stats, fname = msg
                    self._is_previewing = False
                    self.btn_preview.set_enabled(True)

                    self._cur_orig   = orig
                    self._cur_result = result
                    self._cur_binary = binary
                    self._cur_soft   = soft

                    # 비교 탭으로 전환
                    self._switch_tab("compare")

                    # 통계 업데이트
                    self._preview_stat_vars["shadow_pct"].set(
                        f"{stats.get('shadow_pct', 0):.1f} %")
                    self._preview_stat_vars["detect_ms"].set(
                        f"{stats.get('detection_ms', 0):.0f} ms")
                    self._preview_stat_vars["restore_ms"].set(
                        f"{stats.get('restoration_ms', 0):.0f} ms")
                    self._preview_stat_vars["total_ms"].set(
                        f"{stats.get('total_ms', 0):.0f} ms")
                    self._preview_status_lbl.config(
                        text=f"✅ 처리 완료: {fname}\n"
                             "결과가 마음에 들면 [▶ 전체 처리] 버튼을 클릭하세요",
                        fg=GREEN
                    )
                    self._set_status(
                        f"미리보기 완료: {fname}  |  그림자 {stats.get('shadow_pct',0):.1f}%  |  {stats.get('total_ms',0):.0f}ms",
                        GREEN
                    )
                    self._log(
                        f"🔬 미리보기 완료: {fname} | "
                        f"그림자 {stats.get('shadow_pct',0):.1f}% | "
                        f"{stats.get('total_ms',0):.0f}ms",
                        "ok"
                    )

                elif kind == "preview_error":
                    _, err = msg
                    self._is_previewing = False
                    self.btn_preview.set_enabled(True)
                    self._preview_status_lbl.config(
                        text=f"❌ 오류: {err[:80]}", fg=RED)
                    self._set_status("미리보기 실패", RED)
                    self._log(f"❌ 미리보기 오류: {err}", "error")

                # ── 배치 처리 ──────────────────────────────
                elif kind == "progress":
                    _, done, total, fname, log_msg = msg
                    pct = int(done / total * 100) if total else 0
                    self.prog_bar["value"] = pct
                    self.prog_lbl.config(text=f"{done} / {total}  ({pct}%)")
                    self.cur_lbl.config(text=fname or "-")
                    if log_msg:
                        tag = "ok" if log_msg.startswith("✅") else \
                              "warn" if log_msg.startswith("⏭") else "error"
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
                    self._cur_orig   = orig
                    self._cur_result = result
                    self._apply_tab_layout(self._active_tab.get())

                elif kind == "done":
                    _, done, failed, elapsed, cancelled = msg
                    self._is_running = False
                    self.btn_start.set_enabled(True)
                    self.btn_cancel.set_enabled(False)
                    self.btn_preview.set_enabled(True)
                    m, s = divmod(int(elapsed), 60)
                    if cancelled:
                        self._set_status(f"처리 중단  |  완료 {done}장", WARN)
                        self._log(f"⏹  중단됨  {done}장 처리  ({m:02d}:{s:02d})", "warn")
                    else:
                        self._set_status(
                            f"🎉 완료!  {done}장 처리  |  실패 {failed}장  |  {m:02d}:{s:02d}",
                            GREEN)
                        self._log(
                            f"🎉 완료! 총 {done}장  |  실패 {failed}장  |  {m:02d}:{s:02d}",
                            "ok")
                    self.cur_lbl.config(text="-")

        except queue.Empty:
            pass
        self.after(80, self._poll_queue)

    # ──────────────────────────────────────────────────────
    # 헬퍼
    # ──────────────────────────────────────────────────────

    def _set_status(self, text, color=TEXT2):
        self.status_lbl.config(text=text, fg=color)

    def _log(self, text, tag="info"):
        self.log_text.config(state="normal")
        ts = time.strftime("%H:%M:%S")
        self.log_text.insert("end", f"[{ts}] {text}\n", tag)
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
                     troughcolor=DARK2, background=ACCENT,
                     thickness=14)
    style.configure("Vertical.TScrollbar",
                     background=DARK3, troughcolor=DARK2,
                     arrowcolor=TEXT2)
    style.configure("Horizontal.TScrollbar",
                     background=DARK3, troughcolor=DARK2,
                     arrowcolor=TEXT2)

    app.mainloop()


if __name__ == "__main__":
    main()
