"""
Drone Shadow Remover & Color Restorer
Desktop Application  –  Tkinter GUI
────────────────────────────────────
실행: python main.py
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
# 커스텀 위젯: 플랫 버튼 (tk.Label 기반 – Canvas 버그 없음)
# ──────────────────────────────────────────────────────────

class FlatButton(tk.Label):
    """
    Canvas 없이 tk.Label로 구현한 플랫 버튼.
    Windows/macOS/Linux 모든 환경에서 안정적으로 동작.
    """
    def __init__(self, parent, text="", command=None,
                 bg=ACCENT, fg=WHITE, hover=DARK3,
                 width=160, height=36, radius=8,
                 font_size=11, **kw):
        # width/height/radius는 내부 저장용 (Label은 px 단위 크기 직접 지정)
        # kw에서 tk.Label이 모르는 인자 제거
        kw.pop("radius", None)

        # 픽셀 단위 크기 지정을 위해 font 기반으로 설정
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
# 메인 앱 윈도우
# ──────────────────────────────────────────────────────────

class DroneApp(tk.Tk):

    # ── 초기화 ────────────────────────────────────────────

    def __init__(self):
        super().__init__()
        self.title("🛸  Drone Shadow Remover & Color Restorer")
        self.geometry("1380x860")
        self.minsize(1100, 700)
        self.configure(bg=DARK)
        self._set_icon()

        # 상태 변수
        self.input_folder   = tk.StringVar(value="")
        self.output_folder  = tk.StringVar(value="")
        self.recursive_var  = tk.BooleanVar(value=False)
        self.save_compare   = tk.BooleanVar(value=True)
        self.overwrite_var  = tk.BooleanVar(value=False)
        self.worker_var     = tk.IntVar(value=2)

        self._file_list    = []
        self._is_running   = False
        self._cancel_flag  = threading.Event()
        self._queue        = queue.Queue()
        self._preview_orig = None
        self._preview_res  = None

        # 모델 상태 (비동기 로드)
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
            base = os.path.dirname(__file__)
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
        # 메인 3열 레이아웃
        body = tk.Frame(self, bg=DARK)
        body.pack(fill="both", expand=True, padx=12, pady=(0, 10))
        body.columnconfigure(0, weight=0, minsize=280)
        body.columnconfigure(1, weight=1)
        body.columnconfigure(2, weight=0, minsize=310)
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

        # 드론 아이콘 + 제목
        left = tk.Frame(hdr, bg=DARK3)
        left.pack(side="left", padx=20)
        tk.Label(left, text="🛸", bg=DARK3, fg=WHITE,
                  font=("Segoe UI Emoji", 22)).pack(side="left", padx=(0,10))
        title_f = tk.Frame(left, bg=DARK3)
        title_f.pack(side="left")
        tk.Label(title_f, text="Drone Shadow Remover",
                  bg=DARK3, fg=WHITE,
                  font=("Segoe UI", 15, "bold")).pack(anchor="w")
        tk.Label(title_f, text="& Color Restorer  —  AI + Computer Vision",
                  bg=DARK3, fg=TEXT2,
                  font=("Segoe UI", 9)).pack(anchor="w")

        # 모델 상태 배지
        self.model_badge = tk.Label(hdr, text="⏳ 모델 로딩 중",
                                     bg=DARK3, fg=WARN,
                                     font=("Segoe UI", 9))
        self.model_badge.pack(side="right", padx=20)

    # ── 왼쪽 패널 : 폴더 선택 + 설정 ─────────────────────

    def _build_left_panel(self, parent):
        frame = tk.Frame(parent, bg=DARK, width=280)
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

        # 마우스 휠
        def _wheel(e):
            canvas.yview_scroll(-1*(e.delta//120 if e.delta else (-1 if e.num==5 else 1)), "units")
        canvas.bind_all("<MouseWheel>", _wheel)
        canvas.bind_all("<Button-4>",  _wheel)
        canvas.bind_all("<Button-5>",  _wheel)

        p = inner  # 내용 컨테이너

        # ── 폴더 선택 ─────────────────
        self._section(p, "📁  폴더 설정")

        self._folder_row(p, "입력 폴더",  self.input_folder,  self._browse_input)
        self._folder_row(p, "출력 폴더",  self.output_folder, self._browse_output)

        opts = tk.Frame(p, bg=DARK)
        opts.pack(fill="x", padx=10, pady=4)
        self._check(opts, "하위 폴더 포함 (재귀)",  self.recursive_var)
        self._check(opts, "비교 이미지 저장",       self.save_compare)
        self._check(opts, "기존 파일 덮어쓰기",     self.overwrite_var)

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

        btn_f = tk.Frame(p, bg=DARK)
        btn_f.pack(fill="x", padx=10, pady=6)

        self.btn_scan = FlatButton(btn_f, "📂  파일 스캔",
                                    command=self._scan_files,
                                    bg="#2a6496", hover="#1d4f75",
                                    width=120, height=34, font_size=10)
        self.btn_scan.pack(side="left", padx=(0,6))

        self.btn_start = FlatButton(btn_f, "▶  처리 시작",
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

    def _folder_row(self, parent, label, var, cmd):
        f = tk.Frame(parent, bg=DARK)
        f.pack(fill="x", padx=10, pady=3)
        tk.Label(f, text=label, bg=DARK, fg=TEXT2,
                  font=("Segoe UI",9), width=8, anchor="w").pack(side="left")
        tk.Entry(f, textvariable=var, bg=DARK2, fg=TEXT,
                  insertbackground=WHITE, relief="flat",
                  font=("Segoe UI",9)).pack(side="left", fill="x", expand=True, padx=4)
        FlatButton(f, "찾기", command=cmd,
                    bg=DARK3, hover=DARK2, fg=TEXT2,
                    width=50, height=26, font_size=9).pack(side="right", padx=(4,0))

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

    # ── 가운데 패널 : 미리보기 ────────────────────────────

    def _build_center_panel(self, parent):
        frame = tk.Frame(parent, bg=DARK)
        frame.grid(row=0, column=1, sticky="nsew")
        frame.rowconfigure(1, weight=1)
        frame.columnconfigure(0, weight=1)

        # 탭 헤더
        tab_f = tk.Frame(frame, bg=DARK2, height=36)
        tab_f.grid(row=0, column=0, sticky="ew")
        tab_f.pack_propagate(False)

        self._tab_btns = {}
        self._active_tab = tk.StringVar(value="compare")
        for txt, key in [("비교 보기", "compare"), ("원본", "orig"), ("복원", "result")]:
            b = tk.Label(tab_f, text=txt, bg=DARK2, fg=TEXT2,
                          font=("Segoe UI",10), padx=16, pady=8, cursor="hand2")
            b.pack(side="left")
            b.bind("<Button-1>", lambda e, k=key: self._switch_tab(k))
            self._tab_btns[key] = b
        self._switch_tab("compare")

        # 이미지 표시 영역
        img_wrap = tk.Frame(frame, bg=CARD)
        img_wrap.grid(row=1, column=0, sticky="nsew", pady=4)
        img_wrap.rowconfigure(0, weight=1)
        img_wrap.columnconfigure(0, weight=1)

        self.preview_canvas = tk.Canvas(img_wrap, bg=CARD,
                                         highlightthickness=0)
        self.preview_canvas.grid(row=0, column=0, sticky="nsew")
        self.preview_canvas.bind("<Configure>", self._redraw_preview)

        # 기본 안내 텍스트
        self.preview_canvas.create_text(
            400, 300,
            text="📂  폴더를 선택하고 파일을 스캔하세요\n\n처리 후 여기에 결과 미리보기가 표시됩니다",
            fill=TEXT2, font=("Segoe UI", 13), justify="center",
            tags="hint"
        )

        # 파일 리스트 (하단)
        list_f = tk.Frame(frame, bg=DARK2, height=120)
        list_f.grid(row=2, column=0, sticky="ew", pady=(4,0))
        list_f.pack_propagate(False)

        tk.Label(list_f, text="파일 목록", bg=DARK2, fg=TEXT2,
                  font=("Segoe UI",9,"bold"), padx=8).pack(anchor="nw", pady=4)

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
        self._redraw_preview()

    def _redraw_preview(self, event=None):
        w = self.preview_canvas.winfo_width()
        h = self.preview_canvas.winfo_height()
        if w < 10 or h < 10:
            return

        tab = self._active_tab.get()
        img_bgr = None

        if tab == "orig"    and self._preview_orig   is not None: img_bgr = self._preview_orig
        if tab == "result"  and self._preview_res    is not None: img_bgr = self._preview_res
        if tab == "compare" and self._preview_orig is not None and self._preview_res is not None:
            # Side by side (원본 | 복원)
            img_bgr = cv2.hconcat([self._preview_orig, self._preview_res])

        if img_bgr is None:
            return

        pil = cv2pil(img_bgr, w, h)
        tk_img = pil2tk(pil)
        self.preview_canvas.delete("hint")
        self.preview_canvas.delete("preview")
        x = (w - pil.width)  // 2
        y = (h - pil.height) // 2
        self.preview_canvas.create_image(x, y, anchor="nw",
                                          image=tk_img, tags="preview")
        self.preview_canvas._tk_img = tk_img  # 참조 유지

    # ── 오른쪽 패널 : 진행상황 + 통계 ───────────────────

    def _build_right_panel(self, parent):
        frame = tk.Frame(parent, bg=DARK, width=310)
        frame.grid(row=0, column=2, sticky="nsew", padx=(8,0))
        frame.pack_propagate(False)

        # 진행 상황
        self._section_plain(frame, "📊  처리 진행 상황")

        # 전체 프로그레스바
        pg_f = tk.Frame(frame, bg=CARD, pady=10)
        pg_f.pack(fill="x", padx=8, pady=4)
        tk.Label(pg_f, text="전체 진행", bg=CARD, fg=TEXT2,
                  font=("Segoe UI",9)).pack(anchor="w", padx=10)
        self.prog_bar = ttk.Progressbar(pg_f, mode="determinate", length=280)
        self.prog_bar.pack(padx=10, pady=4)
        self.prog_lbl = tk.Label(pg_f, text="0 / 0", bg=CARD, fg=TEXT2,
                                   font=("Segoe UI",9))
        self.prog_lbl.pack(anchor="e", padx=10)

        # 현재 파일
        cur_f = tk.Frame(frame, bg=CARD, pady=8)
        cur_f.pack(fill="x", padx=8, pady=4)
        tk.Label(cur_f, text="현재 처리 중", bg=CARD, fg=TEXT2,
                  font=("Segoe UI",9)).pack(anchor="w", padx=10)
        self.cur_lbl = tk.Label(cur_f, text="-", bg=CARD, fg=WARN,
                                 font=("Consolas",8), wraplength=270, anchor="w")
        self.cur_lbl.pack(fill="x", padx=10)

        # 통계 카드
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

        # 태그 색상
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

    # ── 푸터 : 상태 바 ───────────────────────────────────

    def _build_footer(self):
        ft = tk.Frame(self, bg=DARK3, height=28)
        ft.pack(fill="x", side="bottom")
        ft.pack_propagate(False)
        self.status_lbl = tk.Label(ft, text="준비", bg=DARK3, fg=TEXT2,
                                    font=("Segoe UI",9), padx=12)
        self.status_lbl.pack(side="left", pady=4)
        tk.Label(ft, text="Drone Shadow Remover v1.0",
                  bg=DARK3, fg=TEXT2, font=("Segoe UI",8), padx=12).pack(side="right", pady=4)

    # ──────────────────────────────────────────────────────
    # 이벤트 핸들러
    # ──────────────────────────────────────────────────────

    def _browse_input(self):
        d = filedialog.askdirectory(title="입력 폴더 선택")
        if d:
            self.input_folder.set(d)
            # 출력 폴더 자동 설정
            if not self.output_folder.get():
                self.output_folder.set(os.path.join(d, "output"))

    def _browse_output(self):
        d = filedialog.askdirectory(title="출력 폴더 선택")
        if d:
            self.output_folder.set(d)

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
        self._set_status(f"{len(files)}개 파일 발견", GREEN if files else WARN)
        self._log(f"📂 {folder}\n   → {len(files)}개 이미지 발견", "info")

    def _on_file_select(self, event):
        sel = self.file_listbox.curselection()
        if not sel:
            return
        idx  = sel[0]
        path = self._file_list[idx]
        # 원본 미리보기
        try:
            img = cv2.imread(path)
            if img is not None:
                self._preview_orig = img
                self._preview_res  = None
                self._redraw_preview()
        except Exception:
            pass

    # ──────────────────────────────────────────────────────
    # 처리 시작 / 취소
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

        os.makedirs(out_dir, exist_ok=True)

        self._is_running  = True
        self._cancel_flag.clear()
        self.btn_start.set_enabled(False)
        self.btn_cancel.set_enabled(True)

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
            "detection_mode": self.detect_mode.get(),
            "sensitivity":    float(self.sl_sensitivity.get()),
            "feather":        int(self.sl_feather.get()),
            "radio_strength": float(self.sl_radio.get()),
            "color_strength": float(self.sl_color.get()),
            "retinex_strength": float(self.sl_retinex.get()),
            "use_ai_color":   self.use_ai_color.get(),
            "denoise_h":      int(self.sl_denoise.get()),
            "sharpen_amount": float(self.sl_sharpen.get()),
            "clahe_clip":     float(self.sl_clahe.get()),
        }

    # ──────────────────────────────────────────────────────
    # 배치 처리 스레드
    # ──────────────────────────────────────────────────────

    def _run_batch(self, files, out_dir, params):
        total     = len(files)
        done      = 0
        failed    = 0
        shadow_list = []
        speed_list  = []
        t_start   = time.time()
        n_workers = self.worker_var.get()

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

                # 비교 이미지 저장
                if self.save_compare.get():
                    cmp_path = os.path.join(out_dir, f"{stem}_compare.jpg")
                    cmp = make_compare(img, result, binary, soft)
                    cv2.imwrite(cmp_path, cmp,
                                [cv2.IMWRITE_JPEG_QUALITY, 90])

                return {
                    "status":     "ok",
                    "path":       path,
                    "fname":      fname,
                    "out_path":   out_path,
                    "stats":      stats,
                    "orig":       img,
                    "result":     result,
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

                done += 1
                elapsed = time.time() - t_start

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
                    self._queue.put(("preview", r["orig"], r["result"]))

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
    # 큐 폴링  (메인 스레드에서 UI 업데이트)
    # ──────────────────────────────────────────────────────

    def _poll_queue(self):
        try:
            while True:
                msg = self._queue.get_nowait()
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
                    self._set_status("준비 완료", GREEN)
                    self._log(f"모델 로드 완료: {txt}", "ok")

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

                elif kind == "preview":
                    _, orig, result = msg
                    self._preview_orig = orig
                    self._preview_res  = result
                    self._redraw_preview()

                elif kind == "done":
                    _, done, failed, elapsed, cancelled = msg
                    self._is_running = False
                    self.btn_start.set_enabled(True)
                    self.btn_cancel.set_enabled(False)
                    m, s = divmod(int(elapsed), 60)
                    if cancelled:
                        msg_txt = f"처리 중단  |  완료 {done}장"
                        self._set_status(msg_txt, WARN)
                        self._log(f"⏹  중단됨  {done}장 처리  ({m:02d}:{s:02d})", "warn")
                    else:
                        msg_txt = f"완료  |  {done}장 처리  |  실패 {failed}장  |  {m:02d}:{s:02d}"
                        self._set_status(msg_txt, GREEN)
                        self._log(f"🎉  완료!  총 {done}장 처리  |  실패 {failed}장  |  {m:02d}:{s:02d}", "ok")
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
    # Tk DPI 스케일링 (고해상도 모니터 대응)
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

    app = DroneApp()

    # ttk 스타일
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
