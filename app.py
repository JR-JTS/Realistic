"""
Drone Shadow Removal & Enhancement Tool
Gradio Web UI
"""

import gradio as gr
import cv2
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from pipeline import process_image, create_comparison_image, load_models

# ── Load models at startup ──
print("🚀 Loading models...")
model_status = load_models(model_dir=os.path.join(os.path.dirname(__file__), "models"))
print(f"📊 Model status: {model_status}")


# ────────────────────────────────────────────────
# Processing function for Gradio
# ────────────────────────────────────────────────

def process_drone_image(
    input_img,
    detection_mode,
    shadow_sensitivity,
    mask_feather,
    removal_method,
    removal_strength,
    enhance_mode,
    sharpen_strength,
    denoise_strength,
    clahe_clip,
    brightness,
    contrast,
    saturation,
):
    if input_img is None:
        return None, None, None, "⚠️ 이미지를 업로드하세요."

    try:
        # Convert PIL/numpy to BGR
        img_rgb = np.array(input_img)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # Run pipeline
        result_bgr, shadow_mask, soft_mask, stats = process_image(
            img_bgr,
            detection_mode=detection_mode,
            shadow_sensitivity=shadow_sensitivity,
            mask_feather=int(mask_feather),
            removal_method=removal_method,
            removal_strength=removal_strength,
            enhance_mode=enhance_mode,
            sharpen_strength=sharpen_strength,
            denoise_strength=int(denoise_strength),
            clahe_clip=clahe_clip,
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
        )

        # Convert outputs to RGB for Gradio display
        result_rgb  = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
        compare_bgr = create_comparison_image(img_bgr, result_bgr, shadow_mask)
        compare_rgb = cv2.cvtColor(compare_bgr, cv2.COLOR_BGR2RGB)

        # Colorized shadow mask
        mask_color = np.zeros((*shadow_mask.shape, 3), dtype=np.uint8)
        mask_color[shadow_mask > 0] = [255, 130, 0]

        stats_text = f"""
### 📊 처리 결과

| 항목 | 값 |
|------|-----|
| 🔍 그림자 탐지 시간 | {stats['detection_ms']} ms |
| 🌤️ 그림자 제거 시간 | {stats['removal_ms']} ms |
| ✨ 화질 향상 시간 | {stats['enhance_ms']} ms |
| ⏱️ **총 처리 시간** | **{stats['total_ms']} ms** |
| 🌑 그림자 비율 | {stats['shadow_ratio']} % |

---
**탐지 모드:** `{detection_mode}` &nbsp;&nbsp;|&nbsp;&nbsp;
**제거 방법:** `{removal_method}` &nbsp;&nbsp;|&nbsp;&nbsp;
**향상 모드:** `{enhance_mode}`
"""
        return result_rgb, mask_color, compare_rgb, stats_text

    except Exception as e:
        import traceback
        err = traceback.format_exc()
        return None, None, None, f"❌ 오류 발생:\n```\n{err}\n```"


# ────────────────────────────────────────────────
# Gradio UI Layout
# ────────────────────────────────────────────────

CSS = """
body { font-family: 'Segoe UI', Arial, sans-serif; }

.header-box {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 16px;
    padding: 30px 40px;
    margin-bottom: 20px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}

.header-title {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #00d2ff, #7b2ff7, #00d2ff);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shimmer 3s linear infinite;
}

@keyframes shimmer {
    0% { background-position: 0% center; }
    100% { background-position: 200% center; }
}

.header-sub {
    color: #8bb8e8;
    font-size: 1rem;
    margin-top: 6px;
}

.pipeline-badge {
    display: inline-block;
    background: rgba(255,255,255,0.1);
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 20px;
    padding: 4px 14px;
    color: #e0e0e0;
    font-size: 0.82rem;
    margin: 4px 3px;
}

.section-header {
    font-weight: 700;
    font-size: 1.05rem;
    color: #2c3e50;
    border-left: 4px solid #0f3460;
    padding-left: 10px;
    margin-bottom: 12px;
}

.process-btn {
    background: linear-gradient(135deg, #0f3460, #7b2ff7) !important;
    color: white !important;
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    border-radius: 10px !important;
    padding: 14px 0 !important;
    border: none !important;
    box-shadow: 0 4px 20px rgba(123,47,247,0.4) !important;
    transition: all 0.3s ease !important;
}

.process-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(123,47,247,0.6) !important;
}

.stats-box {
    background: #f8f9fa;
    border-radius: 12px;
    padding: 15px;
    border: 1px solid #e9ecef;
}

footer { display: none !important; }
"""

def build_ui():
    with gr.Blocks(title="🛸 Drone Shadow Remover") as demo:

        # ── Header ──────────────────────────────────
        gr.HTML("""
        <div class="header-box">
            <div class="header-title">🛸 Drone Shadow Remover & Enhancer</div>
            <div class="header-sub">AI + Computer Vision 기반 드론 항공사진 그림자 제거 및 화질 향상 툴</div>
            <div style="margin-top:14px">
                <span class="pipeline-badge">🔍 그림자 탐지</span>
                <span class="pipeline-badge">→</span>
                <span class="pipeline-badge">🌤️ 그림자 제거</span>
                <span class="pipeline-badge">→</span>
                <span class="pipeline-badge">✨ 화질 향상</span>
                <span class="pipeline-badge">→</span>
                <span class="pipeline-badge">🎛️ 전역 조정</span>
            </div>
        </div>
        """)

        # ── Main Layout ──────────────────────────────
        with gr.Row(equal_height=False):

            # ── Left Panel: Controls ─────────────────
            with gr.Column(scale=1, min_width=320):

                gr.HTML('<div class="section-header">📁 이미지 입력</div>')
                input_image = gr.Image(
                    label="드론 사진 업로드",
                    type="pil",
                    image_mode="RGB",
                    height=280,
                    sources=["upload", "clipboard"],
                )

                gr.HTML('<div class="section-header" style="margin-top:20px">🔍 그림자 탐지 설정</div>')
                with gr.Group():
                    detection_mode = gr.Radio(
                        label="탐지 방법",
                        choices=[
                            ("🤖 AI + CV 복합 (권장)", "ai_cv_hybrid"),
                            ("🔬 전통 CV 방법", "cv_only"),
                        ],
                        value="ai_cv_hybrid",
                    )
                    shadow_sensitivity = gr.Slider(0.0, 1.0, value=0.5, step=0.05,
                                                    label="탐지 민감도 (높을수록 더 많이 탐지)")
                    mask_feather = gr.Slider(5, 60, value=25, step=1,
                                              label="마스크 페더링 (경계 부드럽게)")

                gr.HTML('<div class="section-header" style="margin-top:20px">🌤️ 그림자 제거 설정</div>')
                with gr.Group():
                    removal_method = gr.Radio(
                        label="제거 방법",
                        choices=[
                            ("⚡ 복합 (권장)", "combined"),
                            ("☀️ 조명 보정", "illumination"),
                            ("🎨 색상 전환", "color_transfer"),
                        ],
                        value="combined",
                    )
                    removal_strength = gr.Slider(0.0, 1.0, value=0.8, step=0.05,
                                                  label="제거 강도")

                gr.HTML('<div class="section-header" style="margin-top:20px">✨ 화질 향상 설정</div>')
                with gr.Group():
                    enhance_mode = gr.Radio(
                        label="향상 방법",
                        choices=[
                            ("🤖 AI + CV 복합", "ai_cv"),
                            ("🔬 CV 방법만", "cv_only"),
                        ],
                        value="ai_cv",
                    )
                    sharpen_strength = gr.Slider(0.5, 3.0, value=1.5, step=0.1,
                                                  label="선명도 강도")
                    denoise_strength = gr.Slider(1, 15, value=7, step=1,
                                                  label="노이즈 제거 강도")
                    clahe_clip = gr.Slider(1.0, 5.0, value=2.5, step=0.1,
                                            label="CLAHE 대비 강도")

                gr.HTML('<div class="section-header" style="margin-top:20px">🎛️ 전역 색상 조정</div>')
                with gr.Group():
                    brightness = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="밝기")
                    contrast   = gr.Slider(0.5, 2.0, value=1.05, step=0.05, label="대비")
                    saturation = gr.Slider(0.5, 2.0, value=1.1, step=0.05, label="채도")

                process_btn = gr.Button("🚀 처리 시작", elem_classes=["process-btn"])

            # ── Right Panel: Results ─────────────────
            with gr.Column(scale=2):

                with gr.Row():
                    with gr.Column():
                        gr.HTML('<div class="section-header">✅ 처리 결과</div>')
                        output_result = gr.Image(label="처리된 이미지", height=350, interactive=False)
                    with gr.Column():
                        gr.HTML('<div class="section-header">🌑 그림자 마스크</div>')
                        output_mask = gr.Image(label="탐지된 그림자 영역 (주황)", height=350, interactive=False)

                gr.HTML('<div class="section-header" style="margin-top:10px">📊 비교 (원본 | 마스크 | 결과)</div>')
                output_compare = gr.Image(label="3분할 비교 이미지", interactive=False)

                gr.HTML('<div class="section-header" style="margin-top:10px">📈 처리 통계</div>')
                output_stats = gr.Markdown(
                    value="이미지를 업로드하고 **처리 시작** 버튼을 클릭하세요.",
                    elem_classes=["stats-box"],
                )

        # ── Examples ────────────────────────────────
        gr.HTML('<div class="section-header" style="margin-top:20px">💡 사용법 안내</div>')
        gr.Markdown("""
        | 단계 | 설명 |
        |------|------|
        | **1. 이미지 업로드** | 드론 항공사진을 업로드합니다 (JPG/PNG/TIFF 지원) |
        | **2. 탐지 설정** | AI+CV 복합 모드 사용 권장. 민감도로 탐지 범위 조절 |
        | **3. 제거 강도** | 0.6~0.9 권장. 너무 강하면 색상 왜곡 발생 가능 |
        | **4. 선명도** | 1.5~2.0 권장. 그림자 영역의 디테일을 복원합니다 |
        | **5. CLAHE** | 2.0~3.5 권장. 그림자 내부의 대비를 향상시킵니다 |
        | **6. 전역 조정** | 처리 후 전체적인 색감 미세 조정 |

        > 💡 **팁**: 항공사진은 그림자 비율이 높으므로 **제거 강도 0.75~0.85**, **선명도 1.5~2.0** 을 권장합니다.
        """)

        # ── Event Binding ─────────────────────────
        process_btn.click(
            fn=process_drone_image,
            inputs=[
                input_image,
                detection_mode, shadow_sensitivity, mask_feather,
                removal_method, removal_strength,
                enhance_mode, sharpen_strength, denoise_strength, clahe_clip,
                brightness, contrast, saturation,
            ],
            outputs=[output_result, output_mask, output_compare, output_stats],
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        share=False,
        css=CSS,
    )
