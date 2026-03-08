# Drone Shadow Remover & Enhancer

## 프로젝트 개요

드론 항공사진에서 그림자를 자동으로 탐지·제거하고, 그림자 영역의 화질을 향상시키는 AI + CV 기반 툴입니다.

## 처리 파이프라인

```
드론 사진 입력
    │
    ▼
[1] 그림자 탐지 (Shadow Detection)
    ├── AI 모드: UNet-style CNN (모델 파일 있을 때)
    └── CV 모드: HSV + Lab + 색상비율 멀티큐 방식 (기본)
    │
    ▼
[2] 그림자 제거 (Shadow Removal)
    ├── 조명 보정: Multi-Scale Retinex + 채널별 게인 보정
    ├── 색상 전환: Lab 색공간 통계 전환
    └── 복합 모드: 두 방법 혼합 (권장)
    │
    ▼
[3] 화질 향상 (Enhancement)
    ├── AI 향상: 잔차 학습 CNN (모델 파일 있을 때)
    ├── CLAHE: 적응형 히스토그램 평활화
    ├── 노이즈 제거: Non-local Means
    └── 선명도: Unsharp Masking
    │
    ▼
[4] 전역 조정 (Global Adjustment)
    └── 밝기 / 대비 / 채도 미세 조정
    │
    ▼
결과 출력 + 3분할 비교 이미지
```

## 프로젝트 구조

```
webapp/
├── app.py                  # Gradio 웹 UI
├── src/
│   ├── shadow_detection.py # 그림자 탐지 모듈
│   ├── shadow_removal.py   # 그림자 제거 + 화질 향상
│   └── pipeline.py         # 전체 처리 파이프라인
├── models/                 # AI 모델 파일 저장 위치
│   ├── shadow_detector.pth (선택적)
│   └── enhancer.pth        (선택적)
├── uploads/                # 업로드 임시 파일
└── outputs/                # 처리 결과 저장
```

## 설치 및 실행

```bash
pip install gradio opencv-python Pillow torch torchvision scipy numpy

python app.py
# → http://localhost:7860 접속
```

## 주요 기능

### 그림자 탐지
- **AI+CV 복합**: 딥러닝 특징 추출 + CV 멀티큐 방식 융합
- **CV 방법**: HSV 명도, Lab L채널, 청색 비율 3가지 단서 투표 방식
- **소프트 마스크**: 가우시안 페더링으로 자연스러운 경계 처리

### 그림자 제거
- **조명 보정**: 그림자/비그림자 통계 기반 채널별 게인 + Retinex
- **색상 전환**: Lab 색공간 평균/분산 매칭으로 색감 보정
- **강도 조절**: 0.0~1.0 슬라이더로 제거 강도 세밀 조절

### 화질 향상 (그림자 영역)
- **AI 잔차 학습**: 경량 CNN이 보정값을 예측해 더한 방식
- **CLAHE**: 타일 단위 적응형 히스토그램 평활화 (내부 디테일 복원)
- **NL-Means**: 비국소 평균 노이즈 제거
- **Unsharp Mask**: 고주파 성분 강조로 선명도 향상

### AI 모델 (선택적)
사전학습 모델 없이도 CV 방법으로 동작합니다.
`models/` 폴더에 `.pth` 파일을 넣으면 자동으로 AI 모드로 전환됩니다.

## 파라미터 권장값 (항공사진 기준)

| 파라미터 | 권장값 | 설명 |
|---------|--------|------|
| 탐지 민감도 | 0.4~0.6 | 너무 높으면 오탐 증가 |
| 마스크 페더링 | 20~30 | 경계 자연스럽게 |
| 제거 강도 | 0.75~0.85 | 색상 왜곡 없이 최대 효과 |
| 선명도 | 1.5~2.0 | 디테일 복원 |
| CLAHE | 2.0~3.5 | 그림자 내부 대비 |
| 채도 | 1.05~1.15 | 색감 살리기 |
