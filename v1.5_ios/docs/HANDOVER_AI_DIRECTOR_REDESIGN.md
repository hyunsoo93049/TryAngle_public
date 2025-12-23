# 🎬 Try_Angle AI 카메라 감독 시스템 재설계 - 인수인계 문서

**작성일**: 2025-11-18
**작성자**: AI Development Team
**프로젝트**: Try_Angle - AI Photography Guide
**버전**: v3.0 (대규모 재설계)

---

## 📋 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [현재 시스템 문제점 분석](#2-현재-시스템-문제점-분석)
3. [재설계 아키텍처](#3-재설계-아키텍처)
4. [모델 스택 변경](#4-모델-스택-변경)
5. [구현 계획](#5-구현-계획)
6. [코드 위치 및 구조](#6-코드-위치-및-구조)
7. [기술 스택](#7-기술-스택)
8. [참고 자료](#8-참고-자료)

---

## 1. 프로젝트 개요

### 1.1 프로젝트 정보

- **프로젝트명**: Try_Angle
- **목적**: 레퍼런스 사진을 보고 동일한 구도/포즈/스타일로 촬영할 수 있도록 실시간 피드백을 제공하는 AI 카메라 감독 시스템
- **위치**: `/Users/hyunsoo/Try_Angle`
- **Git 브랜치**: `HS_MAC` (main 브랜치에서 작업 중)

### 1.2 개발 환경

```bash
# 시스템
OS: macOS 14.6 (M4 chip)
Python: 3.11 (Conda environment: TA)
Conda Path: /Users/hyunsoo/Try_Angle/TA/

# 백엔드
Framework: FastAPI
Port: 8000

# 프론트엔드
Platform: iOS (Swift/SwiftUI)
IDE: Xcode
Device: iPhone (실제 기기 테스트 가능)

# GPU
Device: Apple M4 (MPS 지원)
```

### 1.3 핵심 컨셉

**"AI 카메라 감독 (AI Director of Photography)"**

단순한 이미지 비교 시스템이 아닌, **사진학적 의도를 이해하고 구체적인 촬영 지시를 내리는 감독**

---

## 2. 현재 시스템 문제점 분석

### 2.1 치명적 문제

#### ❌ **문제 1: CLIP/OpenCLIP 오용**

**현재 코드** (`feature_extractor_v2.py:470-481`):
```python
# 이미지 임베딩만 추출
clip_feat = models["clip_model"].encode_image(clip_in)
clip_feat = (clip_feat / clip_feat.norm(dim=-1, keepdim=True)).cpu().numpy()
# → [0.234, -0.123, 0.456, ...] 512D 벡터

# 문제: 텍스트 프롬프트를 전혀 사용하지 않음!
# "카페 사진", "빈티지 필터" 같은 분류 불가능
```

**의도했던 것**:
- 씬 카테고리 인식 (카페, 눈풍경, 해변 등)
- 필터/보정 스타일 파악 (VSCO A6, 빈티지, 웜톤 등)

**실제 결과**:
- 임베딩 벡터만 추출 → 클러스터링으로 번호만 나옴 (Cluster 7 = 무슨 의미?)
- 씬이나 필터 설명 전혀 불가

#### ❌ **문제 2: 단순 비교 시스템**

**현재 로직**:
```python
reference_features - current_features = difference
→ "왼쪽으로 이동하세요"
```

**문제점**:
1. **주체 불명확**: "왼쪽으로" = 카메라? 피사체?
2. **이유 없음**: 왜 왼쪽으로 가야 하는지 모름
3. **구도 이해 전무**: "3분할 구도", "황금비율" 같은 개념 없음
4. **미러링 미처리**: 전면 카메라 좌우 반전 고려 안 함

#### ❌ **문제 3: 포즈 피드백 부재**

**현재 상태** (`RealtimeAnalyzer.swift`):
- Vision Framework로 얼굴만 감지
- 팔, 다리 각도 비교 전혀 없음
- 포즈 피드백 생성 안 됨

#### ❌ **문제 4: 피드백 추상적**

**현재**:
- "줌 인", "줌 아웃"
- "왼쪽으로 이동"

**원하는 것**:
- "3걸음 앞으로"
- "피사체가 왼쪽으로 50cm (3분할 구도 완성)"
- "카메라를 30cm 낮춰주세요 (로우앵글)"

### 2.2 우선순위 (사용자 명시)

```
구도 > 포즈 > 배경/피사체 위치 >>> 색감
```

**중요**: 색감(CLIP으로 하려던 것)보다 **구도와 포즈**가 훨씬 중요!

---

## 3. 재설계 아키텍처

### 3.1 핵심 철학

```
┌─────────────────────────────────────────────────────┐
│         AI Director of Photography System            │
│                                                       │
│   "이미지를 이해하고, 의도를 파악하고, 지시하는"     │
└─────────────────────────────────────────────────────┘

3-Layer Architecture:

[Layer 1: Understanding]  ← "무엇을, 왜"
[Layer 2: Measurement]    ← "얼마나 정확히"
[Layer 3: Direction]      ← "어떻게 고칠지"
```

### 3.2 Layer 1: Understanding (이해의 뇌)

**모델**: Florence-2 Large

**역할**: 레퍼런스의 사진학적 의도 완전 파악

**입력**: 레퍼런스 이미지

**출력**: Photographic Intent (JSON)

```json
{
    "composition_intent": {
        "primary_rule": "rule_of_thirds",
        "subject_placement": "left_upper_intersection",
        "reasoning": "dynamic asymmetry for storytelling"
    },

    "spatial_intent": {
        "depth_strategy": "shallow_dof",
        "foreground": "coffee_cup_as_anchor",
        "background": "blurred_cafe_context",
        "reasoning": "isolate subject, add depth layers"
    },

    "lighting_intent": {
        "source": "natural_window_left",
        "quality": "soft_diffused",
        "direction": "side_lighting",
        "reasoning": "soft portrait look with dimension"
    },

    "pose_intent": {
        "archetype": "candid_relaxed",
        "key_angles": {
            "shoulder_tilt": "slight_left",
            "head_angle": "looking_away",
            "body_weight": "leaning_forward"
        },
        "reasoning": "natural unstaged moment"
    },

    "emotional_tone": "warm, intimate, contemplative",

    "technique_summary":
        "Portrait using rule of thirds with subject on left intersection.
         Natural window light from left creates soft side lighting.
         Shallow depth of field (f/1.8-2.8) with foreground coffee cup
         to add depth. Candid pose suggests natural moment."
}
```

**핵심 능력**:
- ✅ 구도 "기법" 명시 (rule of thirds, golden ratio, symmetry 등)
- ✅ "왜 그렇게 찍었는지" 추론
- ✅ 전체 맥락 파악 (단순 객체 나열이 아님)
- ✅ 사진학 용어로 표현

### 3.3 Layer 2: Measurement (정밀 측정)

#### **2.1 공간 관계: Grounding DINO**

**역할**: 객체의 정확한 위치 측정

**Florence가 말한 것**:
"Subject on left third intersection"

**Grounding DINO가 측정**:
```json
{
    "subject_bbox": [0.15, 0.25, 0.45, 0.85],
    "subject_center": [0.30, 0.55],

    "thirds_grid": {
        "left_vertical": 0.333,
        "right_vertical": 0.667,
        "upper_horizontal": 0.333,
        "lower_horizontal": 0.667
    },

    "deviation_from_target": {
        "target": "left_upper_intersection",
        "current": [0.30, 0.55],
        "offset_x": -0.033,
        "offset_y": +0.217
    }
}
```

#### **2.2 깊이 분석: Depth Anything V2**

**역할**: 전경/중경/배경 레이어 분리, 거리 측정

**Florence가 말한 것**:
"Shallow depth of field, foreground coffee cup, blurred background"

**Depth Anything V2가 측정**:
```json
{
    "subject_distance": 2.3,
    "foreground_distance": 1.5,
    "background_distance": 5.8,

    "depth_layers": {
        "foreground": {
            "range": [1.2, 1.8],
            "elements": ["coffee_cup"],
            "sharpness": 0.6
        },
        "midground": {
            "range": [2.0, 3.0],
            "elements": ["subject_person"],
            "sharpness": 1.0
        },
        "background": {
            "range": [4.0, 8.0],
            "elements": ["cafe_interior"],
            "sharpness": 0.2
        }
    },

    "estimated_aperture": "f/1.8 - f/2.2"
}
```

#### **2.3 포즈 분석: Sapiens / MoveNet**

**역할**: 신체 각도와 포즈 정밀 측정

**Florence가 말한 것**:
"Candid pose with slight shoulder tilt, looking away"

**Sapiens가 측정**:
```json
{
    "keypoints_3d": [...],

    "body_angles": {
        "shoulder_line": {
            "tilt_angle": -7.3,
            "left_shoulder": [x, y, z],
            "right_shoulder": [x, y, z]
        },
        "head_rotation": {
            "yaw": 25.0,
            "pitch": -5.0,
            "roll": -7.0
        },
        "arm_angles": {
            "left_elbow": 118.0,
            "right_elbow": 95.0
        }
    },

    "pose_archetype_match": {
        "candid_natural": 0.85,
        "posed_formal": 0.10
    }
}
```

#### **2.4 색감 분석: Custom Color Analyzer**

**역할**: 색온도, 톤 커브, 필터 스타일 (낮은 우선순위)

```json
{
    "color_temperature": {
        "kelvin": 4800,
        "label": "golden_hour"
    },

    "filter_match": {
        "name": "vsco_c1_variant",
        "confidence": 0.78
    }
}
```

### 3.4 Layer 3: Direction (지시 생성)

**Director Engine**: Layer 1 + Layer 2 통합 → 명확한 지시

**출력 예시**:
```json
{
    "priority": 1,
    "category": "composition",

    "actor": "subject",
    "action": "move",
    "direction": "user_left",
    "amount": "50cm",

    "reason": "3분할 구도의 왼쪽 상단 교점 달성",

    "visual_overlay": {
        "type": "rule_of_thirds_grid",
        "highlight": "left_upper_intersection"
    },

    "alternative": "또는 📱 카메라를 오른쪽으로 50cm 패닝",

    "current_state": "피사체가 중앙에 위치 (50%, 55%)",
    "target_state": "왼쪽 상단 교점 (33%, 33%)"
}
```

---

## 4. 모델 스택 변경

### 4.1 제거할 모델

| 모델 | 이유 | 절약 |
|------|------|------|
| **CLIP** | Florence-2가 완전 대체, 임베딩만으로는 의도 파악 불가 | ~400MB |
| **OpenCLIP** | 동일 | ~400MB |
| **DINO** | Grounding DINO + Florence-2로 대체 | ~400MB |
| **Contrastive Model** | Florence-2가 더 나음 | ~300MB |

**총 절약**: ~1.5GB

### 4.2 최종 모델 스택

```
┌─────────────────────────────────────────────────┐
│         최종 AI Director 모델 스택               │
└─────────────────────────────────────────────────┘

Layer 1: Understanding
  └─ Florence-2 Large           1.5GB
     microsoft/Florence-2-large

Layer 2: Measurement
  ├─ Grounding DINO             700MB
  │  GroundingDINO/groundingdino
  │
  ├─ Depth Anything V2          400MB
  │  depth-anything/Depth-Anything-V2-Large
  │
  ├─ MoveNet Thunder            200MB
  │  (또는 Sapiens-1B: 800MB)
  │
  └─ Color Grading Analyzer      50MB
     (Custom, 낮은 우선순위)

────────────────────────────────────────────────
Total:                        2.85GB
기존 시스템:                  4.5GB
절약:                        1.65GB
```

### 4.3 모델별 상세 스펙

#### Florence-2 Large

```python
Model: microsoft/Florence-2-large
Size: ~1.5GB
Framework: Transformers (Hugging Face)
Device: CUDA / MPS

Tasks:
- <MORE_DETAILED_CAPTION>: 상세 설명
- <OD>: Object Detection
- <DENSE_REGION_CAPTION>: 영역별 설명
- <OCR>: 텍스트 인식 (선택)

Installation:
pip install transformers torch pillow

Usage:
from transformers import AutoModelForCausalLM, AutoProcessor
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Florence-2-large",
    trust_remote_code=True
)
```

#### Grounding DINO

```python
Model: GroundingDINO/groundingdino_swint_ogc
Size: ~700MB
Framework: Custom (groundingdino library)

특징:
- Open-vocabulary object detection
- Text prompt로 객체 찾기
- 정확한 bbox 좌표

Installation:
git clone https://github.com/IDEA-Research/GroundingDINO.git
pip install -e GroundingDINO/

Usage:
prompt = "person . face . coffee cup"
boxes, scores = model.predict(image, prompt)
```

#### Depth Anything V2

```python
Model: depth-anything/Depth-Anything-V2-Large
Size: ~400MB
Framework: PyTorch

특징:
- MiDaS보다 정확
- 실내/실외 모두 강력
- 미터 단위 절대 깊이 추정 가능

Installation:
git clone https://github.com/DepthAnything/Depth-Anything-V2.git
pip install -e Depth-Anything-V2/

Usage:
depth_map = depth_model.infer_image(image)
```

#### MoveNet Thunder / Sapiens

```python
# Option 1: MoveNet Thunder (추천 - 충분)
Model: movenet_thunder.tflite
Size: ~200MB
Framework: TensorFlow Lite

# Option 2: Sapiens (최고 성능 원하면)
Model: Sapiens-1B
Size: ~800MB
Framework: PyTorch

Installation:
# MoveNet
pip install tensorflow==2.15.0 tensorflow-hub

# Sapiens
git clone https://github.com/facebookresearch/sapiens.git
```

---

## 5. 구현 계획

### 5.1 Phase 1: 핵심 구축 (2주)

**Week 1**:
```
Day 1-2: Florence-2 통합
  - 모델 로드 및 캐싱
  - 기본 태스크 실행 (<CAPTION>, <OD>)
  - 레퍼런스 의도 파싱 로직

Day 3-4: Grounding DINO 통합
  - 설치 및 테스트
  - 3분할 그리드 계산 로직
  - Florence 결과와 통합

Day 5: Director Engine 프로토타입
  - Intent + Measurement → Direction
  - 기본 피드백 생성기
```

**Week 2**:
```
Day 1-2: Depth Anything V2 통합
  - 깊이 맵 추출
  - 전경/중경/배경 분리 로직
  - 거리 계산

Day 3: MoveNet 통합 (기존 활용)
  - 기존 코드 재사용
  - 각도 계산 정밀화

Day 4-5: 통합 테스트
  - 전체 파이프라인 테스트
  - iOS 연동 테스트
```

**산출물**:
- 구도 + 포즈 정확한 피드백
- "3분할 구도로 왼쪽 50cm 이동" 수준 지시

### 5.2 Phase 2: 정교화 (1주)

```
- Custom Color Analyzer 개발
- 다양한 구도 룰 추가 (황금비율, 대각선, 대칭 등)
- AR 오버레이 UI 개선
- 피드백 한국어 자연스럽게 다듬기
```

### 5.3 Phase 3: 최적화 (1주)

```
- 모델 양자화 (FP16, INT8)
- 추론 속도 개선 (배치 처리)
- 캐싱 전략 (레퍼런스 분석 결과 캐시)
- 메모리 최적화
```

---

## 6. 코드 위치 및 구조

### 6.1 현재 파일 구조

```
/Users/hyunsoo/Try_Angle/
│
├── backend/
│   └── main.py                          # FastAPI 서버 (포트 8000)
│
├── src/Multi/version3/
│   ├── analysis/
│   │   ├── image_comparator.py          # [재설계 필요] 현재 비교 로직
│   │   ├── pose_analyzer.py             # 포즈 비교 (MoveNet 사용)
│   │   ├── exif_analyzer.py
│   │   ├── quality_analyzer.py
│   │   └── lighting_analyzer.py
│   │
│   ├── feature_extraction/
│   │   ├── feature_extractor_v2.py      # [재설계 필요] CLIP/DINO 사용
│   │   └── feature_extractor_v3.py      # Contrastive Learning
│   │
│   ├── contrastive/
│   │   └── contrastive_model.py         # [제거 예정]
│   │
│   ├── models/
│   │   ├── movenet_thunder.tflite       # 유지
│   │   └── contrastive/
│   │       └── best_model.pth           # [제거 예정]
│   │
│   ├── utils/
│   │   ├── model_cache.py               # 싱글톤 캐싱
│   │   └── feedback_formatter.py
│   │
│   └── scripts/
│       └── download_movenet.py
│
├── ios/TryAngleApp/
│   ├── ContentView.swift                # 메인 카메라 뷰
│   ├── Services/
│   │   └── RealtimeAnalyzer.swift       # [개선됨] 실시간 분석
│   └── Views/
│       └── FeedbackOverlay.swift        # 피드백 UI
│
├── data/
│   └── test_images/
│
├── docs/
│   ├── MAC_FILE_STRUCTURE.md
│   └── HANDOVER_AI_DIRECTOR_REDESIGN.md # [이 문서]
│
└── README.md
```

### 6.2 신규 생성 파일

```
src/Multi/version3/
│
├── director/                            # [신규]
│   ├── __init__.py
│   ├── understanding_layer.py           # Florence-2 wrapper
│   ├── measurement_layer.py             # Grounding/Depth/Pose 통합
│   ├── direction_engine.py              # 피드백 생성
│   └── composition_rules.py             # 구도 룰 정의
│
├── models/
│   ├── florence2/                       # [신규]
│   ├── grounding_dino/                  # [신규]
│   └── depth_anything_v2/               # [신규]
│
└── analysis/
    ├── color_grading_analyzer.py        # [신규] 색감 분석
    └── spatial_relationship.py          # [신규] 공간 관계 분석
```

---

## 7. 기술 스택

### 7.1 필수 라이브러리

```bash
# 기본
torch==2.1.0
torchvision==0.16.0
transformers==4.35.0
pillow==10.1.0
opencv-python==4.8.1
numpy==1.26.4

# Florence-2
huggingface-hub==0.19.4

# Grounding DINO
groundingdino  # 별도 설치

# Depth Anything V2
# 별도 git clone

# TensorFlow (MoveNet)
tensorflow==2.15.0
tensorflow-hub

# FastAPI
fastapi==0.104.1
uvicorn==0.24.0
python-multipart==0.0.6

# 유틸
pydantic==2.5.0
scipy==1.11.4
scikit-learn==1.3.2
```

### 7.2 설치 스크립트

```bash
#!/bin/bash
# scripts/setup_ai_director.sh

# Conda 환경 활성화
source /Users/hyunsoo/Try_Angle/TA/bin/activate

# 기본 라이브러리
pip install torch torchvision transformers pillow opencv-python

# Florence-2
pip install huggingface-hub

# Grounding DINO
cd /Users/hyunsoo/Try_Angle/external_projects
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
pip install -e .

# Depth Anything V2
cd /Users/hyunsoo/Try_Angle/external_projects
git clone https://github.com/DepthAnything/Depth-Anything-V2.git
cd Depth-Anything-V2
pip install -e .

# TensorFlow (이미 설치됨)
pip install tensorflow==2.15.0 tensorflow-hub

echo "✅ AI Director 환경 설정 완료"
```

---

## 8. 참고 자료

### 8.1 모델 논문/문서

- **Florence-2**: [Microsoft Research](https://huggingface.co/microsoft/Florence-2-large)
- **Grounding DINO**: [arXiv:2303.05499](https://arxiv.org/abs/2303.05499)
- **Depth Anything V2**: [GitHub](https://github.com/DepthAnything/Depth-Anything-V2)
- **Sapiens**: [Meta Research](https://github.com/facebookresearch/sapiens)

### 8.2 사진학 이론

구도 기법:
- Rule of Thirds (3분할 구도)
- Golden Ratio (황금비율)
- Leading Lines (선 유도)
- Frame within Frame (프레임 안의 프레임)
- Symmetry (대칭)
- Negative Space (여백의 미)
- Depth Layers (전경-중경-배경)

### 8.3 기존 작업 커밋

```bash
# 최근 주요 커밋
e755337 - fix: CGFloat 타입 충돌 해결 (iOS 빌드 에러 수정)
cd9eedc - feat: iOS 실시간 피드백 대폭 개선 - 포즈/각도/거리 감지
3a78ac0 - fix: 프레이밍 피드백을 줌 기반에서 거리 기반으로 개선
8296ddc - fix: iOS UI 치명적 오류 수정 및 촬영 버튼 추가
c9011ef - feat: MoveNet을 기본 포즈 모델로 전환
```

---

## 9. 핵심 설계 원칙 (재강조)

### 9.1 5대 원칙

1. **Florence-2 = 뇌**
   - 모든 이해는 Florence-2에서 시작
   - 다른 모델들은 Florence가 말한 것을 "측정"

2. **우선순위 엄수**
   - 구도 > 포즈 > 깊이 > 색감
   - 피드백도 이 순서로 제공
   - 리소스 부족 시 색감부터 생략

3. **명확성 > 정확성**
   - "50cm 왼쪽" > "17% 이동"
   - 사용자가 바로 실행 가능한 지시

4. **이유 설명 필수**
   - "왼쪽으로"만 말하지 말고
   - "3분할 구도 완성을 위해" 항상 추가

5. **대안 제시**
   - 피사체 이동 vs 카메라 이동
   - 사용자가 선택 가능하게

### 9.2 금기사항

- ❌ 단순 특징 비교 (reference_feat - current_feat)
- ❌ 주체 불명확한 지시 ("왼쪽으로" 누가?)
- ❌ 이유 없는 지시 (왜 그래야 하는지)
- ❌ 추상적 표현 ("줌 인", "조금 이동")
- ❌ Florence-2 결과 무시하고 수치만 사용

---

## 10. 테스트 계획

### 10.1 단위 테스트

```python
# tests/test_understanding_layer.py
def test_florence_composition_understanding():
    """Florence-2가 구도를 올바르게 파악하는지"""

# tests/test_measurement_layer.py
def test_grounding_dino_accuracy():
    """Grounding DINO가 정확한 좌표를 반환하는지"""

# tests/test_director_engine.py
def test_direction_generation():
    """피드백이 명확하고 구체적인지"""
```

### 10.2 통합 테스트

```python
# tests/test_full_pipeline.py
def test_full_director_flow():
    """
    레퍼런스 입력 → 의도 파악 → 현재 측정 → 피드백 생성
    전체 플로우가 올바르게 작동하는지
    """
```

### 10.3 실전 테스트

```bash
# 테스트 시나리오
1. 카페 인물 사진 (3분할 구도)
2. 눈 풍경 인물 사진 (중앙 대칭)
3. 로우앵글 전신 사진
4. 하이앵글 상반신 사진
5. 더치앵글 역동적 포즈
```

---

## 11. 다음 개발자를 위한 체크리스트

### 11.1 시작 전 확인사항

- [ ] 맥북 M4 환경인가?
- [ ] Conda 환경 TA 활성화되었나?
- [ ] Git 브랜치 HS_MAC에서 작업 중인가?
- [ ] 이 문서 완전히 읽었나?
- [ ] Florence-2, Grounding DINO 설치 가능한가?

### 11.2 구현 체크리스트

**Phase 1-1: Florence-2 통합**
- [ ] transformers 설치
- [ ] Florence-2 모델 다운로드
- [ ] 기본 캡션 생성 테스트
- [ ] <OD>, <DENSE_REGION_CAPTION> 테스트
- [ ] 레퍼런스 의도 파싱 함수 작성
- [ ] 구도 룰 인식 (rule of thirds 등)

**Phase 1-2: Grounding DINO 통합**
- [ ] GroundingDINO 설치
- [ ] 기본 객체 탐지 테스트
- [ ] 3분할 그리드 계산 로직
- [ ] bbox와 그리드 비교 로직
- [ ] Florence 결과와 통합

**Phase 1-3: Director Engine**
- [ ] Intent 데이터 구조 정의
- [ ] Measurement 데이터 구조 정의
- [ ] Direction 데이터 구조 정의
- [ ] Gap 분석 로직
- [ ] 피드백 생성 로직
- [ ] 한국어 자연어 생성

**Phase 2: Depth & Pose**
- [ ] Depth Anything V2 설치
- [ ] 깊이 맵 추출 및 레이어 분리
- [ ] MoveNet 기존 코드 활용
- [ ] 각도 계산 정밀화

**Phase 3: iOS 연동**
- [ ] FastAPI 엔드포인트 수정
- [ ] iOS RealtimeAnalyzer 업데이트
- [ ] 피드백 UI 개선
- [ ] 실제 기기 테스트

### 11.3 완료 기준

- [ ] 레퍼런스 이미지 입력 시 구도 의도 파악됨
- [ ] "3분할 구도", "황금비율" 같은 용어 사용
- [ ] 피드백에 주체, 행동, 거리, 이유 모두 포함
- [ ] "피사체가 왼쪽으로 50cm (3분할 구도 완성)" 수준
- [ ] iOS 앱에서 실시간 작동
- [ ] 포즈 피드백 생성 ("왼팔을 15도 펴세요")
- [ ] 메모리 사용량 3GB 이하

---

## 12. 긴급 연락처 및 이슈 트래킹

### 12.1 중요 이슈 기록

**Issue #1**: CLIP 오용
→ 해결: Florence-2로 대체

**Issue #2**: 포즈 피드백 부재
→ 해결: MoveNet/Sapiens + Florence-2 통합

**Issue #3**: 주체 불명확
→ 해결: actor 필드 명시 (subject/camera)

### 12.2 알려진 제한사항

1. Florence-2 추론 속도: ~200-500ms
   - 해결: 레퍼런스 분석은 1회만, 캐싱

2. Grounding DINO 메모리: ~700MB
   - 해결: 필요시만 로드, 사용 후 해제

3. iOS 실시간 처리: Vision Framework만 사용
   - 서버 분석은 촬영 후 또는 주기적

---

## 13. 마무리

### 13.1 핵심 요약

**무엇을 바꾸는가?**
- 단순 비교 시스템 → AI 카메라 감독 시스템

**어떻게 바꾸는가?**
- Florence-2로 의도 파악 → 전문 모델로 측정 → 명확한 지시

**왜 바꾸는가?**
- 사용자가 실제로 실행할 수 있는 구체적 피드백 제공
- "왼쪽으로"가 아니라 "피사체가 왼쪽으로 50cm (3분할 구도 완성)"

### 13.2 성공 기준

이 재설계가 성공했다고 판단하는 기준:

1. ✅ 레퍼런스를 보고 "3분할 구도"라고 인식
2. ✅ "피사체가 왼쪽으로 50cm" 같은 구체적 지시
3. ✅ 포즈 피드백 생성 ("왼팔을 15도 펴세요")
4. ✅ 이유 설명 ("3분할 구도 완성을 위해")
5. ✅ iOS 앱에서 실시간 작동

### 13.3 다음 개발자에게

이 시스템은 단순한 CV 모델 조합이 아닙니다.
**"사진을 이해하고 가르치는 선생님"**을 만드는 것입니다.

Florence-2의 결과를 신뢰하고, 그것을 바탕으로 설계하세요.
단순 수치 비교에 빠지지 마세요.

화이팅! 🎬

---

**문서 버전**: 1.0
**최종 수정**: 2025-11-18
**다음 리뷰**: Phase 1 완료 후
