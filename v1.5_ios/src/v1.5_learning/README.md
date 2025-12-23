# TryAngle v1.5 - MVP Learning Pipeline

Windows RTX 4070 Super에서 실행하는 오프라인 학습 파이프라인입니다.

## 개요

분류된 이미지(cafe, park, beach 등)에서 구도 패턴을 추출하여 `patterns_mvp_v1.json`을 생성합니다.

```
분류된 이미지 (300장)
    ↓
Grounding DINO (bbox + 배경객체)
    ↓
Depth Anything V2 (압축감)
    ↓
RTMPose (pose type + angle)
    ↓
통계 계산
    ↓
patterns_mvp_v1.json (500KB)
```

## 환경 설정 (Windows)

### 1. CUDA 확인

```bash
nvidia-smi
# CUDA 11.8+ 필요
```

### 2. Python 환경

```bash
# Anaconda 사용 권장
conda create -n tryangle_v15 python=3.10
conda activate tryangle_v15
```

### 3. PyTorch 설치 (CUDA)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 4. 기본 패키지 설치

```bash
cd src/v1.5_learning
pip install -r requirements.txt
```

### 5. Grounding DINO 설치

```bash
# 방법 1: pip
pip install groundingdino-py

# 방법 2: 소스 설치
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
pip install -e .
```

### 6. MMPose (RTMPose) 설치

```bash
pip install openmim
mim install mmengine mmcv mmdet mmpose
```

### 7. Depth Anything V2 설치

```bash
git clone https://github.com/DepthAnything/Depth-Anything-V2.git
pip install -e Depth-Anything-V2
```

## 사용법

### 1. config.yaml 수정

```yaml
paths:
  # 분류된 이미지 폴더 경로 수정!
  input_dir: "D:/TryAngle/data/classified_images"
  output_dir: "D:/TryAngle/output/v1.5_mvp"
```

폴더 구조:
```
D:/TryAngle/data/classified_images/
├── cafe/          # 150장
├── park/          # 100장
└── beach/         # 50장
```

### 2. 실행

```bash
cd src/v1.5_learning
python extract_features_mvp.py --config config.yaml
```

또는 경로 직접 지정:

```bash
python extract_features_mvp.py \
  --input_dir "D:/images/classified" \
  --output_dir "D:/output/mvp" \
  --device cuda
```

### 3. 결과 확인

```
D:/TryAngle/output/v1.5_mvp/
├── patterns_mvp_v1.json      # 최종 패턴 DB (iOS 앱에서 사용)
└── intermediate_features.json # 디버깅용 중간 결과
```

## 출력: patterns_mvp_v1.json

```json
{
  "cafe_indoor": {
    "theme_description": "Cafe Indoor",
    "total_samples": 150,
    "sub_patterns": {
      "cafe_indoor_half_body": {
        "sample_count": 80,
        "composition": {
          "position": {"mean": [0.35, 0.42], "std": [0.08, 0.10]},
          "size": {"mean": 0.32, "optimal_range": [0.28, 0.38]},
          "margins": {...}
        },
        "camera": {
          "compression_index": {"mean": 0.45, "std": 0.12},
          "angle": {"mean": 12, "std": 5}
        },
        "pose_requirements": {
          "sitting": true,
          "visible_joints": ["shoulders", "hips", "knees"]
        }
      }
    }
  }
}
```

## 예상 소요 시간

- 300장 기준: 약 30분
- 모델 로드: 2-3분
- 이미지 처리: 3초/장

## 문제 해결

### CUDA 메모리 부족

```yaml
# config.yaml
extraction:
  resize_max: 640  # 1024 → 640으로 줄이기

processing:
  batch_size: 2    # 4 → 2로 줄이기
```

### 모델 로드 실패

```bash
# HuggingFace 캐시 확인
pip install huggingface_hub
huggingface-cli scan-cache
```

### MMPose 오류

```bash
# MMPose 재설치
pip uninstall mmpose mmdet mmcv mmengine
mim install mmengine==0.10.3 mmcv==2.1.0 mmdet==3.3.0 mmpose==1.3.1
```

## 다음 단계

1. `patterns_mvp_v1.json`을 iOS 앱으로 복사
2. iOS에서 실시간 피드백 테스트
3. 검증 완료 후 전체 2687장으로 확장

---

**문서 버전:** 1.0
**최종 수정:** 2025-11-30
