# 🍎 TryAngle v3 - Mac 설치 가이드

**크로스 플랫폼 호환 완료!** Windows 코드를 Mac에서 그대로 실행할 수 있습니다.

---

## 📦 1단계: 필요한 파일 복사

맥북으로 복사해야 하는 폴더/파일:

```
try_angle/
├── src/Multi/version3/           # 전체 코드 (필수)
├── feature_models/                # 학습된 모델 (필수) ⭐
├── features/                      # 클러스터 정보 (필수)
└── data/test_images/test3.jpg    # 레퍼런스 이미지 1장 (필수)
```

**❌ 복사 불필요:**
- `data/clustered_images/` (2700장) - 이미 학습 완료됨!
- 기타 테스트 이미지

**압축 방법:**
```bash
# Windows에서
cd C:\try_angle
zip -r tryangle_mac.zip src/Multi/version3 feature_models features data/test_images/test3.jpg
```

---

## 🐍 2단계: Python 환경 구축 (Mac)

### Option A: Anaconda 사용 (추천)

```bash
# Anaconda 설치 (https://www.anaconda.com/download)

# 환경 생성
conda create -n TA python=3.10
conda activate TA

# 필수 패키지
pip install opencv-python numpy pillow pyyaml
pip install torch torchvision  # Mac용 (Apple Silicon이면 자동 최적화됨)
pip install ultralytics  # YOLO
pip install mediapipe
pip install scikit-learn pandas
pip install timm  # Feature extractor
pip install umap-learn
```

### Option B: venv 사용

```bash
# Python 3.10 설치 확인
python3 --version  # 3.10.x 확인

# 가상환경 생성
python3 -m venv ~/TA_env
source ~/TA_env/bin/activate

# 패키지 설치 (위와 동일)
pip install opencv-python numpy pillow pyyaml
pip install torch torchvision
pip install ultralytics mediapipe
pip install scikit-learn pandas timm umap-learn
```

---

## 📁 3단계: 폴더 구조 설정

```bash
# 압축 해제
cd ~/Downloads
unzip tryangle_mac.zip

# 홈 디렉토리로 이동
mv try_angle ~

# 최종 구조 확인
cd ~/try_angle
tree -L 2
```

**예상 결과:**
```
try_angle/
├── src/
│   └── Multi/
├── feature_models/         # 학습된 모델
├── features/               # 클러스터 정보
└── data/
    └── test_images/
```

---

## ▶️ 4단계: 실행

```bash
cd ~/try_angle/src/Multi/version3

# 활성화 (매번 실행 전 필요)
conda activate TA

# 실행!
python camera_realtime.py
```

**조작법:**
- `q`: 종료
- `r`: 레퍼런스 재분석
- `s`: 현재 프레임 저장
- `SPACE`: 분석 일시정지/재개

---

## ⚙️ 5단계: 설정 변경 (선택)

`config.yaml` 파일에서 설정 변경 가능:

```yaml
camera:
  default_index: 0    # 카메라 번호 (0=기본, 1=외장)
  width: 1280
  height: 720
  analysis_interval: 1.0  # 분석 간격 (초)

paths:
  default_reference: test3.jpg  # 다른 이미지로 변경 가능

thresholds:
  depth_ratio: 0.15        # 거리 민감도
  brightness_diff: 20      # 밝기 민감도
  saturation_diff: 0.1     # 채도 민감도
  tilt_diff: 2.0          # 기울기 민감도
```

---

## 🐛 문제 해결

### 1. "config.yaml not found"
```bash
# config.yaml이 version3/ 안에 있는지 확인
ls ~/try_angle/src/Multi/version3/config.yaml
```

### 2. "feature_models not found"
```bash
# 상대 경로가 제대로 설정되었는지 확인
cd ~/try_angle/src/Multi/version3
python -c "from pathlib import Path; print((Path.cwd() / '../../../feature_models').resolve())"
```

### 3. "Camera not found"
```bash
# 카메라 번호 확인
python -c "import cv2; [print(f'Camera {i}: {cv2.VideoCapture(i).isOpened()}') for i in range(5)]"
```

### 4. Apple Silicon (M1/M2/M3) GPU 사용
```bash
# PyTorch MPS 지원 확인
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"

# MPS 사용 (자동 활성화됨)
```

---

## 📊 성능 비교

**Windows vs Mac:**
- 모델 로딩: 동일 (싱글톤 캐싱)
- 분석 속도: Mac M1 이상이면 더 빠를 수 있음
- FPS: 카메라 스펙에 따라 다름

---

## 🎯 핵심 포인트

✅ **하드코딩 경로 0개** - 모든 경로가 상대 경로
✅ **config.yaml** - 모든 설정 파일에서 관리
✅ **pathlib 사용** - Windows/Mac 자동 호환
✅ **학습 데이터 불필요** - 모델 파일만 있으면 OK

---

## 🚀 다음 단계

1. **iPhone 카메라 연동**: Mac에서 Continuity Camera 사용
   ```bash
   # macOS Ventura 이상에서 자동 지원
   # iPhone을 Mac 근처에 두면 카메라로 인식됨
   ```

2. **웹 인터페이스**: 브라우저에서 접속 (추가 개발 필요)

---

**작성일**: 2025-11-15
**버전**: 3.0.0
**플랫폼**: macOS 11+ (Big Sur 이상)
