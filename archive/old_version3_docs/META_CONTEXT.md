# TryAngle Version3 - Meta Context

**목적**: Claude Code와 GPT 간 컨텍스트 공유를 위한 프로젝트 상태 문서
**업데이트**: 2024-11-15 03:40 (KST)

---

## 📁 프로젝트 구조

### 최신 버전: `src/Multi/version3/` (프로덕션)

```
version3/
├── main_feedback.py              # 🎯 메인 실행 파일
├── analysis/                     # 분석 모듈
│   ├── image_analyzer.py         # 통합 이미지 분석 (클러스터+포즈+EXIF+픽셀+구도)
│   ├── image_comparator.py       # 레퍼런스 vs 사용자 비교 & 피드백 생성
│   ├── pose_analyzer.py          # 포즈 분석 (YOLO11 + MediaPipe 하이브리드)
│   └── exif_analyzer.py          # EXIF 메타데이터 추출 (ISO/조리개/셔터속도 등)
├── feature_extraction/           # 특징 추출
│   ├── feature_extractor.py
│   └── feature_extractor_v2.py   # ✅ 사용 중
├── embedder/                     # 임베딩 변환
│   └── embedder.py               # 특징 → 128D UMAP
├── matching/                     # 클러스터 매칭
│   └── cluster_matcher.py        # 임베딩 → 클러스터 ID
└── yolo11s-pose.pt              # YOLO11 포즈 모델 (자동 다운로드)
```

---

## 🔧 주요 모델 & 가중치

### 1. 클러스터링 모델 (K=20)
**위치**: `C:\try_angle\feature_models/`

```
feature_models/
├── kmeans_model.pkl              # ✅ KMeans 모델 (K=20)
├── kmeans_centroids.npy          # 클러스터 중심 (20, 128)
├── umap_128d_model.joblib        # ✅ UMAP 차원 축소 모델
├── scaler_clip.joblib            # CLIP 정규화
├── scaler_openclip.joblib        # OpenCLIP 정규화
├── scaler_dino.joblib            # DINO 정규화
├── scaler_color.joblib           # Color 정규화
├── scaler_midas.joblib           # MiDaS 정규화
├── cluster_labels.npy            # 전체 데이터 클러스터 라벨
├── fusion_128d.npy               # 128D 임베딩 (전체 데이터)
└── weights.json                  # ✅ 최적 가중치
```

**가중치 설정** (weights.json):
```json
{
  "clip": 0.30,
  "openclip": 0.30,
  "dino": 0.25,
  "color": 0.12,
  "midas": 0.03,
  "pose": 0.00
}
```

### 2. 클러스터 정보
**위치**: `C:\try_angle\features\cluster_interpretation.json`
- 20개 클러스터 메타데이터 (depth, tone, brightness, 포즈 등)

### 3. YOLO11 포즈 모델
**위치**: `C:\try_angle\src\Multi\version3\yolo11s-pose.pt`
- 첫 실행 시 자동 다운로드 (19.4MB)
- 17개 키포인트 검출 (COCO format)

---

## 🚀 실행 방법

### 기본 실행
```bash
cd C:\try_angle\src\Multi\version3
"C:\Users\HS\anaconda3\envs\TA\python.exe" main_feedback.py
```

### 환경
- **Conda 환경**: TA
- **Python**: 3.10
- **주요 패키지**:
  - ultralytics 8.3.223 (YOLO11)
  - mediapipe 0.10.21
  - torch, torchvision
  - transformers, timm, open_clip_torch
  - opencv-python, scikit-learn, umap-learn

---

## 📊 시스템 구성 (2024-11-15 기준)

### 분석 파이프라인

```
이미지 입력
    ↓
[1] Feature Extraction (CLIP + OpenCLIP + DINO + MiDaS + Color)
    ↓
[2] Embedding (UMAP 128D)
    ↓
[3] Clustering (KMeans K=20)
    ↓
[4] 통합 분석
    ├─ 포즈 분석 (YOLO11 + MediaPipe)
    ├─ EXIF 추출 (ISO/조리개/셔터속도/초점거리/화이트밸런스)
    ├─ Depth 분석 (MiDaS)
    ├─ 픽셀 분석 (밝기/채도/색온도)
    └─ 구도 분석 (기울기/무게중심/프레이밍)
    ↓
[5] 비교 & 피드백 생성
```

### 우선순위 시스템

```
0순위: 클러스터 (스타일 정보)
0.5순위: 포즈 (YOLO11 + MediaPipe)
1순위: 카메라 설정 (EXIF)
2순위: 거리 (걸음수)
3순위: 밝기 (EV)
4순위: 색감 (채도/색온도)
5순위: 구도 (기울기/프레이밍/줌)
```

---

## 📝 최근 변경 이력 (2024-11-15)

### 1. 포즈 분석 시스템 추가 ✅
**파일**: `analysis/pose_analyzer.py` (신규)

**기능**:
- YOLO11-pose (17 keypoints) + MediaPipe (Pose/Face/Hands) 하이브리드
- 시나리오 자동 판단: face_closeup, full_body, upper_body, hand_gesture, back_view
- 포즈 유사도 계산 (0-100%)
- 각도 비교: 팔꿈치, 어깨, 얼굴 각도
- 위치 비교: 손목, 고개, 어깨 너비

**피드백 예시**:
- "왼팔 팔꿈치를 15도 더 펴세요"
- "얼굴을 55도 더 오른쪽으로 돌리세요"
- "고개를 위로 들어 올리세요"

**confidence 임계값**: 0.3 (0.5에서 낮춤 - 얼굴 클로즈업 대응)

### 2. EXIF 메타데이터 추출 ✅
**파일**: `analysis/exif_analyzer.py` (신규)

**추출 항목**:
- ISO, F-Number (조리개), ExposureTime (셔터속도)
- FocalLength (초점거리), WhiteBalance
- ExposureCompensation, Flash, LensModel

**비교 피드백**:
- "ISO를 400으로 설정하세요 (현재 800)"
- "조리개를 f/2.8로 설정하세요"
- "셔터속도를 1/200s로 설정하세요"

### 3. 구체적인 거리/프레이밍 피드백 ✅
**파일**: `analysis/image_comparator.py` (수정)

**추가 기능**:
- **걸음수 계산**: "약 1걸음 뒤로 가세요" (평균 걸음 70cm 기준)
- **줌 배율**: "화면을 1.3배 확대하세요"
- **프레이밍**: "화면 위쪽 10% 더 포함하세요"
- **크롭 제안**: bbox 위치 비교

**계산 로직**:
```python
# 걸음수: estimated_distance(2.5m) * depth_ratio_diff / 0.7m
# 줌 비율: user_bbox_area / ref_bbox_area
```

### 4. ImageAnalyzer 통합 업데이트 ✅
**파일**: `analysis/image_analyzer.py` (수정)

**변경**:
- `enable_pose=True` 파라미터 추가
- `enable_exif=True` 파라미터 추가
- 반환값에 `pose`, `exif` 필드 추가

### 5. ImageComparator 업데이트 ✅
**파일**: `analysis/image_comparator.py` (수정)

**변경**:
- `_compare_pose()` 메서드 추가
- `_compare_exif()` 메서드 추가
- `_compare_depth()` - 걸음수 계산 추가
- `_compare_composition()` - 줌/프레이밍 피드백 추가
- 우선순위 재조정 (0.5: 포즈, 1: EXIF)

---

## 🔍 알려진 이슈

### 1. EXIF 데이터 없음
- test1.jpg, test2.jpg에는 EXIF 데이터가 없음
- 실제 카메라 촬영 이미지에서는 정상 작동

### 2. MediaPipe 경고 메시지
- TensorFlow Lite 관련 경고 (정상 작동, 무시 가능)
- `inference_feedback_manager` 경고

### 3. Windows 인코딩 이슈
- 이모지 출력 시 cp949 인코딩 오류 가능
- main_feedback.py에서는 처리됨

---

## 💡 주요 함수 & API

### ImageAnalyzer
```python
analyzer = ImageAnalyzer(
    image_path="path/to/image.jpg",
    enable_pose=True,   # 포즈 분석 활성화
    enable_exif=True    # EXIF 추출 활성화
)
result = analyzer.analyze()
# Returns: {cluster, depth, pixels, composition, pose, exif, raw_features}
```

### ImageComparator
```python
comparator = ImageComparator(
    reference_path="ref.jpg",
    user_path="user.jpg"
)
feedback = comparator.get_prioritized_feedback()
# Returns: [{priority, category, message, detail}, ...]
```

### PoseAnalyzer
```python
analyzer = PoseAnalyzer()
pose = analyzer.analyze("image.jpg")
# Returns: {scenario, confidence, bbox, yolo_keypoints, merged_keypoints}

comparison = compare_poses(ref_pose, user_pose)
# Returns: {similarity, angle_differences, position_differences, feedback}
```

### ExifAnalyzer
```python
analyzer = ExifAnalyzer("image.jpg")
settings = analyzer.get_camera_settings()
# Returns: {iso, f_number, shutter_speed, focal_length, white_balance, ...}

comparison = compare_exif(ref_settings, user_settings)
# Returns: {iso_diff, f_number_diff, feedback, ...}
```

---

## 📌 다음 작업 후보

1. **삼분할선 기반 가이드**
   - rule of thirds 그리드 오버레이
   - "피사체를 삼분할선 교차점에 배치하세요"

2. **실시간 카메라 스트림**
   - OpenCV VideoCapture 통합
   - 프레임별 실시간 피드백

3. **모바일/웹 인터페이스**
   - Flask/FastAPI 백엔드
   - 실시간 웹캠 피드백

4. **클러스터 내 유사 이미지 추천**
   - 같은 클러스터 내 가장 유사한 N개 이미지 찾기
   - 다양한 각도의 레퍼런스 제공

5. **포즈 시퀀스 가이드**
   - 현재 포즈 → 목표 포즈까지 단계별 가이드
   - "먼저 왼팔을 올리고 → 그 다음 고개를 돌리세요"

---

## 🐛 디버깅 팁

### 오류 발생 시 체크리스트
1. ✅ TA 환경 활성화 확인
2. ✅ 경로 확인: 절대 경로 사용 (C:\try_angle\...)
3. ✅ 모델 파일 존재 확인:
   - `C:\try_angle\feature_models/` 디렉토리
   - `yolo11s-pose.pt` (자동 다운로드)
4. ✅ 이미지 경로 확인

### 로그 확인
```bash
# 상세 로그 출력
"C:\Users\HS\anaconda3\envs\TA\python.exe" main_feedback.py 2>&1 | tee log.txt
```

### 개별 모듈 테스트
```bash
# 포즈 분석만 테스트
python analysis/pose_analyzer.py

# EXIF 추출만 테스트
python analysis/exif_analyzer.py
```

---

## 📚 참고 문서

- README.md - 전체 시스템 개요
- analysis/pose_analyzer.py - 포즈 분석 상세 로직
- analysis/exif_analyzer.py - EXIF 추출 로직
- analysis/image_comparator.py - 비교 및 피드백 생성

---

**마지막 업데이트**: 2024-11-15 03:40 (KST)
**버전**: 3.0.0
**상태**: 프로덕션 (안정)
