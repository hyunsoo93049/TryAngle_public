# TryAngle version3 - 프로덕션 시스템

AI 기반 실시간 사진 촬영 가이드 시스템 (프로덕션 버전)

## 📁 폴더 구조

```
version3/
├── analysis/               # 이미지 분석 & 비교
│   ├── image_analyzer.py      # 단일 이미지 분석 (클러스터 예측 + 측정)
│   └── image_comparator.py    # 레퍼런스 vs 사용자 이미지 비교
│
├── feature_extraction/    # 특징 추출
│   ├── feature_extractor.py   # CLIP+OpenCLIP+DINO+MiDaS+Color 통합 추출기
│   └── feature_extractor_v2.py
│
├── embedder/              # 임베딩 변환
│   └── embedder.py           # 특징 → 128D 임베딩 (UMAP)
│
├── matching/              # 클러스터 매칭
│   └── cluster_matcher.py    # 임베딩 → 클러스터 ID + 라벨
│
├── training/              # 모델 재학습
│   ├── retrain_clustering.py # KMeans 재학습
│   └── retrain_features.py   # 특징 재추출
│
├── scripts_archive/       # 학습/유틸리티 스크립트 (완료됨)
│   ├── auto_optimize_clustering.py
│   ├── auto_label_and_split.py
│   ├── train_embedding_clusters.py
│   └── create_cluster_interpretation*.py
│
├── main_feedback.py       # 🎯 메인 실행 파일
└── main_test.py          # 테스트 스크립트
```

## 🚀 사용 방법

### 1. 필수 패키지 설치

```bash
pip install torch torchvision
pip install transformers timm open_clip_torch
pip install opencv-python scikit-learn umap-learn
pip install numpy pillow scipy scikit-image
```

### 2. 피드백 시스템 실행

```bash
cd C:\try_angle\src\Multi\version3
python main_feedback.py
```

### 3. 커스터마이징

`main_feedback.py` 파일에서 이미지 경로 수정:

```python
reference_path = r"C:\try_angle\data\clustered_images\cluster_5\IMG_0561.JPG"
user_path = r"C:\try_angle\data\test_images\test1.jpg"
```

## 🎯 시스템 구조

### 파이프라인 (5단계)

```
이미지 입력
    ↓
[1] Feature Extraction (feature_extractor.py)
    ├─ CLIP (512D)
    ├─ OpenCLIP (512D)
    ├─ DINOv2 (384D)
    ├─ MiDaS (2D)
    └─ Color/Texture (119D)
    ↓
[2] Embedding (embedder.py)
    ├─ Scaling (RobustScaler)
    ├─ Concatenation (1529D)
    └─ UMAP (128D)
    ↓
[3] Clustering (cluster_matcher.py)
    └─ KMeans (K=20)
    ↓
[4] Analysis (image_analyzer.py)
    ├─ Cluster info
    ├─ Depth analysis
    ├─ Pixel analysis
    └─ Composition analysis
    ↓
[5] Comparison & Feedback (image_comparator.py)
    └─ Prioritized feedback generation
```

### 주요 모듈

#### 1. `image_analyzer.py`
- **입력**: 이미지 경로
- **출력**: 클러스터 ID, depth, brightness, saturation, composition 등
- **기능**: 단일 이미지 분석 및 측정 가능한 값 추출

#### 2. `image_comparator.py`
- **입력**: 레퍼런스 이미지 + 사용자 이미지
- **출력**: 우선순위 피드백 리스트
- **기능**:
  - 클러스터 비교 (스타일 DNA)
  - Depth 비교 (거리 조정)
  - Brightness 비교 (노출 조정)
  - Color 비교 (색감/채도)
  - Composition 비교 (구도/기울기)

#### 3. `main_feedback.py`
- 전체 시스템 통합 실행
- 우선순위 기반 피드백 출력 (최대 3개 + 1개 정보)

## 📊 모델 정보

### 사용 중인 모델 (K=20)

- **위치**: `C:\try_angle\feature_models/`
- **클러스터 수**: 20개
- **Silhouette Score**: 0.3988
- **최적 가중치**:
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

### 모델 파일

```
feature_models/
├── kmeans_model.pkl          # KMeans 모델 (K=20)
├── kmeans_centroids.npy      # 클러스터 중심 (20, 128)
├── umap_128d_model.joblib    # UMAP 차원 축소 모델
├── scaler_clip.joblib        # CLIP 정규화
├── scaler_openclip.joblib    # OpenCLIP 정규화
├── scaler_dino.joblib        # DINO 정규화
├── scaler_color.joblib       # Color 정규화
├── scaler_midas.joblib       # MiDaS 정규화
├── cluster_labels.npy        # 전체 데이터 클러스터 라벨
├── fusion_128d.npy           # 128D 임베딩 (전체 데이터)
└── weights.json              # 최적 가중치
```

### 클러스터 정보

- **파일**: `C:\try_angle\features\cluster_interpretation.json`
- **내용**: 각 클러스터의 라벨, depth, tone, brightness 등
- **형식**:
  ```json
  {
    "0": {
      "cluster_id": 0,
      "depth_mean": 1050.0,
      "depth_label": "실외 / 멀리",
      "dominant_pose": "반신",
      "tone": "쿨톤",
      "brightness": "중간",
      "auto_label": "실외 / 멀리, 쿨톤, 중간, 반신",
      "sample_count": 84
    },
    ...
  }
  ```

## 📝 백업 정보

### K=10 구버전 모델

- **위치**: `C:\try_angle\feature_models\backup_k10/`
- **설명**: 이전 K=10 클러스터링 모델 (참고용 보관)

### 학습 스크립트

- **위치**: `scripts_archive/`
- **설명**: 모델 학습 및 최적화에 사용된 스크립트들
  - `auto_optimize_clustering.py`: 그리드 서치로 최적 K, 가중치 탐색
  - `train_embedding_clusters.py`: 클러스터링 모델 학습
  - `auto_label_and_split.py`: 자동 라벨링 및 폴더 분리

## 🔧 개발 가이드

### 새 특징 추가

1. `feature_extraction/feature_extractor.py`에 특징 추출 추가
2. `embedder/embedder.py`에 scaler 추가
3. 모델 재학습: `training/retrain_clustering.py`
4. `cluster_interpretation.json` 업데이트

### 클러스터 수 변경

1. `scripts_archive/auto_optimize_clustering.py` 실행
2. 최적 K 찾기
3. `training/retrain_clustering.py`에서 K 수정 후 재학습
4. 모델 파일 교체

### 피드백 로직 수정

- `analysis/image_comparator.py`의 `get_prioritized_feedback()` 함수 수정
- 우선순위: 0(정보) → 1(거리) → 2(밝기) → 3(색감) → 4(구도)

## ⚠️ 주의사항

1. **경로 설정**: 현재 Windows 절대 경로로 하드코딩됨
2. **GPU 필요**: CUDA 사용 가능하면 자동으로 GPU 사용
3. **메모리**: 모델 로딩 시 약 2-3GB RAM 필요
4. **첫 실행**: 모델 다운로드로 인해 느릴 수 있음

## 📚 참고

- **기반 시스템**: `src/Multi/version2/`
- **레퍼런스 데이터**: `data/clustered_images/` (K=20, cluster_00~19)
- **테스트 이미지**: `data/test_images/`

## 🎉 완료된 작업 (2024-11-14)

1. ✅ K=20 클러스터링 모델 학습 완료 (Silhouette 0.3988)
2. ✅ 모델 파일 정리 (feature_models/ 업데이트)
3. ✅ cluster_interpretation.json 생성 (K=20)
4. ✅ 불필요한 파일 정리 (v2_backup 제거, scripts 아카이브)
5. ✅ 폴더 구조 정리 및 문서화

## 🚀 다음 단계

1. 필수 패키지 설치 후 `main_feedback.py` 실행 테스트
2. 실제 사용자 이미지로 피드백 품질 검증
3. 모바일/웹 인터페이스 개발 검토
