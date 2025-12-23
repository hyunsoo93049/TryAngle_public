# 📁 맥북 (HS_MAC) 파일 구조 문서

**생성일**: 2025-11-17
**목적**: Git에 업로드되지 않는 큰 파일들의 구조를 윈도우와 공유

---

## ⚠️ 중요 사항

이 문서는 `.gitignore`로 인해 Git에 **업로드되지 않는** 파일들을 설명합니다.
윈도우 환경과 비교할 때 이 파일들이 맥에만 존재하거나, 다른 위치에 있을 수 있습니다.

---

## 📊 맥북에만 있는 파일들

### 1. MoveNet 모델 (맥에서 다운로드)

**위치**: `/Users/hyunsoo/Try_Angle/src/Multi/version3/models/`

```
models/
├── movenet_thunder.tflite                    # 12MB
└── movenet_thunder_saved_model/              # 35MB
    ├── fingerprint.pb                        # 56B
    ├── saved_model.pb                        # 11MB
    └── variables/
        ├── variables.data-00000-of-00001     # 24MB
        └── variables.index                   # 18KB
```

**생성 방법**: 맥에서 `python scripts/download_movenet.py` 실행
**Git 상태**: `.gitignore`에 의해 제외됨
**윈도우 필요 여부**: 선택 사항 (Phase 2 MoveNet 사용 시 필요)

---

### 2. 대조학습 모델 (Phase 3)

**위치**: `/Users/hyunsoo/Try_Angle/src/Multi/version3/models/contrastive/`

```
contrastive/
├── best_model.pth                # 282MB (최고 성능 모델, Epoch 6)
├── final_model.pth               # 94MB (최종 모델, Epoch 50)
├── checkpoint_epoch_10.pth       # 282MB
├── checkpoint_epoch_20.pth       # 282MB
├── checkpoint_epoch_30.pth       # 282MB
├── checkpoint_epoch_40.pth       # 282MB
├── checkpoint_epoch_50.pth       # 282MB
└── training_history.json         # 5.7KB (학습 이력)
```

**생성 방법**:
1. `python scripts/prepare_contrastive_data.py` (데이터 준비)
2. `python scripts/train_contrastive.py` (모델 훈련 2-3시간)

**Git 상태**: `.gitignore`에 의해 제외됨 (*.pth)
**윈도우 상태**: 윈도우에서 훈련했다면 동일한 파일들이 있어야 함

---

### 3. YOLO 포즈 모델

**위치**: 여러 곳에 중복 존재

```
/Users/hyunsoo/Try_Angle/
├── yolo11s-pose.pt               # 19MB (루트)
├── backend/yolo11s-pose.pt       # 19MB (백엔드용)
└── yolo_models/
    ├── yolo11s.pt                # 22MB
    ├── yolo11s-pose.pt           # 19MB
    └── yolo11s-seg.pt            # 23MB
```

**Git 상태**: `.gitignore`에 의해 제외됨 (*.pt)
**윈도우 상태**: 윈도우에도 동일하게 있어야 함

---

### 4. 특징 추출 모델 (Feature Models)

**위치**: `/Users/hyunsoo/Try_Angle/feature_models/`

```
feature_models/
├── feature_models_v3/            # 20MB (CLIP/OpenCLIP/DINO 모델)
├── features/                     # 32MB (추출된 특징 파일)
├── backup_k10/                   # 19MB (백업)
├── backup_py39/                  # 20MB (백업)
├── fusion_128d.npy               # 1.3MB (융합 임베딩)
├── kmeans_model.pkl              # 24KB (K=20 클러스터)
├── kmeans_centroids.npy          # 12KB
├── umap_128d_model.joblib        # 18MB (차원 축소)
├── scaler_*.joblib               # 각 4-12KB (정규화 모델)
├── cluster_info.json             # 4KB (클러스터 해석)
├── cluster_labels.npy            # 12KB
├── optimization_results.json     # 28KB
└── weights.json                  # 4KB
```

**Git 상태**: 전체 폴더가 `.gitignore`에 의해 제외됨
**윈도우 상태**: 윈도우에도 동일하게 있어야 함 (공유 필요)

---

### 5. Features 폴더

**위치**: `/Users/hyunsoo/Try_Angle/features/` (루트)

**크기**: 19MB
**내용**: 추출된 이미지 특징 캐시 파일들
**Git 상태**: 전체 폴더가 `.gitignore`에 의해 제외됨

---

### 6. 대조학습 데이터셋

**위치**: `/Users/hyunsoo/Try_Angle/data/contrastive_dataset/`

```
contrastive_dataset/
├── train/
│   └── pairs.json                # 훈련 데이터 쌍 (1600개)
└── val/
    └── pairs.json                # 검증 데이터 쌍 (400개)
```

**생성 방법**: `python scripts/prepare_contrastive_data.py`
**Git 상태**: JSON 파일은 업로드 가능하지만, 이미지 파일들은 제외
**윈도우 상태**: 윈도우에서 생성했다면 동일하게 있어야 함

---

## 📂 전체 디렉토리 구조 (Git 제외 파일 포함)

```
/Users/hyunsoo/Try_Angle/
│
├── src/Multi/version3/
│   ├── models/                   ❌ Git 제외
│   │   ├── movenet_thunder.tflite (12MB)
│   │   ├── movenet_thunder_saved_model/ (35MB)
│   │   └── contrastive/          (1.5GB 총합)
│   │       ├── best_model.pth (282MB) ⭐ 필수
│   │       ├── final_model.pth (94MB)
│   │       ├── checkpoint_epoch_*.pth (각 282MB)
│   │       └── training_history.json (5.7KB)
│   │
│   ├── analysis/                 ✅ Git 포함
│   ├── contrastive/              ✅ Git 포함 (코드만)
│   ├── feature_extraction/       ✅ Git 포함
│   ├── matching/                 ✅ Git 포함
│   ├── scripts/                  ✅ Git 포함
│   ├── utils/                    ✅ Git 포함
│   └── *.py                      ✅ Git 포함
│
├── feature_models/               ❌ Git 제외 (110MB)
│   ├── feature_models_v3/
│   ├── features/
│   ├── kmeans_model.pkl
│   ├── umap_128d_model.joblib
│   └── scaler_*.joblib
│
├── features/                     ❌ Git 제외 (19MB)
│
├── data/
│   ├── contrastive_dataset/      ❌ Git 제외 (데이터셋)
│   ├── test_images/              ✅ Git 포함 (작은 이미지)
│   └── sample_images/            ❌ Git 제외
│
├── yolo_models/                  ❌ Git 제외 (64MB)
│   ├── yolo11s.pt
│   ├── yolo11s-pose.pt
│   └── yolo11s-seg.pt
│
├── backend/
│   ├── main.py                   ✅ Git 포함
│   └── yolo11s-pose.pt           ❌ Git 제외
│
├── ios/                          ✅ Git 포함
│
├── docs/                         ✅ Git 포함
├── archive/                      ✅ Git 포함
├── external_projects/            ❌ Git 제외
│
├── yolo11s-pose.pt               ❌ Git 제외 (루트)
├── README.md                     ✅ Git 포함
├── .gitignore                    ✅ Git 포함
└── MAC_FILE_STRUCTURE.md         ✅ Git 포함 (이 파일)
```

---

## 🔄 윈도우와 맥 파일 비교 가이드

### Git으로 동기화되는 것 ✅
- 모든 Python 코드 (*.py)
- 설정 파일 (*.yaml, *.json)
- 문서 (*.md)
- iOS 코드 (*.swift)
- 작은 테스트 이미지 (data/test_images/)

### Git으로 동기화 안 되는 것 ❌
- 모델 파일 (*.pt, *.pth, *.tflite)
- 특징 파일 (*.npz, *.npy, *.pkl, *.joblib)
- 대용량 데이터셋 (clustered_images/, contrastive_dataset/)
- 결과 폴더 (outputs/, results/)

### 윈도우에서 확인해야 할 것
1. **대조학습 모델**: `version3/models/contrastive/best_model.pth` (282MB) 있는지?
2. **YOLO 모델**: `yolo11s-pose.pt` (19MB) 있는지?
3. **특징 모델**: `feature_models/` 폴더 전체 있는지?
4. **대조학습 데이터**: `data/contrastive_dataset/` 있는지?

---

## 💾 파일 공유 방법 (윈도우 ↔ 맥)

### 옵션 1: 압축 파일로 공유
```bash
# 맥에서 필수 모델만 압축
tar -czf mac_models.tar.gz \
  src/Multi/version3/models/contrastive/best_model.pth \
  src/Multi/version3/models/contrastive/training_history.json \
  feature_models/ \
  features/

# 윈도우로 이동 후 압축 해제
tar -xzf mac_models.tar.gz
```

### 옵션 2: Google Drive / Dropbox
- 대용량 파일은 클라우드에 업로드
- 링크 공유

### 옵션 3: Git LFS
- 대용량 파일도 Git으로 관리
- 설정 필요

---

## 📝 주의사항

### 1. 맥 전용 파일
- `movenet_thunder.tflite` - 맥에서만 다운로드
- 윈도우 필요 시: `python scripts/download_movenet.py` 실행

### 2. 공통 파일 (양쪽 모두 필요)
- `feature_models/` - 클러스터링에 필수
- `features/` - 특징 캐시
- `yolo11s-pose.pt` - 포즈 분석에 필수
- `contrastive/best_model.pth` - Phase 3 필요 시

### 3. 윈도우 전용 파일
- 윈도우에만 있는 파일이 있다면 이 문서에 추가 필요

---

## 🔍 파일 크기 요약

| 항목 | 크기 | Git 상태 | 필수 여부 |
|------|------|---------|----------|
| movenet_thunder.tflite | 12MB | ❌ 제외 | 선택 (Phase 2) |
| movenet_thunder_saved_model | 35MB | ❌ 제외 | 선택 |
| contrastive/best_model.pth | 282MB | ❌ 제외 | 필수 (Phase 3) |
| contrastive/checkpoints | 1.4GB | ❌ 제외 | 선택 |
| yolo11s-pose.pt | 19MB | ❌ 제외 | 필수 |
| feature_models/ | 110MB | ❌ 제외 | 필수 |
| features/ | 19MB | ❌ 제외 | 선택 (캐시) |

**총 크기 (Git 제외)**: ~1.8GB

---

## ✅ 체크리스트 (윈도우에서 확인용)

맥과 동일한 환경을 만들려면:

- [ ] `feature_models/` 폴더가 있는가?
- [ ] `yolo11s-pose.pt` 파일이 있는가?
- [ ] `contrastive/best_model.pth` 파일이 있는가?
- [ ] `data/contrastive_dataset/` 폴더가 있는가?
- [ ] Phase 2 사용 시: `movenet_thunder.tflite` 다운로드

---

**생성 환경**: macOS 14.6 (M4)
**Python 환경**: TA (conda, Python 3.11)
**마지막 업데이트**: 2025-11-17
