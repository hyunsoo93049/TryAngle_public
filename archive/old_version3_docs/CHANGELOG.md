# Changelog - TryAngle Version3

변경 이력 추적용 문서 (Claude Code ↔ GPT 컨텍스트 공유)

---

## [3.0.0] - 2024-11-15 03:40 (KST)

### ✨ 추가됨 (Added)

#### 1. 포즈 분석 시스템
- **파일**: `analysis/pose_analyzer.py` (신규 생성)
- **함수**:
  - `PoseAnalyzer.__init__(yolo_model_path)` - YOLO11 + MediaPipe 초기화
  - `PoseAnalyzer.analyze(image_path)` - 포즈 검출 및 시나리오 판단
  - `compare_poses(ref_pose, user_pose)` - 포즈 비교 및 유사도 계산
  - `_compare_angles()` - 관절 각도 비교
  - `_compare_positions()` - 키포인트 위치 비교
  - `_generate_pose_feedback()` - 구체적인 피드백 생성
- **특징**:
  - YOLO11: 17개 키포인트 (전신/뒷모습)
  - MediaPipe Pose: 33개 키포인트 (디테일)
  - MediaPipe Face: 468개 키포인트 (얼굴 클로즈업)
  - MediaPipe Hands: 21개 키포인트/손
- **시나리오**: face_closeup, full_body, upper_body, hand_gesture, back_view
- **confidence 임계값**: 0.3 (얼굴 클로즈업 대응)

#### 2. EXIF 메타데이터 추출
- **파일**: `analysis/exif_analyzer.py` (신규 생성)
- **클래스**: `ExifAnalyzer`
- **추출 데이터**:
  - ISO, F-Number, ExposureTime, FocalLength
  - WhiteBalance, ExposureProgram, ExposureCompensation
  - Flash, LensModel, Camera Make/Model
- **함수**:
  - `get_camera_settings()` - 카메라 설정 딕셔너리 반환
  - `get_shooting_info()` - 사람이 읽기 쉬운 형식 반환
  - `compare_exif(ref, user)` - 설정 비교 및 피드백

#### 3. 구체적인 거리 피드백
- **파일**: `analysis/image_comparator.py` (수정)
- **메서드**: `_compare_depth()`
- **추가 기능**:
  - 걸음수 계산: `steps = distance_change_m / 0.7`
  - 평균 걸음: 70cm
  - 평균 촬영 거리: 2.5m 가정
- **피드백 예시**: "약 1걸음 뒤로 가세요"

#### 4. 프레이밍/줌 피드백
- **파일**: `analysis/image_comparator.py` (수정)
- **메서드**: `_compare_composition()`
- **추가 기능**:
  - bbox 기반 줌 비율 계산
  - 프레이밍 위치 비교 (x, y shift)
  - 구체적인 조정 제안
- **피드백 예시**:
  - "화면을 1.3배 확대하세요"
  - "프레이밍: 화면 위쪽 10% 더 포함하세요"

### 🔧 변경됨 (Changed)

#### 1. ImageAnalyzer 통합 업데이트
- **파일**: `analysis/image_analyzer.py`
- **변경**:
  - 생성자에 `enable_pose=True` 파라미터 추가
  - 생성자에 `enable_exif=True` 파라미터 추가
  - `analyze()` 반환값에 `pose`, `exif` 필드 추가
- **Before**:
  ```python
  def __init__(self, image_path: str)
  # Returns: {cluster, depth, pixels, composition, raw_features}
  ```
- **After**:
  ```python
  def __init__(self, image_path: str, enable_pose=True, enable_exif=True)
  # Returns: {cluster, depth, pixels, composition, pose, exif, raw_features}
  ```

#### 2. ImageComparator 우선순위 재조정
- **파일**: `analysis/image_comparator.py`
- **변경**:
  - 0순위: 클러스터 (동일)
  - **0.5순위: 포즈 (신규)**
  - **1순위: 카메라 설정 (신규)**
  - 2순위: 거리 (이전 1순위)
  - 3순위: 밝기 (이전 2순위)
  - 4순위: 색감 (이전 3순위)
  - 5순위: 구도 (이전 4순위)

#### 3. 포즈 비교 confidence 임계값 조정
- **파일**: `analysis/pose_analyzer.py`
- **함수**: `_compare_angles()`, `_compare_positions()`
- **Before**: 0.5 (고정)
- **After**: 0.3 (기본값), 함수 파라미터로 조정 가능
- **이유**: 얼굴 클로즈업에서 팔이 화면 밖에 있을 때 대응

#### 4. main_feedback.py 출력 추가
- **파일**: `main_feedback.py`
- **추가 출력**:
  - 포즈 정보 (유사도, 각도 차이, 위치 차이)
  - EXIF 설정 비교 (ISO, 조리개, 셔터속도, 초점거리)

### 🐛 수정됨 (Fixed)

#### 1. 포즈 유사도 0% 버그 수정
- **문제**: 얼굴 클로즈업에서 포즈 유사도가 항상 0%로 표시
- **원인**: confidence 임계값 0.5가 너무 높아 팔꿈치/손목 검출 실패
- **해결**: 임계값을 0.3으로 낮춤, 얼굴 키포인트 비교 추가
- **결과**: 유사도 68.58% 정상 계산

#### 2. 걸음수 계산 버그 수정
- **문제**: `steps` 변수가 action="none"일 때 정의되지 않음
- **해결**: 반환 딕셔너리에 `steps` 필드 추가, 0으로 초기화

---

## [2.0.0] - 2024-11-14 00:00 (KST) (이전 버전)

### ✨ 추가됨
- K=20 클러스터링 모델 학습 완료 (Silhouette 0.3988)
- cluster_interpretation.json 생성 (K=20)
- 모델 파일 정리 (feature_models/ 업데이트)

### 🗑️ 제거됨
- K=10 구버전 모델 (backup_k10/로 이동)
- 불필요한 v2_backup 파일

---

## 파일 변경 요약

### 신규 생성 (New)
```
analysis/pose_analyzer.py          # 포즈 분석 (691 lines)
analysis/exif_analyzer.py          # EXIF 추출 (394 lines)
META_CONTEXT.md                    # 메타 컨텍스트 (이 문서)
CHANGELOG.md                       # 변경 이력 추적
```

### 수정됨 (Modified)
```
analysis/image_analyzer.py         # +pose +exif 통합
analysis/image_comparator.py       # +pose +exif +걸음수 +줌/프레이밍
main_feedback.py                   # +pose +exif 출력
```

### 변경 없음 (Unchanged)
```
feature_extraction/                # 특징 추출 (동일)
embedder/                          # 임베딩 (동일)
matching/                          # 클러스터 매칭 (동일)
training/                          # 학습 스크립트 (동일)
```

---

## 데이터 & 모델 상태

### 모델 파일 (feature_models/)
- ✅ kmeans_model.pkl - K=20, 최신
- ✅ umap_128d_model.joblib - 128D, 최신
- ✅ scaler_*.joblib (5개) - 최신
- ✅ weights.json - 최신

### YOLO 모델
- ✅ yolo11s-pose.pt - 19.4MB, 자동 다운로드

### 클러스터 정보
- ✅ features/cluster_interpretation.json - K=20, 최신

---

## 테스트 상태

### ✅ 통과한 테스트
- [x] 포즈 분석 단독 테스트 (test_pose.py)
- [x] 통합 시스템 테스트 (test_integrated.py)
- [x] main_feedback.py 전체 실행
- [x] EXIF 없는 이미지 처리 (test1.jpg, test2.jpg)
- [x] 포즈 유사도 계산 (68.58%)
- [x] 걸음수 계산 (1걸음)

### ⚠️ 알려진 제한사항
- EXIF 데이터가 없는 이미지는 카메라 설정 피드백 생략
- MediaPipe 경고 메시지 (무시 가능)
- Windows cp949 인코딩 이슈 (main_feedback.py에서 처리)

---

## 성능 지표

### 처리 시간 (TA 환경, CPU)
- 이미지 분석: ~5초 (특징 추출 + 클러스터링)
- 포즈 검출: ~2초 (YOLO + MediaPipe)
- EXIF 추출: <0.1초
- 전체 비교: ~10-12초 (2개 이미지)

### 정확도
- 클러스터링 Silhouette: 0.3988
- 포즈 검출 Confidence: 0.91-0.95
- 포즈 유사도: 68.58% (test1 vs test2)

---

**마지막 업데이트**: 2024-11-15 03:40 (KST)
**다음 변경 시 이 파일도 업데이트하세요!**
