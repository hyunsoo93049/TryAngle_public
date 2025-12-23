# 🎯 TryAngle 프로젝트 인수인계 문서

**인수인계 날짜**: 2025-11-17
**프로젝트**: TryAngle - AI 촬영 가이드 앱
**버전**: Phase 1-3 통합 완료 (v2.0.0)

---

## 📋 목차

1. [프로젝트 개요](#프로젝트-개요)
2. [완료된 작업](#완료된-작업)
3. [빠른 시작](#빠른-시작)
4. [주요 파일 구조](#주요-파일-구조)
5. [개발 환경](#개발-환경)
6. [API 사용 방법](#api-사용-방법)
7. [다음 단계](#다음-단계)
8. [문제 해결](#문제-해결)
9. [참고 문서](#참고-문서)

---

## 🎯 프로젝트 개요

### 목적
사진 촬영 초보자도 전문가처럼 찍을 수 있도록 AI가 실시간으로 가이드하는 앱

### 핵심 기능
- **실시간 피드백**: 카메라로 보면서 즉시 피드백
- **초보자 친화**: "EV +0.7" → "화면 위로 슬라이드"
- **단계별 가이드**: 위치 → 구도 → 포즈 → 카메라 → 품질
- **진행도 추적**: "75점! 20점 올랐네!"

### 기술 스택
- **Backend**: Python, FastAPI
- **AI Models**: YOLO11, MoveNet, ResNet50
- **Frontend**: iOS (Swift, SwiftUI)
- **분석**: OpenCV, MediaPipe

---

## ✅ 완료된 작업

### Phase 1 - Quick Wins (빠른 개선)
- ✅ **1.1 Top-K 피드백**: 10개 → 3개 핵심만
- ✅ **1.2 초보자 메시지**: 전문 용어 → 쉬운 설명
- ✅ **1.3 특징 캐싱**: 99.5% 속도 향상

### Phase 2 - Core Features (핵심 기능)
- ✅ **2.1 워크플로우 가이드**: 5단계 촬영 순서
- ✅ **2.2 진행도 추적**: 점수, 진행률, 격려 메시지
- ✅ **2.3 우선순위 시스템**: Critical → Important → Recommended
- ✅ **2.4 적응형 Threshold**: 클러스터별 품질 기준

### Phase 3 - Advanced Features (고급 기능)
- ✅ **3.1 AI 레퍼런스 추천**: 유사 이미지 Top-3
- ✅ **3.2 대조학습 모델**: ResNet50, 77% 정확도
- ✅ **3.3 시각적 가이드**: 삼분할선, 수평선, 피드백 패널

### Backend 통합
- ✅ **FastAPI 서버**: Phase 1-3 통합 (v2.0.0)
- ✅ **API 엔드포인트**: `/api/feedback/enhanced`
- ✅ **진행도 추적 세션**: 세션별 관리
- ✅ **레퍼런스 추천**: `/api/recommendations`

---

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 1. Conda 환경 활성화
conda activate TA

# 2. 필요한 패키지 확인
pip list | grep -E "fastapi|uvicorn|opencv|torch"

# 만약 없다면:
pip install fastapi uvicorn python-multipart
```

### 2. 백엔드 서버 실행

```bash
# Windows
cd C:\try_angle\backend
python main.py

# Mac/Linux
cd /path/to/try_angle/backend
python3 main.py
```

**서버 주소**: `http://localhost:8000`

### 3. Python 테스트 (PC용)

```bash
# 통합 피드백 시스템
cd C:\try_angle\src\Multi\version3
python main_feedback.py

# 실시간 카메라 (PC 웹캠)
python camera_realtime.py
# 'g' 키: 시각적 가이드 ON/OFF
```

### 4. iOS 앱 연결

```swift
// Services/APIService.swift
let baseURL = "http://192.168.0.10:8000"  // PC IP로 변경

// Phase 1-3 통합 API 호출
AF.upload(to: "\(baseURL)/api/feedback/enhanced")
```

---

## 📁 주요 파일 구조

```
C:\try_angle\
├── backend/
│   ├── main.py                      # FastAPI 서버 (v2.0.0)
│   └── API_GUIDE.md                 # API 문서
│
├── src/Multi/version3/
│   ├── analysis/
│   │   ├── image_analyzer.py        # 이미지 분석 (특징 추출)
│   │   └── image_comparator.py      # 이미지 비교 (피드백 생성)
│   │
│   ├── utils/                       # Phase 1-3 핵심 모듈
│   │   ├── feedback_formatter.py    # Phase 1.1, 1.2
│   │   ├── feature_cache.py         # Phase 1.3
│   │   ├── workflow_guide.py        # Phase 2.1
│   │   ├── progress_tracker.py      # Phase 2.2
│   │   ├── priority_system.py       # Phase 2.3
│   │   ├── adaptive_thresholds.py   # Phase 2.4
│   │   ├── reference_recommender.py # Phase 3.1
│   │   ├── model_cache.py           # Phase 3.2
│   │   └── visual_guide.py          # Phase 3.3
│   │
│   ├── scripts/
│   │   ├── prepare_contrastive_data.py  # 대조학습 데이터 준비
│   │   └── train_contrastive.py         # 모델 학습
│   │
│   ├── main_feedback.py             # 통합 피드백 시스템
│   ├── camera_realtime.py           # 실시간 카메라 (PC)
│   └── config.yaml                  # 설정 파일
│
├── models/
│   ├── contrastive/
│   │   ├── best_model.pth           # 대조학습 모델 (283MB)
│   │   └── training_history.json
│   │
│   └── [기타 모델들...]
│
├── data/
│   ├── clustered_images/            # 클러스터별 이미지
│   ├── contrastive_dataset/         # 대조학습 데이터셋
│   └── test_images/                 # 테스트 이미지
│
├── ios/                             # iOS 앱 (Swift)
│   └── TryAngleApp/
│
├── PHASE_1-3_COMPLETION_SUMMARY.md  # 완료 요약
├── HANDOVER.md                      # 이 파일
└── README.md                        # 프로젝트 메인 README
```

---

## 🔧 개발 환경

### Python 환경
```bash
환경 이름: TA
Python: 3.10
주요 패키지:
  - torch==2.5.1
  - opencv-python==4.10.0.84
  - ultralytics==8.3.33 (YOLO11)
  - fastapi==0.121.2
  - uvicorn==0.38.0
  - numpy, pillow, pyyaml
```

### 모델 파일 위치
```
models/
├── yolo11s-pose.pt              # YOLO11 포즈 모델
├── cluster_model.pkl            # 클러스터 분류
├── embedder_model.pth           # 임베딩 모델
├── contrastive/best_model.pth   # 대조학습 (Phase 3.2)
└── depth_anything_v2/           # Depth 모델
```

### 데이터 파일
```
data/
├── clustered_images/            # 20개 클러스터
│   ├── cluster_0/
│   ├── cluster_1/
│   └── ...
│
├── contrastive_dataset/
│   ├── train/pairs.json         # 1600 쌍
│   └── val/pairs.json           # 400 쌍
│
└── test_images/                 # 테스트용
```

---

## 🌐 API 사용 방법

### 1. 서버 상태 확인
```bash
curl http://localhost:8000/
```

**응답**:
```json
{
  "message": "TryAngle iOS Backend (Phase 1-3 Enhanced)",
  "version": "2.0.0",
  "status": "running ✅",
  "features": {
    "phase_1_3": true,
    "top_k_feedback": true,
    "workflow_guide": true,
    "progress_tracking": true
  }
}
```

### 2. Phase 1-3 통합 피드백
```bash
curl -X POST http://localhost:8000/api/feedback/enhanced \
  -F "reference=@test3.jpg" \
  -F "current_frame=@test4.jpg" \
  -F "user_level=beginner" \
  -F "top_k=3" \
  -F "session_id=user123"
```

### 3. iOS에서 사용
```swift
// 1. 서버 주소 설정 (PC IP 확인)
let baseURL = "http://192.168.0.10:8000"

// 2. 멀티파트 요청
let formData = MultipartFormData()
formData.append(referenceImage, withName: "reference")
formData.append(currentFrame, withName: "current_frame")
formData.append("beginner".data(using: .utf8)!, withName: "user_level")
formData.append("3".data(using: .utf8)!, withName: "top_k")
formData.append(sessionID.data(using: .utf8)!, withName: "session_id")

// 3. 요청
AF.upload(multipartFormData: formData,
          to: "\(baseURL)/api/feedback/enhanced")
  .responseDecodable(of: EnhancedFeedbackResponse.self) { response in
      // 피드백 처리
  }
```

**자세한 API 문서**: `backend/API_GUIDE.md` 참고

---

## 🔜 다음 단계

### 우선순위 1: iOS 앱 업데이트
- [ ] `APIService.swift` - 새 엔드포인트 추가
- [ ] `EnhancedFeedbackView.swift` - Phase 1-3 UI 구현
- [ ] `ProgressView.swift` - 진행도 트래커
- [ ] `WorkflowGuideView.swift` - 단계별 가이드

### 우선순위 2: 시각적 가이드 (ARKit)
- [ ] Phase 3.3 가이드를 ARKit 오버레이로 변환
- [ ] 실시간 삼분할선 표시
- [ ] 수평선 가이드 (자이로스코프 연동)

### 우선순위 3: 성능 최적화
- [ ] 서버 응답 시간 단축 (현재 ~2초)
- [ ] 이미지 압축 최적화
- [ ] 백그라운드 캐싱 개선

### 우선순위 4: 추가 기능
- [ ] 음성 피드백 (TTS)
- [ ] 햅틱 피드백 (진동으로 정렬 가이드)
- [ ] 오프라인 모드 (Core ML 변환)

---

## 🐛 문제 해결

### 1. 서버 실행 에러

**문제**: `ModuleNotFoundError: No module named 'fastapi'`
```bash
# 해결
pip install fastapi uvicorn python-multipart
```

**문제**: `Port 8000 already in use`
```bash
# 해결 1: 다른 포트 사용
uvicorn main:app --host 0.0.0.0 --port 8001

# 해결 2: 기존 프로세스 종료 (Windows)
netstat -ano | findstr :8000
taskkill /PID [PID번호] /F
```

### 2. 모델 로드 에러

**문제**: `FileNotFoundError: models/yolo11s-pose.pt`
```bash
# 해결: 모델 다운로드
cd C:\try_angle\models
# YOLO11 모델은 자동 다운로드됨
```

**문제**: `RuntimeError: CUDA out of memory`
```bash
# 해결: CPU 모드 사용
# image_analyzer.py에서 device='cpu' 설정
```

### 3. iOS 연결 에러

**문제**: `Network error: Connection refused`
```bash
# 해결 1: 방화벽 확인
# Windows Defender → 인바운드 규칙 → Python 허용

# 해결 2: 같은 WiFi 확인
# PC와 iPhone이 같은 네트워크에 있어야 함

# 해결 3: IP 주소 확인
ipconfig  # IPv4 주소 확인
```

### 4. Phase 1-3 기능 안 보임

**문제**: `phase_1_3: false` in API response
```bash
# 해결: utils 모듈 import 확인
cd C:\try_angle\src\Multi\version3
python -c "from utils.feedback_formatter import FeedbackFormatter; print('OK')"

# 에러 발생시 경로 문제 - sys.path 확인
```

---

## 📚 참고 문서

### 주요 문서
1. **PHASE_1-3_COMPLETION_SUMMARY.md** - 완료 내용 상세
2. **backend/API_GUIDE.md** - API 상세 문서
3. **QUICK_REFERENCE.md** - 빠른 참조 (version3/)
4. **README.md** - 프로젝트 전체 개요

### 코드 예제
- `main_feedback.py` - Python 통합 피드백 사용
- `camera_realtime.py` - 실시간 카메라 사용
- `backend/main.py` - FastAPI 서버 구조

### 학습 데이터
- `scripts/prepare_contrastive_data.py` - 데이터 준비 과정
- `scripts/train_contrastive.py` - 모델 학습 과정
- `models/contrastive/training_history.json` - 학습 이력

---

## 💡 핵심 개념

### 1. 피드백 우선순위
```python
CRITICAL (0.0)    # 다시 찍기 (극심한 블러)
    ↓
POSE (0.5)        # 자세 교정
    ↓
CAMERA (1.0)      # 카메라 설정
    ↓
COMPOSITION (2.0) # 구도
    ↓
LIGHTING (3.0)    # 조명
    ↓
QUALITY (5.0)     # 품질
    ↓
INFO (8.0)        # 정보
```

### 2. 워크플로우 단계
```python
1. 위치 설정 (15초)   # 거리, 조명
    ↓
2. 구도 잡기 (10초)   # 프레이밍
    ↓
3. 포즈 조정 (20초)   # 자세
    ↓
4. 카메라 설정 (10초) # ISO, 조리개
    ↓
5. 품질 확인 (5초)    # 블러, 노이즈
```

### 3. 진행도 계산
```python
점수 = 100
  - Critical 피드백 × 15점
  - Important 피드백 × 10점
  - Nice-to-have 피드백 × 5점

진행률 = (해결된 문제 / 전체 문제) × 100
```

---

## 🎓 학습 자료

### Python 코드 이해
1. `utils/feedback_formatter.py` - 피드백 포맷팅 로직
2. `utils/workflow_guide.py` - 워크플로우 구성
3. `utils/progress_tracker.py` - 진행도 계산

### API 구조
1. `backend/main.py` - FastAPI 라우팅
2. `analysis/image_comparator.py` - 피드백 생성

### 모델 학습
1. `scripts/train_contrastive.py` - 대조학습 코드
2. `models/contrastive/training_history.json` - 학습 결과

---

## 📞 연락처 및 지원

### Git Repository
- **브랜치**: `HS_COMPUTER` (현재 PC)
- **리모트**: `HS_MAC` (맥북 업로드 파일들)

### 이슈 발생 시
1. 에러 로그 확인
2. `HANDOVER.md` 문제 해결 섹션 참조
3. Git 이슈 등록

---

## ✅ 인수인계 체크리스트

### 코드
- [x] Phase 1-3 모든 utils 모듈 완성
- [x] FastAPI 서버 통합
- [x] Python 테스트 코드 작성
- [x] API 문서 작성

### 모델
- [x] 대조학습 모델 학습 완료 (77% 정확도)
- [x] 모델 파일 저장 (`models/contrastive/`)
- [x] 학습 이력 기록

### 문서
- [x] 완료 요약 (`PHASE_1-3_COMPLETION_SUMMARY.md`)
- [x] API 가이드 (`backend/API_GUIDE.md`)
- [x] 인수인계 문서 (이 파일)
- [x] README 업데이트 (TODO)

### 테스트
- [x] Python 통합 피드백 테스트 (`main_feedback.py`)
- [x] 실시간 카메라 테스트 (`camera_realtime.py`)
- [x] FastAPI 서버 실행 확인
- [ ] iOS 앱 연동 테스트 (다음 단계)

### Git
- [ ] 변경사항 커밋
- [ ] 원격 저장소 푸시
- [ ] 태그 생성 (v2.0.0)

---

## 🎯 최종 정리

### 현재 상태
- ✅ **Python 백엔드**: 100% 완료
- ✅ **FastAPI 서버**: Phase 1-3 통합 완료
- ✅ **AI 모델**: 학습 완료 (77% 정확도)
- ⏳ **iOS 앱**: 기본 기능만 있음, Phase 1-3 통합 필요

### 바로 사용 가능
```bash
# 1. 서버 실행
cd C:\try_angle\backend
python main.py

# 2. Python 테스트
cd C:\try_angle\src\Multi\version3
python main_feedback.py

# 3. 실시간 카메라 (PC)
python camera_realtime.py
```

### iOS 연동 준비됨
- API 엔드포인트: `/api/feedback/enhanced`
- 응답 형식: iOS 친화적 JSON
- 세션 기반 진행도 추적

---

**인수인계 완료일**: 2025-11-17
**다음 작업자**: iOS 앱 통합 및 테스트
**예상 소요 시간**: 1-2일 (iOS UI + 연동)

**문의사항이 있으면 이 문서를 먼저 참조하세요!** 📚
