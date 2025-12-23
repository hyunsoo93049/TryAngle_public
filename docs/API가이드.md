# 🚀 TryAngle iOS Backend API 가이드 (Phase 1-3)

**버전**: 2.0.0
**업데이트**: 2025-11-17

---

## 📍 서버 주소

**로컬 개발**:
- `http://localhost:8000`
- `http://YOUR_PC_IP:8000` (iOS에서 접속)

**API 문서** (자동 생성):
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## 🔧 서버 실행

### Windows
```bash
cd C:\try_angle\backend
python main.py
```

### Mac/Linux
```bash
cd /path/to/try_angle/backend
python3 main.py
```

---

## 📡 엔드포인트

### 1. 서버 상태 확인
```http
GET /
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
    "progress_tracking": true,
    "recommendations": true
  }
}
```

---

### 2. 기본 실시간 분석 (구버전 호환)
```http
POST /api/analyze/realtime
Content-Type: multipart/form-data
```

**파라미터**:
- `reference`: 레퍼런스 이미지 파일
- `current_frame`: 현재 프레임 이미지 파일
- `pose_model`: (선택) "yolo11" 또는 "movenet"

**응답**:
```json
{
  "userFeedback": [
    {
      "priority": 1,
      "icon": "👤",
      "message": "왼쪽 어깨 더 올리세요",
      "category": "pose",
      "currentValue": 0,
      "targetValue": 15.5,
      "tolerance": 5,
      "unit": "도"
    }
  ],
  "cameraSettings": {
    "iso": 400,
    "wbKelvin": 5500,
    "evCompensation": 0.7
  },
  "processingTime": "2.345s",
  "timestamp": 1731852000.123
}
```

---

### 3. ⭐ Phase 1-3 통합 피드백 (신규)
```http
POST /api/feedback/enhanced
Content-Type: multipart/form-data
```

**파라미터**:
- `reference`: 레퍼런스 이미지 파일
- `current_frame`: 현재 프레임 이미지 파일
- `user_level`: (선택) "beginner" | "intermediate" | "expert" (기본: beginner)
- `top_k`: (선택) 표시할 피드백 개수 (기본: 3)
- `session_id`: (선택) 진행도 추적용 세션 ID (예: "user123")

**응답**:
```json
{
  "feedback": {
    "primary": [
      {
        "priority": 0.5,
        "category": "pose",
        "message": "왼팔을 15° 올리세요",
        "detail": "레퍼런스와 15° 차이",
        "icon": "🤸"
      },
      {
        "priority": 2.0,
        "category": "distance",
        "message": "2걸음 뒤로",
        "detail": "거리 비율: 1.45",
        "icon": "📏"
      }
    ],
    "secondary": [...],
    "display_text": "포맷된 텍스트",
    "critical_count": 0
  },
  "workflow": {
    "steps": {
      "position": {
        "step": 1,
        "label": "위치 설정",
        "items": [...],
        "completed": false,
        "duration": 15
      },
      "composition": {...},
      "pose": {...},
      "camera": {...},
      "quality": {...}
    },
    "text": "워크플로우 가이드 텍스트",
    "current_step": 1
  },
  "priorities": {
    "critical": [],
    "important": [
      {
        "priority": 2.0,
        "category": "distance",
        "message": "2걸음 뒤로",
        "detail": "거리 비율: 1.45",
        "icon": "📏"
      }
    ],
    "recommended": [...]
  },
  "progress": {
    "score": 75,
    "progress_percent": 60,
    "attempt": 3,
    "text": "진행도 포맷 텍스트",
    "encouragement": "👏 잘하고 있어요! 조금만 더!",
    "is_first": false
  },
  "processing_time": "1.234s",
  "timestamp": 1731852000.123
}
```

**Features**:
- ✅ Top-K 피드백 (Phase 1.1)
- ✅ 초보자 친화 메시지 (Phase 1.2)
- ✅ 워크플로우 가이드 (Phase 2.1)
- ✅ 진행도 추적 (Phase 2.2)
- ✅ 우선순위 분류 (Phase 2.3)

---

### 4. 진행도 초기화
```http
POST /api/progress/reset
Content-Type: multipart/form-data
```

**파라미터**:
- `session_id`: 초기화할 세션 ID

**응답**:
```json
{
  "status": "reset",
  "session_id": "user123"
}
```

---

### 5. AI 레퍼런스 추천 (Phase 3.1)
```http
GET /api/recommendations
Content-Type: multipart/form-data
```

**파라미터**:
- `user_image`: 사용자 이미지 파일
- `top_k`: (선택) 추천 개수 (기본: 3)

**응답**:
```json
{
  "recommendations": [
    {
      "image_path": "/path/to/IMG_1234.jpg",
      "cluster_id": 5,
      "similarity": 0.92,
      "quality_score": 0.85,
      "reason": "매우 유사하면서 고품질이에요!"
    },
    {
      "image_path": "/path/to/IMG_5678.jpg",
      "cluster_id": 5,
      "similarity": 0.88,
      "quality_score": 0.78,
      "reason": "비슷한 스타일이에요"
    }
  ],
  "cluster_id": 5,
  "cluster_label": "실외 / 멀리, 쿨톤, 중간, 반신"
}
```

---

## 🎯 사용 시나리오

### Scenario 1: 기본 실시간 피드백 (구버전 호환)

```swift
// iOS Swift
let formData = MultipartFormData()
formData.append(referenceImage, withName: "reference", fileName: "ref.jpg", mimeType: "image/jpeg")
formData.append(currentFrame, withName: "current_frame", fileName: "frame.jpg", mimeType: "image/jpeg")

AF.upload(multipartFormData: formData, to: "http://\(serverIP):8000/api/analyze/realtime")
    .responseDecodable(of: RealtimeResponse.self) { response in
        // 기본 피드백 처리
    }
```

---

### Scenario 2: Phase 1-3 통합 피드백 (신규)

```swift
// iOS Swift
let formData = MultipartFormData()
formData.append(referenceImage, withName: "reference", fileName: "ref.jpg", mimeType: "image/jpeg")
formData.append(currentFrame, withName: "current_frame", fileName: "frame.jpg", mimeType: "image/jpeg")
formData.append("beginner".data(using: .utf8)!, withName: "user_level")
formData.append("3".data(using: .utf8)!, withName: "top_k")
formData.append(sessionID.data(using: .utf8)!, withName: "session_id")

AF.upload(multipartFormData: formData, to: "http://\(serverIP):8000/api/feedback/enhanced")
    .responseDecodable(of: EnhancedFeedbackResponse.self) { response in
        guard let data = response.value else { return }

        // Top-K 피드백 표시
        displayPrimaryFeedback(data.feedback.primary)

        // 워크플로우 단계별 가이드
        displayWorkflowStep(data.workflow.current_step, items: data.workflow.steps)

        // 진행도 표시
        if let progress = data.progress {
            updateProgressBar(progress.progress_percent)
            showEncouragement(progress.encouragement)
        }

        // 우선순위별 정리
        if !data.priorities.critical.isEmpty {
            showCriticalAlert(data.priorities.critical)
        }
    }
```

---

### Scenario 3: 진행도 추적 세션

```swift
// 1회차 촬영 (세션 시작)
uploadWithSession(sessionID: "user123")  // is_first: true, score: 60

// 2회차 촬영 (개선)
uploadWithSession(sessionID: "user123")  // is_first: false, score: 75, progress: 60%

// 3회차 촬영 (거의 완료)
uploadWithSession(sessionID: "user123")  // is_first: false, score: 90, progress: 90%

// 세션 종료 (초기화)
resetProgress(sessionID: "user123")
```

---

## 📊 응답 필드 설명

### Feedback Object
```typescript
interface Feedback {
  priority: number;      // 우선순위 (낮을수록 중요)
  category: string;      // 카테고리 (pose, distance, brightness, ...)
  message: string;       // 사용자 친화적 메시지
  detail: string;        // 상세 설명
  icon: string;          // 이모지 아이콘
}
```

### Workflow Object
```typescript
interface WorkflowStep {
  step: number;          // 단계 번호 (1-5)
  label: string;         // 단계 이름
  items: Feedback[];     // 해당 단계 피드백 목록
  completed: boolean;    // 완료 여부
  duration: number;      // 예상 소요 시간 (초)
}
```

### Progress Object
```typescript
interface Progress {
  score: number;              // 전체 점수 (0-100)
  progress_percent: number;   // 진행률 (0-100)
  attempt: number;            // 시도 횟수
  text: string;              // 진행도 포맷 텍스트
  encouragement: string;      // 격려 메시지
  is_first: boolean;          // 첫 시도 여부
}
```

---

## 🔥 성능 최적화 팁

1. **세션 재사용**: 같은 사용자는 동일한 `session_id` 사용
2. **캐싱**: 레퍼런스 이미지는 한 번만 업로드 (서버 캐싱)
3. **이미지 크기**: 720p 이하로 리사이즈 (1280x720)
4. **네트워크**: WiFi 사용 권장 (모바일 데이터는 느림)

---

## 🐛 에러 처리

### 503 Service Unavailable
```json
{
  "error": "Phase 1-3 features not available"
}
```
→ Phase 1-3 utils가 제대로 import되지 않음

### 500 Internal Server Error
```json
{
  "error": "오류 메시지"
}
```
→ 서버 로그 확인 필요

### 400 Bad Request
```json
{
  "error": "Could not extract embedding"
}
```
→ 이미지 분석 실패 (레퍼런스 추천)

---

## 📝 버전 히스토리

### v2.0.0 (2025-11-17)
- ✅ Phase 1-3 통합
- ✅ `/api/feedback/enhanced` 엔드포인트 추가
- ✅ 워크플로우 가이드
- ✅ 진행도 추적
- ✅ AI 레퍼런스 추천

### v1.0.0 (이전)
- 기본 실시간 분석
- 포즈 피드백

---

## 📞 문의

**프로젝트**: TryAngle - AI 촬영 가이드
**GitHub**: [링크]
**개발자**: Claude (Anthropic)
