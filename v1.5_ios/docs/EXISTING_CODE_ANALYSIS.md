# 📊 기존 코드 분석 및 V1 설계 매핑

**작성일**: 2025-11-18
**목적**: 기존 코드와 V1 설계 비교, 마이그레이션 전략 수립

---

## 1. 기존 코드 현황

### 1.1 파일 구조

```
ios/TryAngleApp/
├── Models/
│   └── Feedback.swift                    # FeedbackItem 모델
│
├── Services/
│   ├── CameraManager.swift               # 카메라 관리
│   ├── RealtimeAnalyzer.swift            # 🔥 실시간 분석 (핵심)
│   └── APIService.swift                  # 서버 통신 (V1에서 불필요)
│
├── Views/
│   ├── CameraView.swift                  # 카메라 프리뷰
│   ├── FeedbackOverlay.swift             # 피드백 UI
│   ├── GridOverlay.swift                 # 그리드 오버레이
│   └── ReferenceSelector.swift           # 레퍼런스 선택
│
├── ContentView.swift                     # 메인 화면
└── TryAngleApp.swift                     # 앱 엔트리
```

---

## 2. 기존 코드 vs V1 설계 매핑

### 2.1 컴포넌트 매핑

| V1 설계 | 기존 코드 | 상태 | 비고 |
|---------|----------|------|------|
| **RealtimeDirector** | RealtimeAnalyzer | ✅ **있음** | 이름만 다름, 기능 유사 |
| **AnalysisEngine** | RealtimeAnalyzer 내부 | ✅ **통합됨** | Vision Framework 사용 |
| **RuleEngine** | RealtimeAnalyzer 내부 | △ **부분적** | 간단한 비교만 |
| **ComparisonEngine** | RealtimeAnalyzer 내부 | ✅ **통합됨** | 피드백 생성 로직 |
| **CompositionAnalyzer** | ❌ **없음** | ❌ | 3분할/황금비율 분석 필요 |
| **CameraAngleDetector** | 간단히 구현 | △ **부분적** | Y 위치만 체크 |
| **AngleCalculator** | calculateAngle() | ✅ **있음** | 3점 각도 계산 |
| **AdaptivePoseComparator** | comparePoseKeypoints() | △ **부분적** | 적응형 아님 |
| **Feedback Model** | FeedbackItem | ✅ **있음** | 거의 동일 |
| **CameraManager** | CameraManager | ✅ **있음** | AVFoundation 래핑 |

### 2.2 기능 매핑

| 기능 | V1 설계 | 기존 코드 | 격차 |
|------|---------|----------|------|
| **얼굴 감지** | Vision Framework | ✅ 구현됨 | 동일 |
| **포즈 감지** | Vision Framework | ✅ 구현됨 | 동일 |
| **구도 분석** | 3분할, 황금비율, 중앙 | ❌ 단순 위치만 | **개선 필요** |
| **카메라 앵글** | 기하학적 3가지 방법 | △ Y 위치만 | **개선 필요** |
| **얼굴 각도** | yaw, pitch, roll | ✅ yaw, pitch 구현 | 거의 동일 |
| **포즈 비교** | 적응형 (보이는 부분만) | △ 단순 비교 | **개선 필요** |
| **거리 피드백** | 걸음 수 기반 | ✅ 구현됨 | 동일 |
| **완성도 감지** | isPerfect | ✅ 구현됨 | 동일 |
| **히스테리시스** | 안정화 로직 | ✅ 구현됨 | 동일 |

---

## 3. 기존 코드 장점

### ✅ **이미 잘 구현된 부분**

```
1. Vision Framework 통합 완벽
   - VNDetectFaceLandmarksRequest
   - VNDetectHumanBodyPoseRequest
   - 17개 keypoints 추출
   - yaw, pitch 각도

2. 실시간 성능 최적화
   - 100ms 간격 분석 (analysisInterval)
   - 히스테리시스 (연속 3프레임 감지)
   - "완벽" 상태 10프레임 연속 체크

3. 사용자 경험
   - 진행도 추적 (currentValue, targetValue)
   - 완성도 점수 (perfectScore)
   - 우선순위 시스템 (priority)

4. 카메라 관리
   - AVFoundation 완벽 래핑
   - 줌, 플래시, 카메라 전환
   - FPS 추적

5. 피드백 UI
   - 실시간 오버레이
   - 진행도 바
   - 카테고리별 아이콘
```

### 💪 **코드 품질**

```
✅ Swift 네이티브
✅ Combine 사용 (@Published)
✅ Vision Framework 최신 API
✅ 메모리 효율적 (샘플링, 캐싱)
✅ 에러 처리
✅ 주석 명확
```

---

## 4. V1 설계와 격차 (개선 필요)

### ❌ **없는 기능**

#### 4.1 구도 분석 (Composition Analysis)

```swift
// 현재 (기존 코드):
if let refFace = reference.faceRect, let curFace = currentFaceRect {
    let xDiff = (curFace.midX - refFace.midX) * 100
    let yDiff = (curFace.midY - refFace.midY) * 100
    // → 단순 위치 차이만 계산
}

// V1 설계:
- 3분할 그리드 자동 감지
- 황금비율 그리드
- 중앙 집중 감지
- "왼쪽 상단 3분할 교점" 같은 명확한 분류
```

**해결**: CompositionAnalyzer 추가 필요

#### 4.2 카메라 앵글 정밀 감지

```swift
// 현재 (기존 코드):
if faceY < 0.33 {
    return "low"   // 로우앵글
} else if faceY > 0.67 {
    return "high"  // 하이앵글
}
// → Y 위치만 사용, 단순함

// V1 설계:
- 얼굴 랜드마크 Y 비율 (눈/코)
- 어깨-머리 depth 차이 (ARKit)
- 수평선 감지 (더치 틸트)
- 3가지 방법 Fusion
```

**해결**: CameraAngleDetector 개선 필요

#### 4.3 적응형 포즈 비교

```swift
// 현재 (기존 코드):
// 왼팔, 오른팔만 비교
// 항상 17개 keypoints 기대

// V1 설계:
- 보이는 keypoints만 필터링
- 포즈 타입 자동 분류 (전신/상반신/흉상)
- 공통 keypoints만 비교
- 보이지 않는 부분 안내
```

**해결**: AdaptivePoseComparator 추가 필요

#### 4.4 모듈화 구조

```swift
// 현재 (기존 코드):
// RealtimeAnalyzer에 모든 로직 통합 (545줄)

// V1 설계:
// 명확한 책임 분리:
- AnalysisEngine (Vision)
- RuleEngine (구도, 앵글)
- ComparisonEngine (비교, 피드백)
```

**해결**: 리팩터링 필요

---

## 5. 마이그레이션 전략

### 🎯 **전략: 점진적 개선 (기존 코드 기반)**

```
원칙:
✅ 기존 코드는 유지 (작동 중)
✅ V1 기능을 점진적으로 추가
✅ 하위 호환성 유지
❌ 대규모 리팩터링 피하기
```

### 5.1 Phase 1: 기존 코드 정리 (3일)

```
목표: V1 구조에 맞춰 파일 분리

작업:
✅ RealtimeAnalyzer → RealtimeDirector로 이름 변경
✅ Vision 로직 → VisionAnalyzer 분리
✅ 구도 로직 → CompositionAnalyzer 신규 생성
✅ 앵글 로직 → CameraAngleDetector 신규 생성
✅ 포즈 로직 → AdaptivePoseComparator 신규 생성

결과:
  - 기능 변화 없음
  - 구조만 V1에 맞춤
  - 테스트 통과
```

### 5.2 Phase 2: V1 기능 추가 (1주)

```
목표: V1 설계의 고급 기능 구현

작업:
✅ CompositionAnalyzer 구현
   - 3분할 그리드
   - 황금비율 그리드
   - 자동 구도 분류

✅ CameraAngleDetector 개선
   - 3가지 방법 통합
   - Fusion 로직

✅ AdaptivePoseComparator 구현
   - 포즈 타입 자동 감지
   - 공통 keypoints만 비교

결과:
  - 정확도 향상
  - "3분할 구도" 같은 명확한 피드백
  - 부분 포즈 대응
```

### 5.3 Phase 3: 최적화 (3일)

```
목표: 성능 및 UX 개선

작업:
✅ 메모리 최적화
✅ 배터리 소모 최소화
✅ 한국어 메시지 다듬기
✅ 애니메이션 추가
✅ 버그 수정

결과:
  - V1 완성
  - 앱스토어 출시 가능
```

---

## 6. 구체적 작업 목록

### 6.1 파일 생성 (신규)

```
Services/Analysis/
├── VisionAnalyzer.swift              # Vision 로직 분리

Services/RuleEngine/
├── CompositionAnalyzer.swift         # 🆕 구도 분석
├── CameraAngleDetector.swift         # 🆕 앵글 감지
├── AngleCalculator.swift             # 기존 로직 분리
└── AdaptivePoseComparator.swift      # 🆕 적응형 포즈

Services/Comparison/
├── GapAnalyzer.swift                 # 차이 분석
├── FeedbackGenerator.swift           # 피드백 생성
└── PriorityManager.swift             # 우선순위

Utils/
├── GeometryHelpers.swift             # 기하학 계산
└── GridGenerator.swift               # 그리드 생성
```

### 6.2 파일 수정

```
RealtimeAnalyzer.swift → RealtimeDirector.swift
  - 이름 변경
  - 로직 분리 (VisionAnalyzer, CompositionAnalyzer 호출)
  - 인터페이스 유지

Feedback.swift → Feedback.swift
  - 변경 없음 (그대로 사용)

ContentView.swift
  - @StateObject var analyzer → director
  - 나머지 동일
```

### 6.3 파일 제거 (선택)

```
APIService.swift → V1에서는 불필요
  - 나중에 V2에서 사용
  - 지금은 주석 처리 또는 무시
```

---

## 7. 마이그레이션 상세 계획

### Step 1: 백업 생성 (필수!)

```bash
cd /Users/hyunsoo/Try_Angle
git checkout -b v1-migration
git add .
git commit -m "백업: V1 마이그레이션 전"
```

### Step 2: 이름 변경 및 분리

```swift
// 1. RealtimeAnalyzer → RealtimeDirector
// ios/TryAngleApp/Services/RealtimeDirector.swift

class RealtimeDirector: ObservableObject {
    // 기존 RealtimeAnalyzer 코드 그대로
    // 나중에 점진적으로 리팩터링
}

// 2. ContentView 수정
// @StateObject private var analyzer = RealtimeAnalyzer()
@StateObject private var director = RealtimeDirector()

// 나머지는 analyzer → director로 찾아 바꾸기
```

### Step 3: Vision 로직 분리

```swift
// 신규: ios/TryAngleApp/Services/Analysis/VisionAnalyzer.swift

class VisionAnalyzer {
    // RealtimeAnalyzer의 Vision 관련 로직만 이동
    func detectFace(image: UIImage) -> FaceResult? { ... }
    func detectPose(image: UIImage) -> PoseResult? { ... }
}

// RealtimeDirector에서 사용
class RealtimeDirector {
    private let visionAnalyzer = VisionAnalyzer()

    func analyzeFrame(_ image: UIImage) {
        let face = visionAnalyzer.detectFace(image)
        let pose = visionAnalyzer.detectPose(image)
        // ...
    }
}
```

### Step 4: CompositionAnalyzer 추가

```swift
// 신규: ios/TryAngleApp/Services/RuleEngine/CompositionAnalyzer.swift

class CompositionAnalyzer {
    func classifyComposition(subjectPosition: CGPoint) -> CompositionType {
        // 3분할 그리드 체크
        if isRuleOfThirdsLeftUpper(subjectPosition) {
            return .ruleOfThirdsLeftUpper
        }
        // 황금비율 체크
        else if isGoldenRatioLeft(subjectPosition) {
            return .goldenRatioLeft
        }
        // 중앙 집중
        else if isCenterFocus(subjectPosition) {
            return .centerFocus
        }
        // 커스텀
        else {
            return .custom(position: subjectPosition)
        }
    }

    private func isRuleOfThirdsLeftUpper(_ pos: CGPoint) -> Bool {
        // (0.333, 0.333) ± 5% 허용
        return abs(pos.x - 0.333) < 0.05 && abs(pos.y - 0.333) < 0.05
    }

    // ... 나머지 메서드
}
```

### Step 5: 통합 및 테스트

```swift
// RealtimeDirector에서 CompositionAnalyzer 사용

class RealtimeDirector {
    private let compositionAnalyzer = CompositionAnalyzer()

    func analyzeReference(_ image: UIImage) {
        // 기존 로직
        let faceRect = ...

        // 🆕 구도 분류
        let subjectPosition = CGPoint(x: faceRect.midX, y: faceRect.midY)
        let compositionType = compositionAnalyzer.classifyComposition(
            subjectPosition: subjectPosition
        )

        print("📸 레퍼런스 구도: \(compositionType)")
        // → "3분할 왼쪽 상단" 같은 명확한 출력

        // ReferenceData에 저장
        referenceAnalysis = FrameAnalysis(
            ...,
            compositionType: compositionType  // 🆕 추가
        )
    }
}
```

---

## 8. 권장 접근법

### 🥇 **Option 1: 점진적 개선 (추천!)**

```
장점:
  ✅ 기존 코드 활용
  ✅ 작동하는 상태 유지
  ✅ 리스크 최소화
  ✅ 빠른 진행

단점:
  ⚠️ 구조가 완전히 V1과 다를 수 있음
  ⚠️ 코드 중복 가능

일정:
  - Week 1: 파일 분리 및 구조 정리
  - Week 2: CompositionAnalyzer 추가
  - Week 3: CameraAngleDetector, AdaptivePoseComparator
  - Week 4: 최적화 및 테스트
```

### Option 2: 처음부터 V1 설계대로 재작성

```
장점:
  ✅ V1 설계 완벽 구현
  ✅ 깨끗한 구조
  ✅ 모듈화 완벽

단점:
  ❌ 시간 많이 소요 (3-4주)
  ❌ 기존 코드 버림 (아까움)
  ❌ 버그 발생 위험

일정:
  - Week 1-2: 전체 재작성
  - Week 3: 통합 및 테스트
  - Week 4: 버그 수정

→ 권장하지 않음
```

---

## 9. 다음 단계

### 즉시 시작 가능한 작업

```bash
# 1. 브랜치 생성
git checkout -b v1-enhancement

# 2. 백업 커밋
git add .
git commit -m "백업: V1 개선 전"

# 3. 파일 이름 변경
# RealtimeAnalyzer.swift → RealtimeDirector.swift
mv ios/TryAngleApp/Services/RealtimeAnalyzer.swift \
   ios/TryAngleApp/Services/RealtimeDirector.swift

# 4. 디렉토리 생성
mkdir -p ios/TryAngleApp/Services/Analysis
mkdir -p ios/TryAngleApp/Services/RuleEngine
mkdir -p ios/TryAngleApp/Services/Comparison
mkdir -p ios/TryAngleApp/Utils

# 5. 첫 파일 생성
# CompositionAnalyzer.swift 구현 시작
```

---

## 10. 최종 권장사항

```
✅ 기존 코드는 훌륭한 기반
✅ V1 설계의 70% 이미 구현됨
✅ 점진적으로 개선하는 게 최선
✅ 1주일이면 V1 완성 가능

다음 작업:
  1. RealtimeAnalyzer → RealtimeDirector 이름 변경
  2. CompositionAnalyzer 추가
  3. V1 설계의 고급 기능 통합
  4. 테스트 및 출시
```

**결론: 기존 코드를 버리지 말고, V1 설계를 점진적으로 적용하세요!**

---

**문서 버전**: 1.0
**최종 수정**: 2025-11-18
**다음 단계**: 실제 마이그레이션 시작
