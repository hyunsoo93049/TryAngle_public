# 🔄 Try_Angle V1 → V2 마이그레이션 가이드

**작성일**: 2025-11-18
**목적**: V1 (온디바이스) → V2 (하이브리드) 전환 가이드

---

## 📋 목차

1. [마이그레이션 개요](#1-마이그레이션-개요)
2. [단계별 전환 계획](#2-단계별-전환-계획)
3. [클라이언트 변경사항](#3-클라이언트-변경사항)
4. [서버 구축](#4-서버-구축)
5. [테스트 전략](#5-테스트-전략)
6. [롤백 계획](#6-롤백-계획)

---

## 1. 마이그레이션 개요

### 1.1 마이그레이션 철학

```
"Zero Breaking Changes"

핵심 원칙:
  ✅ V1 코드는 한 줄도 수정하지 않음
  ✅ V2는 V1 위에 레이어로 추가
  ✅ 서버 없으면 자동으로 V1 모드
  ✅ 언제든 V1으로 되돌릴 수 있음
```

### 1.2 변경 범위

| 영역 | V1 | V2 추가 | 수정 |
|------|----|---------| -----|
| **Core Engine** | 유지 | - | ❌ 없음 |
| **Director** | RealtimeDirector | HybridDirector | ❌ V1 코드 수정 없음 |
| **Network** | - | ServerClient, SmartSampler | ✅ 신규 |
| **Models** | 유지 | HybridReferenceData, etc. | ✅ 확장 |
| **UI** | 유지 | 애니메이션 추가 | △ 소폭 수정 |
| **Server** | - | 전체 | ✅ 신규 |

---

## 2. 단계별 전환 계획

### Phase 1: 서버 개발 (V1과 병렬, 2주)

```
목표: 서버 단독 개발 및 테스트

작업:
✅ FastAPI 프로젝트 생성
✅ Florence-2 통합
✅ Grounding DINO 통합
✅ Depth Anything V2 통합
✅ AIDirectorEngine 구현
✅ API 엔드포인트 구현
✅ Postman으로 단독 테스트

환경:
  - 로컬 PC (RTX 4070 Super)
  - Python 3.11 (Conda)
  - PyTorch + CUDA

완료 기준:
  - POST /api/v2/analyze-reference 작동
  - POST /api/v2/analyze-precise 작동
  - 응답 시간 <200ms

V1 앱:
  - 전혀 영향 없음
  - 계속 개발/테스트 가능
```

### Phase 2: 클라이언트 하이브리드 통합 (1주)

```
목표: V1 위에 V2 레이어 추가

작업:
✅ HybridDirector 구현
✅ ServerClient 구현
✅ SmartSampler 구현
✅ FusionLayer 구현
✅ 데이터 모델 확장
✅ 설정 화면 추가 (서버 URL)

변경 파일:
  - 신규: Services/HybridDirector.swift
  - 신규: Network/ServerClient.swift
  - 신규: Hybrid/SmartSampler.swift
  - 신규: Hybrid/FusionLayer.swift
  - 확장: Models/HybridReferenceData.swift
  - 소폭 수정: Views/CameraView.swift

V1 코드:
  - RealtimeDirector.swift: 수정 없음
  - 모든 V1 컴포넌트: 수정 없음

완료 기준:
  - HybridDirector가 RealtimeDirector 호출
  - 서버 연결 성공 시 Tier 3 작동
  - 서버 없으면 V1 모드로 Fallback
```

### Phase 3: 통합 테스트 (3일)

```
목표: V1/V2 모드 전환 테스트

테스트 시나리오:
✅ V1 모드 단독 테스트 (서버 OFF)
✅ V2 모드 단독 테스트 (서버 ON)
✅ V1 → V2 전환 (서버 연결)
✅ V2 → V1 Fallback (서버 끊김)
✅ 네트워크 불안정 시나리오
✅ 성능 비교 (FPS, 배터리)

완료 기준:
  - 모든 시나리오 통과
  - V1 성능 저하 없음
  - V2 정확도 향상 확인
```

### Phase 4: 베타 배포 (1주)

```
목표: 실사용자 테스트

배포:
  - TestFlight (iOS)
  - 내부 테스트 (Android)
  - 베타 테스터 10-20명

서버:
  - 자택 PC (RTX 4070 Super)
  - Ngrok/Cloudflare Tunnel
  - 또는 AWS g4dn.xlarge (1주 테스트)

피드백 수집:
  - 정확도 개선 체감도
  - 버그 리포트
  - 성능 이슈

완료 기준:
  - 치명적 버그 없음
  - 정확도 향상 확인
  - 비용 측정 완료
```

### Phase 5: 정식 출시 (1주)

```
목표: 프로덕션 배포

배포:
  - 앱스토어 (iOS)
  - 플레이스토어 (Android)

서버:
  Option A: V2 기능 비활성화 (무료 버전)
  Option B: 클라우드 서버 (유료 버전)
  Option C: 서버 기능은 "Pro 구독"으로

출시 전략:
  - 무료: V1 모드 (온디바이스만)
  - Pro ($4.99/month): V2 모드 (서버 연동)

완료 기준:
  - 앱 승인 통과
  - 서버 안정성 확인
  - 모니터링 구축
```

---

## 3. 클라이언트 변경사항

### 3.1 신규 파일 목록

```
ios/TryAngleApp/Services/
├── HybridDirector.swift          # 🆕 메인 컨트롤러 (V1 확장)

ios/TryAngleApp/Services/Network/
├── ServerClient.swift             # 🆕 HTTP 클라이언트
├── WebSocketClient.swift          # 🆕 (선택) WebSocket
└── NetworkMonitor.swift           # 🆕 연결 상태 감시

ios/TryAngleApp/Services/Hybrid/
├── SmartSampler.swift             # 🆕 변화 감지
├── FusionLayer.swift              # 🆕 피드백 통합
└── FeedbackAnimator.swift         # 🆕 전환 애니메이션

ios/TryAngleApp/Models/
├── HybridReferenceData.swift     # 🆕 V1 확장 모델
├── ServerReferenceData.swift     # 🆕 서버 응답 모델
└── HybridFeedback.swift          # 🆕 V1 확장 모델

ios/TryAngleApp/Views/
└── SettingsView.swift            # 🆕 설정 화면 (서버 URL)
```

### 3.2 수정 파일

#### **CameraView.swift** (소폭 수정)

```swift
// Before (V1):
@StateObject private var director = RealtimeDirector()

// After (V2):
@StateObject private var director = HybridDirector()
// HybridDirector가 내부적으로 RealtimeDirector 사용
// 인터페이스 동일하므로 UI 코드 변경 없음
```

#### **FeedbackOverlay.swift** (애니메이션 추가)

```swift
// V2 추가: 피드백 전환 애니메이션
.transition(.opacity.combined(with: .scale))
.animation(.easeInOut(duration: 0.3), value: feedbacks)
```

### 3.3 구현 예시 (HybridDirector)

```swift
// ios/TryAngleApp/Services/HybridDirector.swift

import Foundation
import SwiftUI

class HybridDirector: ObservableObject {

    // V1 엔진 (그대로 사용)
    private let v1Director = RealtimeDirector()

    // V2 추가 컴포넌트
    private let serverClient = ServerClient()
    private let smartSampler = SmartSampler()
    private let fusionLayer = FusionLayer()

    // 상태
    @Published var isServerAvailable: Bool = false
    @Published var currentReference: HybridReferenceData?
    @Published var feedbacks: [HybridFeedback] = []

    // 캐시
    private var serverFeedbackCache: [HybridFeedback]?
    private var lastServerSyncTime: Date = .distantPast


    // ━━━ 레퍼런스 분석 ━━━
    func analyzeReference(_ image: UIImage) {
        // Step 1: V1 온디바이스 분석 (즉시)
        let deviceData = v1Director.analyzeReference(image)

        // 즉시 UI 업데이트
        self.currentReference = HybridReferenceData(
            deviceData: deviceData,
            serverData: nil
        )

        // Step 2: 서버 전송 (백그라운드)
        Task {
            await sendReferenceToServer(image)
        }
    }

    private func sendReferenceToServer(_ image: UIImage) async {
        guard isServerAvailable else { return }

        do {
            let serverData = try await serverClient.analyzeReference(image)

            // 서버 응답 도착 → UI 업데이트
            await MainActor.run {
                self.currentReference?.serverData = serverData
            }
        } catch {
            print("서버 분석 실패: \(error)")
            // V1 데이터는 이미 있으므로 계속 작동
        }
    }


    // ━━━ 실시간 프레임 처리 ━━━
    func processFrame(_ frame: CVPixelBuffer) {
        // Tier 1: V1 온디바이스 (항상 실행)
        let tier1Feedbacks = v1Director.processFrame(frame)

        // 즉시 UI 업데이트
        self.feedbacks = tier1Feedbacks.map { HybridFeedback(v1: $0) }

        // Tier 2: Smart Sampling (조건 충족 시만)
        if smartSampler.shouldSample(frame) {
            Task {
                await sendFrameToServer(frame)
            }
        }
    }

    private func sendFrameToServer(_ frame: CVPixelBuffer) async {
        guard isServerAvailable,
              let reference = currentReference else { return }

        do {
            let response = try await serverClient.analyzePrecise(
                frame: frame,
                referenceId: reference.serverData?.referenceId ?? ""
            )

            // Tier 3 피드백 도착
            await MainActor.run {
                self.serverFeedbackCache = response.feedbacks

                // Fusion: Tier 1 + Tier 3 통합
                let merged = fusionLayer.merge(
                    tier1: self.feedbacks,
                    tier3: response.feedbacks
                )

                // 부드러운 전환
                withAnimation(.easeInOut(duration: 0.3)) {
                    self.feedbacks = merged
                }
            }
        } catch {
            // 서버 오류 → V1 피드백 유지
            print("서버 실시간 분석 실패: \(error)")
        }
    }


    // ━━━ 서버 연결 상태 체크 ━━━
    func checkServerConnection() async {
        do {
            let isHealthy = try await serverClient.checkHealth()
            await MainActor.run {
                self.isServerAvailable = isHealthy
            }
        } catch {
            await MainActor.run {
                self.isServerAvailable = false
            }
        }
    }
}

// V1 Feedback → HybridFeedback 변환
extension HybridFeedback {
    init(v1: Feedback) {
        self.init(
            id: v1.id,
            priority: v1.priority,
            category: v1.category,
            icon: v1.icon,
            message: v1.message,
            reason: v1.reason,
            source: .device,  // V1은 항상 device
            alternativeAction: nil,
            technicalDetail: nil,
            currentValue: v1.currentValue,
            targetValue: v1.targetValue,
            unit: v1.unit,
            tolerance: v1.tolerance,
            isCompleted: v1.isCompleted
        )
    }
}
```

---

## 4. 서버 구축

### 4.1 서버 프로젝트 생성

```bash
# 1. 디렉토리 생성
cd /Users/hyunsoo/Try_Angle
mkdir server
cd server

# 2. Python 가상환경 (Conda)
conda create -n tryangle_server python=3.11 -y
conda activate tryangle_server

# 3. 필수 패키지 설치
pip install fastapi uvicorn python-multipart
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers pillow opencv-python

# 4. 프로젝트 구조 생성
mkdir -p api services models schemas utils config
touch main.py
touch requirements.txt
```

### 4.2 FastAPI 기본 구조

```python
# server/main.py

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(title="Try_Angle V2 Server")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 개발용, 프로덕션에서는 제한 필요
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# V2 API
from api.v2_routes import router as v2_router
app.include_router(v2_router, prefix="/api/v2")

# Health Check
@app.get("/")
def root():
    return {"status": "OK", "version": "2.0"}

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
```

```python
# server/api/v2_routes.py

from fastapi import APIRouter, UploadFile, File, Form
from schemas.responses import AnalyzeReferenceResponse, AnalyzePreciseResponse
from services.ai_director_engine import AIDirectorEngine

router = APIRouter()
director_engine = AIDirectorEngine()

@router.post("/analyze-reference", response_model=AnalyzeReferenceResponse)
async def analyze_reference(image: UploadFile = File(...)):
    """레퍼런스 이미지 분석"""
    image_data = await image.read()
    result = director_engine.analyze_reference(image_data)
    return result

@router.post("/analyze-precise", response_model=AnalyzePreciseResponse)
async def analyze_precise(
    image: UploadFile = File(...),
    reference_id: str = Form(...)
):
    """실시간 프레임 정밀 분석"""
    image_data = await image.read()
    result = director_engine.analyze_precise(image_data, reference_id)
    return result

@router.get("/health")
def health():
    """서버 상태 체크"""
    return {
        "status": "healthy",
        "gpu_available": director_engine.gpu_available,
        "models_loaded": director_engine.models_loaded
    }
```

### 4.3 AI 엔진 구현

```python
# server/services/ai_director_engine.py

from models.florence2_manager import Florence2Manager
from models.grounding_dino_manager import GroundingDINOManager
from models.depth_manager import DepthManager
import torch
import numpy as np
from PIL import Image
import io

class AIDirectorEngine:

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 모델 로딩
        self.florence2 = Florence2Manager(device=self.device)
        self.grounding_dino = GroundingDINOManager(device=self.device)
        self.depth = DepthManager(device=self.device)

        # 캐시
        self.reference_cache = {}

        self.gpu_available = torch.cuda.is_available()
        self.models_loaded = True


    def analyze_reference(self, image_bytes: bytes):
        """레퍼런스 분석 (1회)"""

        # 이미지 로딩
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Florence-2: 의도 파악
        caption = self.florence2.generate_caption(image)
        objects = self.florence2.detect_objects(image)
        intent = self._parse_intent(caption, objects)

        # Grounding DINO: 정밀 위치
        bbox = self.grounding_dino.detect(image, "person")
        precise_position = self._calculate_position(bbox, image.size)

        # Depth: 깊이 분석
        depth_map = self.depth.estimate(image)
        depth_analysis = self._analyze_depth(depth_map)

        # 캐시 저장
        reference_id = self._generate_id()
        self.reference_cache[reference_id] = {
            "intent": intent,
            "position": precise_position,
            "depth": depth_analysis
        }

        return {
            "reference_id": reference_id,
            "composition_intent": intent["composition"],
            "lighting_intent": intent["lighting"],
            "pose_intent": intent["pose"],
            "precise_position": precise_position,
            "depth_analysis": depth_analysis,
            "processing_time_ms": 209  # TODO: 실제 측정
        }


    def analyze_precise(self, image_bytes: bytes, reference_id: str):
        """실시간 프레임 정밀 분석"""

        # 캐시에서 레퍼런스 로드
        reference = self.reference_cache.get(reference_id)
        if not reference:
            raise ValueError("Reference not found")

        # 현재 프레임 분석
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Grounding DINO: 현재 위치
        bbox = self.grounding_dino.detect(image, "person")
        current_position = self._calculate_position(bbox, image.size)

        # Gap 계산
        gaps = self._calculate_gaps(
            reference["position"],
            current_position
        )

        # 피드백 생성
        feedbacks = self._generate_feedbacks(gaps, reference["intent"])

        return {
            "feedbacks": feedbacks,
            "composition_alignment": 0.72,  # TODO: 실제 계산
            "completion_status": False,
            "processing_time_ms": 80  # TODO: 실제 측정
        }


    def _parse_intent(self, caption, objects):
        """Florence-2 출력 → 의도 파싱"""
        # TODO: 실제 파싱 로직
        return {
            "composition": {
                "primary_rule": "rule_of_thirds",
                "subject_placement": "left_upper_intersection",
                "reasoning": "..."
            },
            "lighting": {...},
            "pose": {...}
        }

    # ... (기타 헬퍼 메서드)
```

### 4.4 서버 실행

```bash
# 개발 모드
cd /Users/hyunsoo/Try_Angle/server
conda activate tryangle_server
python main.py

# 또는 uvicorn 직접
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# 프로덕션 모드 (나중에)
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

---

## 5. 테스트 전략

### 5.1 단위 테스트

```swift
// Tests/HybridDirectorTests.swift

import XCTest
@testable import TryAngleApp

class HybridDirectorTests: XCTestCase {

    func testV1FallbackWhenServerUnavailable() {
        let director = HybridDirector()
        director.isServerAvailable = false

        let image = UIImage(named: "test_reference")!
        director.analyzeReference(image)

        // V1 데이터는 있어야 함
        XCTAssertNotNil(director.currentReference?.deviceData)
        // 서버 데이터는 없어야 함
        XCTAssertNil(director.currentReference?.serverData)
    }

    func testSmartSamplerThrottling() {
        let sampler = SmartSampler()

        // 첫 프레임은 통과
        XCTAssertTrue(sampler.shouldSample(frame1))

        // 0.5초 이내는 차단
        usleep(500_000)  // 0.5초
        XCTAssertFalse(sampler.shouldSample(frame2))

        // 1초 후는 통과
        usleep(500_000)  // +0.5초 = 1초
        XCTAssertTrue(sampler.shouldSample(frame3))
    }
}
```

### 5.2 통합 테스트

```swift
func testEndToEndHybridFlow() async {
    let director = HybridDirector()

    // 1. 서버 연결
    await director.checkServerConnection()
    XCTAssertTrue(director.isServerAvailable)

    // 2. 레퍼런스 분석
    let reference = UIImage(named: "test_reference")!
    director.analyzeReference(reference)

    // 3. V1 데이터 즉시 확인
    XCTAssertNotNil(director.currentReference?.deviceData)

    // 4. 서버 응답 대기 (최대 2초)
    try await Task.sleep(nanoseconds: 2_000_000_000)
    XCTAssertNotNil(director.currentReference?.serverData)

    // 5. 실시간 프레임 처리
    let frame = createTestFrame()
    director.processFrame(frame)

    // 6. Tier 1 피드백 즉시 확인
    XCTAssertFalse(director.feedbacks.isEmpty)

    // 7. Tier 3 피드백 대기
    try await Task.sleep(nanoseconds: 2_000_000_000)
    let tier3Count = director.feedbacks.filter { $0.source == .server }.count
    XCTAssertGreaterThan(tier3Count, 0)
}
```

### 5.3 성능 테스트

```swift
func testPerformance() {
    let director = HybridDirector()
    let frame = createTestFrame()

    measure {
        // Tier 1 처리 시간 측정
        director.processFrame(frame)
    }

    // 목표: <50ms
}
```

---

## 6. 롤백 계획

### 6.1 V2 → V1 되돌리기

```
문제 발생 시:

Step 1: 서버 기능 비활성화
  - HybridDirector.isServerAvailable = false
  - 자동으로 V1 모드로 Fallback
  - 사용자는 온디바이스만 사용

Step 2: 코드 되돌리기 (필요 시)
  - CameraView.swift:
    @StateObject private var director = RealtimeDirector()
  - V2 파일들은 남겨두되 사용 안 함

Step 3: 앱 업데이트 배포
  - V1 모드 기본값으로 설정
  - "서버 기능 일시 중단" 공지

중요:
  ✅ V1 코드는 손상 없음
  ✅ 즉시 되돌릴 수 있음
  ✅ 사용자 데이터 손실 없음
```

### 6.2 데이터 호환성

```
V1 ReferenceData ↔ V2 HybridReferenceData

마이그레이션:
  - V1 데이터는 그대로 유지
  - V2는 serverData만 추가
  - V1으로 되돌려도 deviceData는 있음

호환성 보장:
  ✅ V1 앱은 V1 데이터 읽기
  ✅ V2 앱은 V1/V2 데이터 모두 읽기
  ✅ 양방향 호환
```

---

## 7. 체크리스트

### 7.1 Phase 1 완료 조건

```
서버:
  ✅ FastAPI 실행됨
  ✅ Florence-2 로딩됨
  ✅ Grounding DINO 로딩됨
  ✅ POST /api/v2/analyze-reference 응답
  ✅ POST /api/v2/analyze-precise 응답
  ✅ 응답 시간 <200ms
  ✅ Postman 테스트 통과
```

### 7.2 Phase 2 완료 조건

```
클라이언트:
  ✅ HybridDirector 구현됨
  ✅ ServerClient 구현됨
  ✅ SmartSampler 구현됨
  ✅ FusionLayer 구현됨
  ✅ 서버 연결 성공
  ✅ Tier 1 + Tier 3 통합 작동
  ✅ 서버 끊김 시 V1 Fallback
```

### 7.3 Phase 3 완료 조건

```
테스트:
  ✅ V1 단독 작동
  ✅ V2 단독 작동
  ✅ V1 ↔ V2 전환
  ✅ 네트워크 불안정 대응
  ✅ 성능 저하 없음
  ✅ 정확도 향상 확인
```

### 7.4 출시 준비 완료 조건

```
배포:
  ✅ 베타 테스트 완료
  ✅ 치명적 버그 없음
  ✅ 서버 안정성 확인
  ✅ 비용 산정 완료
  ✅ 모니터링 구축
  ✅ 문서 작성
```

---

## 8. FAQ

### Q1: V1을 먼저 출시하고 나중에 V2 업데이트 가능한가?
**A**: 가능합니다. 오히려 권장합니다.
- V1으로 먼저 출시
- 사용자 피드백 수집
- V2는 나중에 무료 업데이트

### Q2: V2 업데이트 시 V1 사용자는?
**A**: 아무 영향 없습니다.
- V2는 V1 위에 추가됨
- 서버 없으면 자동 V1 모드
- 기존 기능 100% 유지

### Q3: 서버 비용이 부담되면?
**A**: V1 모드만 제공하면 됩니다.
- 무료 버전: V1 (온디바이스)
- Pro 버전: V2 (서버 연동)
- 유연한 비즈니스 모델

### Q4: V1 코드를 수정해야 하나?
**A**: 아니요, 한 줄도 수정 안 함.
- V1 코드는 그대로
- V2는 별도 레이어
- 완전한 하위 호환성

---

**문서 버전**: 1.0
**최종 수정**: 2025-11-18
**관련 문서**: DESIGN_V1_ONDEVICE_ONLY.md, DESIGN_V2_HYBRID.md
