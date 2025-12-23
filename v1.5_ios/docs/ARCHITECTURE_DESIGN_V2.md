# TryAngle v1.5 - Architecture Design v2.0

## 개요
본 문서는 TryAngle v1.5 피드백 시스템의 개선된 아키텍처 설계를 정의합니다.

**설계 일자**: 2025-12-02
**버전**: 2.0
**목표**: 유지보수성, 확장성, 테스트 용이성 개선

---

## 현재 설계의 문제점

### 1. 책임 분리 부족
- `reference_comparison.py` (546줄): 분석 + 비교 + 피드백 생성 + 시각화
- `unified_feedback_system.py` (721줄): 전체 기능 통합
- 단일 책임 원칙(SRP) 위반
- 코드 중복 발생

### 2. 성능 문제
- 매 인스턴스마다 모델 로드 (8-9초)
- 메모리 비효율
- 캐싱 부족

### 3. 유지보수성 문제
- 하드코딩된 임계값 (0.1, 0.03 등)
- 하드코딩된 메시지
- 다국어 지원 불가
- A/B 테스트 불가

### 4. 확장성 문제
- 새 피드백 룰 추가 시 기존 코드 수정 필요
- 우선순위 변경 어려움
- 플러그인 구조 부재

---

## 개선된 아키텍처

### 계층 구조

```
┌─────────────────────────────────────────┐
│      Presentation Layer                 │
│  - API Server (FastAPI)                 │
│  - CLI Interface                        │
│  - Test Runners                         │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│      Application Layer                  │
│  - FeedbackOrchestrator                 │
│    ├── ReferenceFeedbackService         │
│    ├── PatternFeedbackService           │
│    └── HybridFeedbackService            │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│      Domain Layer                       │
│  - AnalysisEngine                       │
│    ├── PersonDetector                   │
│    ├── DepthAnalyzer                    │
│    └── CompositionAnalyzer              │
│  - ComparisonEngine                     │
│  - FeedbackGenerator                    │
│    ├── PriorityCalculator               │
│    ├── ActionGenerator                  │
│    └── MessageFormatter                 │
│  - VisualizationEngine                  │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│      Infrastructure Layer               │
│  - ModelManager (싱글톤)                │
│  - CacheManager                         │
│  - ConfigManager                        │
└─────────────────────────────────────────┘
```

### 디렉토리 구조

```
v1.5_ios/src/v1.5_realtime/
├── core/                           # 핵심 도메인
│   ├── __init__.py
│   ├── models.py                   # 데이터 모델 (ImageAnalysis, FeedbackAction 등)
│   │
│   ├── analyzers/                  # 분석 엔진
│   │   ├── __init__.py
│   │   ├── person_detector.py
│   │   ├── depth_analyzer.py
│   │   └── composition_analyzer.py
│   │
│   ├── comparison/                 # 비교 엔진
│   │   ├── __init__.py
│   │   └── comparison_engine.py
│   │
│   └── feedback/                   # 피드백 생성
│       ├── __init__.py
│       ├── base_rule.py            # FeedbackRule 베이스 클래스
│       ├── rules/                  # 피드백 룰 플러그인
│       │   ├── __init__.py
│       │   ├── aspect_ratio_rule.py
│       │   ├── position_rule.py
│       │   ├── compression_rule.py
│       │   └── margin_rule.py
│       ├── generator.py            # FeedbackEngine
│       └── contextualizer.py       # 상황별 메시지 최적화
│
├── services/                       # 애플리케이션 서비스
│   ├── __init__.py
│   ├── reference_service.py
│   ├── pattern_service.py
│   └── hybrid_service.py
│
├── infrastructure/                 # 인프라
│   ├── __init__.py
│   ├── model_manager.py            # 싱글톤 모델 매니저
│   ├── cache_manager.py
│   └── config_manager.py
│
├── config/                         # 설정 파일
│   ├── feedback_rules.yaml         # 룰 설정
│   ├── messages_ko.yaml            # 한국어 메시지
│   ├── messages_en.yaml            # 영어 메시지
│   └── models.yaml                 # 모델 설정
│
├── tests/                          # 테스트
│   ├── unit/
│   │   ├── test_person_detector.py
│   │   ├── test_aspect_ratio_rule.py
│   │   └── test_position_rule.py
│   ├── integration/
│   │   ├── test_analysis_pipeline.py
│   │   └── test_feedback_generation.py
│   └── e2e/
│       └── test_full_workflow.py
│
└── legacy/                         # 기존 코드 (마이그레이션 후 제거)
    ├── reference_comparison.py
    └── unified_feedback_system.py
```

---

## 핵심 컴포넌트 설계

### 1. ModelManager (싱글톤 패턴)

**목적**: 모델 로딩 최적화, 메모리 효율

```python
class ModelManager:
    """모델 관리 싱글톤"""
    _instance = None
    _models = {}

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get_model(self, model_name: str):
        """지연 로딩 + 캐싱"""
        if model_name not in self._models:
            self._models[model_name] = self._load_model(model_name)
        return self._models[model_name]
```

**장점**:
- 모델 한 번만 로드 (8-9초 → 0초)
- 메모리 효율 향상
- 전역 접근 가능

---

### 2. FeedbackRule (플러그인 아키텍처)

**목적**: 룰 추가/수정/제거 용이

```python
from abc import ABC, abstractmethod

class FeedbackRule(ABC):
    """피드백 룰 베이스 클래스"""

    @property
    @abstractmethod
    def priority(self) -> int:
        """우선순위 (낮을수록 먼저)"""
        pass

    @abstractmethod
    def evaluate(self, current, reference) -> Optional[FeedbackAction]:
        """룰 평가"""
        pass
```

**구현 예시**:

```python
class AspectRatioRule(FeedbackRule):
    priority = 1

    def __init__(self, config: FeedbackConfig):
        self.threshold = config.get_rule("aspect_ratio")["threshold"]
        self.impact = config.get_rule("aspect_ratio")["impact_score"]

    def evaluate(self, current, reference):
        if abs(current.aspect_ratio - reference.aspect_ratio) > self.threshold:
            return FeedbackAction(
                type="aspect_ratio",
                action=self.config.get_message("feedback.aspect_ratio.change"),
                amount=f"{reference.aspect_ratio}로 변경",
                impact=f"+{self.impact}점"
            )
        return None
```

**장점**:
- 새 룰 추가 = 새 클래스만 생성
- 기존 코드 수정 없음
- 독립적 테스트 가능
- A/B 테스트 가능

---

### 3. 설정 기반 시스템

**config/feedback_rules.yaml**:
```yaml
aspect_ratio:
  threshold: 0.1
  priority: 1
  message_key: "feedback.aspect_ratio.change"
  impact_score: 30
  required: true

position:
  threshold: 0.03
  priority: 2
  message_key: "feedback.position.adjust"
  impact_score: 10

compression:
  threshold: 0.2
  priority: 3
  message_key: "feedback.compression.adjust"
  impact_score: 5
```

**config/messages_ko.yaml**:
```yaml
feedback:
  aspect_ratio:
    change: "카메라 종횡비 변경"
    description: "{target}로 변경하세요"

  position:
    adjust: "위치 조정"
    up: "피사체를 프레임 아래쪽으로 (카메라 ↑)"
    down: "피사체를 프레임 위쪽으로 (카메라 ↓)"
    left: "피사체를 프레임 오른쪽으로 (카메라 ←)"
    right: "피사체를 프레임 왼쪽으로 (카메라 →)"

  compression:
    adjust: "렌즈 초점거리 조정"
    increase: "망원 렌즈 사용 (85mm 이상)"
    decrease: "광각 렌즈 사용 (35mm 이하)"
```

**config/messages_en.yaml**:
```yaml
feedback:
  aspect_ratio:
    change: "Change camera aspect ratio"
    description: "Change to {target}"

  position:
    adjust: "Adjust position"
    up: "Move subject to lower frame (camera ↑)"
    # ...
```

**장점**:
- 코드 수정 없이 설정 변경
- 다국어 지원
- A/B 테스트 가능
- 버전 관리 용이

---

### 4. FeedbackEngine

**목적**: 룰 실행 및 우선순위 정렬

```python
class FeedbackEngine:
    def __init__(self, config: FeedbackConfig):
        self.config = config
        self.rules: List[FeedbackRule] = []
        self._register_rules()

    def _register_rules(self):
        """룰 등록"""
        self.rules = [
            AspectRatioRule(self.config),
            PositionRule(self.config),
            CompressionRule(self.config),
            MarginRule(self.config),
        ]
        self.rules.sort(key=lambda r: r.priority)

    def generate_feedback(self, current, reference) -> List[FeedbackAction]:
        """모든 룰 실행"""
        actions = []
        for rule in self.rules:
            action = rule.evaluate(current, reference)
            if action:
                action.priority = rule.priority
                actions.append(action)

        return sorted(actions, key=lambda a: a.priority)
```

---

### 5. FeedbackContextualizer

**목적**: 상황별 메시지 최적화

```python
class FeedbackContextualizer:
    def contextualize(self,
                     action: FeedbackAction,
                     mode: str,  # 'realtime' or 'post'
                     scene_type: str  # 'selfie', 'portrait', 'landscape'
                     ) -> str:

        if mode == 'realtime':
            # 실시간: 간결
            return self._format_realtime(action, scene_type)

        elif mode == 'post':
            # 사후: 상세
            return self._format_post(action, scene_type)

    def _format_post(self, action, scene_type):
        """사후 평가용 상세 메시지"""
        camera = self._get_camera_instruction(action)
        subject = self._get_subject_instruction(action)

        if scene_type == 'selfie':
            return f"{camera} (피사체: {subject})"
        else:
            return f"옵션 1: {camera}\n옵션 2: {subject}"
```

---

## 마이그레이션 전략

### Phase 1: Infrastructure Layer (1-2시간)
1. `infrastructure/model_manager.py` 구현
2. `infrastructure/config_manager.py` 구현
3. 기존 코드에 점진적 적용

### Phase 2: Domain Layer - Feedback Rules (3-4시간)
1. `core/feedback/base_rule.py` 구현
2. 기존 로직을 룰로 추출:
   - `aspect_ratio_rule.py`
   - `position_rule.py`
   - `compression_rule.py`
   - `margin_rule.py`
3. `core/feedback/generator.py` 구현

### Phase 3: Configuration (2-3시간)
1. YAML 설정 파일 생성
2. 메시지 외부화 (한국어/영어)
3. ConfigManager 통합

### Phase 4: Service Layer (2-3시간)
1. `services/reference_service.py` 리팩토링
2. 기존 `reference_comparison.py` 래핑
3. 점진적 마이그레이션

### Phase 5: Testing & Validation (2-3시간)
1. 단위 테스트 작성
2. 통합 테스트
3. 기존 기능 검증

### Phase 6: Cleanup (1시간)
1. 레거시 코드 이동 (`legacy/`)
2. 문서 업데이트
3. Git 커밋

**총 예상 시간**: 11-16시간

---

## 검증 체크리스트

### 기능 검증
- [ ] 종횡비 우선 체크 작동
- [ ] 통합 방향 계산 (충돌 없음)
- [ ] 한국어 메시지 출력
- [ ] 설정 파일로 임계값 변경 가능
- [ ] 새 룰 추가 시 기존 코드 수정 없음

### 성능 검증
- [ ] 모델 로딩 시간 단축 (8초 → 0초)
- [ ] 메모리 사용량 감소
- [ ] 분석 속도 유지

### 테스트 검증
- [ ] 모든 단위 테스트 통과
- [ ] 통합 테스트 통과
- [ ] 기존 테스트 케이스 통과

### 코드 품질
- [ ] 각 클래스가 단일 책임
- [ ] 의존성 주입 사용
- [ ] 하드코딩 제거
- [ ] 문서화 완료

---

## Git Commit 전략

```bash
# Phase 1
git commit -m "feat: Add ModelManager singleton and ConfigManager"

# Phase 2
git commit -m "feat: Implement FeedbackRule plugin architecture"

# Phase 3
git commit -m "feat: Externalize configuration to YAML files"

# Phase 4
git commit -m "refactor: Migrate reference_service to new architecture"

# Phase 5
git commit -m "test: Add comprehensive unit and integration tests"

# Phase 6
git commit -m "chore: Move legacy code and update documentation"

# Final
git commit -m "feat: Complete architecture v2.0 migration

- Implement plugin-based feedback rules
- Add configuration-based system
- Improve performance with singleton ModelManager
- Support multi-language messages
- Enable A/B testing capabilities

BREAKING CHANGE: API changes in ReferenceComparison class"
```

---

## 향후 확장 계획

### v2.1: 실시간 경량 엔진
- iOS Vision Framework 통합
- 경량 모델 사용
- 60fps 목표

### v2.2: 패턴 DB 통합
- 클러스터링 결과 활용
- 테마별 최적 구도 추천
- 학습 데이터 확장

### v2.3: A/B 테스트 프레임워크
- 실험 관리 시스템
- 메트릭 수집
- 자동 최적화

---

## 참고 문서

- `FEEDBACK_SYSTEM_SUMMARY.md`: 현재 시스템 요약
- `reference_comparison.py`: 기존 구현
- `unified_feedback_system.py`: 통합 시스템

**작성자**: Claude
**검토자**: TBD
**승인일**: TBD
