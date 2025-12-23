# 📱 TryAngle iOS App

> AI 기반 실시간 사진 촬영 가이드 - iOS 네이티브 앱

---

## 🎯 개요

TryAngle의 iOS 버전입니다. 안드로이드 [camera2app](https://github.com/juneunseo/camera2app) 구조를 기반으로 재구성했습니다.

### 주요 기능
- ✅ **실시간 카메라 프리뷰** (AVFoundation)
- ✅ **레퍼런스 이미지 선택** (Photos 연동)
- ✅ **실시간 AI 피드백** (1초마다 분석)
- ✅ **자동 카메라 설정** (ISO, 화이트밸런스, EV)
- ✅ **전후면 카메라 전환**

---

## 📁 프로젝트 구조

```
TryAngleApp/
├── TryAngleApp.swift          # 앱 진입점
├── ContentView.swift           # 메인 화면
├── Info.plist                  # 권한 설정
├── Models/
│   ├── Feedback.swift          # API 응답 모델
│   └── CameraSettings.swift    # 카메라 설정 모델
├── Views/
│   ├── CameraView.swift        # AVFoundation 프리뷰
│   ├── FeedbackOverlay.swift   # 피드백 UI
│   └── ReferenceSelector.swift # 레퍼런스 선택기
└── Services/
    ├── CameraManager.swift     # 카메라 제어
    └── APIService.swift        # FastAPI 통신
```

---

## 🛠️ 설치 및 실행

### 1. Xcode로 프로젝트 열기

```bash
cd /Users/hyunsoo/Try_Angle/ios/TryAngleApp
open TryAngleApp.xcodeproj  # 또는 Xcode에서 폴더 열기
```

⚠️ **xcodeproj 파일이 없다면** Xcode에서 직접 프로젝트 생성 필요:

1. Xcode 열기
2. File > New > Project
3. iOS > App 선택
4. Product Name: **TryAngleApp**
5. Interface: **SwiftUI**
6. Language: **Swift**
7. 생성 후 위의 소스 파일들을 추가

### 2. 백엔드 서버 실행 (필수!)

iOS 앱은 맥북의 FastAPI 서버와 통신합니다.

```bash
# 터미널 1: 백엔드 서버 실행
cd /Users/hyunsoo/Try_Angle/backend
python main.py

# 출력:
# 🚀 TryAngle iOS Backend Server Starting...
# 📱 iOS 앱에서 접속할 주소:
#    http://YOUR_PC_IP:8000
```

### 3. 서버 IP 주소 설정

`Services/APIService.swift` 파일에서 IP 주소 변경:

```swift
// ⚠️ 맥북의 실제 IP로 변경하세요!
private let baseURL = "http://192.168.0.10:8000"

// 맥북 IP 확인 방법:
// 시스템 설정 > 네트워크 > WiFi > 세부정보 > IP 주소
```

### 4. iOS 앱 실행

1. Xcode에서 타겟 디바이스 선택 (iPhone 또는 시뮬레이터)
2. ⌘ + R (실행)
3. 카메라 권한 허용

⚠️ **시뮬레이터 제한사항**: 카메라를 사용할 수 없으므로 **실제 iPhone 필요**

---

## 📱 사용 방법

### 1단계: 레퍼런스 선택
- 하단 왼쪽 "레퍼런스" 버튼 클릭
- 사진 라이브러리에서 원하는 스타일 사진 선택

### 2단계: 실시간 촬영
- 레퍼런스 선택 후 자동으로 분석 시작 (1초마다)
- 화면에 실시간 피드백 표시:
  - 👤 포즈 조정
  - 📏 거리 조정
  - 📐 구도 조정

### 3단계: 자동 카메라 설정
- 앱이 자동으로 ISO, 화이트밸런스, 노출 조정
- 사용자는 포즈/거리/구도만 신경쓰면 됨!

---

## 🔧 주요 컴포넌트 설명

### 1. CameraManager.swift
AVFoundation 기반 카메라 제어 클래스
- ✅ 실시간 프리뷰
- ✅ 카메라 설정 적용 (ISO, EV, WB)
- ✅ 전후면 전환
- ✅ 프레임 캡처 (UIImage 변환)

### 2. APIService.swift
FastAPI 서버 통신 클래스
- ✅ Multipart/form-data 전송
- ✅ 레퍼런스 + 현재 프레임 업로드
- ✅ JSON 응답 파싱

### 3. ContentView.swift
메인 UI 및 로직 통합
- ✅ 1초마다 자동 분석
- ✅ 피드백 표시
- ✅ 에러 핸들링

---

## 🎨 UI 구성

```
┌─────────────────────────────┐
│   ⚡ 0.8s (처리 시간)        │ ← 상단
│                             │
│                             │
│   📷 카메라 프리뷰          │ ← 중앙
│                             │
│   👤 왼팔을 15도 올리세요   │ ← 피드백
│   📏 2걸음 뒤로 가세요      │
│                             │
│                             │
│  [레퍼런스]      [카메라전환]│ ← 하단
└─────────────────────────────┘
```

---

## 🔄 Android → iOS 변환 매핑

| Android | iOS |
|---------|-----|
| `Camera2Controller` | `CameraManager` (AVFoundation) |
| `TextureView` | `AVCaptureVideoPreviewLayer` |
| `MainActivity` | `ContentView` (SwiftUI) |
| `CameraCaptureSession` | `AVCaptureSession` |
| `ImageReader` | `AVCaptureVideoDataOutput` |
| `SeekBar` | `Slider` (사용 안 함) |
| `Retrofit` | `URLSession` |

---

## 📊 성능

- **FPS**: 30 FPS (실시간 프리뷰)
- **분석 주기**: 1초 (조정 가능)
- **네트워크**: WiFi 필수
- **처리 시간**: ~0.5-1초 (서버 분석 포함)

---

## ⚠️ 주의사항

### 1. 네트워크
- **iPhone과 맥북이 같은 WiFi에 연결**되어야 합니다
- 모바일 데이터로는 로컬 서버 접근 불가

### 2. 카메라 권한
- 앱 최초 실행 시 카메라 권한 허용 필요
- 거부 시 설정 > TryAngle에서 수동 허용

### 3. 실제 디바이스 필요
- 시뮬레이터는 카메라 미지원
- iPhone 6s 이상 권장

---

## 🐛 문제 해결

### "서버 연결 실패"
```bash
# 1. 백엔드 서버 실행 중인지 확인
cd /Users/hyunsoo/Try_Angle/backend
python main.py

# 2. IP 주소 확인
# APIService.swift의 baseURL이 맥북 IP와 일치하는지 확인

# 3. 방화벽 확인
# 시스템 설정 > 네트워크 > 방화벽에서 Python 허용
```

### "카메라 권한 없음"
```
설정 > 개인정보 보호 > 카메라 > TryAngle 활성화
```

### "피드백이 안 나옴"
```
1. 레퍼런스 이미지를 선택했는지 확인
2. 서버 로그에서 분석 요청 확인
3. 네트워크 상태 확인
```

---

## 📚 참고

- [안드로이드 원본 앱](https://github.com/juneunseo/camera2app)
- [AVFoundation 가이드](https://developer.apple.com/av-foundation/)
- [SwiftUI 튜토리얼](https://developer.apple.com/tutorials/swiftui)

---

## 🚀 다음 단계

### Phase 1 (현재)
- ✅ 기본 카메라 프리뷰
- ✅ 실시간 피드백 표시
- ✅ 자동 카메라 설정

### Phase 2 (계획)
- [ ] 촬영 버튼 추가
- [ ] 촬영한 사진 저장
- [ ] 히스토리 기능

### Phase 3 (미래)
- [ ] 온디바이스 AI (Core ML)
- [ ] 오프라인 모드
- [ ] 고급 UI/UX

---

**Made with ❤️ and AI**
