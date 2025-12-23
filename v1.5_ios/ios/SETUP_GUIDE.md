# 🔧 iOS 앱 Xcode 프로젝트 생성 가이드

> 소스 파일들이 준비되었으니, Xcode에서 프로젝트를 생성하는 방법입니다.

---

## 📝 단계별 가이드

### 1. Xcode 열기
```bash
# Xcode가 없다면 App Store에서 설치 (무료)
# Xcode 14 이상 권장
```

### 2. 새 프로젝트 생성
1. **Xcode 실행**
2. **Create a new Xcode project** 클릭
3. 템플릿 선택:
   - **iOS** 탭
   - **App** 선택
   - **Next** 클릭

### 3. 프로젝트 설정
다음 정보 입력:
- **Product Name**: `TryAngleApp`
- **Team**: None (또는 본인 Apple ID)
- **Organization Identifier**: `com.tryangle` (또는 원하는 값)
- **Interface**: **SwiftUI** ← 중요!
- **Language**: **Swift** ← 중요!
- **Use Core Data**: ❌ (체크 해제)
- **Include Tests**: ❌ (체크 해제, 선택사항)

**Next** 클릭

### 4. 저장 위치 선택
```
/Users/hyunsoo/Try_Angle/ios/
```
위 경로로 이동 후 **Create** 클릭

⚠️ **주의**: 이미 `TryAngleApp` 폴더가 존재하므로, Xcode가 덮어쓰기 물어보면 **Merge** 선택

---

## 📁 파일 추가 (중요!)

Xcode가 생성한 기본 파일들을 우리 파일로 교체합니다.

### 자동 생성된 파일 삭제
Xcode 좌측 Project Navigator에서 다음 파일 **삭제** (Move to Trash):
- `ContentView.swift` (기본 생성된 것)
- `TryAngleAppApp.swift` (있다면)

### 우리 파일 추가
1. Finder에서 다음 파일들을 **드래그앤드롭**으로 Xcode에 추가:

```
TryAngleApp/
├── TryAngleApp.swift          ← 추가
├── ContentView.swift           ← 추가
├── Info.plist                  ← 추가 (중요!)
├── Models/                     ← 폴더째 추가
│   └── Feedback.swift
├── Views/                      ← 폴더째 추가
│   ├── CameraView.swift
│   ├── FeedbackOverlay.swift
│   └── ReferenceSelector.swift
└── Services/                   ← 폴더째 추가
    ├── CameraManager.swift
    └── APIService.swift
```

2. 드래그앤드롭 시 옵션:
   - ✅ **Copy items if needed** (체크)
   - ✅ **Create groups** (선택)
   - ✅ **Add to targets: TryAngleApp** (체크)

---

## ⚙️ Info.plist 설정 (중요!)

### 방법 1: 기존 Info.plist 사용
Xcode 좌측에서 `Info.plist` 선택 후 우리가 만든 파일로 교체

### 방법 2: Xcode에서 직접 설정
1. 좌측 Project Navigator에서 **TryAngleApp (프로젝트 아이콘)** 클릭
2. **Info** 탭 선택
3. **Custom iOS Target Properties** 섹션에서 `+` 버튼 클릭
4. 다음 항목 추가:

```
Key: Privacy - Camera Usage Description
Type: String
Value: TryAngle은 실시간 촬영 가이드를 제공하기 위해 카메라 접근이 필요합니다.

Key: Privacy - Photo Library Usage Description
Type: String
Value: 레퍼런스 이미지를 선택하기 위해 사진 라이브러리 접근이 필요합니다.
```

---

## 🔧 API 서버 주소 설정

`Services/APIService.swift` 파일 열기:

```swift
// ⚠️ 여기를 맥북의 실제 IP로 변경!
private let baseURL = "http://192.168.0.10:8000"
```

### 맥북 IP 확인 방법
```bash
# 터미널에서:
ifconfig | grep "inet " | grep -v 127.0.0.1

# 또는 GUI로:
# 시스템 설정 > 네트워크 > WiFi > 세부정보 > IP 주소
```

---

## ▶️ 빌드 및 실행

### 1. 시뮬레이터 선택 (테스트용)
- 상단 메뉴: **Any iOS Device** 클릭
- **iPhone 14 Pro** (또는 원하는 모델) 선택

⚠️ **시뮬레이터 제한**: 카메라 기능 사용 불가

### 2. 실제 디바이스 선택 (권장)
1. iPhone을 USB로 맥북에 연결
2. iPhone에서 **"이 컴퓨터를 신뢰합니까?"** → **신뢰** 클릭
3. Xcode 상단에서 연결된 iPhone 선택

### 3. 실행
- **⌘ + R** (Command + R) 누르기
- 또는 상단 ▶️ 버튼 클릭

### 4. 신뢰 설정 (실제 디바이스만)
앱이 실행 안 되고 "Untrusted Developer" 에러 발생 시:

```
iPhone에서:
설정 > 일반 > VPN 및 기기 관리
→ 개발자 앱 항목에서 본인 Apple ID 선택
→ 신뢰 클릭
```

---

## ✅ 테스트

### 1. 백엔드 서버 실행
```bash
# 터미널 1
cd /Users/hyunsoo/Try_Angle/backend
python main.py
```

### 2. iOS 앱 실행
- Xcode에서 **⌘ + R**

### 3. 기능 테스트
1. 카메라 권한 허용
2. 하단 "레퍼런스" 버튼 → 사진 선택
3. 실시간 피드백 확인!

---

## 🐛 자주 발생하는 에러

### "Command CodeSign failed"
```
해결: Signing & Capabilities 탭에서
Team: None으로 설정 (또는 본인 Apple ID)
```

### "Module 'AVFoundation' not found"
```
해결: Build Phases > Link Binary With Libraries에서
+ 버튼 → AVFoundation.framework 추가
```

### "Info.plist not found"
```
해결: Build Settings 검색 → "Info.plist File"
값: TryAngleApp/Info.plist
```

### 빌드는 되는데 앱이 켜지자마자 꺼짐
```
1. Xcode 하단 Console 탭 확인
2. 에러 메시지 읽기
3. 대부분 Info.plist 권한 설정 누락
```

---

## 📱 다음 단계

프로젝트 생성 완료 후:

1. ✅ 백엔드 서버 실행 확인
2. ✅ IP 주소 올바른지 확인
3. ✅ iPhone과 맥북이 같은 WiFi 사용 확인
4. ✅ 레퍼런스 이미지 선택
5. ✅ 실시간 피드백 테스트!

---

궁금한 점이 있으면 README.md를 참고하세요! 🚀
