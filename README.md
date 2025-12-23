# Try_Angle

> AI 기반 사진 구도 분석 및 피드백 시스템

[![iOS](https://img.shields.io/badge/iOS-15.0+-blue.svg)](https://www.apple.com/ios/)
[![Python](https://img.shields.io/badge/Python-3.9+-green.svg)](https://www.python.org/)

---

## 📸 소개

Try_Angle은 AI를 활용하여 사진 촬영 시 실시간으로 구도를 분석하고
개선 방안을 제시하는 모바일 애플리케이션입니다.

### 주요 기능

- 📐 **실시간 구도 분석**: 촬영 중 실시간으로 구도 피드백
- 🎯 **포즈 검출**: AI 기반 인물 포즈 인식
- 🎨 **테마별 추천**: 다양한 촬영 테마에 맞는 구도 제안
- 📊 **비교 분석**: 레퍼런스 사진과 현재 구도 비교

---

## 🛠️ 기술 스택

### iOS
- Swift / SwiftUI
- Core ML (온디바이스 AI)
- Vision Framework

### Backend (Demo)
- Python 3.9+
- FastAPI
- Computer Vision (OpenCV, PIL)

---

## 📋 프로젝트 구조

```
Try_Angle/
├── ios/                    # iOS 앱 (UI 및 인터페이스)
├── backend/               # 데모 백엔드 서버
├── docs/                  # 문서
├── data/                  # 샘플 데이터
└── README.md
```

---

## 🚀 시작하기

### 필수 조건

- iOS 15.0 이상
- Python 3.9 이상 (백엔드 실행 시)

### 설치

```bash
# Repository 클론
git clone https://github.com/hyunsoo93049/Try_Angle_Public.git
cd Try_Angle_Public

# Python 의존성 설치
pip install -r requirements.txt
```

### iOS 앱 실행

1. Xcode에서 `ios/TryAngle/TryAngle.xcodeproj` 열기
2. 시뮬레이터 또는 실제 기기 선택
3. Build & Run

---

## 📖 문서

자세한 문서는 [docs/](./docs) 폴더를 참조하세요.

---

## ⚠️ 주의사항

**이 저장소는 Try_Angle 프로젝트의 공개 버전입니다.**

핵심 분석 알고리즘 및 학습된 모델은 proprietary이며 이 저장소에 포함되지 않습니다.

전체 기능을 사용하려면 별도의 라이선스가 필요합니다.

---

## 📧 문의

비즈니스 문의 또는 협업 제안:
- GitHub Issues: https://github.com/hyunsoo93049/Try_Angle_Public/issues

---

## 📄 라이선스

이 프로젝트의 공개 부분은 MIT 라이선스 하에 배포됩니다.
핵심 알고리즘은 별도의 상용 라이선스가 적용됩니다.

---

© 2024 Try_Angle Team. All rights reserved.
