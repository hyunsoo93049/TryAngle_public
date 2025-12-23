# TryAngle v3 - Quality & Lighting Analysis 재설계

**작성**: Claude Code
**날짜**: 2025-11-15
**목적**: GPT 제안(15가지 카메라/환경 정보 추출)을 TryAngle에 통합하기 위한 설계 문서

---

## 📋 목차

1. [GPT 제안 요약](#1-gpt-제안-요약)
2. [현재 구현 vs 미구현 분석](#2-현재-구현-vs-미구현-분석)
3. [우선순위 결정](#3-우선순위-결정)
4. [파일 구조 재설계](#4-파일-구조-재설계)
5. [API 설계](#5-api-설계)
6. [구현 방법 및 알고리즘](#6-구현-방법-및-알고리즘)
7. [피드백 우선순위 재조정](#7-피드백-우선순위-재조정)
8. [구현 로드맵](#8-구현-로드맵)

---

## 1. GPT 제안 요약

### 핵심 아이디어
**EXIF 없이도 이미지 분석만으로 15가지 카메라/환경 정보를 추출해서 실용적인 피드백 제공**

### 15가지 정보 (출처: gpt_answer.txt)

#### ⭐ 필수 6종
1. **노출** (밝기) - 히스토그램
2. **노이즈** (≈ISO) - 고주파 성분
3. **블러** (≈셔터속도) - Laplacian variance
4. **색온도** (화이트밸런스) - 색상 분포
5. **DOF / 배경흐림** - MiDaS depth
6. **선명도** (초점) - Edge density

#### ⭐ 선택 6종
7. **기울기** (Leveling) - Hough Line
8. **HDR 여부** - 히스토그램 분포
9. **조명 방향** - 얼굴 영역 밝기 그래디언트
10. **역광** - 전경/배경 밝기 비교
11. **색 대비** (Contrast) - HSV 분산
12. **채도** (Saturation) - HSV

#### ⭐ 고급 3종
13. **광각 왜곡** - 직선 검출 후 왜곡 분석
14. **손떨림 흔들림** - 모션 블러 검출
15. **피사체 움직임 분석** - Optical flow

---

## 2. 현재 구현 vs 미구현 분석

### ✅ 이미 구현됨 (5개)

| 번호 | GPT 제안 | 현재 TryAngle | 위치 | 상태 |
|------|---------|---------------|------|------|
| 1 | 노출 (밝기) | `pixels.brightness` | image_analyzer.py:192 | ✅ 완료 |
| 4 | 색온도 | `pixels.temperature` | image_analyzer.py:200 | ✅ 완료 |
| 5 | DOF/배경흐림 | `depth` (MiDaS) | image_analyzer.py:150 | ✅ 완료 |
| 7 | 기울기 | `composition.tilt_angle` | image_analyzer.py:239 | ⚠️ 개선 필요 (threshold 높음) |
| 12 | 채도 | `pixels.saturation` | image_analyzer.py:196 | ✅ 완료 |

### ❌ 미구현 (10개)

#### 필수 4종 (Phase 1)
- **2. 노이즈** (ISO 추정) - Laplacian variance
- **3. 블러** (셔터속도/손떨림) - Laplacian variance
- **6. 선명도** (초점) - Edge density in ROI
- **11. 색 대비** (Contrast) - HSV V channel std

#### 선택 3종 (Phase 2)
- **8. HDR 여부** - Histogram distribution
- **9. 조명 방향** - Face bbox brightness gradient
- **10. 역광** - Foreground/background brightness

#### 고급 3종 (Phase 3, 나중에)
- **13. 광각 왜곡** - Line detection + distortion
- **14. 손떨림** - Motion blur detection
- **15. 피사체 움직임** - Optical flow

---

## 3. 우선순위 결정

### Phase 1: 필수 품질 분석 (즉시 구현 권장)
이미지 품질에 가장 직접적인 영향

| 기능 | 피드백 예시 | 중요도 | 난이도 |
|------|------------|--------|-------|
| 노이즈 | "노이즈가 많아요. 더 밝은 곳에서 찍어보세요" | ⭐⭐⭐ | ⭐ 쉬움 |
| 블러 | "사진이 흔들렸어요. 손을 고정하거나 연사를 써보세요" | ⭐⭐⭐ | ⭐ 쉬움 |
| 선명도 | "인물 얼굴에 초점이 안 맞았어요" | ⭐⭐⭐ | ⭐⭐ 보통 |
| 색 대비 | "대비가 낮아서 밋밋해보여요" | ⭐⭐ | ⭐ 쉬움 |

**예상 코드량**: ~400줄
**예상 소요 시간**: 1-2일

### Phase 2: 실용 환경 분석 (촬영 환경 개선)

| 기능 | 피드백 예시 | 중요도 | 난이도 |
|------|------------|--------|-------|
| 조명 방향 | "광원이 정면에 없어 얼굴이 어두워요" | ⭐⭐⭐ | ⭐⭐ 보통 |
| 역광 | "역광입니다. 180도 돌아서 찍어보세요" | ⭐⭐⭐ | ⭐⭐ 보통 |
| HDR | "HDR 효과가 강해서 부자연스러워요" | ⭐⭐ | ⭐ 쉬움 |

**예상 코드량**: ~350줄
**예상 소요 시간**: 1-2일

### Phase 3: 고급 분석 (선택, 나중에)

| 기능 | 피드백 예시 | 중요도 | 난이도 |
|------|------------|--------|-------|
| 광각 왜곡 | "광각 왜곡으로 얼굴이 퍼져보여요" | ⭐ | ⭐⭐⭐ 어려움 |
| 손떨림 | "손떨림이 감지됐어요" | ⭐ | ⭐⭐ 보통 |
| 움직임 | "움직이는 대상은 셔터가 빨라야 해요" | ⭐ | ⭐⭐⭐ 어려움 |

**예상 코드량**: ~500줄
**예상 소요 시간**: 3-5일

---

## 4. 파일 구조 재설계

### 제안하는 새로운 구조

```
version3/
├── analysis/
│   ├── image_analyzer.py         # ✅ 기존 (통합 컨트롤러)
│   ├── image_comparator.py       # ✅ 기존 (비교 & 피드백)
│   ├── pose_analyzer.py          # ✅ 기존 (YOLO + MediaPipe)
│   ├── exif_analyzer.py          # ✅ 기존 (EXIF 추출)
│   │
│   ├── quality_analyzer.py       # 🆕 Phase 1: 이미지 품질
│   │   # - detect_noise()         노이즈 검출
│   │   # - detect_blur()          블러 검출
│   │   # - analyze_sharpness()    선명도 분석
│   │   # - analyze_contrast()     색 대비 분석
│   │   # - compare_quality()      품질 비교 함수
│   │
│   ├── lighting_analyzer.py      # 🆕 Phase 2: 조명 환경
│   │   # - detect_light_direction()  조명 방향
│   │   # - detect_backlight()        역광 여부
│   │   # - detect_hdr()               HDR 여부
│   │   # - compare_lighting()        조명 비교 함수
│   │
│   └── distortion_analyzer.py    # 🆕 Phase 3: 렌즈/모션 (나중에)
│       # - detect_distortion()       광각 왜곡
│       # - detect_motion()           피사체 움직임
```

### 기존 파일 수정 사항

#### `image_analyzer.py` (수정)
```python
# 추가할 파라미터:
def __init__(self, image_path: str,
             enable_pose=True,
             enable_exif=True,
             enable_quality=True,      # 🆕
             enable_lighting=True):    # 🆕

# 반환값에 추가:
result = {
    "cluster": ...,
    "depth": ...,
    "pixels": ...,
    "composition": ...,
    "pose": ...,
    "exif": ...,
    "quality": {...},     # 🆕
    "lighting": {...},    # 🆕
    "raw_features": ...
}
```

#### `image_comparator.py` (수정)
```python
# 비교 메서드 추가:
def _compare_quality(self) -> dict
def _compare_lighting(self) -> dict

# 우선순위 재조정:
# 0   : 클러스터
# 0.5 : 블러/흔들림 (다시 찍어야 함) 🆕
# 1   : 선명도/초점 (다시 찍어야 함) 🆕
# 1.5 : 역광 🆕
# 2   : 포즈
# 2.5 : 조명 방향 🆕
# 3   : 카메라 설정 (EXIF)
# 4   : 거리
# 5   : 밝기
# 6   : 노이즈 🆕
# 7   : 색 대비 🆕
# 8   : 색감
# 9   : 구도
```

---

## 5. API 설계

### 5.1 quality_analyzer.py (신규)

```python
class QualityAnalyzer:
    """이미지 품질 분석 (노이즈, 블러, 선명도, 대비)"""

    def __init__(self, image_path: str):
        """
        Args:
            image_path (str): 분석할 이미지 경로
        """
        self.image_path = image_path
        self.img = cv2.imread(image_path)
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

    def analyze_all(self) -> dict:
        """
        전체 품질 분석 (통합 함수)

        Returns:
            dict: {
                "noise": dict,      # detect_noise() 결과
                "blur": dict,       # detect_blur() 결과
                "sharpness": dict,  # analyze_sharpness() 결과
                "contrast": dict    # analyze_contrast() 결과
            }
        """
        return {
            "noise": self.detect_noise(),
            "blur": self.detect_blur(),
            "sharpness": self.analyze_sharpness(),
            "contrast": self.analyze_contrast()
        }

    def detect_noise(self) -> dict:
        """
        노이즈 검출 (고주파 성분 분석)

        알고리즘:
            - Laplacian 고주파 분석
            - variance가 높을수록 노이즈 많음

        Returns:
            dict: {
                "noise_level": float,      # 0-1 (0=없음, 1=심함)
                "severity": str,           # "low" / "medium" / "high"
                "variance": float          # 원본 variance 값
            }
        """
        pass

    def detect_blur(self) -> dict:
        """
        블러 검출 (손떨림/모션블러)

        알고리즘:
            - Laplacian variance
            - variance < 100 → 흐림

        Returns:
            dict: {
                "blur_score": float,       # Laplacian variance
                "is_blurred": bool,        # True if blur_score < 100
                "severity": str            # "none" / "slight" / "severe"
            }
        """
        pass

    def analyze_sharpness(self, roi=None) -> dict:
        """
        선명도 분석 (초점 맞았는지)

        알고리즘:
            - ROI 영역의 edge density 계산
            - 얼굴 bbox 있으면 우선 사용

        Args:
            roi (tuple, optional): (x, y, w, h) bbox

        Returns:
            dict: {
                "sharpness_score": float,  # 0-1 (0=흐림, 1=선명)
                "focus_quality": str,      # "good" / "poor"
                "roi_used": bool           # ROI 사용 여부
            }
        """
        pass

    def analyze_contrast(self) -> dict:
        """
        색 대비 분석

        알고리즘:
            - HSV의 V 채널 표준편차

        Returns:
            dict: {
                "contrast": float,         # 0-1 (정규화)
                "level": str,              # "low" / "normal" / "high"
                "std_dev": float           # 원본 표준편차
            }
        """
        pass


def compare_quality(ref_quality: dict, user_quality: dict) -> dict:
    """
    품질 비교 및 피드백 생성 (상대적 평가 기반)

    Args:
        ref_quality (dict): 레퍼런스 품질 (analyze_all() 결과)
        user_quality (dict): 사용자 품질 (analyze_all() 결과)

    Returns:
        dict: {
            "available": bool,
            "feedback": list[dict],       # 상세 피드백 (아래 형식)
            "has_critical_issues": bool,  # 치명적 문제 있는지
            "has_style_differences": bool # 스타일 차이 있는지
        }

        feedback 항목 형식:
        {
            "category": str,              # "blur" / "noise" / "sharpness" / "contrast"
            "ref_value": float,           # 레퍼런스 값
            "user_value": float,          # 사용자 값
            "difference_percent": int,    # 차이율 (%) 예: 90
            "direction": str,             # "sharper"/"blurrier"/"more"/"less"
            "is_critical": bool,          # 치명적 문제 여부
            "is_style": bool,             # 스타일 차이 여부
            "message": str,               # "레퍼런스보다 90% 더 선명해요"
            "adjustment": str,            # "셔터속도를 1/30s로 낮추세요"
            "adjustment_numeric": dict,   # {"shutter_speed": "1/30s", ...}
            "priority": float             # 동적 우선순위 (0.5~9)
        }
    """
    pass
```

### 5.2 lighting_analyzer.py (신규)

```python
class LightingAnalyzer:
    """조명 환경 분석 (조명 방향, 역광, HDR)"""

    def __init__(self, image_path: str, pose_data=None, depth_data=None):
        """
        Args:
            image_path (str): 분석할 이미지 경로
            pose_data (dict, optional): 포즈 분석 결과 (얼굴 bbox 활용)
            depth_data (dict, optional): depth 분석 결과 (역광 검출에 활용)
        """
        self.image_path = image_path
        self.img = cv2.imread(image_path)
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.pose_data = pose_data
        self.depth_data = depth_data

    def analyze_all(self) -> dict:
        """
        전체 조명 분석 (통합 함수)

        Returns:
            dict: {
                "light_direction": dict,
                "backlight": dict,
                "hdr": dict
            }
        """
        return {
            "light_direction": self.detect_light_direction(),
            "backlight": self.detect_backlight(),
            "hdr": self.detect_hdr()
        }

    def detect_light_direction(self) -> dict:
        """
        조명 방향 분석 (얼굴 영역 밝기 그래디언트)

        알고리즘:
            - pose_data에서 얼굴 bbox 추출
            - 얼굴 영역을 4분할 (left, right, top, bottom)
            - 각 영역의 평균 밝기 비교

        Returns:
            dict: {
                "direction": str,          # "front" / "left" / "right" / "top" / "back"
                "confidence": float,       # 0-1
                "brightness_map": dict,    # {"left": float, "right": float, ...}
                "available": bool          # pose_data 없으면 False
            }
        """
        pass

    def detect_backlight(self) -> dict:
        """
        역광 검출 (배경 vs 전경 밝기 비교)

        알고리즘:
            - depth_data로 전경/배경 분리
            - 배경 밝기 > 전경 밝기 * 1.5 → 역광

        Returns:
            dict: {
                "is_backlight": bool,
                "severity": float,         # 0-1 (심각도)
                "bg_brightness": float,
                "fg_brightness": float,
                "ratio": float,            # bg / fg
                "available": bool          # depth_data 없으면 False
            }
        """
        pass

    def detect_hdr(self) -> dict:
        """
        HDR 여부 검출 (히스토그램 분포)

        알고리즘:
            - 히스토그램 양 끝 (0-30, 225-255) 비율 확인
            - 양쪽 다 < 5% → HDR 처리됨

        Returns:
            dict: {
                "is_hdr": bool,
                "dynamic_range": float,    # 동적 범위
                "shadow_ratio": float,     # 어두운 영역 비율
                "highlight_ratio": float   # 밝은 영역 비율
            }
        """
        pass


def compare_lighting(ref_lighting: dict, user_lighting: dict) -> dict:
    """
    조명 비교 및 피드백 생성

    Args:
        ref_lighting (dict): 레퍼런스 조명 (analyze_all() 결과)
        user_lighting (dict): 사용자 조명 (analyze_all() 결과)

    Returns:
        dict: {
            "available": bool,
            "direction_match": bool,
            "backlight_diff": bool,
            "hdr_diff": bool,
            "feedback": list[str],
            "has_issues": bool
        }
    """
    pass
```

---

## 6. 구현 방법 및 알고리즘

### 6.1 Phase 1: 품질 분석

#### 1. 노이즈 검출

```python
def detect_noise(self) -> dict:
    """노이즈 검출"""
    laplacian = cv2.Laplacian(self.gray, cv2.CV_64F)
    noise_variance = laplacian.var()

    # 정규화 (경험적 임계값)
    noise_level = min(1.0, noise_variance / 1000)

    if noise_level < 0.3:
        severity = "low"
    elif noise_level < 0.6:
        severity = "medium"
    else:
        severity = "high"

    return {
        "noise_level": float(noise_level),
        "severity": severity,
        "variance": float(noise_variance)
    }
```

**알고리즘**:
- Laplacian 고주파 성분 분석
- variance가 높을수록 노이즈 많음
- 임계값: 1000 기준으로 정규화

#### 2. 블러 검출

```python
def detect_blur(self) -> dict:
    """블러 검출"""
    laplacian = cv2.Laplacian(self.gray, cv2.CV_64F)
    blur_score = laplacian.var()

    # 임계값 (경험적)
    is_blurred = blur_score < 100

    if blur_score > 500:
        severity = "none"
    elif blur_score > 100:
        severity = "slight"
    else:
        severity = "severe"

    return {
        "blur_score": float(blur_score),
        "is_blurred": is_blurred,
        "severity": severity
    }
```

**알고리즘**:
- Laplacian variance
- < 100: 심각한 흐림
- 100-500: 약간 흐림
- > 500: 선명

#### 3. 선명도 분석

```python
def analyze_sharpness(self, roi=None) -> dict:
    """선명도 분석 (얼굴 ROI 우선)"""
    # ROI 결정
    if roi is None:
        # 전체 이미지
        target = self.gray
        roi_used = False
    else:
        # ROI만 추출
        x, y, w, h = roi
        target = self.gray[y:y+h, x:x+w]
        roi_used = True

    # Edge density 계산
    edges = cv2.Canny(target, 50, 150)
    edge_ratio = np.sum(edges > 0) / edges.size

    # 정규화 (경험적)
    sharpness_score = min(1.0, edge_ratio * 10)

    focus_quality = "good" if sharpness_score > 0.5 else "poor"

    return {
        "sharpness_score": float(sharpness_score),
        "focus_quality": focus_quality,
        "roi_used": roi_used,
        "edge_ratio": float(edge_ratio)
    }
```

**알고리즘**:
- Canny edge detection
- Edge pixel 비율 계산
- ROI(얼굴) 있으면 우선 사용

#### 4. 색 대비 분석

```python
def analyze_contrast(self) -> dict:
    """색 대비 분석"""
    hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]

    # V 채널 표준편차
    std_dev = v_channel.std()

    # 정규화
    contrast = std_dev / 255.0

    if contrast < 0.2:
        level = "low"
    elif contrast < 0.4:
        level = "normal"
    else:
        level = "high"

    return {
        "contrast": float(contrast),
        "level": level,
        "std_dev": float(std_dev)
    }
```

**알고리즘**:
- HSV의 V(밝기) 채널 표준편차
- 표준편차가 클수록 대비 높음

### 6.2 Phase 2: 조명 분석

#### 1. 조명 방향 검출

```python
def detect_light_direction(self) -> dict:
    """조명 방향 분석"""
    if not self.pose_data or 'bbox' not in self.pose_data:
        return {"available": False}

    # 얼굴 bbox 추출
    x, y, w, h = self.pose_data['bbox']
    face = self.gray[y:y+h, x:x+w]

    # 4분할
    left_bright = face[:, :w//2].mean()
    right_bright = face[:, w//2:].mean()
    top_bright = face[:h//2, :].mean()
    bottom_bright = face[h//2:, :].mean()

    brightness_map = {
        "left": float(left_bright),
        "right": float(right_bright),
        "top": float(top_bright),
        "bottom": float(bottom_bright)
    }

    # 방향 결정 (가장 밝은 쪽)
    max_side = max(brightness_map, key=brightness_map.get)
    min_bright = min(brightness_map.values())
    max_bright = max(brightness_map.values())

    confidence = (max_bright - min_bright) / 255.0

    if confidence < 0.1:
        direction = "front"  # 균일한 조명
    else:
        direction = max_side

    return {
        "direction": direction,
        "confidence": float(confidence),
        "brightness_map": brightness_map,
        "available": True
    }
```

**알고리즘**:
- 얼굴 bbox를 4분할 (left, right, top, bottom)
- 각 영역의 평균 밝기 비교
- 가장 밝은 쪽이 광원 방향

#### 2. 역광 검출

```python
def detect_backlight(self) -> dict:
    """역광 검출"""
    if not self.depth_data:
        return {"available": False}

    # depth map으로 전경/배경 분리
    depth_map = self.depth_data  # 가정: numpy array
    fg_mask = depth_map < np.percentile(depth_map, 30)  # 가까운 30%

    # 전경/배경 밝기
    fg_brightness = self.gray[fg_mask].mean()
    bg_brightness = self.gray[~fg_mask].mean()

    ratio = bg_brightness / (fg_brightness + 1e-6)
    is_backlight = ratio > 1.5

    severity = min(1.0, (ratio - 1.0) / 2.0)  # 1.0~3.0 → 0~1

    return {
        "is_backlight": bool(is_backlight),
        "severity": float(severity),
        "bg_brightness": float(bg_brightness),
        "fg_brightness": float(fg_brightness),
        "ratio": float(ratio),
        "available": True
    }
```

**알고리즘**:
- Depth map으로 전경(가까운 30%) 분리
- 배경 밝기 > 전경 밝기 * 1.5 → 역광

#### 3. HDR 검출

```python
def detect_hdr(self) -> dict:
    """HDR 여부 검출"""
    hist = cv2.calcHist([self.gray], [0], None, [256], [0, 256])
    total_pixels = hist.sum()

    # 양 끝 비율
    shadow_ratio = hist[0:30].sum() / total_pixels
    highlight_ratio = hist[225:256].sum() / total_pixels

    # HDR: 양쪽 다 적음
    is_hdr = (shadow_ratio < 0.05) and (highlight_ratio < 0.05)

    dynamic_range = highlight_ratio + shadow_ratio

    return {
        "is_hdr": bool(is_hdr),
        "dynamic_range": float(dynamic_range),
        "shadow_ratio": float(shadow_ratio),
        "highlight_ratio": float(highlight_ratio)
    }
```

**알고리즘**:
- 히스토그램 양 끝 (0-30, 225-255) 비율
- 양쪽 다 < 5% → HDR 처리됨

---

## 6.5 절대적 vs 상대적 평가 기준

### 핵심 철학
**"레퍼런스 이미지의 스타일/감성을 따라하기 위한 피드백"**

- 흐림/노이즈/대비 등은 "나쁜 것"이 아니라 **"의도된 스타일"일 수 있음**
- 절대적 평가는 치명적 문제만 (초점 실패, 극심한 노출 오류)
- 나머지는 모두 레퍼런스 기준 상대적 평가

### 절대적 평가 (Critical Issues)
**무조건 다시 찍어야 하는 치명적 문제**

| 지표 | 임계값 | 판단 | 피드백 | 우선순위 |
|------|-------|------|--------|---------|
| 초점 실패 | sharpness < 0.1 | 아무것도 안 보임 | "초점이 완전히 실패했어요. 다시 찍으세요" | 0.5 |
| 노출 오버 | highlight_clip > 80% | 화면이 새하얗게 | "노출 오버입니다. 밝기를 -2 EV 낮추세요" | 0.5 |
| 노출 언더 | shadow_clip > 80% | 화면이 까맣게 | "노출 언더입니다. 밝기를 +2 EV 올리세요" | 0.5 |

→ **우선순위: 0.5** (최우선, 다시 찍어야 함)

### 상대적 평가 (Style Differences)
**레퍼런스 스타일을 따라하기 위한 조정**

| 지표 | 허용 범위 | 피드백 형식 | 우선순위 |
|------|----------|------------|---------|
| 블러 | ±30% | "레퍼런스보다 X% 더 선명/흐려요. [조정 방법]" | 동적 (5~8) |
| 노이즈 | ±30% | "레퍼런스보다 X% 노이즈 많음/적음. [조정 방법]" | 동적 (6~8) |
| 선명도 | ±20% | "레퍼런스보다 X% 더 선명/흐려요. [조정 방법]" | 동적 (6~8) |
| 대비 | ±20% | "레퍼런스보다 X% 대비 높음/낮음. [조정 방법]" | 7~8 |
| 색온도 | ±500K | "레퍼런스보다 X°K 따뜻함/차가움. [조정 방법]" | 8 |

→ **우선순위: 동적 조정** (레퍼런스가 의도적으로 흐리면 낮은 우선순위)

### 동적 우선순위 결정 로직

```python
def _determine_priority(category, ref_value, user_value, is_critical):
    """우선순위 동적 결정"""

    if is_critical:
        return 0.5  # 치명적 문제 = 최우선

    # 블러의 경우
    if category == "blur":
        if ref_value < 100:  # 레퍼런스가 흐림 = 의도된 스타일
            return 8.0  # 스타일이므로 낮은 우선순위
        else:  # 레퍼런스가 선명 = 품질 요구
            if user_value < ref_value * 0.5:  # 너무 흐림
                return 1.0  # 다시 찍어야 할 수도
            else:
                return 6.0  # 후보정 가능

    # 노이즈의 경우
    elif category == "noise":
        if ref_value > 0.6:  # 레퍼런스가 노이즈 많음 = 필름 느낌 의도
            return 7.0
        else:
            return 6.0

    # 기타
    return 7.0
```

### 조정 수치 계산 로직

```python
def _calculate_adjustment(category, ref_value, user_value):
    """구체적 조정 수치 계산"""

    if category == "blur":
        ratio = user_value / (ref_value + 1e-6)

        if ratio > 1.5:  # 너무 선명함
            # 레퍼런스가 흐린 경우 → 흔들림 효과 추가
            shutter = "1/30s" if ratio > 3 else "1/60s"
            return {
                "message": f"레퍼런스보다 {int((ratio-1)*100)}% 더 선명해요",
                "adjustment": f"셔터속도를 {shutter}로 낮추고 카메라를 살짝 움직이세요",
                "numeric": {"shutter_speed": shutter, "method": "camera_shake"}
            }
        elif ratio < 0.7:  # 너무 흐림
            return {
                "message": f"레퍼런스보다 {int((1-ratio)*100)}% 더 흐려요",
                "adjustment": "손을 더 고정하거나 셔터속도를 높이세요 (1/125s 이상)",
                "numeric": {"shutter_speed": "1/125s+", "method": "stabilize"}
            }

    elif category == "noise":
        diff = user_value - ref_value

        if diff < -0.3:  # 노이즈 너무 적음
            iso = "1600" if diff < -0.5 else "800"
            grain = int(abs(diff) * 100)
            return {
                "message": f"레퍼런스보다 노이즈가 {int(abs(diff)*100)}% 적어요",
                "adjustment": f"ISO를 {iso}으로 올리거나 후보정에서 그레인 +{grain}% 추가",
                "numeric": {"iso": iso, "post_grain": f"+{grain}%"}
            }
        elif diff > 0.3:  # 노이즈 너무 많음
            return {
                "message": f"레퍼런스보다 노이즈가 {int(diff*100)}% 많아요",
                "adjustment": "ISO를 낮추거나 후보정에서 노이즈 제거 필터 적용",
                "numeric": {"iso": "400 이하", "post_denoise": "ON"}
            }

    elif category == "contrast":
        diff = user_value - ref_value
        adjust_percent = int(diff * 100)

        return {
            "message": f"레퍼런스보다 대비가 {abs(adjust_percent)}% {'높아요' if diff > 0 else '낮아요'}",
            "adjustment": f"대비를 {-adjust_percent:+d}% 조정하세요 (후보정 가능)",
            "numeric": {"contrast_adjust": f"{-adjust_percent:+d}%"}
        }

    # 기본
    return None
```

### 피드백 예시

#### 예시 1: 의도적 흔들림 (모션 블러)
```
레퍼런스: blur_score = 50 (흐림)
사용자: blur_score = 500 (선명)

피드백:
{
    "category": "blur",
    "ref_value": 50,
    "user_value": 500,
    "difference_percent": 90,
    "direction": "sharper",
    "is_critical": false,
    "is_style": true,
    "message": "레퍼런스보다 90% 더 선명해요 (레퍼런스는 흔들림 효과)",
    "adjustment": "셔터속도를 1/30s로 낮추고 카메라를 살짝 움직이세요",
    "adjustment_numeric": {"shutter_speed": "1/30s", "method": "camera_shake"},
    "priority": 8.0
}
```

#### 예시 2: 필름 느낌 노이즈
```
레퍼런스: noise_level = 0.7 (높음, 필름 느낌)
사용자: noise_level = 0.3 (낮음)

피드백:
{
    "category": "noise",
    "ref_value": 0.7,
    "user_value": 0.3,
    "difference_percent": 57,
    "direction": "less_noisy",
    "is_critical": false,
    "is_style": true,
    "message": "레퍼런스보다 노이즈가 57% 적어요 (레퍼런스는 필름 느낌)",
    "adjustment": "ISO를 800으로 올리거나 후보정에서 그레인 +40% 추가",
    "adjustment_numeric": {"iso": "800", "post_grain": "+40%"},
    "priority": 7.0
}
```

#### 예시 3: 선명한 레퍼런스에서 흔들림 (치명적 X, 하지만 높은 우선순위)
```
레퍼런스: blur_score = 400 (선명)
사용자: blur_score = 80 (흐림)

피드백:
{
    "category": "blur",
    "ref_value": 400,
    "user_value": 80,
    "difference_percent": 80,
    "direction": "blurrier",
    "is_critical": false,
    "is_style": false,
    "message": "레퍼런스보다 80% 더 흐려요",
    "adjustment": "손을 더 고정하거나 셔터속도를 1/125s 이상으로 높이세요",
    "adjustment_numeric": {"shutter_speed": "1/125s+", "method": "stabilize"},
    "priority": 1.0  # 품질 문제이므로 높은 우선순위
}
```

---

## 7. 피드백 우선순위 재조정

### 새로운 우선순위 (철학: 후보정 불가능한 문제를 최우선)

```
우선순위   카테고리          피드백 예시                                    이유
─────────────────────────────────────────────────────────────────────────────
0         클러스터          "같은 스타일입니다 (Cluster 2)"                 정보성

─────── 📸 다시 찍어야 하는 문제 (높은 우선순위) ────────────────────────────
0.5       블러/흔들림 🆕     "사진이 흔들렸어요. 손 고정/연사"              후보정 불가
1         선명도/초점 🆕     "얼굴에 초점이 안 맞았어요"                    후보정 불가

─────── 🎬 촬영 시 조정 가능한 문제 ──────────────────────────────────────
1.5       역광 🆕           "역광입니다. 180도 돌아서 찍어보세요"           위치 변경
2         포즈              "왼팔을 15도 더 올리세요"                      자세 조정
2.5       조명 방향 🆕       "광원이 왼쪽에 있어 얼굴 오른쪽이 어두워요"    광원 위치
3         카메라 설정       "ISO 400으로 설정하세요"                       설정 변경
4         거리              "약 1걸음 뒤로 가세요"                         거리 조정

─────── 🎨 후보정 가능한 문제 (낮은 우선순위) ─────────────────────────────
5         밝기              "밝기를 0.5 EV 올리면 좋아요"                  편집 가능
6         노이즈 🆕          "노이즈가 있어요"                              필터 적용
7         색 대비 🆕         "대비가 낮아서 밋밋해요"                       편집 가능
8         색감              "색온도를 따뜻하게 조정하세요"                  편집 가능
9         구도              "화면을 시계방향으로 2도 회전하세요"            크롭 가능
```

### 기존 vs 새로운 우선순위 비교

| 기존 | 새로운 | 카테고리 | 변경 이유 |
|------|--------|----------|----------|
| - | 0.5 | 블러/흔들림 | 후보정 불가 → 최우선 |
| - | 1 | 선명도/초점 | 후보정 불가 → 최우선 |
| 0.5 | 2 | 포즈 | 조정 가능 → 중간 |
| 1 | 3 | 카메라 설정 | 조정 가능 → 중간 |
| 2 | 4 | 거리 | 변경 없음 |
| 3 | 5 | 밝기 | 변경 없음 |
| 4 | 8 | 색감 | 후보정 가능 → 낮춤 |
| 5 | 9 | 구도 | 후보정 가능 → 낮춤 |

---

## 8. 구현 로드맵

### Phase 1: 필수 품질 분석 (1-2일)

**목표**: 노이즈, 블러, 선명도, 대비 추가

**작업 순서**:
1. `quality_analyzer.py` 생성
   - `QualityAnalyzer` 클래스 구현
   - 4개 메서드: `detect_noise()`, `detect_blur()`, `analyze_sharpness()`, `analyze_contrast()`
   - `compare_quality()` 함수 구현

2. `image_analyzer.py` 통합
   - `enable_quality` 파라미터 추가
   - `analyze()` 반환값에 `quality` 필드 추가

3. `image_comparator.py` 업데이트
   - `_compare_quality()` 메서드 추가
   - 우선순위 재조정 (0.5: 블러, 1: 선명도)
   - 피드백 메시지 작성

4. 테스트 및 검증
   - `main_feedback.py` 실행
   - 출력 확인
   - 피드백 메시지 조정

**예상 코드량**: ~400줄

### Phase 2: 조명 환경 분석 (1-2일)

**목표**: 조명 방향, 역광, HDR 추가

**작업 순서**:
1. `lighting_analyzer.py` 생성
   - `LightingAnalyzer` 클래스 구현
   - 3개 메서드: `detect_light_direction()`, `detect_backlight()`, `detect_hdr()`
   - `compare_lighting()` 함수 구현

2. `image_analyzer.py` 통합
   - `enable_lighting` 파라미터 추가
   - `lighting_analyzer`에 `pose_data`, `depth_data` 전달

3. `image_comparator.py` 업데이트
   - `_compare_lighting()` 메서드 추가
   - 우선순위 추가 (1.5: 역광, 2.5: 조명 방향)

4. 테스트 및 검증
   - depth map과 pose bbox 연동 확인
   - 피드백 정확도 검증

**예상 코드량**: ~350줄

### Phase 3: 고급 기능 (선택, 3-5일)

**목표**: 광각 왜곡, 피사체 움직임 (우선순위 낮음)

**작업 순서**:
1. `distortion_analyzer.py` 생성
2. 광각 왜곡 검출 (직선 검출 + 왜곡 분석)
3. 모션 분석 (optical flow)

**예상 코드량**: ~500줄

---

## 📌 다음 작업자를 위한 가이드

### 시작 전 체크리스트
- [ ] QUICK_REFERENCE.md "현재 작업 컨텍스트" 확인
- [ ] 이 문서 (DESIGN_QUALITY_LIGHTING.md) 전체 읽기
- [ ] test1.jpg, test2.jpg 확인 (기울기 0° 정상, EXIF 없음 정상)
- [ ] TA 환경 활성화

### 구현 시작하기

**Option 1: Phase 1만 구현** (추천)
```bash
cd C:\try_angle\src\Multi\version3\analysis
# quality_analyzer.py 생성 시작
```

**Option 2: 우선순위 조정만**
```bash
# image_comparator.py 수정
# get_prioritized_feedback() 메서드만 조정
```

**Option 3: 문서만 업데이트**
```bash
# 필요 시 이 문서 보완
```

### 완료 후 할 일
1. QUICK_REFERENCE.md "현재 작업 컨텍스트" 업데이트 (덮어쓰기)
2. CHANGELOG.md에 변경사항 추가
3. `main_feedback.py` 실행해서 테스트
4. 토큰 사용률 확인 (70% 이상이면 저장 후 GPT로 넘기기)

---

**문서 끝**
**작성**: Claude Code
**날짜**: 2025-11-15
