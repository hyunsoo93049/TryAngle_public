# ============================================================
# 🎯 TryAngle - Quality Analyzer
# 이미지 품질 분석 (노이즈, 블러, 선명도, 대비)
# ============================================================

import cv2
import numpy as np
from typing import Dict, Optional


class QualityAnalyzer:
    """이미지 품질 분석 (노이즈, 블러, 선명도, 대비)"""

    def __init__(self, image_path: str):
        """
        Args:
            image_path (str): 분석할 이미지 경로
        """
        self.image_path = image_path
        self.img = cv2.imread(image_path)
        if self.img is None:
            raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image_path}")
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
        laplacian = cv2.Laplacian(self.gray, cv2.CV_64F)
        noise_variance = laplacian.var()

        # 정규화 (경험적 임계값: 1000)
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

    def analyze_sharpness(self, roi: Optional[tuple] = None) -> dict:
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
                "roi_used": bool,          # ROI 사용 여부
                "edge_ratio": float        # Edge pixel 비율
            }
        """
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


# ============================================================
# 품질 비교 함수
# ============================================================

def compare_quality(ref_quality: dict, user_quality: dict) -> dict:
    """
    품질 비교 및 피드백 생성 (상대적 평가 기반)

    Args:
        ref_quality (dict): 레퍼런스 품질 (analyze_all() 결과)
        user_quality (dict): 사용자 품질 (analyze_all() 결과)

    Returns:
        dict: {
            "available": bool,
            "feedback": list[dict],       # 상세 피드백
            "has_critical_issues": bool,  # 치명적 문제 있는지
            "has_style_differences": bool # 스타일 차이 있는지
        }
    """
    if not ref_quality or not user_quality:
        return {
            "available": False,
            "feedback": [],
            "has_critical_issues": False,
            "has_style_differences": False
        }

    feedback_list = []

    # 1. 블러 비교
    blur_feedback = _compare_blur(
        ref_quality["blur"],
        user_quality["blur"]
    )
    if blur_feedback:
        feedback_list.append(blur_feedback)

    # 2. 선명도 비교
    sharpness_feedback = _compare_sharpness(
        ref_quality["sharpness"],
        user_quality["sharpness"]
    )
    if sharpness_feedback:
        feedback_list.append(sharpness_feedback)

    # 3. 노이즈 비교
    noise_feedback = _compare_noise(
        ref_quality["noise"],
        user_quality["noise"]
    )
    if noise_feedback:
        feedback_list.append(noise_feedback)

    # 4. 대비 비교
    contrast_feedback = _compare_contrast(
        ref_quality["contrast"],
        user_quality["contrast"]
    )
    if contrast_feedback:
        feedback_list.append(contrast_feedback)

    # 치명적 문제 / 스타일 차이 판단
    has_critical = any(fb["is_critical"] for fb in feedback_list)
    has_style = any(fb["is_style"] for fb in feedback_list)

    return {
        "available": True,
        "feedback": feedback_list,
        "has_critical_issues": has_critical,
        "has_style_differences": has_style
    }


def _compare_blur(ref_blur: dict, user_blur: dict) -> Optional[dict]:
    """블러 비교"""
    ref_score = ref_blur["blur_score"]
    user_score = user_blur["blur_score"]

    # 치명적 문제: 사용자 이미지가 극심하게 흐림
    if user_score < 50:
        return {
            "category": "blur",
            "ref_value": ref_score,
            "user_value": user_score,
            "difference_percent": 0,
            "direction": "critical_blur",
            "is_critical": True,
            "is_style": False,
            "message": "사진이 극심하게 흐려요 (초점 실패)",
            "adjustment": "다시 찍으세요. 초점을 맞추고 손을 고정하세요",
            "adjustment_numeric": {"action": "retake"},
            "priority": 0.5
        }

    # 상대적 평가
    ratio = user_score / (ref_score + 1e-6)
    diff_percent = int(abs(ratio - 1.0) * 100)

    # ±30% 이내면 OK
    if 0.7 <= ratio <= 1.3:
        return None

    # 레퍼런스가 흐린지 판단
    ref_is_blurry = ref_score < 100

    if ratio > 1.3:  # 사용자가 더 선명
        if ref_is_blurry:
            # 레퍼런스가 의도적으로 흐림 → 스타일
            shutter = "1/30s" if ratio > 2.5 else "1/60s"
            return {
                "category": "blur",
                "ref_value": ref_score,
                "user_value": user_score,
                "difference_percent": diff_percent,
                "direction": "sharper",
                "is_critical": False,
                "is_style": True,
                "message": f"레퍼런스보다 {diff_percent}% 더 선명해요 (레퍼런스는 흔들림 효과)",
                "adjustment": f"셔터속도를 {shutter}로 낮추고 카메라를 살짝 움직이세요",
                "adjustment_numeric": {"shutter_speed": shutter, "method": "camera_shake"},
                "priority": 8.0
            }
        else:
            # 레퍼런스가 선명 → 사용자가 더 선명한 건 좋은 것
            return None

    elif ratio < 0.7:  # 사용자가 더 흐림
        if ref_is_blurry:
            # 레퍼런스도 흐림 → 더 흐린 건 과도
            return {
                "category": "blur",
                "ref_value": ref_score,
                "user_value": user_score,
                "difference_percent": diff_percent,
                "direction": "blurrier",
                "is_critical": False,
                "is_style": True,
                "message": f"레퍼런스보다 {diff_percent}% 더 흐려요",
                "adjustment": "적당히 흔들리게 하세요 (너무 과하면 안 보임)",
                "adjustment_numeric": {"method": "less_shake"},
                "priority": 6.0
            }
        else:
            # 레퍼런스가 선명 → 사용자가 흐린 건 문제
            if ratio < 0.5:
                priority = 1.0  # 심각
            else:
                priority = 3.0  # 중간

            return {
                "category": "blur",
                "ref_value": ref_score,
                "user_value": user_score,
                "difference_percent": diff_percent,
                "direction": "blurrier",
                "is_critical": False,
                "is_style": False,
                "message": f"레퍼런스보다 {diff_percent}% 더 흐려요",
                "adjustment": "손을 더 고정하거나 셔터속도를 1/125s 이상으로 높이세요",
                "adjustment_numeric": {"shutter_speed": "1/125s+", "method": "stabilize"},
                "priority": priority
            }

    return None


def _compare_sharpness(ref_sharp: dict, user_sharp: dict) -> Optional[dict]:
    """선명도 비교 (초점)"""
    ref_score = ref_sharp["sharpness_score"]
    user_score = user_sharp["sharpness_score"]

    # 치명적 문제: 초점 완전 실패
    if user_score < 0.1:
        return {
            "category": "sharpness",
            "ref_value": ref_score,
            "user_value": user_score,
            "difference_percent": 0,
            "direction": "critical_unfocused",
            "is_critical": True,
            "is_style": False,
            "message": "초점이 완전히 실패했어요",
            "adjustment": "다시 찍으세요. 피사체에 초점을 맞추세요",
            "adjustment_numeric": {"action": "retake"},
            "priority": 0.5
        }

    # 상대적 평가
    diff = user_score - ref_score
    diff_percent = int(abs(diff) * 100)

    # ±20% 이내면 OK
    if abs(diff) < 0.2:
        return None

    if diff > 0.2:  # 사용자가 더 선명
        # 더 선명한 건 대체로 좋은 것
        return None

    elif diff < -0.2:  # 사용자가 덜 선명
        return {
            "category": "sharpness",
            "ref_value": ref_score,
            "user_value": user_score,
            "difference_percent": diff_percent,
            "direction": "less_sharp",
            "is_critical": False,
            "is_style": False,
            "message": f"레퍼런스보다 {diff_percent}% 덜 선명해요",
            "adjustment": "피사체에 정확히 초점을 맞추세요 (탭해서 초점)",
            "adjustment_numeric": {"action": "focus_better"},
            "priority": 2.0
        }

    return None


def _compare_noise(ref_noise: dict, user_noise: dict) -> Optional[dict]:
    """노이즈 비교"""
    ref_level = ref_noise["noise_level"]
    user_level = user_noise["noise_level"]

    diff = user_level - ref_level
    diff_percent = int(abs(diff) * 100)

    # ±30% 이내면 OK
    if abs(diff) < 0.3:
        return None

    # 레퍼런스가 노이즈 많은지 판단
    ref_is_noisy = ref_level > 0.6

    if diff < -0.3:  # 사용자가 노이즈 적음
        if ref_is_noisy:
            # 레퍼런스가 필름 느낌
            iso = "1600" if diff < -0.5 else "800"
            grain = int(abs(diff) * 100)
            return {
                "category": "noise",
                "ref_value": ref_level,
                "user_value": user_level,
                "difference_percent": diff_percent,
                "direction": "less_noisy",
                "is_critical": False,
                "is_style": True,
                "message": f"레퍼런스보다 노이즈가 {diff_percent}% 적어요 (레퍼런스는 필름 느낌)",
                "adjustment": f"ISO를 {iso}으로 올리거나 후보정에서 그레인 +{grain}% 추가",
                "adjustment_numeric": {"iso": iso, "post_grain": f"+{grain}%"},
                "priority": 7.0
            }
        else:
            # 노이즈 적은 건 좋은 것
            return None

    elif diff > 0.3:  # 사용자가 노이즈 많음
        if ref_is_noisy:
            # 레퍼런스도 노이즈 많음 → 더 많은 건 과도
            return {
                "category": "noise",
                "ref_value": ref_level,
                "user_value": user_level,
                "difference_percent": diff_percent,
                "direction": "more_noisy",
                "is_critical": False,
                "is_style": True,
                "message": f"레퍼런스보다 노이즈가 {diff_percent}% 많아요",
                "adjustment": "ISO를 조금 낮추세요",
                "adjustment_numeric": {"iso": "lower"},
                "priority": 7.0
            }
        else:
            # 레퍼런스가 깨끗 → 노이즈 많은 건 문제
            return {
                "category": "noise",
                "ref_value": ref_level,
                "user_value": user_level,
                "difference_percent": diff_percent,
                "direction": "more_noisy",
                "is_critical": False,
                "is_style": False,
                "message": f"레퍼런스보다 노이즈가 {diff_percent}% 많아요",
                "adjustment": "ISO를 낮추거나 후보정에서 노이즈 제거 필터 적용",
                "adjustment_numeric": {"iso": "400 이하", "post_denoise": "ON"},
                "priority": 6.0
            }

    return None


def _compare_contrast(ref_contrast: dict, user_contrast: dict) -> Optional[dict]:
    """대비 비교"""
    ref_level = ref_contrast["contrast"]
    user_level = user_contrast["contrast"]

    diff = user_level - ref_level
    diff_percent = int(abs(diff) * 100)

    # ±20% 이내면 OK
    if abs(diff) < 0.2:
        return None

    adjust_percent = int(diff * 100)

    return {
        "category": "contrast",
        "ref_value": ref_level,
        "user_value": user_level,
        "difference_percent": diff_percent,
        "direction": "higher" if diff > 0 else "lower",
        "is_critical": False,
        "is_style": True,
        "message": f"레퍼런스보다 대비가 {diff_percent}% {'높아요' if diff > 0 else '낮아요'}",
        "adjustment": f"대비를 {-adjust_percent:+d}% 조정하세요 (후보정 가능)",
        "adjustment_numeric": {"contrast_adjust": f"{-adjust_percent:+d}%"},
        "priority": 7.0
    }


# ============================================================
# 테스트
# ============================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python quality_analyzer.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    analyzer = QualityAnalyzer(image_path)
    result = analyzer.analyze_all()

    print("\n" + "="*60)
    print("Quality Analysis Result".center(60))
    print("="*60)

    print(f"\n📊 Noise:")
    print(f"   Level: {result['noise']['noise_level']:.2f}")
    print(f"   Severity: {result['noise']['severity']}")

    print(f"\n📷 Blur:")
    print(f"   Score: {result['blur']['blur_score']:.2f}")
    print(f"   Severity: {result['blur']['severity']}")

    print(f"\n🔍 Sharpness:")
    print(f"   Score: {result['sharpness']['sharpness_score']:.2f}")
    print(f"   Focus: {result['sharpness']['focus_quality']}")

    print(f"\n🎨 Contrast:")
    print(f"   Level: {result['contrast']['contrast']:.2f}")
    print(f"   Category: {result['contrast']['level']}")

    print("\n" + "="*60)
