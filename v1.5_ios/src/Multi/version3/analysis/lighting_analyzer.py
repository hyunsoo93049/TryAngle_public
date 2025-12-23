# ============================================================
# 🎯 TryAngle - Lighting Analyzer
# 조명 환경 분석: 조명 방향, 역광, HDR
# ============================================================

import cv2
import numpy as np
from typing import Optional, Dict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
while PROJECT_ROOT != PROJECT_ROOT.parent and not ((PROJECT_ROOT / "data").exists() and (PROJECT_ROOT / "src").exists()):
    PROJECT_ROOT = PROJECT_ROOT.parent


class LightingAnalyzer:
    """
    조명 환경 분석 (조명 방향, 역광, HDR)
    """

    def __init__(self, image_path: str, pose_data: Optional[Dict] = None, depth_data: Optional[np.ndarray] = None):
        """
        Args:
            image_path (str): 분석할 이미지 경로
            pose_data (dict, optional): 포즈 분석 결과 (얼굴 bbox 활용)
            depth_data (np.ndarray, optional): depth map (역광 검출에 활용)
        """
        self.image_path = image_path
        self.img = cv2.imread(image_path)

        if self.img is None:
            raise FileNotFoundError(f"❌ Image not found: {image_path}")

        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.pose_data = pose_data
        self.depth_data = depth_data

    def analyze_all(self) -> Dict:
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

    def detect_light_direction(self) -> Dict:
        """
        조명 방향 분석 (얼굴 영역 밝기 그래디언트)

        알고리즘:
            - pose_data에서 얼굴 bbox 추출
            - 얼굴 영역을 4분할 (left, right, top, bottom)
            - 각 영역의 평균 밝기 비교

        Returns:
            dict: {
                "direction": str,          # "front" / "left" / "right" / "top" / "bottom"
                "confidence": float,       # 0-1
                "brightness_map": dict,    # {"left": float, "right": float, ...}
                "available": bool          # pose_data 없으면 False
            }
        """
        # 전체 이미지 분석 함수 (재사용)
        def analyze_full_image():
            h, w = self.gray.shape

            left_bright = float(self.gray[:, :w//2].mean())
            right_bright = float(self.gray[:, w//2:].mean())
            top_bright = float(self.gray[:h//2, :].mean())
            bottom_bright = float(self.gray[h//2:, :].mean())

            brightness_map = {
                "left": left_bright,
                "right": right_bright,
                "top": top_bright,
                "bottom": bottom_bright
            }

            max_side = max(brightness_map, key=brightness_map.get)
            min_bright = min(brightness_map.values())
            max_bright = max(brightness_map.values())

            confidence = (max_bright - min_bright) / 255.0

            if confidence < 0.1:
                direction = "front"
            else:
                direction = max_side

            return {
                "direction": direction,
                "confidence": float(confidence),
                "brightness_map": brightness_map,
                "available": True,
                "note": "전체 이미지 기반"
            }

        # pose_data 없으면 전체 이미지로 분석
        if not self.pose_data:
            return analyze_full_image()

        # pose_data 있으면 얼굴 bbox 사용
        bbox = None

        # bbox 추출 (다양한 형식 지원)
        if isinstance(self.pose_data, dict):
            if 'bbox' in self.pose_data:
                bbox = self.pose_data['bbox']
            elif 'keypoints' in self.pose_data:
                # keypoints로 bbox 계산
                keypoints = self.pose_data['keypoints']
                if len(keypoints) > 0:
                    # 코, 눈 등 얼굴 keypoints (0-4)
                    face_kpts = keypoints[:5] if len(keypoints) >= 5 else keypoints
                    xs = [kp[0] for kp in face_kpts if kp[2] > 0.3]  # confidence > 0.3
                    ys = [kp[1] for kp in face_kpts if kp[2] > 0.3]

                    if xs and ys:
                        x_min, x_max = int(min(xs)), int(max(xs))
                        y_min, y_max = int(min(ys)), int(max(ys))

                        # bbox 확장 (얼굴 전체)
                        w = x_max - x_min
                        h = y_max - y_min
                        margin = int(max(w, h) * 0.3)

                        x_min = max(0, x_min - margin)
                        y_min = max(0, y_min - margin)
                        x_max = min(self.gray.shape[1], x_max + margin)
                        y_max = min(self.gray.shape[0], y_max + margin)

                        bbox = (x_min, y_min, x_max - x_min, y_max - y_min)

        if bbox is None:
            # bbox 없으면 전체 이미지로 폴백
            return analyze_full_image()

        # 얼굴 bbox 추출 (정수로 변환!)
        x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        # 경계 체크
        if x < 0 or y < 0 or x + w > self.gray.shape[1] or y + h > self.gray.shape[0]:
            # 범위 벗어나면 전체 이미지로 폴백
            return analyze_full_image()

        face = self.gray[y:y+h, x:x+w]

        # 얼굴 너무 작으면 무시
        if face.shape[0] < 20 or face.shape[1] < 20:
            return analyze_full_image()

        # 4분할
        left_bright = float(face[:, :w//2].mean())
        right_bright = float(face[:, w//2:].mean())
        top_bright = float(face[:h//2, :].mean())
        bottom_bright = float(face[h//2:, :].mean())

        brightness_map = {
            "left": left_bright,
            "right": right_bright,
            "top": top_bright,
            "bottom": bottom_bright
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
            "available": True,
            "note": "얼굴 bbox 기반"
        }

    def detect_backlight(self) -> Dict:
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
        if self.depth_data is None:
            # depth 없으면 간단한 휴리스틱 사용
            # 중앙부가 어둡고 가장자리가 밝으면 역광
            h, w = self.gray.shape

            # 중앙부 (전경 가정)
            center_y1, center_y2 = h // 4, 3 * h // 4
            center_x1, center_x2 = w // 4, 3 * w // 4
            center_region = self.gray[center_y1:center_y2, center_x1:center_x2]
            fg_brightness = float(center_region.mean())

            # 가장자리 (배경 가정)
            # 상단 가장자리
            edge_region = self.gray[:h//4, :]
            bg_brightness = float(edge_region.mean())

            ratio = bg_brightness / (fg_brightness + 1e-6)
            is_backlight = ratio > 1.5

            severity = min(1.0, (ratio - 1.0) / 2.0)  # 1.0~3.0 → 0~1

            return {
                "is_backlight": bool(is_backlight),
                "severity": float(severity),
                "bg_brightness": bg_brightness,
                "fg_brightness": fg_brightness,
                "ratio": float(ratio),
                "available": True,
                "note": "depth 없음, 휴리스틱 사용"
            }

        # depth_data 있으면 정확한 분리
        depth_map = self.depth_data

        # depth map이 이미지 크기와 다르면 리사이즈
        if depth_map.shape[:2] != self.gray.shape[:2]:
            depth_map = cv2.resize(depth_map, (self.gray.shape[1], self.gray.shape[0]))

        # 전경/배경 분리 (가까운 30%)
        fg_threshold = np.percentile(depth_map, 30)
        fg_mask = depth_map < fg_threshold

        # 전경/배경 밝기
        fg_brightness = float(self.gray[fg_mask].mean())
        bg_brightness = float(self.gray[~fg_mask].mean())

        ratio = bg_brightness / (fg_brightness + 1e-6)
        is_backlight = ratio > 1.5

        severity = min(1.0, (ratio - 1.0) / 2.0)  # 1.0~3.0 → 0~1

        return {
            "is_backlight": bool(is_backlight),
            "severity": float(severity),
            "bg_brightness": bg_brightness,
            "fg_brightness": fg_brightness,
            "ratio": float(ratio),
            "available": True,
            "note": "depth map 기반"
        }

    def detect_hdr(self) -> Dict:
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
        hist = cv2.calcHist([self.gray], [0], None, [256], [0, 256])
        total_pixels = hist.sum()

        # 양 끝 비율
        shadow_ratio = float(hist[0:30].sum() / total_pixels)
        highlight_ratio = float(hist[225:256].sum() / total_pixels)

        # HDR: 양쪽 다 적음 (클리핑 없음)
        is_hdr = (shadow_ratio < 0.05) and (highlight_ratio < 0.05)

        dynamic_range = float(highlight_ratio + shadow_ratio)

        return {
            "is_hdr": bool(is_hdr),
            "dynamic_range": dynamic_range,
            "shadow_ratio": shadow_ratio,
            "highlight_ratio": highlight_ratio
        }


def compare_lighting(ref_lighting: Dict, user_lighting: Dict) -> Dict:
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
            "feedback": list[dict],
            "has_issues": bool
        }
    """
    feedback_list = []

    # ==========================================
    # 1. 조명 방향 비교
    # ==========================================
    ref_dir = ref_lighting["light_direction"]
    user_dir = user_lighting["light_direction"]

    direction_match = True

    if ref_dir["available"] and user_dir["available"]:
        if ref_dir["direction"] != user_dir["direction"]:
            direction_match = False

            # 방향 한글 변환
            dir_map = {
                "left": "왼쪽",
                "right": "오른쪽",
                "top": "위쪽",
                "bottom": "아래쪽",
                "front": "정면"
            }

            ref_dir_kr = dir_map.get(ref_dir["direction"], ref_dir["direction"])
            user_dir_kr = dir_map.get(user_dir["direction"], user_dir["direction"])

            feedback_list.append({
                "category": "lighting_direction",
                "priority": 7.0,  # 낮은 우선순위 (조명은 바꾸기 어려움)
                "message": f"조명 방향이 달라요",
                "detail": f"레퍼런스는 {ref_dir_kr} 조명, 현재는 {user_dir_kr} 조명",
                "adjustment": f"{ref_dir_kr}에서 조명이 오도록 위치를 바꾸거나, 같은 시간대/장소에서 촬영하세요",
                "adjustment_numeric": {
                    "target_direction": ref_dir["direction"],
                    "current_direction": user_dir["direction"]
                }
            })

    # ==========================================
    # 2. 역광 비교
    # ==========================================
    ref_back = ref_lighting["backlight"]
    user_back = user_lighting["backlight"]

    backlight_diff = False

    if ref_back["available"] and user_back["available"]:
        # 레퍼런스는 역광 아닌데 사용자는 역광
        if not ref_back["is_backlight"] and user_back["is_backlight"]:
            backlight_diff = True

            feedback_list.append({
                "category": "backlight",
                "priority": 4.0,  # 중간 우선순위 (중요한 문제)
                "message": "역광이 있어요",
                "detail": f"레퍼런스는 역광 없음, 현재는 역광 (심각도 {user_back['severity']:.0%})",
                "adjustment": "광원을 등지지 말고, 광원이 얼굴을 비추도록 위치를 바꾸세요. 또는 노출 보정 +1~2 EV",
                "adjustment_numeric": {
                    "ev_adjustment": "+1.5",
                    "severity": user_back["severity"]
                }
            })

        # 레퍼런스는 역광인데 사용자는 아님
        elif ref_back["is_backlight"] and not user_back["is_backlight"]:
            backlight_diff = True

            feedback_list.append({
                "category": "backlight",
                "priority": 7.0,  # 낮은 우선순위 (의도된 스타일)
                "message": "역광 효과가 필요해요",
                "detail": f"레퍼런스는 역광 효과 (심각도 {ref_back['severity']:.0%}), 현재는 역광 없음",
                "adjustment": "광원을 등지고 촬영하세요. 노출 보정 +1~2 EV로 얼굴을 밝게",
                "adjustment_numeric": {
                    "ev_adjustment": "+1.5",
                    "target_severity": ref_back["severity"]
                }
            })

    # ==========================================
    # 3. HDR 비교
    # ==========================================
    ref_hdr = ref_lighting["hdr"]
    user_hdr = user_lighting["hdr"]

    hdr_diff = False

    # 레퍼런스는 HDR인데 사용자는 아님
    if ref_hdr["is_hdr"] and not user_hdr["is_hdr"]:
        hdr_diff = True

        # 클리핑 정보
        if user_hdr["shadow_ratio"] > 0.1:
            issue = "어두운 부분이 너무 많아요"
            adjustment = "노출을 +1~2 EV 올리거나, HDR 모드를 켜세요"
        elif user_hdr["highlight_ratio"] > 0.1:
            issue = "밝은 부분이 날아갔어요"
            adjustment = "노출을 -1~2 EV 낮추거나, HDR 모드를 켜세요"
        else:
            issue = "HDR 처리가 필요해요"
            adjustment = "HDR 모드를 켜거나, 후보정에서 섀도우/하이라이트 복구하세요"

        feedback_list.append({
            "category": "hdr",
            "priority": 6.0,  # 중간 우선순위
            "message": issue,
            "detail": f"레퍼런스는 HDR 처리됨 (섀도우 {ref_hdr['shadow_ratio']:.1%}, 하이라이트 {ref_hdr['highlight_ratio']:.1%})",
            "adjustment": adjustment,
            "adjustment_numeric": {
                "hdr_mode": True,
                "shadow_ratio": user_hdr["shadow_ratio"],
                "highlight_ratio": user_hdr["highlight_ratio"]
            }
        })

    # 레퍼런스는 HDR 아닌데 사용자는 HDR
    elif not ref_hdr["is_hdr"] and user_hdr["is_hdr"]:
        hdr_diff = True

        feedback_list.append({
            "category": "hdr",
            "priority": 8.0,  # 낮은 우선순위 (큰 문제 아님)
            "message": "HDR 모드를 꺼야 해요",
            "detail": f"레퍼런스는 HDR 없음, 현재는 HDR 처리됨",
            "adjustment": "HDR 모드를 끄고 촬영하세요",
            "adjustment_numeric": {
                "hdr_mode": False
            }
        })

    return {
        "available": True,
        "direction_match": direction_match,
        "backlight_diff": backlight_diff,
        "hdr_diff": hdr_diff,
        "feedback": feedback_list,
        "has_issues": len(feedback_list) > 0
    }


# ============================================================
# 테스트
# ============================================================
if __name__ == "__main__":
    test_img = PROJECT_ROOT / "data" / "test_images" / "test1.jpg"
    
    try:
        analyzer = LightingAnalyzer(str(test_img))
        result = analyzer.analyze_all()

        print("\n" + "="*60)
        print("💡 LIGHTING ANALYSIS RESULT")
        print("="*60)

        # 조명 방향
        light_dir = result["light_direction"]
        if light_dir["available"]:
            print(f"\n🔦 Light Direction: {light_dir['direction']}")
            print(f"   └ Confidence: {light_dir['confidence']:.2f}")
            print(f"   └ Brightness map:")
            for side, bright in light_dir["brightness_map"].items():
                print(f"      - {side}: {bright:.1f}")
        else:
            print(f"\n🔦 Light Direction: 사용 불가")

        # 역광
        backlight = result["backlight"]
        if backlight["available"]:
            print(f"\n🌅 Backlight: {'있음' if backlight['is_backlight'] else '없음'}")
            print(f"   └ Severity: {backlight['severity']:.2f}")
            print(f"   └ FG brightness: {backlight['fg_brightness']:.1f}")
            print(f"   └ BG brightness: {backlight['bg_brightness']:.1f}")
            print(f"   └ Ratio: {backlight['ratio']:.2f}")
        else:
            print(f"\n🌅 Backlight: 사용 불가")

        # HDR
        hdr = result["hdr"]
        print(f"\n🎨 HDR: {'처리됨' if hdr['is_hdr'] else '없음'}")
        print(f"   └ Shadow ratio: {hdr['shadow_ratio']:.2%}")
        print(f"   └ Highlight ratio: {hdr['highlight_ratio']:.2%}")
        print(f"   └ Dynamic range: {hdr['dynamic_range']:.2%}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
