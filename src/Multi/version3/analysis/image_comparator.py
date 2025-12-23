# ============================================================
# 🔄 TryAngle - Image Comparator
# 레퍼런스 vs 사용자 이미지 비교 + 피드백 생성
# ============================================================

import numpy as np
from typing import List, Dict

# ImageAnalyzer import
import sys
from pathlib import Path

ANALYSIS_DIR = Path(__file__).resolve().parent
VERSION3_DIR = ANALYSIS_DIR.parent
PROJECT_ROOT = VERSION3_DIR
while PROJECT_ROOT != PROJECT_ROOT.parent and not ((PROJECT_ROOT / "data").exists() and (PROJECT_ROOT / "src").exists()):
    PROJECT_ROOT = PROJECT_ROOT.parent

if str(VERSION3_DIR) not in sys.path:
    sys.path.append(str(VERSION3_DIR))
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.append(str(ANALYSIS_DIR))

from image_analyzer import ImageAnalyzer

# Phase 1: Feedback Formatter
try:
    import sys
    from pathlib import Path
    utils_dir = VERSION3_DIR / "utils"
    if str(utils_dir) not in sys.path:
        sys.path.append(str(utils_dir))
    from feedback_formatter import FeedbackFormatter, BeginnerMessageAdapter
    FEEDBACK_FORMATTER_AVAILABLE = True
except ImportError:
    FEEDBACK_FORMATTER_AVAILABLE = False
    print("⚠️ Feedback Formatter not available (Phase 1)")

# Pose comparison
try:
    from analysis.pose_analyzer import compare_poses
    POSE_COMPARE_AVAILABLE = True
except ImportError:
    POSE_COMPARE_AVAILABLE = False
    print("⚠️ Pose comparison not available")

# EXIF comparison
try:
    from analysis.exif_analyzer import compare_exif
    EXIF_COMPARE_AVAILABLE = True
except ImportError:
    EXIF_COMPARE_AVAILABLE = False
    print("⚠️ EXIF comparison not available")

# Quality comparison
try:
    from analysis.quality_analyzer import compare_quality
    QUALITY_COMPARE_AVAILABLE = True
except ImportError:
    QUALITY_COMPARE_AVAILABLE = False
    print("⚠️ Quality comparison not available")

# Lighting comparison
try:
    from analysis.lighting_analyzer import compare_lighting
    LIGHTING_COMPARE_AVAILABLE = True
except ImportError:
    LIGHTING_COMPARE_AVAILABLE = False
    print("⚠️ Lighting comparison not available")


class ImageComparator:
    """
    레퍼런스 vs 사용자 이미지 비교
    클러스터 정보 + 픽셀 분석 모두 활용
    """

    def __init__(self, reference_path: str, user_path: str, use_movenet: bool = False):
        """
        Args:
            reference_path: 레퍼런스 이미지 경로
            user_path: 사용자 이미지 경로
            use_movenet: True면 MoveNet 사용, False면 YOLO11 사용 (Phase 2-4)
        """
        print("\n" + "="*60)
        print("📸 레퍼런스 이미지 분석")
        print("="*60)
        self.ref_analyzer = ImageAnalyzer(reference_path, use_movenet=use_movenet)
        self.ref_data = self.ref_analyzer.analyze()

        print("\n" + "="*60)
        print("📸 사용자 이미지 분석")
        print("="*60)
        self.user_analyzer = ImageAnalyzer(user_path, use_movenet=use_movenet)
        self.user_data = self.user_analyzer.analyze()
        
    def compare(self) -> Dict:
        """
        모든 차원에서 비교
        """

        return {
            "cluster_comparison": self._compare_clusters(),
            "pose_comparison": self._compare_pose(),
            "exif_comparison": self._compare_exif(),
            "quality_comparison": self._compare_quality(),
            "lighting_comparison": self._compare_lighting(),
            "depth_comparison": self._compare_depth(),
            "brightness_comparison": self._compare_brightness(),
            "color_comparison": self._compare_color(),
            "composition_comparison": self._compare_composition(),
        }
    
    def _compare_clusters(self) -> Dict:
        """클러스터 비교 (스타일 DNA)"""
        ref_cluster = self.ref_data["cluster"]["cluster_id"]
        user_cluster = self.user_data["cluster"]["cluster_id"]
        
        same_style = (ref_cluster == user_cluster)
        
        # 임베딩 거리 계산 (128D)
        ref_emb = self.ref_data["cluster"]["embedding_128d"]
        user_emb = self.user_data["cluster"]["embedding_128d"]
        embedding_distance = float(np.linalg.norm(ref_emb - user_emb))
        
        return {
            "same_cluster": same_style,
            "reference_cluster": ref_cluster,
            "user_cluster": user_cluster,
            "reference_label": self.ref_data["cluster"]["cluster_label"],
            "user_label": self.user_data["cluster"]["cluster_label"],
            "embedding_distance": embedding_distance,
            "style_match": "similar" if embedding_distance < 0.5 else "different"
        }
    
    def _compare_depth(self) -> Dict:
        """거리 비교 (MiDaS)"""
        ref_depth = self.ref_data["depth"]["depth_mean"]
        user_depth = self.user_data["depth"]["depth_mean"]

        depth_ratio = user_depth / (ref_depth + 1e-8)

        # 피드백 계산 (구체적인 걸음수 포함)
        if depth_ratio > 1.15:  # 15% 이상 차이
            percent_diff = int((depth_ratio - 1) * 100)

            # 걸음수 계산 (평균 걸음 70cm, 일반 촬영거리 2-3m 가정)
            estimated_distance_m = 2.5  # 평균 촬영 거리
            distance_change_m = estimated_distance_m * (depth_ratio - 1)
            steps = max(1, round(distance_change_m / 0.7))  # 0.7m per step

            feedback = f"피사체에 약 {steps}걸음 더 가까이 가세요 (약 {percent_diff}% 더 가까이)"
            action = "move_closer"
        elif depth_ratio < 0.85:
            percent_diff = int((1 - depth_ratio) * 100)

            # 걸음수 계산
            estimated_distance_m = 2.5
            distance_change_m = estimated_distance_m * (1 - depth_ratio)
            steps = max(1, round(distance_change_m / 0.7))

            feedback = f"피사체에서 약 {steps}걸음 뒤로 가세요 (약 {percent_diff}% 더 멀리)"
            action = "move_away"
        else:
            feedback = "거리는 적절합니다"
            action = "none"

        return {
            "ref_depth": ref_depth,
            "user_depth": user_depth,
            "ratio": depth_ratio,
            "feedback": feedback,
            "action": action,
            "steps": steps if action != "none" else 0
        }
    
    def _compare_brightness(self) -> Dict:
        """밝기 비교"""
        ref_brightness = self.ref_data["pixels"]["brightness"]
        user_brightness = self.user_data["pixels"]["brightness"]
        
        diff = user_brightness - ref_brightness
        ev_adjustment = diff / 25.0 * 0.3  # 대략 변환 (약 25밝기 = 0.3 EV)
        
        if abs(diff) > 15:
            feedback = f"노출을 {abs(ev_adjustment):.1f} EV {'올리세요' if diff < 0 else '낮추세요'}"
            action = "increase_exposure" if diff < 0 else "decrease_exposure"
        else:
            feedback = "밝기는 적절합니다"
            action = "none"
        
        return {
            "ref_brightness": ref_brightness,
            "user_brightness": user_brightness,
            "difference": diff,
            "ev_adjustment": ev_adjustment,
            "feedback": feedback,
            "action": action
        }
    
    def _compare_color(self) -> Dict:
        """색감 비교"""
        ref_saturation = self.ref_data["pixels"]["saturation"]
        user_saturation = self.user_data["pixels"]["saturation"]
        
        ref_temp = self.ref_data["pixels"]["color_temperature"]
        user_temp = self.user_data["pixels"]["color_temperature"]
        
        sat_diff = user_saturation - ref_saturation
        
        feedback_list = []
        
        # 채도
        if abs(sat_diff) > 0.1:
            percent = abs(sat_diff) * 100
            feedback_list.append(
                f"채도를 {'높이세요' if sat_diff < 0 else '낮추세요'} (약 {percent:.0f}%)"
            )
        
        # 색온도
        if ref_temp != user_temp:
            if ref_temp == "warm" and user_temp != "warm":
                feedback_list.append("색감을 더 따뜻하게 조정하세요 (화이트밸런스 조정)")
            elif ref_temp == "cool" and user_temp != "cool":
                feedback_list.append("색감을 더 차갑게 조정하세요 (화이트밸런스 조정)")
        
        if not feedback_list:
            feedback_list = ["색감은 적절합니다"]
        
        return {
            "ref_saturation": ref_saturation,
            "user_saturation": user_saturation,
            "saturation_diff": sat_diff,
            "ref_temperature": ref_temp,
            "user_temperature": user_temp,
            "feedback": feedback_list
        }
    
    def _compare_composition(self) -> Dict:
        """구도 비교 (프레이밍 포함)"""
        ref_comp = self.ref_data["composition"]
        user_comp = self.user_data["composition"]

        tilt_diff = user_comp["tilt_angle"] - ref_comp["tilt_angle"]

        feedback_list = []

        # 프레이밍/줌 비교 (포즈 bbox 활용)
        if self.ref_data.get("pose") and self.user_data.get("pose"):
            ref_pose = self.ref_data["pose"]
            user_pose = self.user_data["pose"]

            if ref_pose.get("bbox") and user_pose.get("bbox"):
                ref_bbox = ref_pose["bbox"]  # [x1, y1, x2, y2] normalized
                user_bbox = user_pose["bbox"]

                # bbox 크기 계산
                ref_width = ref_bbox[2] - ref_bbox[0]
                ref_height = ref_bbox[3] - ref_bbox[1]
                ref_area = ref_width * ref_height

                user_width = user_bbox[2] - user_bbox[0]
                user_height = user_bbox[3] - user_bbox[1]
                user_area = user_width * user_height

                # 줌 비율 계산
                zoom_ratio = user_area / (ref_area + 1e-8)

                if zoom_ratio < 0.7:  # 사용자가 너무 줌아웃
                    zoom_needed = 1 / zoom_ratio
                    percent = int((zoom_needed - 1) * 100)
                    feedback_list.append(f"화면을 {zoom_needed:.1f}배 확대하세요 (줌 {percent}% 늘리기)")

                elif zoom_ratio > 1.4:  # 사용자가 너무 줌인
                    zoom_needed = zoom_ratio
                    percent = int((zoom_needed - 1) * 100)
                    feedback_list.append(f"화면을 {1/zoom_needed:.1f}배 축소하세요 (줌 {percent}% 줄이기)")

                # 크롭 제안 (bbox 위치 비교)
                ref_center_x = (ref_bbox[0] + ref_bbox[2]) / 2
                ref_center_y = (ref_bbox[1] + ref_bbox[3]) / 2
                user_center_x = (user_bbox[0] + user_bbox[2]) / 2
                user_center_y = (user_bbox[1] + user_bbox[3]) / 2

                x_shift = user_center_x - ref_center_x
                y_shift = user_center_y - ref_center_y

                if abs(y_shift) > 0.1:
                    percent = abs(int(y_shift * 100))
                    if y_shift > 0:
                        feedback_list.append(f"프레이밍: 화면 위쪽 {percent}% 더 포함하세요")
                    else:
                        feedback_list.append(f"프레이밍: 화면 아래쪽 {percent}% 더 포함하세요")

                if abs(x_shift) > 0.1:
                    percent = abs(int(x_shift * 100))
                    if x_shift > 0:
                        feedback_list.append(f"프레이밍: 화면 왼쪽 {percent}% 더 포함하세요")
                    else:
                        feedback_list.append(f"프레이밍: 화면 오른쪽 {percent}% 더 포함하세요")

        # 기울기
        if abs(tilt_diff) > 3:
            feedback_list.append(
                f"카메라를 {'왼쪽' if tilt_diff > 0 else '오른쪽'}으로 {abs(tilt_diff):.1f}도 기울이세요"
            )

        # 무게중심 비교
        ref_center = ref_comp["center_of_mass"]
        user_center = user_comp["center_of_mass"]

        x_diff = user_center["x"] - ref_center["x"]
        y_diff = user_center["y"] - ref_center["y"]

        if abs(x_diff) > 0.15:
            feedback_list.append(
                f"피사체를 화면의 {'왼쪽' if x_diff > 0 else '오른쪽'}으로 이동하세요"
            )

        if abs(y_diff) > 0.15:
            feedback_list.append(
                f"피사체를 화면의 {'위쪽' if y_diff > 0 else '아래쪽'}으로 이동하세요"
            )

        if not feedback_list:
            feedback_list = ["구도는 적절합니다"]

        return {
            "ref_tilt": ref_comp["tilt_angle"],
            "user_tilt": user_comp["tilt_angle"],
            "tilt_diff": tilt_diff,
            "ref_center": ref_center,
            "user_center": user_center,
            "feedback": feedback_list
        }

    def _compare_pose(self) -> Dict:
        """포즈 비교 (YOLO + MediaPipe)"""
        if not POSE_COMPARE_AVAILABLE:
            return {
                "available": False,
                "feedback": []
            }

        ref_pose = self.ref_data.get("pose")
        user_pose = self.user_data.get("pose")

        if ref_pose is None or user_pose is None:
            return {
                "available": False,
                "feedback": ["포즈 분석을 사용할 수 없습니다"]
            }

        # 포즈 비교 실행
        try:
            comparison = compare_poses(ref_pose, user_pose)
            return {
                "available": True,
                "similarity": comparison["similarity"],
                "angle_differences": comparison.get("angle_differences", {}),
                "position_differences": comparison.get("position_differences", {}),
                "feedback": comparison["feedback"]
            }
        except Exception as e:
            print(f"  ⚠️ Pose comparison failed: {e}")
            return {
                "available": False,
                "feedback": [f"포즈 비교 실패: {str(e)}"]
            }

    def _compare_exif(self) -> Dict:
        """EXIF 비교 (카메라 설정)"""
        if not EXIF_COMPARE_AVAILABLE:
            return {
                "available": False,
                "feedback": []
            }

        ref_exif_data = self.ref_data.get("exif")
        user_exif_data = self.user_data.get("exif")

        if ref_exif_data is None or user_exif_data is None:
            return {
                "available": False,
                "feedback": ["EXIF 데이터가 없습니다"]
            }

        ref_settings = ref_exif_data["camera_settings"]
        user_settings = user_exif_data["camera_settings"]

        # EXIF 비교 실행
        try:
            comparison = compare_exif(ref_settings, user_settings)
            return {
                "available": True,
                "has_differences": comparison.get("has_differences", False),
                "iso_diff": comparison.get("iso_diff"),
                "f_number_diff": comparison.get("f_number_diff"),
                "shutter_speed_ratio": comparison.get("shutter_speed_ratio"),
                "focal_length_diff": comparison.get("focal_length_diff"),
                "white_balance_match": comparison.get("white_balance_match"),
                "feedback": comparison["feedback"],
                "ref_settings": ref_settings,
                "user_settings": user_settings
            }
        except Exception as e:
            print(f"  ⚠️ EXIF comparison failed: {e}")
            return {
                "available": False,
                "feedback": [f"EXIF 비교 실패: {str(e)}"]
            }

    def _compare_quality(self) -> Dict:
        """Quality 비교 (노이즈, 블러, 선명도, 대비)"""
        if not QUALITY_COMPARE_AVAILABLE:
            return {
                "available": False,
                "feedback": []
            }

        ref_quality = self.ref_data.get("quality")
        user_quality = self.user_data.get("quality")

        if ref_quality is None or user_quality is None:
            return {
                "available": False,
                "feedback": []
            }

        # Quality 비교 실행 (상대적 평가 기반)
        try:
            comparison = compare_quality(ref_quality, user_quality)
            return comparison
        except Exception as e:
            print(f"  ⚠️ Quality comparison failed: {e}")
            return {
                "available": False,
                "feedback": []
            }

    def _compare_lighting(self) -> Dict:
        """Lighting 비교 (조명 방향, 역광, HDR)"""
        if not LIGHTING_COMPARE_AVAILABLE:
            return {
                "available": False,
                "feedback": []
            }

        ref_lighting = self.ref_data.get("lighting")
        user_lighting = self.user_data.get("lighting")

        if ref_lighting is None or user_lighting is None:
            return {
                "available": False,
                "feedback": []
            }

        # Lighting 비교 실행
        try:
            comparison = compare_lighting(ref_lighting, user_lighting)
            return comparison
        except Exception as e:
            print(f"  ⚠️ Lighting comparison failed: {e}")
            return {
                "available": False,
                "feedback": []
            }

    def get_prioritized_feedback(self) -> List[Dict]:
        """
        우선순위에 따라 피드백 정렬
        0순위: 클러스터 (정보성)
        0.5순위: 포즈 (매우 중요!)
        1순위: 카메라 설정 (EXIF)
        2순위: 거리 (depth)
        3순위: 밝기
        4순위: 색감
        5순위: 구도
        """
        comparison = self.compare()

        feedback_list = []

        # 0순위: 클러스터 (정보성)
        cluster_comp = comparison["cluster_comparison"]
        if not cluster_comp["same_cluster"]:
            feedback_list.append({
                "priority": 0,
                "category": "style",
                "message": f"⚠️ 스타일이 다릅니다",
                "detail": (
                    f"레퍼런스: Cluster {cluster_comp['reference_cluster']} - {cluster_comp['reference_label']}\n"
                    f"     사용자: Cluster {cluster_comp['user_cluster']} - {cluster_comp['user_label']}"
                )
            })
        else:
            feedback_list.append({
                "priority": 0,
                "category": "style",
                "message": f"✅ 같은 스타일입니다 (Cluster {cluster_comp['reference_cluster']})",
                "detail": f"{cluster_comp['reference_label']}"
            })

        # 0.5순위: 포즈 (매우 중요!)
        pose_comp = comparison["pose_comparison"]
        if pose_comp["available"] and pose_comp["feedback"]:
            if pose_comp["feedback"][0] != "✅ 포즈가 적절합니다":
                for fb in pose_comp["feedback"]:
                    similarity = pose_comp.get("similarity", 0.0)
                    feedback_list.append({
                        "priority": 0.5,
                        "category": "pose",
                        "message": fb,
                        "detail": f"포즈 유사도: {similarity:.2%}"
                    })
            else:
                # 포즈가 적절한 경우
                similarity = pose_comp.get("similarity", 1.0)
                feedback_list.append({
                    "priority": 0.5,
                    "category": "pose",
                    "message": "✅ 포즈가 적절합니다",
                    "detail": f"유사도: {similarity:.2%}"
                })

        # 1순위: 카메라 설정 (EXIF)
        exif_comp = comparison["exif_comparison"]
        if exif_comp["available"] and exif_comp.get("has_differences", False):
            for fb in exif_comp["feedback"]:
                feedback_list.append({
                    "priority": 1,
                    "category": "camera_settings",
                    "message": fb,
                    "detail": "카메라 설정을 조정하세요"
                })

        # Quality: 동적 우선순위 (0.5~8.0)
        quality_comp = comparison["quality_comparison"]
        if quality_comp["available"] and quality_comp["feedback"]:
            for fb_item in quality_comp["feedback"]:
                # fb_item = {category, ref_value, user_value, difference_percent,
                #           direction, is_critical, is_style, message, adjustment,
                #           adjustment_numeric, priority}
                feedback_list.append({
                    "priority": fb_item["priority"],
                    "category": fb_item["category"],
                    "message": fb_item["message"],
                    "detail": fb_item["adjustment"]
                })

        # Lighting: 동적 우선순위 (4~8)
        lighting_comp = comparison["lighting_comparison"]
        if lighting_comp.get("available", False) and lighting_comp.get("has_issues", False):
            for fb_item in lighting_comp["feedback"]:
                # fb_item = {category, priority, message, detail, adjustment, adjustment_numeric}
                feedback_list.append({
                    "priority": fb_item["priority"],
                    "category": fb_item["category"],
                    "message": fb_item["message"],
                    "detail": fb_item["adjustment"]
                })

        # 2순위: 거리
        depth_comp = comparison["depth_comparison"]
        if depth_comp["action"] != "none":
            feedback_list.append({
                "priority": 2,
                "category": "distance",
                "message": depth_comp["feedback"],
                "detail": f"레퍼런스 depth={depth_comp['ref_depth']:.1f}, 현재={depth_comp['user_depth']:.1f} (비율: {depth_comp['ratio']:.2f})"
            })

        # 3순위: 밝기
        brightness_comp = comparison["brightness_comparison"]
        if brightness_comp["action"] != "none":
            feedback_list.append({
                "priority": 3,
                "category": "exposure",
                "message": brightness_comp["feedback"],
                "detail": f"레퍼런스 밝기={brightness_comp['ref_brightness']:.1f}, 현재={brightness_comp['user_brightness']:.1f} (차이: {brightness_comp['difference']:.1f})"
            })

        # 4순위: 색감
        color_comp = comparison["color_comparison"]
        if color_comp["feedback"][0] != "색감은 적절합니다":
            for fb in color_comp["feedback"]:
                feedback_list.append({
                    "priority": 4,
                    "category": "color",
                    "message": fb,
                    "detail": f"레퍼런스 채도={color_comp['ref_saturation']:.2f}, 현재={color_comp['user_saturation']:.2f} (차이: {color_comp['saturation_diff']:.2f})"
                })

        # 5순위: 구도
        comp_comp = comparison["composition_comparison"]
        if comp_comp["feedback"][0] != "구도는 적절합니다":
            for fb in comp_comp["feedback"]:
                feedback_list.append({
                    "priority": 5,
                    "category": "composition",
                    "message": fb,
                    "detail": f"기울기 차이={comp_comp['tilt_diff']:.1f}도"
                })
        
        # 우선순위 정렬
        feedback_list.sort(key=lambda x: x["priority"])

        return feedback_list

    def get_top_feedback(
        self,
        top_k: int = 3,
        include_style: bool = False,
        user_level: str = 'beginner'
    ) -> Dict:
        """
        Phase 1.1: Top-K 피드백 + 초보자 모드

        Args:
            top_k: 상위 몇 개까지 표시 (기본 3개)
            include_style: 클러스터 정보도 Top-K에 포함할지
            user_level: 'beginner', 'intermediate', 'expert'

        Returns:
            {
                'style': 스타일 정보,
                'primary': Top-K 피드백,
                'secondary': 나머지 피드백,
                'total_count': 전체 개수,
                'more_count': 더보기 개수,
                'critical_count': 치명적 문제 개수,
                'display_text': UI용 텍스트
            }
        """
        # 전체 피드백 가져오기
        feedback_list = self.get_prioritized_feedback()

        if not FEEDBACK_FORMATTER_AVAILABLE:
            # Fallback: formatter 없으면 기존 방식
            return {
                'primary': feedback_list[:top_k],
                'secondary': feedback_list[top_k:],
                'total_count': len(feedback_list),
                'more_count': len(feedback_list[top_k:]),
                'display_text': '\n'.join([f"{i+1}. {fb['message']}"
                                          for i, fb in enumerate(feedback_list[:top_k])])
            }

        # Phase 1: Formatter 사용
        formatter = FeedbackFormatter(user_level=user_level)
        adapter = BeginnerMessageAdapter()

        # 초보자 모드 적용
        if user_level == 'beginner':
            feedback_list = [adapter.adapt_message(fb, user_level)
                           for fb in feedback_list]

        # Top-K 추출
        top_k_result = formatter.format_top_k(
            feedback_list,
            top_k=top_k,
            include_style=include_style
        )

        # UI용 텍스트 생성
        top_k_result['display_text'] = formatter.format_for_display(top_k_result)
        top_k_result['secondary_text'] = formatter.format_secondary(top_k_result)

        return top_k_result

    def get_beginner_friendly_feedback(self) -> str:
        """
        Phase 1.2: 초보자용 간단한 피드백

        Returns:
            초보자가 이해하기 쉬운 텍스트
        """
        top_k_result = self.get_top_feedback(top_k=3, user_level='beginner')
        return top_k_result['display_text']

    def get_expert_feedback(self) -> str:
        """
        전문가용 상세 피드백

        Returns:
            기술적 세부사항 포함
        """
        feedback_list = self.get_prioritized_feedback()

        lines = []
        for i, fb in enumerate(feedback_list, 1):
            lines.append(f"{i}. [{fb['category'].upper()}] {fb['message']}")
            if fb.get('detail'):
                lines.append(f"   {fb['detail']}")
            lines.append(f"   Priority: {fb['priority']}")
            lines.append("")

        return "\n".join(lines)


# ============================================================
# 테스트
# ============================================================
if __name__ == "__main__":
    ref_path = PROJECT_ROOT / "data" / "test_images" / "test1.jpg"
    user_path = PROJECT_ROOT / "data" / "test_images" / "test1.jpg"  # 같은 이미지로 테스트
    
    try:
        comparator = ImageComparator(str(ref_path), str(user_path))
        feedback = comparator.get_prioritized_feedback()
        
        print("\n" + "="*60)
        print("📋 촬영 가이드")
        print("="*60)
        
        for i, fb in enumerate(feedback, 1):
            print(f"\n{i}. [{fb['category'].upper()}]")
            print(f"   {fb['message']}")
            print(f"   └ {fb['detail']}")
        
        print("\n" + "="*60)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
