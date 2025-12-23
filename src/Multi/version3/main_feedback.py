# ============================================================
# 🎯 TryAngle - Main Feedback System (Enhanced)
# 레퍼런스 vs 사용자 이미지 비교 및 피드백 제공
# Phase 1-3 통합: 사용자 친화적 피드백 시스템
# ============================================================

import os
import sys
from pathlib import Path
from typing import Optional, Dict, List

# 경로 설정
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR
while PROJECT_ROOT != PROJECT_ROOT.parent and not ((PROJECT_ROOT / "data").exists() and (PROJECT_ROOT / "src").exists()):
    PROJECT_ROOT = PROJECT_ROOT.parent

VERSION3_DIR = PROJECT_ROOT / "src" / "Multi" / "version3"
ANALYSIS_DIR = VERSION3_DIR / "analysis"
UTILS_DIR = VERSION3_DIR / "utils"

if str(VERSION3_DIR) not in sys.path:
    sys.path.append(str(VERSION3_DIR))
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.append(str(ANALYSIS_DIR))
if str(UTILS_DIR) not in sys.path:
    sys.path.append(str(UTILS_DIR))

from analysis.image_comparator import ImageComparator

# Phase 1-3 imports
try:
    from utils.feedback_formatter import FeedbackFormatter
    from utils.workflow_guide import WorkflowGuide
    from utils.progress_tracker import ProgressTracker
    from utils.priority_system import PriorityClassifier
    from utils.adaptive_thresholds import AdaptiveThresholdManager
    from utils.reference_recommender import ReferenceRecommender
    ENHANCED_FEATURES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ 일부 고급 기능을 사용할 수 없습니다: {e}")
    ENHANCED_FEATURES_AVAILABLE = False


# ============================================================
# 글로벌 진행도 트래커 (세션 간 유지)
# ============================================================
_progress_tracker = None


def get_enhanced_feedback(
    reference_path: str,
    user_path: str,
    user_level: str = 'beginner',
    top_k: int = 3,
    use_workflow: bool = True,
    track_progress: bool = True,
    recommend_references: bool = False
) -> Dict:
    """
    Phase 1-3 통합 피드백 시스템

    Args:
        reference_path: 레퍼런스 이미지 경로
        user_path: 사용자 이미지 경로
        user_level: 사용자 수준 ('beginner', 'intermediate', 'expert')
        top_k: 표시할 최대 피드백 개수
        use_workflow: 워크플로우 기반 정렬 사용 여부
        track_progress: 진행도 추적 사용 여부
        recommend_references: 레퍼런스 추천 사용 여부

    Returns:
        {
            'formatted_feedback': 포맷된 피드백 딕셔너리,
            'workflow_steps': 워크플로우 단계별 피드백 (if use_workflow),
            'progress': 진행도 정보 (if track_progress),
            'recommendations': 추천 레퍼런스 (if recommend_references),
            'display_text': 사용자 친화적 출력 텍스트
        }
    """
    global _progress_tracker

    if not ENHANCED_FEATURES_AVAILABLE:
        raise RuntimeError("Enhanced features are not available. Check imports.")

    # ==========================================
    # 1. 기본 이미지 비교
    # ==========================================
    comparator = ImageComparator(reference_path, user_path)
    raw_feedback = comparator.get_prioritized_feedback()

    # ==========================================
    # 2. Phase 2.3: 우선순위 분류
    # ==========================================
    priority_groups = PriorityClassifier.group_by_priority(raw_feedback)

    # ==========================================
    # 3. Phase 1.1 & 1.2: 피드백 포맷팅
    # ==========================================
    formatter = FeedbackFormatter(user_level=user_level)
    formatted = formatter.format_top_k(raw_feedback, top_k=top_k, include_style=True)
    display_text = formatter.format_for_display(formatted)

    result = {
        'formatted_feedback': formatted,
        'priority_groups': priority_groups,
        'display_text': display_text
    }

    # ==========================================
    # 4. Phase 2.1: 워크플로우 가이드
    # ==========================================
    if use_workflow:
        workflow_guide = WorkflowGuide()
        workflow_steps = workflow_guide.organize_by_workflow(raw_feedback)
        workflow_text = workflow_guide.format_workflow_text(workflow_steps, show_all=False)
        result['workflow_steps'] = workflow_steps
        result['workflow_text'] = workflow_text

    # ==========================================
    # 5. Phase 2.2: 진행도 추적
    # ==========================================
    if track_progress:
        if _progress_tracker is None:
            _progress_tracker = ProgressTracker()
            _progress_tracker.set_initial_state(raw_feedback)
            progress = {
                'overall_score': _progress_tracker.history[0]['score'],
                'initial_score': _progress_tracker.history[0]['score'],
                'score_improvement': 0,
                'progress_percent': 0,
                'improved_items': [],
                'remaining_items': raw_feedback,
                'new_issues': [],
                'celebration': False,
                'attempt_number': 1,
                'is_first_attempt': True
            }
        else:
            progress = _progress_tracker.update_progress(raw_feedback)
            progress['is_first_attempt'] = False

        progress_text = _progress_tracker.format_progress_text(progress)
        encouragement = _progress_tracker.get_encouragement_message(progress)

        result['progress'] = progress
        result['progress_text'] = progress_text
        result['encouragement'] = encouragement

    # ==========================================
    # 6. Phase 3.1: 레퍼런스 추천
    # ==========================================
    if recommend_references:
        try:
            # 사용자 이미지의 클러스터 정보 가져오기
            comparison = comparator.compare()
            user_cluster = comparison['cluster_comparison']['user_cluster']
            user_embedding = comparator.user_analyzer.features.get('embedding', None)

            if user_embedding is not None:
                recommender = ReferenceRecommender()
                recommendations = recommender.recommend(
                    user_image_path=user_path,
                    user_cluster_id=user_cluster,
                    user_embedding=user_embedding,
                    top_k=3
                )

                rec_text = recommender.format_recommendations(recommendations)
                result['recommendations'] = recommendations
                result['recommendations_text'] = rec_text
        except Exception as e:
            print(f"⚠️ 레퍼런스 추천 실패: {e}")
            result['recommendations'] = []

    return result


def print_enhanced_feedback(
    reference_path: str,
    user_path: str,
    user_level: str = 'beginner',
    top_k: int = 3,
    show_workflow: bool = True,
    show_progress: bool = True,
    show_recommendations: bool = False,
    show_detailed: bool = False
):
    """
    사용자 친화적 피드백 출력

    Phase 1-3 통합 버전
    """
    print("\n" + "="*70)
    print("🎯 TryAngle - 스마트 촬영 가이드".center(70))
    print("="*70)
    print(f"📸 레퍼런스: {Path(reference_path).name}")
    print(f"👤 사용자  : {Path(user_path).name}")
    print(f"📊 수준    : {user_level.upper()}")

    try:
        # 향상된 피드백 가져오기
        result = get_enhanced_feedback(
            reference_path=reference_path,
            user_path=user_path,
            user_level=user_level,
            top_k=top_k,
            use_workflow=show_workflow,
            track_progress=show_progress,
            recommend_references=show_recommendations
        )

        # ==========================================
        # 진행도 표시 (Phase 2.2)
        # ==========================================
        if show_progress and 'progress' in result:
            print("\n" + result['progress_text'])
            print(f"\n💬 {result['encouragement']}")

        # ==========================================
        # 워크플로우 가이드 (Phase 2.1)
        # ==========================================
        if show_workflow and 'workflow_text' in result:
            print("\n" + "="*70)
            print("📋 촬영 워크플로우 가이드".center(70))
            print("="*70)
            print(result['workflow_text'])
        else:
            # 기본 피드백 표시 (Phase 1.1 & 1.2)
            print("\n" + "="*70)
            print("📋 촬영 가이드".center(70))
            print("="*70)
            print(result['display_text'])

        # ==========================================
        # 우선순위 그룹 표시 (Phase 2.3)
        # ==========================================
        priority_groups = result.get('priority_groups', {})
        if any(priority_groups.values()):
            print("\n" + "="*70)
            print("🎯 우선순위별 조정사항".center(70))
            print("="*70)
            print(PriorityClassifier.format_grouped_feedback(priority_groups))

        # ==========================================
        # 레퍼런스 추천 (Phase 3.1)
        # ==========================================
        if show_recommendations and 'recommendations_text' in result:
            print("\n" + result['recommendations_text'])

        # ==========================================
        # 상세 정보 (선택)
        # ==========================================
        if show_detailed:
            comparator = ImageComparator(reference_path, user_path)
            comparison = comparator.compare()

            print("\n" + "="*70)
            print("📊 상세 비교 정보".center(70))
            print("="*70)

            # 클러스터
            cluster = comparison["cluster_comparison"]
            print(f"\n🎯 Cluster:")
            print(f"   레퍼런스: {cluster['reference_cluster']} - {cluster['reference_label']}")
            print(f"   사용자  : {cluster['user_cluster']} - {cluster['user_label']}")
            print(f"   임베딩 거리: {cluster['embedding_distance']:.4f}")

            # Depth
            depth = comparison["depth_comparison"]
            print(f"\n📏 Depth:")
            print(f"   레퍼런스: {depth['ref_depth']:.1f}")
            print(f"   사용자  : {depth['user_depth']:.1f}")
            print(f"   거리 조정: {depth.get('feedback', '적절함')}")

            # 밝기
            brightness = comparison["brightness_comparison"]
            print(f"\n💡 Brightness:")
            print(f"   레퍼런스: {brightness['ref_brightness']:.1f}")
            print(f"   사용자  : {brightness['user_brightness']:.1f}")
            print(f"   차이    : {brightness['difference']:.1f}")

            # 포즈
            pose = comparison["pose_comparison"]
            if pose["available"]:
                print(f"\n🤸 Pose:")
                print(f"   유사도: {pose.get('similarity', 0.0):.2%}")
                if pose.get("feedback"):
                    for fb in pose["feedback"][:3]:
                        print(f"   • {fb.get('message', '')}")

        print("\n" + "="*70)

    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()


def main():
    """
    메인 함수 - Phase 1-3 통합 버전
    """

    # ==========================================
    # 이미지 경로 설정
    # ==========================================
    #reference_path = PROJECT_ROOT / "data" / "clustered_images" / "cluster_5" / "IMG_0561.JPG"
    reference_path = PROJECT_ROOT / "data" / "test_images" / "test3.jpg"
    user_path = PROJECT_ROOT / "data" / "test_images" / "test4.jpg"

    # 경로 확인
    if not reference_path.exists():
        print(f"❌ 레퍼런스 이미지를 찾을 수 없습니다: {reference_path}")
        return

    if not user_path.exists():
        print(f"❌ 사용자 이미지를 찾을 수 없습니다: {user_path}")
        return

    # ==========================================
    # Phase 1-3 통합 피드백 사용
    # ==========================================
    if ENHANCED_FEATURES_AVAILABLE:
        print_enhanced_feedback(
            reference_path=str(reference_path),
            user_path=str(user_path),
            user_level='beginner',  # 'beginner', 'intermediate', 'expert'
            top_k=3,
            show_workflow=True,
            show_progress=True,
            show_recommendations=False,  # 레퍼런스 추천 (느릴 수 있음)
            show_detailed=False  # 상세 정보 표시 여부
        )
        return
    
    print("\n" + "🎯 TryAngle - 촬영 피드백 시스템".center(60, "="))
    print(f"레퍼런스: {reference_path.name}")
    print(f"사용자  : {user_path.name}")
    
    try:
        # ==========================================
        # 이미지 비교
        # ==========================================
        comparator = ImageComparator(str(reference_path), str(user_path))
        
        # ==========================================
        # 우선순위 피드백 받기
        # ==========================================
        feedback = comparator.get_prioritized_feedback()
        
        # ==========================================
        # 결과 출력
        # ==========================================
        print("\n" + "="*60)
        print("📋 촬영 가이드".center(60))
        print("="*60)
        
        # 정보성 메시지 (클러스터)
        info_messages = [fb for fb in feedback if fb["priority"] == 0]
        actionable_messages = [fb for fb in feedback if fb["priority"] > 0]
        
        if info_messages:
            print("\n📊 스타일 정보:")
            for fb in info_messages:
                print(f"   {fb['message']}")
                print(f"   └ {fb['detail']}")
        
        if actionable_messages:
            print("\n🎬 촬영 조정 사항:")
            for i, fb in enumerate(actionable_messages, 1):
                print(f"\n   {i}. [{fb['category'].upper()}] {fb['message']}")
                print(f"      └ {fb['detail']}")
        else:
            print("\n✅ 완벽합니다! 조정 사항이 없습니다.")
        
        print("\n" + "="*60)
        
        # ==========================================
        # 상세 비교 정보 (옵션)
        # ==========================================
        comparison = comparator.compare()
        
        print("\n" + "="*60)
        print("📊 상세 비교 정보".center(60))
        print("="*60)
        
        # 클러스터
        cluster = comparison["cluster_comparison"]
        print(f"\n🎯 Cluster:")
        print(f"   레퍼런스: {cluster['reference_cluster']} - {cluster['reference_label']}")
        print(f"   사용자  : {cluster['user_cluster']} - {cluster['user_label']}")
        print(f"   임베딩 거리: {cluster['embedding_distance']:.4f}")
        
        # Depth
        depth = comparison["depth_comparison"]
        print(f"\n📏 Depth:")
        print(f"   레퍼런스: {depth['ref_depth']:.1f}")
        print(f"   사용자  : {depth['user_depth']:.1f}")
        print(f"   비율    : {depth['ratio']:.2f}")
        
        # 밝기
        brightness = comparison["brightness_comparison"]
        print(f"\n💡 Brightness:")
        print(f"   레퍼런스: {brightness['ref_brightness']:.1f}")
        print(f"   사용자  : {brightness['user_brightness']:.1f}")
        print(f"   차이    : {brightness['difference']:.1f}")
        print(f"   EV 조정 : {brightness['ev_adjustment']:.2f}")
        
        # 색감
        color = comparison["color_comparison"]
        print(f"\n🎨 Color:")
        print(f"   레퍼런스: 채도={color['ref_saturation']:.2f}, 색온도={color['ref_temperature']}")
        print(f"   사용자  : 채도={color['user_saturation']:.2f}, 색온도={color['user_temperature']}")
        
        # 구도
        comp = comparison["composition_comparison"]
        print(f"\n📐 Composition:")
        print(f"   레퍼런스 기울기: {comp['ref_tilt']:.1f}°")
        print(f"   사용자 기울기  : {comp['user_tilt']:.1f}°")
        print(f"   차이          : {comp['tilt_diff']:.1f}°")

        # 포즈
        pose = comparison["pose_comparison"]
        if pose["available"]:
            print(f"\n🤸 Pose:")
            print(f"   유사도: {pose.get('similarity', 0.0):.2%}")
            if pose["angle_differences"]:
                print(f"   각도 차이: {len(pose['angle_differences'])}개 관절")
            if pose["position_differences"]:
                print(f"   위치 차이: {len(pose['position_differences'])}개 키포인트")
        else:
            print(f"\n🤸 Pose: 사용 불가")

        # EXIF (카메라 설정)
        exif = comparison["exif_comparison"]
        if exif["available"]:
            print(f"\n📷 Camera Settings:")
            ref_settings = exif.get("ref_settings", {})
            user_settings = exif.get("user_settings", {})

            if "iso" in ref_settings:
                print(f"   ISO: 레퍼런스={ref_settings.get('iso')}, 현재={user_settings.get('iso')}")
            if "f_number" in ref_settings:
                print(f"   조리개: 레퍼런스=f/{ref_settings.get('f_number'):.1f}, 현재=f/{user_settings.get('f_number'):.1f}")
            if "shutter_speed_display" in ref_settings:
                print(f"   셔터속도: 레퍼런스={ref_settings.get('shutter_speed_display')}, 현재={user_settings.get('shutter_speed_display')}")
            if "focal_length" in ref_settings:
                print(f"   초점거리: 레퍼런스={ref_settings.get('focal_length'):.0f}mm, 현재={user_settings.get('focal_length'):.0f}mm")

            if exif.get("has_differences"):
                print(f"   ⚠️ 카메라 설정 조정 필요")
        else:
            print(f"\n📷 Camera Settings: EXIF 데이터 없음")

        # Quality (노이즈, 블러, 선명도, 대비)
        quality = comparison["quality_comparison"]
        if quality["available"]:
            print(f"\n🔍 Quality:")
            if quality["feedback"]:
                print(f"   피드백 {len(quality['feedback'])}개:")
                for i, fb in enumerate(quality["feedback"], 1):
                    print(f"   {i}. [{fb['category'].upper()}] {fb['message']}")
                    print(f"      조정: {fb['adjustment']}")
                    if fb.get('adjustment_numeric'):
                        print(f"      수치: {fb['adjustment_numeric']}")
            else:
                print(f"   ✅ 품질이 적절합니다")
        else:
            print(f"\n🔍 Quality: 분석 불가")

        # Lighting (조명 방향, 역광, HDR)
        lighting = comparison["lighting_comparison"]
        if lighting.get("available", False):
            print(f"\n💡 Lighting:")
            if lighting.get("has_issues", False) and lighting.get("feedback"):
                print(f"   피드백 {len(lighting['feedback'])}개:")
                for i, fb in enumerate(lighting["feedback"], 1):
                    print(f"   {i}. [{fb['category'].upper()}] {fb['message']}")
                    print(f"      조정: {fb['adjustment']}")
                    if fb.get('adjustment_numeric'):
                        print(f"      수치: {fb['adjustment_numeric']}")
            else:
                print(f"   ✅ 조명이 적절합니다")
        else:
            print(f"\n💡 Lighting: 분석 불가")

        print("\n" + "="*60)
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
