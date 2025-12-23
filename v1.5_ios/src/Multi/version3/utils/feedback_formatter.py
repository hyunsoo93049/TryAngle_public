# ============================================================
# 📝 Feedback Formatter
# Phase 1.1: Top-K 피드백 + 사용자 친화적 포맷팅
# ============================================================

from typing import List, Dict, Optional


class FeedbackFormatter:
    """
    피드백을 사용자 친화적으로 포맷팅

    Features:
    - Top-K 필터링
    - 초보자/전문가 모드
    - 단계별 그룹화
    """

    def __init__(self, user_level: str = 'beginner'):
        """
        Args:
            user_level: 'beginner', 'intermediate', 'expert'
        """
        self.user_level = user_level

    def format_top_k(
        self,
        feedback_list: List[Dict],
        top_k: int = 3,
        include_style: bool = False
    ) -> Dict:
        """
        Top-K 피드백 추출 + 나머지는 "더보기"

        Args:
            feedback_list: get_prioritized_feedback()의 출력
            top_k: 상위 몇 개까지 표시할지
            include_style: 클러스터 정보(priority=0)도 Top-K에 포함할지

        Returns:
            {
                'primary': [최우선 피드백 top_k개],
                'secondary': [나머지 피드백],
                'total_count': 전체 개수,
                'more_count': 더보기 개수,
                'critical_count': 치명적 문제 개수
            }
        """
        # 스타일 정보(priority=0) 분리
        style_feedback = [fb for fb in feedback_list if fb['category'] == 'style']
        action_feedback = [fb for fb in feedback_list if fb['category'] != 'style']

        # Critical 항목 (priority < 1.0) 카운트
        critical_items = [fb for fb in action_feedback if fb['priority'] < 1.0]

        # Top-K 추출
        if include_style:
            primary_feedback = feedback_list[:top_k]
            secondary_feedback = feedback_list[top_k:]
        else:
            primary_feedback = action_feedback[:top_k]
            secondary_feedback = action_feedback[top_k:]

        return {
            'style': style_feedback,
            'primary': primary_feedback,
            'secondary': secondary_feedback,
            'total_count': len(feedback_list),
            'action_count': len(action_feedback),
            'more_count': len(secondary_feedback),
            'critical_count': len(critical_items),
            'has_critical': len(critical_items) > 0
        }

    def format_for_display(self, top_k_result: Dict) -> str:
        """
        UI 표시용 텍스트 생성

        Returns:
            포맷팅된 피드백 문자열
        """
        lines = []

        # 스타일 정보
        if top_k_result['style']:
            style = top_k_result['style'][0]
            lines.append(f"\n{style['message']}")
            lines.append(f"  {style['detail']}")
            lines.append("")

        # Critical 경고
        if top_k_result['has_critical']:
            lines.append("⚠️  중요한 조정이 필요합니다!\n")

        # Primary 피드백 (Top-K)
        lines.append("🎯 지금 조정하세요:")
        for i, fb in enumerate(top_k_result['primary'], 1):
            category_emoji = self._get_category_emoji(fb['category'])
            lines.append(f"\n{i}. {category_emoji} [{fb['category'].upper()}] {fb['message']}")
            if fb.get('detail'):
                lines.append(f"   → {fb['detail']}")

        # 더보기
        if top_k_result['more_count'] > 0:
            lines.append(f"\n... 추가 {top_k_result['more_count']}개 조정 사항 (우선순위 낮음)")

        return "\n".join(lines)

    def format_secondary(self, top_k_result: Dict) -> str:
        """
        Secondary 피드백 포맷팅 (더보기 클릭 시)
        """
        if not top_k_result['secondary']:
            return "모든 피드백이 표시되었습니다."

        lines = ["\n📋 추가 조정 사항:\n"]

        for i, fb in enumerate(top_k_result['secondary'], 1):
            category_emoji = self._get_category_emoji(fb['category'])
            lines.append(f"{i}. {category_emoji} [{fb['category'].upper()}] {fb['message']}")
            if fb.get('detail'):
                lines.append(f"   → {fb['detail']}")
            lines.append("")

        return "\n".join(lines)

    def _get_category_emoji(self, category: str) -> str:
        """카테고리별 이모지"""
        emoji_map = {
            'pose': '🤸',
            'camera_settings': '📷',
            'distance': '📏',
            'exposure': '💡',
            'color': '🎨',
            'composition': '🖼️',
            'quality': '✨',
            'blur': '🌫️',
            'noise': '📊',
            'sharpness': '🔍',
            'lighting': '☀️',
            'backlight': '🌅',
            'lighting_direction': '💡',
            'style': 'ℹ️'
        }
        return emoji_map.get(category, '•')


class BeginnerMessageAdapter:
    """
    초보자용 메시지 변환
    Phase 1.2: 기술 용어 → 쉬운 설명
    """

    # 메시지 템플릿
    BEGINNER_TEMPLATES = {
        # 노출 관련
        'exposure_up': '사진을 더 밝게 찍으세요\n   💡 Tip: 화면을 터치한 후 위로 슬라이드하세요',
        'exposure_down': '사진을 더 어둡게 찍으세요\n   💡 Tip: 화면을 터치한 후 아래로 슬라이드하세요',

        # ISO
        'iso_up': 'ISO를 높이세요 (사진이 더 밝아져요)',
        'iso_down': 'ISO를 낮추세요 (노이즈가 줄어들어요)',
        'iso_auto': 'ISO는 자동(AUTO)으로 두세요',

        # 조리개
        'aperture': '조리개 조정은 고급 기능이에요\n   💡 Tip: "포트레이트 모드"를 사용해보세요',

        # 셔터 스피드
        'shutter_speed': '셔터 속도 조정은 고급 기능이에요\n   💡 Tip: 흔들림 방지를 위해 카메라를 안정적으로 잡으세요',

        # 색감
        'saturation_up': '사진을 더 선명하게 (채도 높이기)',
        'saturation_down': '사진을 더 부드럽게 (채도 낮추기)',

        # 화이트밸런스
        'white_balance': '색온도 조정은 고급 기능이에요\n   💡 Tip: 자동(AUTO)으로 두거나 "일광/흐림" 프리셋을 사용하세요'
    }

    @staticmethod
    def adapt_message(feedback_item: Dict, user_level: str = 'beginner') -> Dict:
        """
        피드백 메시지를 사용자 레벨에 맞게 변환

        Args:
            feedback_item: {priority, category, message, detail}
            user_level: 'beginner', 'intermediate', 'expert'

        Returns:
            변환된 피드백 (원본 유지 or 변환)
        """
        if user_level != 'beginner':
            return feedback_item

        # 초보자 모드: 메시지 간소화
        category = feedback_item['category']
        message = feedback_item['message']

        # 간단한 패턴 매칭으로 변환
        adapted = feedback_item.copy()

        # EV 관련
        if 'EV' in message or '노출' in message:
            if '올리' in message or '밝' in message:
                adapted['message'] = BeginnerMessageAdapter.BEGINNER_TEMPLATES['exposure_up']
            elif '낮추' in message or '어둡' in message:
                adapted['message'] = BeginnerMessageAdapter.BEGINNER_TEMPLATES['exposure_down']

        # ISO
        elif 'ISO' in message:
            if '높이' in message:
                adapted['message'] = BeginnerMessageAdapter.BEGINNER_TEMPLATES['iso_up']
            elif '낮추' in message:
                adapted['message'] = BeginnerMessageAdapter.BEGINNER_TEMPLATES['iso_down']
            else:
                adapted['message'] = BeginnerMessageAdapter.BEGINNER_TEMPLATES['iso_auto']

        # 조리개
        elif '조리개' in message or 'f/' in message:
            adapted['message'] = BeginnerMessageAdapter.BEGINNER_TEMPLATES['aperture']
            adapted['detail'] = '프로 모드에서 조정 가능해요'

        # 셔터 스피드
        elif '셔터' in message or '1/' in message:
            adapted['message'] = BeginnerMessageAdapter.BEGINNER_TEMPLATES['shutter_speed']

        # 채도
        elif '채도' in message:
            if '높이' in message:
                adapted['message'] = BeginnerMessageAdapter.BEGINNER_TEMPLATES['saturation_up']
            else:
                adapted['message'] = BeginnerMessageAdapter.BEGINNER_TEMPLATES['saturation_down']

        # 화이트밸런스
        elif '색온도' in message or '화이트밸런스' in message:
            adapted['message'] = BeginnerMessageAdapter.BEGINNER_TEMPLATES['white_balance']

        return adapted


# ============================================================
# 사용 예시
# ============================================================

if __name__ == "__main__":
    # 예제 피드백
    sample_feedback = [
        {
            "priority": 0,
            "category": "style",
            "message": "✅ 같은 스타일입니다 (Cluster 5)",
            "detail": "실외/멀리/웜톤/반신"
        },
        {
            "priority": 0.5,
            "category": "pose",
            "message": "왼팔 팔꿈치를 15° 더 펴세요",
            "detail": "포즈 유사도: 68.58%"
        },
        {
            "priority": 1,
            "category": "camera_settings",
            "message": "ISO를 400으로 설정하세요",
            "detail": "카메라 설정을 조정하세요"
        },
        {
            "priority": 2,
            "category": "distance",
            "message": "2걸음 뒤로 가세요",
            "detail": "레퍼런스 depth=250.0, 현재=180.0 (비율: 0.72)"
        },
        {
            "priority": 3,
            "category": "exposure",
            "message": "노출을 0.7 EV 올리세요",
            "detail": "레퍼런스 밝기=120.5, 현재=95.2 (차이: -25.3)"
        },
        {
            "priority": 4,
            "category": "color",
            "message": "채도를 10% 높이세요",
            "detail": "레퍼런스 채도=0.65, 현재=0.55 (차이: -0.10)"
        }
    ]

    # 1. Top-3 피드백
    formatter = FeedbackFormatter(user_level='beginner')
    top_k = formatter.format_top_k(sample_feedback, top_k=3, include_style=False)

    print("="*60)
    print("Top-3 피드백 (일반 모드)")
    print("="*60)
    print(formatter.format_for_display(top_k))

    print("\n" + "="*60)
    print("더보기")
    print("="*60)
    print(formatter.format_secondary(top_k))

    # 2. 초보자 모드 적용
    print("\n" + "="*60)
    print("초보자 모드 메시지 변환")
    print("="*60)

    adapter = BeginnerMessageAdapter()
    for fb in sample_feedback:
        adapted = adapter.adapt_message(fb, user_level='beginner')
        if adapted['message'] != fb['message']:
            print(f"\n원본: {fb['message']}")
            print(f"변환: {adapted['message']}")
