# ============================================================
# 🎯 Priority System
# Phase 2.3: 계층적 우선순위 시스템
# ============================================================

from enum import Enum
from typing import Dict, List


class PriorityLevel(Enum):
    """
    우선순위 레벨 정의

    사용자 관점:
    - CRITICAL: "다시 찍어야 해요" (초점 실패, 극심한 블러)
    - POSE: "자세를 바꿔주세요" (포즈 교정)
    - CAMERA: "카메라 설정을 조정하세요" (ISO, 조리개)
    - COMPOSITION: "구도를 잡아주세요" (거리, 프레이밍)
    - LIGHTING: "조명을 확인하세요" (밝기, 색감)
    - QUALITY: "품질을 개선하세요" (선명도, 노이즈)
    - INFO: "참고하세요" (정보성)
    """
    CRITICAL = 0.0      # 치명적 - 다시 찍기
    POSE = 0.5          # 포즈 - 매우 중요
    CAMERA = 1.0        # 카메라 설정
    COMPOSITION = 2.0   # 구도 (거리, 프레이밍)
    LIGHTING = 3.0      # 조명 (밝기, 색감)
    QUALITY = 5.0       # 품질 (조정 가능)
    INFO = 8.0          # 정보 (스타일 등)


# 카테고리별 우선순위 매핑
CATEGORY_PRIORITY_MAP = {
    # Critical
    'critical_blur': PriorityLevel.CRITICAL,
    'critical_focus': PriorityLevel.CRITICAL,
    'critical_exposure': PriorityLevel.CRITICAL,

    # Pose
    'pose': PriorityLevel.POSE,

    # Camera
    'camera_settings': PriorityLevel.CAMERA,
    'iso': PriorityLevel.CAMERA,
    'aperture': PriorityLevel.CAMERA,
    'shutter_speed': PriorityLevel.CAMERA,

    # Composition
    'distance': PriorityLevel.COMPOSITION,
    'framing': PriorityLevel.COMPOSITION,
    'composition': PriorityLevel.COMPOSITION,

    # Lighting
    'exposure': PriorityLevel.LIGHTING,
    'brightness': PriorityLevel.LIGHTING,
    'color': PriorityLevel.LIGHTING,
    'saturation': PriorityLevel.LIGHTING,
    'white_balance': PriorityLevel.LIGHTING,
    'backlight': PriorityLevel.LIGHTING,
    'lighting_direction': PriorityLevel.LIGHTING,

    # Quality
    'blur': PriorityLevel.QUALITY,
    'sharpness': PriorityLevel.QUALITY,
    'noise': PriorityLevel.QUALITY,
    'quality': PriorityLevel.QUALITY,

    # Info
    'style': PriorityLevel.INFO,
    'cluster': PriorityLevel.INFO
}


class PriorityClassifier:
    """
    피드백을 계층적 우선순위로 분류

    사용자 관점:
    - "먼저 이것부터 하세요" (Critical, Pose)
    - "그다음 이것을" (Camera, Composition)
    - "여유되면 이것도" (Lighting, Quality)
    """

    @staticmethod
    def classify(feedback_item: Dict) -> Dict:
        """
        피드백에 명확한 우선순위 부여

        Args:
            feedback_item: {priority, category, message, detail}

        Returns:
            우선순위 정보가 추가된 피드백
        """
        category = feedback_item['category']

        # 기본 우선순위
        if category in CATEGORY_PRIORITY_MAP:
            priority_level = CATEGORY_PRIORITY_MAP[category]
            base_priority = priority_level.value
        else:
            # Unknown category
            base_priority = PriorityLevel.QUALITY.value

        # Critical 항목 자동 감지
        message = feedback_item.get('message', '').lower()

        if any(word in message for word in ['다시', '실패', '불가능', '극심', '치명적']):
            base_priority = PriorityLevel.CRITICAL.value
            priority_level = PriorityLevel.CRITICAL

        # 원본 priority와 비교하여 더 높은 것 사용
        original_priority = feedback_item.get('priority', 5.0)
        final_priority = min(base_priority, original_priority)

        # 우선순위 레벨 결정
        if final_priority <= 0.5:
            level_name = 'Critical/Pose'
            level_color = '🔴'
            actionable = True
        elif final_priority <= 2.0:
            level_name = 'Important'
            level_color = '🟡'
            actionable = True
        elif final_priority <= 5.0:
            level_name = 'Recommended'
            level_color = '🟢'
            actionable = True
        else:
            level_name = 'Optional'
            level_color = '⚪'
            actionable = False

        return {
            **feedback_item,
            'priority': final_priority,
            'priority_level': level_name,
            'priority_color': level_color,
            'actionable': actionable
        }

    @staticmethod
    def group_by_priority(feedback_list: List[Dict]) -> Dict:
        """
        우선순위별로 그룹화

        Returns:
            {
                'critical': [...],
                'important': [...],
                'recommended': [...],
                'optional': [...]
            }
        """
        classified = [PriorityClassifier.classify(fb) for fb in feedback_list]

        groups = {
            'critical': [],
            'important': [],
            'recommended': [],
            'optional': []
        }

        for fb in classified:
            level = fb['priority_level']

            if 'Critical' in level or 'Pose' in level:
                groups['critical'].append(fb)
            elif level == 'Important':
                groups['important'].append(fb)
            elif level == 'Recommended':
                groups['recommended'].append(fb)
            else:
                groups['optional'].append(fb)

        return groups

    @staticmethod
    def format_grouped_feedback(groups: Dict) -> str:
        """
        그룹화된 피드백을 사용자 친화적으로 표시
        """
        lines = []

        # Critical
        if groups['critical']:
            lines.append("🔴 먼저 이것부터! (필수)")
            for i, fb in enumerate(groups['critical'], 1):
                lines.append(f"   {i}. {fb['message']}")
            lines.append("")

        # Important
        if groups['important']:
            lines.append("🟡 그다음 이것을 (중요)")
            for i, fb in enumerate(groups['important'], 1):
                lines.append(f"   {i}. {fb['message']}")
            lines.append("")

        # Recommended
        if groups['recommended']:
            lines.append("🟢 여유되면 이것도 (추천)")
            for i, fb in enumerate(groups['recommended'], 1):
                lines.append(f"   {i}. {fb['message']}")
            lines.append("")

        # Optional
        if groups['optional']:
            lines.append(f"⚪ 참고 사항 ({len(groups['optional'])}개)")

        return "\n".join(lines)


# ============================================================
# 사용 예시
# ============================================================

if __name__ == "__main__":
    # 예제 피드백
    sample_feedback = [
        {"priority": 0.5, "category": "blur", "message": "다시 찍으세요 (극심한 블러)", "detail": ""},
        {"priority": 0.5, "category": "pose", "message": "왼팔을 15° 올리세요", "detail": ""},
        {"priority": 1.0, "category": "camera_settings", "message": "ISO 400", "detail": ""},
        {"priority": 2.0, "category": "distance", "message": "2걸음 뒤로", "detail": ""},
        {"priority": 3.0, "category": "exposure", "message": "노출 +0.7 EV", "detail": ""},
        {"priority": 4.0, "category": "color", "message": "채도 10% 올리기", "detail": ""},
        {"priority": 5.0, "category": "sharpness", "message": "선명도 개선", "detail": ""},
        {"priority": 8.0, "category": "style", "message": "같은 스타일", "detail": ""}
    ]

    print("="*60)
    print("Phase 2.3: 계층적 우선순위 시스템 테스트")
    print("="*60)

    # 1. 개별 분류
    print("\n1. 개별 피드백 분류:")
    for fb in sample_feedback[:3]:
        classified = PriorityClassifier.classify(fb)
        print(f"{classified['priority_color']} {classified['priority_level']}: {classified['message']}")

    # 2. 그룹화
    print("\n2. 우선순위별 그룹화:")
    groups = PriorityClassifier.group_by_priority(sample_feedback)
    for level, items in groups.items():
        if items:
            print(f"{level.upper()}: {len(items)}개")

    # 3. 사용자 친화적 표시
    print("\n3. 사용자 친화적 표시:")
    print(PriorityClassifier.format_grouped_feedback(groups))
