# ============================================================
# 📸 Workflow Guide
# Phase 2.1: 단계별 촬영 가이드 시스템
# ============================================================

from typing import List, Dict
from collections import OrderedDict


class WorkflowGuide:
    """
    피드백을 촬영 흐름에 맞게 재구성

    실제 촬영 순서:
    1. 위치 설정 (거리, 조명)
    2. 구도 설정 (프레이밍, 기울기)
    3. 포즈 조정
    4. 카메라 설정 (ISO, 조리개, 노출)
    5. 품질 확인 (블러, 노이즈)
    """

    # 촬영 단계 정의
    WORKFLOW_STEPS = OrderedDict([
        ('position', {
            'step': 1,
            'name': '위치 설정',
            'icon': '📍',
            'categories': ['distance', 'backlight', 'lighting_direction'],
            'estimated_time_per_item': 15,  # 초
            'description': '촬영 위치와 조명 환경을 설정합니다'
        }),
        ('composition', {
            'step': 2,
            'name': '구도 설정',
            'icon': '🖼️',
            'categories': ['composition', 'framing'],
            'estimated_time_per_item': 10,
            'description': '화면 구성과 프레이밍을 조정합니다'
        }),
        ('pose', {
            'step': 3,
            'name': '포즈 조정',
            'icon': '🤸',
            'categories': ['pose'],
            'estimated_time_per_item': 20,
            'description': '피사체의 자세를 조정합니다'
        }),
        ('camera', {
            'step': 4,
            'name': '카메라 설정',
            'icon': '📷',
            'categories': ['camera_settings', 'exposure', 'color'],
            'estimated_time_per_item': 10,
            'description': '카메라 설정을 조정합니다'
        }),
        ('quality', {
            'step': 5,
            'name': '품질 확인',
            'icon': '✨',
            'categories': ['quality', 'blur', 'sharpness', 'noise'],
            'estimated_time_per_item': 5,
            'description': '촬영 후 품질을 확인합니다'
        })
    ])

    def organize_by_workflow(self, feedback_list: List[Dict]) -> Dict:
        """
        피드백을 촬영 단계별로 재구성

        Args:
            feedback_list: get_prioritized_feedback() 출력

        Returns:
            {
                'steps': [단계별 피드백],
                'total_time': 예상 소요 시간(초),
                'current_step': 현재 작업 단계,
                'progress': 진행률
            }
        """
        # 스타일 정보 제외
        action_feedback = [fb for fb in feedback_list if fb['category'] != 'style']

        # 단계별로 분류
        steps_with_feedback = []
        total_time = 0

        for step_key, step_info in self.WORKFLOW_STEPS.items():
            step_feedback = [
                fb for fb in action_feedback
                if fb['category'] in step_info['categories']
            ]

            if step_feedback:
                # 우선순위 높은 순
                step_feedback.sort(key=lambda x: x['priority'])

                estimated_time = len(step_feedback) * step_info['estimated_time_per_item']
                total_time += estimated_time

                steps_with_feedback.append({
                    'step': step_info['step'],
                    'name': step_info['name'],
                    'icon': step_info['icon'],
                    'description': step_info['description'],
                    'feedback': step_feedback,
                    'count': len(step_feedback),
                    'estimated_time': estimated_time,
                    'completed': False
                })

        return {
            'steps': steps_with_feedback,
            'total_steps': len(steps_with_feedback),
            'total_time': total_time,
            'current_step': 1 if steps_with_feedback else 0,
            'progress_percent': 0
        }

    def format_workflow_text(self, workflow: Dict, show_all: bool = False) -> str:
        """
        촬영 가이드 텍스트 생성

        Args:
            workflow: organize_by_workflow() 출력
            show_all: True면 모든 단계, False면 현재 단계만

        Returns:
            포맷팅된 텍스트
        """
        if not workflow['steps']:
            return "✅ 모든 설정이 완벽합니다!"

        lines = []

        # 전체 요약
        lines.append("="*60)
        lines.append(f"📸 촬영 가이드 ({workflow['total_steps']}단계, 약 {workflow['total_time']}초 소요)")
        lines.append("="*60)

        if show_all:
            # 모든 단계 표시
            for i, step in enumerate(workflow['steps'], 1):
                lines.append(f"\n{step['icon']} {step['step']}단계: {step['name']} ({step['estimated_time']}초 소요)")
                lines.append(f"   {step['description']}")

                for j, fb in enumerate(step['feedback'], 1):
                    lines.append(f"   {j}. {fb['message']}")
                    if fb.get('detail'):
                        lines.append(f"      → {fb['detail']}")

                if i < len(workflow['steps']):
                    lines.append("")
        else:
            # 현재 단계만 표시
            current = workflow['steps'][0]  # 첫 번째 = 현재 단계
            lines.append(f"\n{current['icon']} {current['step']}단계: {current['name']} ({current['estimated_time']}초 소요)")
            lines.append(f"   {current['description']}\n")

            for j, fb in enumerate(current['feedback'], 1):
                lines.append(f"   ✓ {fb['message']}")
                if fb.get('detail'):
                    lines.append(f"      → {fb['detail']}")

            # 다음 단계 미리보기
            if len(workflow['steps']) > 1:
                lines.append(f"\n⏭️  다음: {workflow['steps'][1]['icon']} {workflow['steps'][1]['name']}")

            lines.append(f"\n[{workflow['current_step']}/{workflow['total_steps']} 완료]")

        return "\n".join(lines)

    def mark_step_completed(self, workflow: Dict, step_number: int) -> Dict:
        """
        단계 완료 표시

        Args:
            workflow: organize_by_workflow() 출력
            step_number: 완료한 단계 번호 (1-based)

        Returns:
            업데이트된 workflow
        """
        for step in workflow['steps']:
            if step['step'] == step_number:
                step['completed'] = True

        # 진행률 계산
        completed_count = sum(1 for step in workflow['steps'] if step['completed'])
        workflow['progress_percent'] = (completed_count / workflow['total_steps']) * 100

        # 현재 단계 업데이트
        for step in workflow['steps']:
            if not step['completed']:
                workflow['current_step'] = step['step']
                break
        else:
            # 모두 완료
            workflow['current_step'] = workflow['total_steps'] + 1

        return workflow

    def get_next_action(self, workflow: Dict) -> str:
        """
        다음에 할 행동 안내

        Returns:
            "2걸음 뒤로 가세요" 같은 즉시 실행 가능한 행동
        """
        if not workflow['steps']:
            return "✅ 촬영하세요!"

        current = workflow['steps'][0]
        if current['feedback']:
            # 첫 번째 피드백의 메시지
            return current['feedback'][0]['message']

        return "✅ 다음 단계로 이동하세요"


# ============================================================
# 사용 예시
# ============================================================

if __name__ == "__main__":
    # 예제 피드백
    sample_feedback = [
        {"priority": 0, "category": "style", "message": "✅ 같은 스타일입니다", "detail": ""},
        {"priority": 0.5, "category": "pose", "message": "왼팔을 15° 올리세요", "detail": "포즈 유사도: 68%"},
        {"priority": 0.5, "category": "pose", "message": "오른쪽 다리를 앞으로", "detail": ""},
        {"priority": 1, "category": "camera_settings", "message": "ISO 400으로 설정", "detail": ""},
        {"priority": 2, "category": "distance", "message": "2걸음 뒤로 가세요", "detail": ""},
        {"priority": 3, "category": "exposure", "message": "노출 +0.7 EV", "detail": ""},
        {"priority": 4, "category": "color", "message": "채도 10% 높이기", "detail": ""},
        {"priority": 5, "category": "composition", "message": "카메라 3도 왼쪽으로 기울이기", "detail": ""},
        {"priority": 5, "category": "framing", "message": "1.2배 확대", "detail": ""}
    ]

    guide = WorkflowGuide()

    print("="*60)
    print("Phase 2.1: 단계별 촬영 가이드 테스트")
    print("="*60)

    # 1. 워크플로우 구성
    workflow = guide.organize_by_workflow(sample_feedback)

    print(f"\n총 {workflow['total_steps']}단계, 예상 시간: {workflow['total_time']}초")

    # 2. 현재 단계만 표시
    print("\n" + "="*60)
    print("현재 단계만 표시")
    print("="*60)
    print(guide.format_workflow_text(workflow, show_all=False))

    # 3. 모든 단계 표시
    print("\n" + "="*60)
    print("모든 단계 표시")
    print("="*60)
    print(guide.format_workflow_text(workflow, show_all=True))

    # 4. 다음 행동
    print("\n" + "="*60)
    print("다음 행동")
    print("="*60)
    print(f"👉 {guide.get_next_action(workflow)}")

    # 5. 단계 완료
    print("\n" + "="*60)
    print("1단계 완료 후")
    print("="*60)
    workflow = guide.mark_step_completed(workflow, step_number=1)
    print(f"진행률: {workflow['progress_percent']:.0f}%")
    print(f"현재 단계: {workflow['current_step']}")
    print(guide.format_workflow_text(workflow, show_all=False))
