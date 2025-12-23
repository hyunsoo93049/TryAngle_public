# ============================================================
# 📈 Progress Tracker
# Phase 2.2: 실시간 진행도 추적 시스템
# ============================================================

from typing import List, Dict, Optional
import json
from pathlib import Path


class ProgressTracker:
    """
    촬영 진행도 추적

    사용자 관점:
    - "내가 얼마나 개선했지?" → 시각적 진행률
    - "뭘 고쳤지?" → 개선된 항목 목록
    - "뭐가 남았지?" → 남은 항목 목록
    - "언제 끝나지?" → 예상 완료 시간
    """

    def __init__(self):
        """초기 상태 저장"""
        self.initial_feedback = None
        self.history = []  # [{timestamp, feedback, score}, ...]

    def set_initial_state(self, feedback_list: List[Dict]):
        """
        첫 촬영의 피드백을 저장 (기준점)

        Args:
            feedback_list: get_prioritized_feedback() 출력
        """
        self.initial_feedback = feedback_list
        self.history = []

        # 초기 스코어 계산
        initial_score = self._calculate_score(feedback_list)

        self.history.append({
            'attempt': 1,
            'feedback': feedback_list,
            'score': initial_score,
            'issues_count': self._count_issues(feedback_list)
        })

    def update_progress(self, current_feedback: List[Dict]) -> Dict:
        """
        현재 촬영의 진행도 계산

        Args:
            current_feedback: 현재 피드백 리스트

        Returns:
            {
                'overall_score': 0-100 점수,
                'progress_percent': 진행률,
                'improved_items': 개선된 항목,
                'remaining_items': 남은 항목,
                'new_issues': 새로 생긴 문제,
                'celebration': 축하 메시지 여부
            }
        """
        if self.initial_feedback is None:
            raise ValueError("초기 상태가 설정되지 않았습니다. set_initial_state()를 먼저 호출하세요.")

        # 현재 스코어 계산
        current_score = self._calculate_score(current_feedback)

        # 히스토리 추가
        self.history.append({
            'attempt': len(self.history) + 1,
            'feedback': current_feedback,
            'score': current_score,
            'issues_count': self._count_issues(current_feedback)
        })

        # 개선/남은/새로운 항목 분석
        improved, remaining, new_issues = self._analyze_changes(
            self.initial_feedback,
            current_feedback
        )

        # 진행률 계산
        initial_issues = self._count_issues(self.initial_feedback)
        resolved_count = len(improved)
        progress = (resolved_count / initial_issues * 100) if initial_issues > 0 else 100

        # 축하 메시지
        celebration = progress >= 90 or current_score >= 85

        return {
            'overall_score': current_score,
            'initial_score': self.history[0]['score'],
            'score_improvement': current_score - self.history[0]['score'],
            'progress_percent': min(progress, 100),
            'improved_items': improved,
            'remaining_items': remaining,
            'new_issues': new_issues,
            'celebration': celebration,
            'attempt_number': len(self.history),
            'total_attempts': len(self.history)
        }

    def _calculate_score(self, feedback_list: List[Dict]) -> float:
        """
        피드백 기반 점수 계산 (0-100)

        로직:
        - 완벽 (피드백 없음): 100점
        - 각 피드백마다 감점
        - Critical (priority < 1): -15점
        - Important (priority 1-3): -10점
        - Nice-to-have (priority > 3): -5점
        """
        score = 100.0

        for fb in feedback_list:
            if fb['category'] == 'style':
                continue  # 스타일 정보는 점수에 영향 없음

            priority = fb['priority']

            if priority < 1.0:
                # Critical: 포즈, 심각한 블러 등
                score -= 15
            elif 1.0 <= priority <= 3.0:
                # Important: 카메라 설정, 거리, 밝기
                score -= 10
            else:
                # Nice-to-have: 색감, 구도
                score -= 5

        return max(score, 0.0)

    def _count_issues(self, feedback_list: List[Dict]) -> int:
        """
        문제 개수 카운트 (스타일 정보 제외)
        """
        return len([fb for fb in feedback_list if fb['category'] != 'style'])

    def _analyze_changes(
        self,
        initial: List[Dict],
        current: List[Dict]
    ) -> tuple:
        """
        초기 대비 변화 분석

        Returns:
            (improved, remaining, new_issues)
        """
        # 카테고리별로 그룹화
        initial_by_cat = {fb['category']: fb for fb in initial if fb['category'] != 'style'}
        current_by_cat = {fb['category']: fb for fb in current if fb['category'] != 'style'}

        improved = []  # 개선된 항목
        remaining = []  # 여전히 남은 항목
        new_issues = []  # 새로 생긴 문제

        # 초기에 있던 문제들 체크
        for cat, initial_fb in initial_by_cat.items():
            if cat not in current_by_cat:
                # 해결됨!
                improved.append({
                    'category': cat,
                    'message': initial_fb['message'],
                    'status': '✅ 해결됨'
                })
            else:
                # 여전히 존재
                current_fb = current_by_cat[cat]

                # 개선되었는지 체크 (priority 증가 = 덜 중요해짐 = 개선)
                if current_fb['priority'] > initial_fb['priority']:
                    improved.append({
                        'category': cat,
                        'message': current_fb['message'],
                        'status': '⬆️ 개선 중'
                    })
                else:
                    remaining.append({
                        'category': cat,
                        'message': current_fb['message'],
                        'priority': current_fb['priority']
                    })

        # 새로 생긴 문제
        for cat, current_fb in current_by_cat.items():
            if cat not in initial_by_cat:
                new_issues.append({
                    'category': cat,
                    'message': current_fb['message'],
                    'status': '⚠️ 새 문제'
                })

        return improved, remaining, new_issues

    def format_progress_text(self, progress: Dict) -> str:
        """
        진행도 UI 텍스트 생성

        사용자 친화적 표현:
        - 진행률 바
        - "거의 다 됐어요!" 같은 격려
        - 개선된 것 강조
        """
        lines = []

        # 1. 점수 & 진행률
        score = progress['overall_score']
        progress_pct = progress['progress_percent']

        lines.append("="*60)
        lines.append("📊 촬영 진행도")
        lines.append("="*60)

        # 진행률 바
        bar_length = 20
        filled = int(bar_length * progress_pct / 100)
        bar = "█" * filled + "░" * (bar_length - filled)
        lines.append(f"\n{bar} {progress_pct:.0f}%")

        # 점수
        lines.append(f"점수: {score:.0f}점 ({progress['score_improvement']:+.0f})")

        # 시도 횟수
        lines.append(f"촬영 시도: {progress['attempt_number']}회")

        # 2. 축하 메시지
        if progress['celebration']:
            if progress_pct >= 95:
                lines.append("\n🎉 완벽해요! 이제 촬영하세요!")
            elif progress_pct >= 90:
                lines.append("\n🎊 거의 다 됐어요! 조금만 더!")
            elif score >= 85:
                lines.append("\n👍 잘하고 있어요!")

        # 3. 개선된 항목
        if progress['improved_items']:
            lines.append(f"\n✅ 개선됨 ({len(progress['improved_items'])}개):")
            for item in progress['improved_items'][:3]:  # 최대 3개만
                lines.append(f"   {item['status']} {item['category']}")

        # 4. 남은 항목
        if progress['remaining_items']:
            lines.append(f"\n⏳ 남은 조정 ({len(progress['remaining_items'])}개):")
            for item in progress['remaining_items'][:3]:  # 최대 3개만
                lines.append(f"   • {item['message']}")

        # 5. 새 문제 (경고)
        if progress['new_issues']:
            lines.append(f"\n⚠️  새로운 문제 ({len(progress['new_issues'])}개):")
            for item in progress['new_issues']:
                lines.append(f"   {item['status']} {item['message']}")

        return "\n".join(lines)

    def get_encouragement_message(self, progress: Dict) -> str:
        """
        격려 메시지 생성

        사용자를 응원하는 메시지
        """
        score = progress['overall_score']
        progress_pct = progress['progress_percent']
        attempt = progress['attempt_number']

        if score >= 95:
            return "🌟 완벽합니다! 프로처럼 찍으셨어요!"
        elif score >= 85:
            return "🎯 훌륭해요! 거의 완성이에요!"
        elif score >= 70:
            return "👏 잘하고 있어요! 조금만 더!"
        elif progress_pct >= 50:
            return f"💪 절반 완료! 이미 {len(progress['improved_items'])}개 개선했어요!"
        elif attempt == 2:
            return "🔥 좋아요! 계속 개선하고 있어요!"
        else:
            return "📸 하나씩 차근차근 해볼까요?"

    def save_history(self, filepath: str):
        """히스토리 저장 (나중에 분석용)"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)

    def load_history(self, filepath: str):
        """히스토리 로드"""
        with open(filepath, 'r', encoding='utf-8') as f:
            self.history = json.load(f)

        if self.history:
            self.initial_feedback = self.history[0]['feedback']


# ============================================================
# 사용 예시
# ============================================================

if __name__ == "__main__":
    # 시뮬레이션: 사용자가 점점 개선하는 과정

    # 초기 촬영 (문제 많음)
    initial = [
        {"priority": 0, "category": "style", "message": "같은 스타일"},
        {"priority": 0.5, "category": "pose", "message": "왼팔 15° 올리기"},
        {"priority": 1, "category": "camera_settings", "message": "ISO 400"},
        {"priority": 2, "category": "distance", "message": "2걸음 뒤로"},
        {"priority": 3, "category": "exposure", "message": "노출 +0.7 EV"},
        {"priority": 4, "category": "color", "message": "채도 10% 올리기"}
    ]

    tracker = ProgressTracker()
    tracker.set_initial_state(initial)

    print("="*60)
    print("Phase 2.2: 실시간 진행도 추적 테스트")
    print("="*60)

    print("\n📸 1회차: 초기 촬영")
    print(f"점수: {tracker.history[0]['score']:.0f}점")
    print(f"문제: {tracker.history[0]['issues_count']}개")

    # 2회차: 거리와 노출 개선
    attempt2 = [
        {"priority": 0, "category": "style", "message": "같은 스타일"},
        {"priority": 0.5, "category": "pose", "message": "왼팔 15° 올리기"},
        {"priority": 1, "category": "camera_settings", "message": "ISO 400"},
        {"priority": 4, "category": "color", "message": "채도 10% 올리기"}
        # distance, exposure 해결됨!
    ]

    print("\n📸 2회차: 거리와 노출 개선")
    progress = tracker.update_progress(attempt2)
    print(tracker.format_progress_text(progress))
    print(f"\n💬 {tracker.get_encouragement_message(progress)}")

    # 3회차: 포즈까지 개선
    attempt3 = [
        {"priority": 0, "category": "style", "message": "같은 스타일"},
        {"priority": 1, "category": "camera_settings", "message": "ISO 400"},
        {"priority": 4, "category": "color", "message": "채도 10% 올리기"}
        # pose 해결됨!
    ]

    print("\n" + "="*60)
    print("\n📸 3회차: 포즈까지 개선")
    progress = tracker.update_progress(attempt3)
    print(tracker.format_progress_text(progress))
    print(f"\n💬 {tracker.get_encouragement_message(progress)}")

    # 4회차: 거의 완성
    attempt4 = [
        {"priority": 0, "category": "style", "message": "같은 스타일"}
        # 거의 완벽!
    ]

    print("\n" + "="*60)
    print("\n📸 4회차: 거의 완성!")
    progress = tracker.update_progress(attempt4)
    print(tracker.format_progress_text(progress))
    print(f"\n💬 {tracker.get_encouragement_message(progress)}")
