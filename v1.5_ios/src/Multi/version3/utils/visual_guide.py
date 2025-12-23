# ============================================================
# 👁️ Visual Guide Overlay
# Phase 3.3: 시각적 가이드 시스템
# ============================================================

import cv2
import numpy as np
from typing import Dict, Optional, Tuple, List


class VisualGuideOverlay:
    """
    카메라 화면에 가이드 라인 표시

    사용자 관점:
    - 말로 설명하면 모르겠어요 → 눈으로 보고 따라해요!
    - "3도 기울이기" → 화면에 수평선 표시
    - "2걸음 뒤로" → 목표 위치 박스 표시
    - "삼분할" → 그리드 라인 표시
    """

    # 색상 정의 (BGR)
    COLORS = {
        'guide': (0, 255, 0),       # 녹색 - 가이드 라인
        'target': (0, 255, 255),    # 노란색 - 목표 위치
        'current': (0, 0, 255),     # 빨간색 - 현재 상태
        'good': (0, 255, 0),        # 녹색 - 좋음
        'warning': (0, 165, 255),   # 주황색 - 경고
        'error': (0, 0, 255),       # 빨간색 - 오류
        'text': (255, 255, 255)     # 흰색 - 텍스트
    }

    def __init__(self):
        """초기화"""
        pass

    def draw_rule_of_thirds(self, frame: np.ndarray, color=None, thickness=1) -> np.ndarray:
        """
        삼분할선 그리기

        사용자: "어디에 피사체를 두어야 하죠?"
        → 삼분할선 표시로 구도 가이드
        """
        if color is None:
            color = self.COLORS['guide']

        h, w = frame.shape[:2]

        # 세로선 2개
        cv2.line(frame, (w//3, 0), (w//3, h), color, thickness)
        cv2.line(frame, (2*w//3, 0), (2*w//3, h), color, thickness)

        # 가로선 2개
        cv2.line(frame, (0, h//3), (w, h//3), color, thickness)
        cv2.line(frame, (0, 2*h//3), (w, 2*h//3), color, thickness)

        return frame

    def draw_horizon_line(
        self,
        frame: np.ndarray,
        current_tilt: float,
        target_tilt: float = 0.0
    ) -> np.ndarray:
        """
        수평선 그리기

        사용자: "카메라를 똑바로 잡아야 하나요?"
        → 수평선 표시 (현재 기울기 vs 목표 기울기)

        Args:
            current_tilt: 현재 기울기 (도)
            target_tilt: 목표 기울기 (도)
        """
        h, w = frame.shape[:2]
        center_y = h // 2

        # 목표 수평선 (녹색, 점선)
        if abs(target_tilt) > 0.5:
            # 기울어진 목표선
            angle_rad = np.deg2rad(target_tilt)
            dx = int(w/2 * np.cos(angle_rad))
            dy = int(w/2 * np.sin(angle_rad))

            cv2.line(
                frame,
                (w//2 - dx, center_y - dy),
                (w//2 + dx, center_y + dy),
                self.COLORS['target'],
                2,
                cv2.LINE_AA
            )

        # 현재 수평선 (빨간색/녹색)
        angle_rad = np.deg2rad(current_tilt)
        dx = int(w/2 * np.cos(angle_rad))
        dy = int(w/2 * np.sin(angle_rad))

        color = self.COLORS['good'] if abs(current_tilt - target_tilt) < 2 else self.COLORS['error']

        cv2.line(
            frame,
            (w//2 - dx, center_y - dy),
            (w//2 + dx, center_y + dy),
            color,
            3,
            cv2.LINE_AA
        )

        # 각도 표시
        text = f"{current_tilt:.1f}°"
        if abs(current_tilt - target_tilt) < 2:
            text += " ✓"

        cv2.putText(
            frame,
            text,
            (w//2 - 50, center_y - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            color,
            2,
            cv2.LINE_AA
        )

        return frame

    def draw_target_bbox(
        self,
        frame: np.ndarray,
        target_bbox: Tuple[int, int, int, int],
        current_bbox: Optional[Tuple[int, int, int, int]] = None,
        label: str = "목표 위치"
    ) -> np.ndarray:
        """
        목표 위치 박스 그리기

        사용자: "피사체를 어디에 둬야 하죠?"
        → 목표 박스 표시 (레퍼런스 위치)

        Args:
            target_bbox: (x, y, w, h) 목표 박스
            current_bbox: (x, y, w, h) 현재 박스 (있으면)
        """
        tx, ty, tw, th = target_bbox

        # 목표 박스 (노란색, 점선)
        self._draw_dashed_rect(
            frame,
            (tx, ty, tx+tw, ty+th),
            self.COLORS['target'],
            2
        )

        # 라벨
        cv2.putText(
            frame,
            label,
            (tx, ty - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            self.COLORS['target'],
            2,
            cv2.LINE_AA
        )

        # 현재 박스 (있으면)
        if current_bbox is not None:
            cx, cy, cw, ch = current_bbox

            # 겹침 정도로 색상 결정
            overlap = self._calculate_overlap(target_bbox, current_bbox)

            if overlap > 0.8:
                color = self.COLORS['good']
                status = "Good!"
            elif overlap > 0.5:
                color = self.COLORS['warning']
                status = "Almost"
            else:
                color = self.COLORS['error']
                status = "Move"

            cv2.rectangle(
                frame,
                (cx, cy),
                (cx+cw, cy+ch),
                color,
                2
            )

            cv2.putText(
                frame,
                status,
                (cx, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
                cv2.LINE_AA
            )

        return frame

    def draw_pose_guide(
        self,
        frame: np.ndarray,
        target_keypoints: Dict,
        current_keypoints: Optional[Dict] = None
    ) -> np.ndarray:
        """
        포즈 가이드 그리기

        사용자: "어떤 자세를 취해야 하죠?"
        → 목표 포즈 스켈레톤 표시

        Args:
            target_keypoints: 목표 포즈 keypoints
            current_keypoints: 현재 포즈 keypoints (있으면)
        """
        # 스켈레톤 연결 (COCO 17 keypoints)
        skeleton = [
            (0, 1), (0, 2),  # 코 - 눈
            (1, 3), (2, 4),  # 눈 - 귀
            (0, 5), (0, 6),  # 코 - 어깨
            (5, 7), (7, 9),  # 왼팔
            (6, 8), (8, 10), # 오른팔
            (5, 11), (6, 12),  # 어깨 - 골반
            (11, 13), (13, 15),  # 왼다리
            (12, 14), (14, 16)   # 오른다리
        ]

        # 목표 포즈 (노란색, 반투명)
        if target_keypoints:
            overlay = frame.copy()
            self._draw_skeleton(
                overlay,
                target_keypoints,
                self.COLORS['target'],
                thickness=3
            )
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        # 현재 포즈 (녹색/빨간색)
        if current_keypoints:
            self._draw_skeleton(
                frame,
                current_keypoints,
                self.COLORS['good'],
                thickness=2
            )

        return frame

    def draw_feedback_panel(
        self,
        frame: np.ndarray,
        feedback_messages: List[str],
        position: str = 'top'
    ) -> np.ndarray:
        """
        피드백 패널 그리기

        사용자: "지금 뭘 해야 하죠?"
        → 화면에 간단한 지시사항 표시

        Args:
            feedback_messages: 피드백 메시지 리스트
            position: 'top', 'bottom', 'left', 'right'
        """
        h, w = frame.shape[:2]

        # 배경 패널 (반투명)
        panel_height = min(150, 50 + len(feedback_messages) * 35)

        if position == 'top':
            panel_y = 0
        else:  # bottom
            panel_y = h - panel_height

        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (0, panel_y),
            (w, panel_y + panel_height),
            (0, 0, 0),
            -1
        )
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # 피드백 메시지
        y_offset = panel_y + 30

        for i, msg in enumerate(feedback_messages[:3]):  # 최대 3개
            # 번호 + 메시지
            text = f"{i+1}. {msg}"

            cv2.putText(
                frame,
                text,
                (15, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                self.COLORS['text'],
                2,
                cv2.LINE_AA
            )

            y_offset += 40

        return frame

    def _draw_dashed_rect(self, frame: np.ndarray, bbox: Tuple, color: Tuple, thickness: int):
        """점선 사각형 그리기"""
        x1, y1, x2, y2 = bbox

        # 점선 간격
        dash_length = 10

        # 상단
        for x in range(x1, x2, dash_length*2):
            cv2.line(frame, (x, y1), (min(x+dash_length, x2), y1), color, thickness)

        # 하단
        for x in range(x1, x2, dash_length*2):
            cv2.line(frame, (x, y2), (min(x+dash_length, x2), y2), color, thickness)

        # 좌측
        for y in range(y1, y2, dash_length*2):
            cv2.line(frame, (x1, y), (x1, min(y+dash_length, y2)), color, thickness)

        # 우측
        for y in range(y1, y2, dash_length*2):
            cv2.line(frame, (x2, y), (x2, min(y+dash_length, y2)), color, thickness)

    def _calculate_overlap(self, bbox1: Tuple, bbox2: Tuple) -> float:
        """두 박스의 겹침 정도 계산 (IoU)"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        # 교집합
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1+w1, x2+w2)
        yi2 = min(y1+h1, y2+h2)

        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

        # 합집합
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0

    def _draw_skeleton(self, frame: np.ndarray, keypoints: Dict, color: Tuple, thickness: int):
        """스켈레톤 그리기 (간단 버전)"""
        # 실제 구현에서는 keypoints 딕셔너리에서 좌표 추출
        # 여기서는 플레이스홀더
        pass


# ============================================================
# 사용 예시
# ============================================================

if __name__ == "__main__":
    # 테스트 이미지 생성
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    guide = VisualGuideOverlay()

    print("="*60)
    print("Phase 3.3: 시각적 가이드 오버레이 테스트")
    print("="*60)

    # 1. 삼분할선
    frame_grid = frame.copy()
    frame_grid = guide.draw_rule_of_thirds(frame_grid)
    cv2.imwrite("test_grid.jpg", frame_grid)
    print("\n✅ 삼분할선 → test_grid.jpg")

    # 2. 수평선
    frame_horizon = frame.copy()
    frame_horizon = guide.draw_horizon_line(frame_horizon, current_tilt=5.2, target_tilt=0.0)
    cv2.imwrite("test_horizon.jpg", frame_horizon)
    print("✅ 수평선 → test_horizon.jpg")

    # 3. 목표 박스
    frame_bbox = frame.copy()
    frame_bbox = guide.draw_target_bbox(
        frame_bbox,
        target_bbox=(400, 200, 480, 640),
        current_bbox=(350, 180, 500, 660),
        label="레퍼런스 위치"
    )
    cv2.imwrite("test_bbox.jpg", frame_bbox)
    print("✅ 목표 박스 → test_bbox.jpg")

    # 4. 피드백 패널
    frame_feedback = frame.copy()
    frame_feedback = guide.draw_feedback_panel(
        frame_feedback,
        ["2걸음 뒤로 가세요", "왼팔을 15° 올리세요", "카메라를 3° 왼쪽으로"],
        position='top'
    )
    cv2.imwrite("test_feedback.jpg", frame_feedback)
    print("✅ 피드백 패널 → test_feedback.jpg")

    # 5. 전체 통합
    frame_all = frame.copy()
    frame_all = guide.draw_rule_of_thirds(frame_all)
    frame_all = guide.draw_target_bbox(
        frame_all,
        target_bbox=(400, 200, 480, 640),
        current_bbox=(350, 180, 500, 660)
    )
    frame_all = guide.draw_feedback_panel(
        frame_all,
        ["2걸음 뒤로", "좋아요! 계속!"],
        position='top'
    )
    cv2.imwrite("test_all.jpg", frame_all)
    print("✅ 전체 통합 → test_all.jpg")

    print("\n💬 사용자 관점:")
    print("   복잡한 텍스트 설명 대신")
    print("   → 화면에 가이드 라인 보면서 따라하기!")
