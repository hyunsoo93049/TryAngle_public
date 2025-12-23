# ============================================================
# TryAngle v1.5 - Pose Type Classifier
# 포즈 타입 분류 및 검증
# ============================================================

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class PoseClassificationResult:
    """포즈 분류 결과"""
    pose_type: str                     # closeup, upper_body, half_body, full_body
    confidence: float                  # 분류 신뢰도
    sitting: bool                      # 앉은 자세 여부
    visible_joints: List[str]          # 보이는 관절 목록
    details: Dict                      # 상세 정보


class PoseClassifier:
    """
    RTMPose 결과를 기반으로 Pose Type 분류

    Features:
        - Pose type 판정 (closeup/upper/half/full)
        - 앉음/서있음 판정
        - Occlusion 처리
    """

    def __init__(self, confidence_threshold: float = 0.3):
        self.threshold = confidence_threshold

        # 관절 그룹 정의
        self.joint_groups = {
            "face": ["nose", "left_eye", "right_eye", "left_ear", "right_ear"],
            "shoulders": ["left_shoulder", "right_shoulder"],
            "arms": ["left_elbow", "right_elbow", "left_wrist", "right_wrist"],
            "hips": ["left_hip", "right_hip"],
            "legs": ["left_knee", "right_knee"],
            "feet": ["left_ankle", "right_ankle"]
        }

    def classify(
        self,
        keypoints: List[Dict],
        bbox: Optional[Tuple[float, float, float, float]] = None
    ) -> PoseClassificationResult:
        """
        포즈 분류 수행

        Args:
            keypoints: [{"name": str, "x": float, "y": float, "confidence": float}, ...]
            bbox: (x1, y1, x2, y2) normalized

        Returns:
            PoseClassificationResult
        """
        # 키포인트를 dict로 변환
        kp_dict = {kp["name"]: kp for kp in keypoints if kp.get("name")}

        # 각 그룹별 가시성 확인
        visibility = {}
        visible_joints = []

        for group_name, joint_names in self.joint_groups.items():
            visible_count = 0
            for joint_name in joint_names:
                if joint_name in kp_dict:
                    if kp_dict[joint_name].get("confidence", 0) >= self.threshold:
                        visible_count += 1
                        visible_joints.append(joint_name)

            visibility[group_name] = visible_count >= len(joint_names) * 0.5

        # Pose Type 판정
        pose_type, type_confidence = self._determine_pose_type(visibility, kp_dict)

        # 앉음/서있음 판정
        sitting = self._determine_sitting(kp_dict, visibility)

        return PoseClassificationResult(
            pose_type=pose_type,
            confidence=type_confidence,
            sitting=sitting,
            visible_joints=visible_joints,
            details={
                "visibility": visibility,
                "joint_count": len(visible_joints)
            }
        )

    def _determine_pose_type(
        self,
        visibility: Dict[str, bool],
        kp_dict: Dict
    ) -> Tuple[str, float]:
        """Pose type 결정"""

        # 발목 보임 → 전신
        if visibility.get("feet", False):
            return "full_body", 0.95

        # 무릎 보임 → 반신
        if visibility.get("legs", False):
            return "half_body", 0.90

        # 골반 보임 → 반신 (최소)
        if visibility.get("hips", False):
            return "half_body", 0.85

        # 어깨 보임 → 상반신
        if visibility.get("shoulders", False):
            return "upper_body", 0.85

        # 얼굴만 → 클로즈업
        if visibility.get("face", False):
            return "closeup", 0.80

        # 기본값 (Conservative)
        return "upper_body", 0.50

    def _determine_sitting(
        self,
        kp_dict: Dict,
        visibility: Dict[str, bool]
    ) -> bool:
        """앉은 자세 판정"""

        # 무릎과 골반이 모두 보여야 판정 가능
        if not (visibility.get("hips") and visibility.get("legs")):
            return False

        try:
            # 골반과 무릎의 y 좌표 비교
            l_hip = kp_dict.get("left_hip", {})
            r_hip = kp_dict.get("right_hip", {})
            l_knee = kp_dict.get("left_knee", {})
            r_knee = kp_dict.get("right_knee", {})

            hip_y = (l_hip.get("y", 0) + r_hip.get("y", 0)) / 2
            knee_y = (l_knee.get("y", 0) + r_knee.get("y", 0)) / 2

            # 무릎과 골반의 y 차이가 작으면 앉은 자세
            # (서있으면 무릎이 골반보다 훨씬 아래)
            vertical_diff = knee_y - hip_y

            # 앉은 자세: 골반-무릎 거리가 짧음 (0.05 ~ 0.15 정도)
            # 서있는 자세: 골반-무릎 거리가 김 (0.15 ~ 0.30 정도)
            if 0 < vertical_diff < 0.15:
                return True

        except (KeyError, TypeError, ZeroDivisionError):
            pass

        return False

    def get_pose_requirements(self, pose_type: str) -> Dict:
        """포즈 타입별 요구 사항 반환"""
        requirements = {
            "closeup": {
                "required_visible": ["face"],
                "min_bbox_ratio": 0.15,  # 얼굴이 최소 15%
                "description": "클로즈업 (얼굴 위주)"
            },
            "upper_body": {
                "required_visible": ["face", "shoulders"],
                "min_bbox_ratio": 0.20,
                "description": "상반신 (어깨까지)"
            },
            "half_body": {
                "required_visible": ["shoulders", "hips"],
                "optional_visible": ["legs"],
                "min_bbox_ratio": 0.30,
                "description": "반신 (허리~무릎)"
            },
            "full_body": {
                "required_visible": ["shoulders", "hips", "feet"],
                "min_bbox_ratio": 0.40,
                "description": "전신 (발목까지)"
            }
        }
        return requirements.get(pose_type, requirements["upper_body"])


# ============================================================
# Quick Test
# ============================================================
if __name__ == "__main__":
    classifier = PoseClassifier(confidence_threshold=0.3)

    # 테스트 키포인트 (반신)
    test_keypoints = [
        {"name": "nose", "x": 0.5, "y": 0.2, "confidence": 0.9},
        {"name": "left_shoulder", "x": 0.4, "y": 0.3, "confidence": 0.8},
        {"name": "right_shoulder", "x": 0.6, "y": 0.3, "confidence": 0.8},
        {"name": "left_hip", "x": 0.4, "y": 0.5, "confidence": 0.7},
        {"name": "right_hip", "x": 0.6, "y": 0.5, "confidence": 0.7},
        {"name": "left_knee", "x": 0.4, "y": 0.7, "confidence": 0.6},
        {"name": "right_knee", "x": 0.6, "y": 0.7, "confidence": 0.6},
    ]

    result = classifier.classify(test_keypoints)

    print(f"Pose Type: {result.pose_type}")
    print(f"Confidence: {result.confidence}")
    print(f"Sitting: {result.sitting}")
    print(f"Visible Joints: {result.visible_joints}")
