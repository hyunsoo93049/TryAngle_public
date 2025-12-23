# ============================================================
# TryAngle v1.5 - Composition Analyzer
# 구도 특징 계산 (위치, 크기, 여백, 삼분할)
# ============================================================

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class CompositionFeatures:
    """구도 특징"""
    # 위치 (normalized 0-1)
    position: Tuple[float, float]          # (center_x, center_y)

    # 크기
    size_ratio: float                       # bbox 면적 / 이미지 면적
    aspect_ratio: float                     # bbox 가로/세로 비율

    # 여백 (normalized 0-1)
    margins: Dict[str, float]               # left, right, top, bottom

    # 삼분할
    rule_of_thirds_score: float             # 0-1, 높을수록 삼분할 점 근처
    nearest_third_point: Tuple[float, float]

    # 비대칭성
    horizontal_asymmetry: str               # left_heavy, center, right_heavy
    vertical_asymmetry: str                 # top_heavy, center, bottom_heavy


class CompositionAnalyzer:
    """
    Bbox 기반 구도 분석

    Usage:
        analyzer = CompositionAnalyzer()
        features = analyzer.analyze(bbox=(0.2, 0.1, 0.6, 0.9))
    """

    # 삼분할 점 (normalized)
    THIRD_POINTS = [
        (1/3, 1/3),   # 좌상
        (2/3, 1/3),   # 우상
        (1/3, 2/3),   # 좌하
        (2/3, 2/3)    # 우하
    ]

    def __init__(self):
        pass

    def analyze(
        self,
        bbox: Tuple[float, float, float, float],
        image_size: Optional[Tuple[int, int]] = None
    ) -> CompositionFeatures:
        """
        Bbox 기반 구도 분석

        Args:
            bbox: (x1, y1, x2, y2) normalized [0-1]
            image_size: (width, height) - 참고용

        Returns:
            CompositionFeatures
        """
        x1, y1, x2, y2 = bbox

        # 중심점
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        # 크기
        width = x2 - x1
        height = y2 - y1
        size_ratio = width * height
        aspect_ratio = width / height if height > 0 else 1.0

        # 여백
        margins = {
            "left": x1,
            "right": 1.0 - x2,
            "top": y1,
            "bottom": 1.0 - y2
        }

        # 삼분할 점수
        rule_of_thirds_score, nearest_point = self._calculate_thirds_score(
            center_x, center_y
        )

        # 비대칭성
        horizontal_asymmetry = self._classify_horizontal_asymmetry(center_x)
        vertical_asymmetry = self._classify_vertical_asymmetry(center_y)

        return CompositionFeatures(
            position=(center_x, center_y),
            size_ratio=size_ratio,
            aspect_ratio=aspect_ratio,
            margins=margins,
            rule_of_thirds_score=rule_of_thirds_score,
            nearest_third_point=nearest_point,
            horizontal_asymmetry=horizontal_asymmetry,
            vertical_asymmetry=vertical_asymmetry
        )

    def _calculate_thirds_score(
        self,
        center_x: float,
        center_y: float
    ) -> Tuple[float, Tuple[float, float]]:
        """
        삼분할 점수 계산

        Returns:
            (score, nearest_point)
        """
        min_distance = float('inf')
        nearest_point = self.THIRD_POINTS[0]

        for point in self.THIRD_POINTS:
            dx = center_x - point[0]
            dy = center_y - point[1]
            distance = np.sqrt(dx**2 + dy**2)

            if distance < min_distance:
                min_distance = distance
                nearest_point = point

        # 최대 가능 거리 (코너에서 가장 가까운 삼분할 점까지)
        max_distance = np.sqrt(2) / 3  # 약 0.47

        # 점수: 가까울수록 높음
        score = max(0, 1.0 - (min_distance / max_distance))

        return score, nearest_point

    def _classify_horizontal_asymmetry(self, center_x: float) -> str:
        """수평 비대칭성 분류"""
        if center_x < 0.4:
            return "left_heavy"
        elif center_x > 0.6:
            return "right_heavy"
        else:
            return "center"

    def _classify_vertical_asymmetry(self, center_y: float) -> str:
        """수직 비대칭성 분류"""
        if center_y < 0.4:
            return "top_heavy"
        elif center_y > 0.6:
            return "bottom_heavy"
        else:
            return "center"

    def calculate_composition_score(
        self,
        features: CompositionFeatures,
        reference_pattern: Dict
    ) -> Dict[str, float]:
        """
        레퍼런스 패턴 대비 구도 점수 계산

        Args:
            features: 현재 구도 특징
            reference_pattern: 레퍼런스 패턴 통계

        Returns:
            {
                "position_score": float,
                "size_score": float,
                "margin_score": float,
                "thirds_score": float,
                "total_score": float
            }
        """
        scores = {}

        # 1. 위치 점수
        if "position" in reference_pattern:
            ref_pos = reference_pattern["position"]["mean"]
            ref_std = reference_pattern["position"].get("std", [0.1, 0.1])

            dx = abs(features.position[0] - ref_pos[0])
            dy = abs(features.position[1] - ref_pos[1])

            # 표준편차 내면 높은 점수
            pos_score_x = max(0, 1.0 - dx / (ref_std[0] * 2))
            pos_score_y = max(0, 1.0 - dy / (ref_std[1] * 2))
            scores["position_score"] = (pos_score_x + pos_score_y) / 2
        else:
            scores["position_score"] = 0.5

        # 2. 크기 점수
        if "size" in reference_pattern:
            ref_size = reference_pattern["size"]["mean"]
            ref_range = reference_pattern["size"].get("optimal_range", [0.2, 0.5])

            if ref_range[0] <= features.size_ratio <= ref_range[1]:
                scores["size_score"] = 1.0
            else:
                diff = min(
                    abs(features.size_ratio - ref_range[0]),
                    abs(features.size_ratio - ref_range[1])
                )
                scores["size_score"] = max(0, 1.0 - diff * 5)
        else:
            scores["size_score"] = 0.5

        # 3. 여백 점수
        if "margins" in reference_pattern:
            margin_scores = []
            for side in ["left", "right", "top", "bottom"]:
                if side in reference_pattern["margins"]:
                    ref_margin = reference_pattern["margins"][side]["mean"]
                    ref_std = reference_pattern["margins"][side].get("std", 0.1)
                    diff = abs(features.margins[side] - ref_margin)
                    margin_scores.append(max(0, 1.0 - diff / (ref_std * 2)))
            scores["margin_score"] = np.mean(margin_scores) if margin_scores else 0.5
        else:
            scores["margin_score"] = 0.5

        # 4. 삼분할 점수 (기본값 사용)
        scores["thirds_score"] = features.rule_of_thirds_score

        # 총점 (가중 평균)
        weights = {
            "position_score": 0.30,
            "size_score": 0.25,
            "margin_score": 0.25,
            "thirds_score": 0.20
        }

        total = sum(scores[k] * weights[k] for k in weights)
        scores["total_score"] = total

        return scores


# ============================================================
# Quick Test
# ============================================================
if __name__ == "__main__":
    analyzer = CompositionAnalyzer()

    # 테스트 bbox (우측 삼분할 점 근처)
    test_bbox = (0.4, 0.2, 0.8, 0.8)

    features = analyzer.analyze(test_bbox)

    print(f"Position: {features.position}")
    print(f"Size Ratio: {features.size_ratio:.3f}")
    print(f"Margins: {features.margins}")
    print(f"Rule of Thirds Score: {features.rule_of_thirds_score:.3f}")
    print(f"Nearest Third Point: {features.nearest_third_point}")
    print(f"H-Asymmetry: {features.horizontal_asymmetry}")
    print(f"V-Asymmetry: {features.vertical_asymmetry}")
