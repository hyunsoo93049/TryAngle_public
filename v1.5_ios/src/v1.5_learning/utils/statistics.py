# ============================================================
# TryAngle v1.5 - Pattern Statistics
# 테마/포즈별 패턴 통계 계산 및 JSON 생성
# ============================================================

import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import numpy as np
from collections import defaultdict


@dataclass
class PatternStats:
    """단일 패턴 통계"""
    theme: str
    pose_type: str
    sample_count: int

    # 구도 통계
    position_mean: List[float]
    position_std: List[float]
    size_mean: float
    size_std: float
    margins_mean: Dict[str, float]
    margins_std: Dict[str, float]

    # 카메라 통계
    compression_mean: float
    compression_std: float
    angle_mean: float
    angle_std: float

    # 포즈 통계
    sitting_ratio: float
    common_visible_joints: List[str]

    # 메타데이터 (점수 기반 선정된 exemplars)
    exemplars: List[Dict]  # [{filename, score}, ...]


class PatternStatistics:
    """
    패턴 통계 계산 및 관리

    Usage:
        stats = PatternStatistics(min_samples=20)
        stats.add_sample(theme, pose_type, features)
        ...
        stats.calculate_all()
        stats.save_json("patterns.json")
    """

    def __init__(
        self,
        min_samples: int = 20,
        remove_outliers: bool = True,
        iqr_multiplier: float = 1.5
    ):
        self.min_samples = min_samples
        self.remove_outliers = remove_outliers
        self.iqr_multiplier = iqr_multiplier

        # 데이터 저장소: {(theme, pose_type): [sample_data, ...]}
        self.samples: Dict[tuple, List[Dict]] = defaultdict(list)

        # 계산된 패턴
        self.patterns: Dict[str, PatternStats] = {}

    def add_sample(
        self,
        theme: str,
        pose_type: str,
        features: Dict[str, Any],
        filename: Optional[str] = None
    ):
        """
        샘플 추가

        Args:
            theme: 테마 이름
            pose_type: 포즈 타입
            features: 추출된 특징 딕셔너리
            filename: 파일명 (exemplar 선정용)
        """
        key = (theme, pose_type)

        sample = {
            "filename": filename,
            **features
        }

        self.samples[key].append(sample)

    def calculate_all(self) -> Dict[str, PatternStats]:
        """
        모든 패턴 통계 계산

        Returns:
            {pattern_name: PatternStats}
        """
        self.patterns = {}

        for (theme, pose_type), samples in self.samples.items():
            if len(samples) < self.min_samples:
                print(f"[Skip] {theme}_{pose_type}: {len(samples)} samples < {self.min_samples}")
                continue

            pattern_name = f"{theme}_{pose_type}"
            pattern_stats = self._calculate_pattern(theme, pose_type, samples)
            self.patterns[pattern_name] = pattern_stats

            print(f"[OK] {pattern_name}: {len(samples)} samples")

        return self.patterns

    def _calculate_pattern(
        self,
        theme: str,
        pose_type: str,
        samples: List[Dict]
    ) -> PatternStats:
        """단일 패턴 통계 계산"""

        # 배열로 변환
        positions = []
        sizes = []
        margins = {"left": [], "right": [], "top": [], "bottom": []}
        compressions = []
        angles = []
        sitting_count = 0
        visible_joints_counter = defaultdict(int)
        filenames = []

        for sample in samples:
            # 위치
            if "position" in sample:
                positions.append(sample["position"])

            # 크기
            if "size_ratio" in sample:
                sizes.append(sample["size_ratio"])

            # 여백
            if "margins" in sample:
                for side in margins:
                    if side in sample["margins"]:
                        margins[side].append(sample["margins"][side])

            # 압축감
            if "compression_index" in sample:
                compressions.append(sample["compression_index"])

            # 앵글
            if "estimated_angle" in sample:
                angles.append(sample["estimated_angle"])

            # 앉음 여부
            if sample.get("sitting", False):
                sitting_count += 1

            # 보이는 관절
            if "visible_joints" in sample:
                for joint in sample["visible_joints"]:
                    visible_joints_counter[joint] += 1

            # 파일명
            if sample.get("filename"):
                filenames.append(sample["filename"])

        # Outlier 제거
        if self.remove_outliers:
            positions = self._remove_outliers_2d(positions)
            sizes = self._remove_outliers_1d(sizes)
            compressions = self._remove_outliers_1d(compressions)
            angles = self._remove_outliers_1d(angles)

        # 통계 계산
        position_mean = list(np.mean(positions, axis=0)) if positions else [0.5, 0.5]
        position_std = list(np.std(positions, axis=0)) if positions else [0.1, 0.1]

        size_mean = float(np.mean(sizes)) if sizes else 0.3
        size_std = float(np.std(sizes)) if sizes else 0.1

        margins_mean = {}
        margins_std = {}
        for side, values in margins.items():
            margins_mean[side] = float(np.mean(values)) if values else 0.25
            margins_std[side] = float(np.std(values)) if values else 0.1

        compression_mean = float(np.mean(compressions)) if compressions else 0.5
        compression_std = float(np.std(compressions)) if compressions else 0.15

        angle_mean = float(np.mean(angles)) if angles else 0.0
        angle_std = float(np.std(angles)) if angles else 10.0

        sitting_ratio = sitting_count / len(samples)

        # 공통 관절 (50% 이상 등장)
        threshold = len(samples) * 0.5
        common_joints = [
            joint for joint, count in visible_joints_counter.items()
            if count >= threshold
        ]

        # Exemplars: 패턴 평균에 가장 가까운 샘플 5개 선정
        pattern_mean = {
            "position": position_mean,
            "size_ratio": size_mean,
            "compression_index": compression_mean,
            "estimated_angle": angle_mean
        }
        exemplars = self._select_exemplars(samples, pattern_mean, top_k=5)

        return PatternStats(
            theme=theme,
            pose_type=pose_type,
            sample_count=len(samples),
            position_mean=position_mean,
            position_std=position_std,
            size_mean=size_mean,
            size_std=size_std,
            margins_mean=margins_mean,
            margins_std=margins_std,
            compression_mean=compression_mean,
            compression_std=compression_std,
            angle_mean=angle_mean,
            angle_std=angle_std,
            sitting_ratio=sitting_ratio,
            common_visible_joints=common_joints,
            exemplars=exemplars
        )

    def _remove_outliers_1d(self, data: List[float]) -> List[float]:
        """1D IQR 기반 outlier 제거"""
        if len(data) < 4:
            return data

        arr = np.array(data)
        q1, q3 = np.percentile(arr, [25, 75])
        iqr = q3 - q1

        lower = q1 - self.iqr_multiplier * iqr
        upper = q3 + self.iqr_multiplier * iqr

        return [x for x in data if lower <= x <= upper]

    def _remove_outliers_2d(self, data: List[List[float]]) -> List[List[float]]:
        """2D IQR 기반 outlier 제거"""
        if len(data) < 4:
            return data

        arr = np.array(data)
        result = []

        for i, point in enumerate(data):
            is_outlier = False
            for dim in range(arr.shape[1]):
                col = arr[:, dim]
                q1, q3 = np.percentile(col, [25, 75])
                iqr = q3 - q1
                lower = q1 - self.iqr_multiplier * iqr
                upper = q3 + self.iqr_multiplier * iqr

                if not (lower <= point[dim] <= upper):
                    is_outlier = True
                    break

            if not is_outlier:
                result.append(point)

        return result if result else data

    def _select_exemplars(
        self,
        samples: List[Dict],
        pattern_mean: Dict,
        top_k: int = 5
    ) -> List[Dict]:
        """
        패턴 평균에 가장 가까운 샘플들을 exemplar로 선정

        Args:
            samples: 모든 샘플 리스트
            pattern_mean: 패턴 평균값 딕셔너리
            top_k: 선정할 exemplar 개수

        Returns:
            [{filename, score}, ...] 점수 높은 순
        """
        scored_samples = []

        for sample in samples:
            if not sample.get("filename"):
                continue

            score = self._calculate_sample_score(sample, pattern_mean)
            scored_samples.append({
                "filename": sample["filename"],
                "score": score
            })

        # 점수 높은 순 정렬 (1.0에 가까울수록 좋음)
        scored_samples.sort(key=lambda x: x["score"], reverse=True)

        return scored_samples[:top_k]

    def _calculate_sample_score(
        self,
        sample: Dict,
        pattern_mean: Dict
    ) -> float:
        """
        샘플이 패턴 평균에 얼마나 가까운지 점수 계산

        Returns:
            0.0 ~ 1.0 (1.0 = 완벽히 일치)
        """
        scores = []
        weights = []

        # 1. 위치 점수 (가중치 0.35)
        if "position" in sample and "position" in pattern_mean:
            pos = sample["position"]
            mean_pos = pattern_mean["position"]
            # 유클리드 거리 (최대 ~1.4)
            dist = np.sqrt((pos[0] - mean_pos[0])**2 + (pos[1] - mean_pos[1])**2)
            pos_score = max(0, 1.0 - dist / 0.5)  # 0.5 이상 차이나면 0점
            scores.append(pos_score)
            weights.append(0.35)

        # 2. 크기 점수 (가중치 0.25)
        if "size_ratio" in sample and "size_ratio" in pattern_mean:
            size = sample["size_ratio"]
            mean_size = pattern_mean["size_ratio"]
            size_diff = abs(size - mean_size)
            size_score = max(0, 1.0 - size_diff / 0.2)  # 0.2 이상 차이나면 0점
            scores.append(size_score)
            weights.append(0.25)

        # 3. 압축감 점수 (가중치 0.25)
        if "compression_index" in sample and "compression_index" in pattern_mean:
            comp = sample["compression_index"]
            mean_comp = pattern_mean["compression_index"]
            comp_diff = abs(comp - mean_comp)
            comp_score = max(0, 1.0 - comp_diff / 0.3)  # 0.3 이상 차이나면 0점
            scores.append(comp_score)
            weights.append(0.25)

        # 4. 앵글 점수 (가중치 0.15)
        if "estimated_angle" in sample and "estimated_angle" in pattern_mean:
            angle = sample["estimated_angle"]
            mean_angle = pattern_mean["estimated_angle"]
            angle_diff = abs(angle - mean_angle)
            angle_score = max(0, 1.0 - angle_diff / 20)  # 20도 이상 차이나면 0점
            scores.append(angle_score)
            weights.append(0.15)

        if not scores:
            return 0.0

        # 가중 평균
        total_weight = sum(weights)
        weighted_score = sum(s * w for s, w in zip(scores, weights)) / total_weight

        return round(weighted_score, 4)

    def to_dict(self) -> Dict:
        """JSON 변환용 딕셔너리"""
        output = {}

        # 테마별로 그룹화
        themes = defaultdict(dict)
        for pattern_name, stats in self.patterns.items():
            theme = stats.theme
            pose_type = stats.pose_type

            if theme not in themes:
                themes[theme] = {
                    "theme_description": theme.replace("_", " ").title(),
                    "total_samples": 0,
                    "sub_patterns": {}
                }

            themes[theme]["total_samples"] += stats.sample_count
            themes[theme]["sub_patterns"][pattern_name] = {
                "theme": theme,
                "pose_type": pose_type,
                "sample_count": stats.sample_count,
                "composition": {
                    "position": {
                        "mean": stats.position_mean,
                        "std": stats.position_std,
                        "acceptable_range": {
                            "x": [
                                stats.position_mean[0] - stats.position_std[0] * 2,
                                stats.position_mean[0] + stats.position_std[0] * 2
                            ],
                            "y": [
                                stats.position_mean[1] - stats.position_std[1] * 2,
                                stats.position_mean[1] + stats.position_std[1] * 2
                            ]
                        }
                    },
                    "size": {
                        "mean": stats.size_mean,
                        "std": stats.size_std,
                        "optimal_range": [
                            max(0, stats.size_mean - stats.size_std * 2),
                            min(1, stats.size_mean + stats.size_std * 2)
                        ]
                    },
                    "margins": {
                        side: {"mean": stats.margins_mean[side], "std": stats.margins_std[side]}
                        for side in ["left", "right", "top", "bottom"]
                    }
                },
                "camera": {
                    "compression_index": {
                        "mean": stats.compression_mean,
                        "std": stats.compression_std,
                        "range": [
                            max(0, stats.compression_mean - stats.compression_std * 2),
                            min(1, stats.compression_mean + stats.compression_std * 2)
                        ]
                    },
                    "angle": {
                        "mean": stats.angle_mean,
                        "std": stats.angle_std,
                        "range": [
                            stats.angle_mean - stats.angle_std * 2,
                            stats.angle_mean + stats.angle_std * 2
                        ]
                    }
                },
                "pose_requirements": {
                    "sitting": stats.sitting_ratio > 0.5,
                    "sitting_ratio": stats.sitting_ratio,
                    "visible_joints": stats.common_visible_joints
                },
                "exemplars": stats.exemplars  # [{filename, score}, ...]
            }

        return dict(themes)

    def save_json(self, path: str):
        """JSON 파일로 저장"""
        output = self.to_dict()

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"[Saved] {path}")
        print(f"  - Themes: {len(output)}")
        print(f"  - Total patterns: {len(self.patterns)}")

    def load_json(self, path: str):
        """JSON 파일에서 로드"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data

    def summary(self) -> str:
        """통계 요약"""
        lines = ["=" * 50, "Pattern Statistics Summary", "=" * 50]

        for name, stats in self.patterns.items():
            lines.append(f"\n{name}:")
            lines.append(f"  Samples: {stats.sample_count}")
            lines.append(f"  Position: {stats.position_mean[0]:.2f}, {stats.position_mean[1]:.2f}")
            lines.append(f"  Size: {stats.size_mean:.2f} ± {stats.size_std:.2f}")
            lines.append(f"  Compression: {stats.compression_mean:.2f}")
            lines.append(f"  Angle: {stats.angle_mean:.1f}°")
            lines.append(f"  Sitting: {stats.sitting_ratio*100:.0f}%")

        return "\n".join(lines)


# ============================================================
# Quick Test
# ============================================================
if __name__ == "__main__":
    stats = PatternStatistics(min_samples=5)  # 테스트용 낮은 threshold

    # 테스트 샘플 추가
    for i in range(10):
        stats.add_sample(
            theme="cafe_indoor",
            pose_type="half_body",
            features={
                "position": [0.35 + np.random.normal(0, 0.05), 0.42 + np.random.normal(0, 0.05)],
                "size_ratio": 0.32 + np.random.normal(0, 0.03),
                "margins": {
                    "left": 0.22 + np.random.normal(0, 0.02),
                    "right": 0.46 + np.random.normal(0, 0.04),
                    "top": 0.18 + np.random.normal(0, 0.02),
                    "bottom": 0.38 + np.random.normal(0, 0.05)
                },
                "compression_index": 0.45 + np.random.normal(0, 0.05),
                "estimated_angle": 12 + np.random.normal(0, 3),
                "sitting": True,
                "visible_joints": ["nose", "left_shoulder", "right_shoulder"]
            },
            filename=f"test_{i}.jpg"
        )

    stats.calculate_all()
    print(stats.summary())

    # JSON 저장 테스트
    # stats.save_json("test_patterns.json")
