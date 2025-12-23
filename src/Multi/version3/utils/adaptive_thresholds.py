# ============================================================
# 🎚️ Adaptive Thresholds
# Phase 2.4: 클러스터 기반 동적 threshold
# ============================================================

import json
from pathlib import Path
from typing import Dict, Optional
import sys

# Project paths
UTILS_DIR = Path(__file__).resolve().parent
VERSION3_DIR = UTILS_DIR.parent
PROJECT_ROOT = VERSION3_DIR
while PROJECT_ROOT != PROJECT_ROOT.parent and not ((PROJECT_ROOT / "data").exists() and (PROJECT_ROOT / "src").exists()):
    PROJECT_ROOT = PROJECT_ROOT.parent


class AdaptiveThresholdManager:
    """
    클러스터별 맞춤형 Threshold 관리

    사용자 관점:
    - 인물 클로즈업: 얼굴 흐리면 안 됨 → 블러 엄격
    - 풍경 사진: 약간 흐려도 OK → 블러 관대
    - 제품 사진: 디테일 중요 → 선명도 엄격
    """

    # 기본 Threshold (클러스터 정보 없을 때)
    DEFAULT_THRESHOLDS = {
        'blur': {
            'critical': 50,      # 이것보다 낮으면 "다시 찍기"
            'acceptable': 100,   # 이것보다 높으면 "괜찮음"
            'good': 150          # 이것보다 높으면 "좋음"
        },
        'noise': {
            'good': 0.1,         # 낮을수록 좋음
            'acceptable': 0.2,
            'critical': 0.4
        },
        'sharpness': {
            'critical': 30,
            'acceptable': 50,
            'good': 70
        }
    }

    # 클러스터 타입별 조정 계수
    CLUSTER_TYPE_ADJUSTMENTS = {
        # 인물 클로즈업 (얼굴이 중요)
        'closeup': {
            'blur': 1.3,        # 더 엄격 (threshold 높임)
            'noise': 0.8,       # 더 엄격 (threshold 낮춤)
            'sharpness': 1.2
        },
        # 전신/반신 (일반)
        'portrait': {
            'blur': 1.0,
            'noise': 1.0,
            'sharpness': 1.0
        },
        # 풍경
        'landscape': {
            'blur': 0.8,        # 더 관대
            'noise': 1.2,       # 더 관대
            'sharpness': 0.9
        },
        # 제품 (디테일 중요)
        'product': {
            'blur': 1.4,        # 매우 엄격
            'noise': 0.7,
            'sharpness': 1.3
        }
    }

    def __init__(self, cluster_info_path: Optional[str] = None):
        """
        Args:
            cluster_info_path: cluster_interpretation.json 경로
        """
        if cluster_info_path is None:
            cluster_info_path = PROJECT_ROOT / "features" / "cluster_interpretation.json"

        self.cluster_info_path = Path(cluster_info_path)
        self.cluster_info = {}

        if self.cluster_info_path.exists():
            with open(self.cluster_info_path, 'r', encoding='utf-8') as f:
                self.cluster_info = json.load(f)

    def _detect_cluster_type(self, cluster_id: int) -> str:
        """
        클러스터 타입 자동 감지

        Returns:
            'closeup', 'portrait', 'landscape', 'product'
        """
        if str(cluster_id) not in self.cluster_info:
            return 'portrait'  # 기본값

        cluster_data = self.cluster_info[str(cluster_id)]
        label = cluster_data.get('auto_label', '').lower()

        # 라벨 기반 타입 감지
        if '클로즈업' in label or 'closeup' in label or '얼굴' in label:
            return 'closeup'
        elif '풍경' in label or 'landscape' in label or '실외' in label:
            return 'landscape'
        elif '제품' in label or 'product' in label:
            return 'product'
        else:
            return 'portrait'

    def get_threshold(
        self,
        metric: str,
        level: str,
        cluster_id: Optional[int] = None
    ) -> float:
        """
        클러스터별 맞춤 threshold 계산

        Args:
            metric: 'blur', 'noise', 'sharpness'
            level: 'critical', 'acceptable', 'good'
            cluster_id: 클러스터 ID (None이면 기본값)

        Returns:
            조정된 threshold
        """
        # 기본 threshold
        base_threshold = self.DEFAULT_THRESHOLDS.get(metric, {}).get(level, 0)

        if cluster_id is None:
            return base_threshold

        # 클러스터 타입 감지
        cluster_type = self._detect_cluster_type(cluster_id)

        # 조정 계수 적용
        adjustment = self.CLUSTER_TYPE_ADJUSTMENTS.get(cluster_type, {}).get(metric, 1.0)

        adjusted_threshold = base_threshold * adjustment

        return adjusted_threshold

    def evaluate_quality(
        self,
        metric: str,
        value: float,
        cluster_id: Optional[int] = None
    ) -> Dict:
        """
        품질 평가

        Args:
            metric: 'blur', 'noise', 'sharpness'
            value: 측정값
            cluster_id: 클러스터 ID

        Returns:
            {
                'level': 'critical'/'acceptable'/'good'/'excellent',
                'message': 사용자 메시지,
                'threshold_used': 사용된 threshold,
                'cluster_adjusted': 클러스터 조정 여부
            }
        """
        cluster_type = self._detect_cluster_type(cluster_id) if cluster_id is not None else 'portrait'

        # Threshold 가져오기
        critical = self.get_threshold(metric, 'critical', cluster_id)
        acceptable = self.get_threshold(metric, 'acceptable', cluster_id)
        good = self.get_threshold(metric, 'good', cluster_id)

        # 평가 (노이즈는 반대 - 낮을수록 좋음)
        if metric == 'noise':
            if value <= good:
                level = 'excellent'
                message = "✅ 노이즈가 거의 없어요"
            elif value <= acceptable:
                level = 'good'
                message = "👍 노이즈가 적당해요"
            elif value <= critical:
                level = 'acceptable'
                message = "⚠️ 노이즈가 조금 있어요"
            else:
                level = 'critical'
                message = f"🔴 노이즈가 심해요 (ISO를 낮추세요)"
        else:
            # blur, sharpness는 높을수록 좋음
            if value >= good:
                level = 'excellent'
                message = f"✅ {metric}가 훌륭해요"
            elif value >= acceptable:
                level = 'good'
                message = f"👍 {metric}가 적당해요"
            elif value >= critical:
                level = 'acceptable'
                message = f"⚠️ {metric}를 개선하세요"
            else:
                level = 'critical'
                message = f"🔴 {metric} 심각 - 다시 찍으세요"

        return {
            'level': level,
            'message': message,
            'value': value,
            'threshold_used': {
                'critical': critical,
                'acceptable': acceptable,
                'good': good
            },
            'cluster_type': cluster_type,
            'cluster_adjusted': cluster_id is not None
        }

    def get_user_friendly_message(self, evaluation: Dict) -> str:
        """
        사용자 친화적 메시지 생성
        """
        level = evaluation['level']
        cluster_type = evaluation['cluster_type']

        # 클러스터 타입별 설명 추가
        type_descriptions = {
            'closeup': '(인물 클로즈업이라 더 엄격하게 평가했어요)',
            'portrait': '',
            'landscape': '(풍경 사진이라 조금 너그럽게 평가했어요)',
            'product': '(제품 사진이라 더 엄격하게 평가했어요)'
        }

        type_desc = type_descriptions.get(cluster_type, '')

        message = evaluation['message']

        if evaluation['cluster_adjusted']:
            message += f" {type_desc}"

        return message


# ============================================================
# 사용 예시
# ============================================================

if __name__ == "__main__":
    manager = AdaptiveThresholdManager()

    print("="*60)
    print("Phase 2.4: 클러스터 기반 동적 threshold 테스트")
    print("="*60)

    # 시뮬레이션: 같은 블러 값 (120)이 클러스터별로 다르게 평가됨

    blur_value = 120

    print(f"\n📊 Blur 값: {blur_value}")
    print("-"*60)

    # 1. 인물 클로즈업 (엄격)
    print("\n1. 인물 클로즈업 (Cluster 0)")
    eval_closeup = manager.evaluate_quality('blur', blur_value, cluster_id=0)
    print(manager.get_user_friendly_message(eval_closeup))
    print(f"   레벨: {eval_closeup['level']}")
    print(f"   Threshold: {eval_closeup['threshold_used']}")

    # 2. 일반 인물 (보통)
    print("\n2. 일반 인물 (Cluster 5)")
    eval_portrait = manager.evaluate_quality('blur', blur_value, cluster_id=5)
    print(manager.get_user_friendly_message(eval_portrait))
    print(f"   레벨: {eval_portrait['level']}")

    # 3. 풍경 (관대)
    print("\n3. 풍경 사진 (가상 Cluster)")
    # 풍경 클러스터가 실제로 있다면
    eval_landscape = manager.evaluate_quality('blur', blur_value, cluster_id=None)  # 기본값
    print(manager.get_user_friendly_message(eval_landscape))
    print(f"   레벨: {eval_landscape['level']}")

    # 노이즈 테스트
    print("\n" + "="*60)
    print("노이즈 평가 테스트")
    print("="*60)

    noise_value = 0.15

    print(f"\n📊 Noise 값: {noise_value}")

    for cluster_type in ['closeup', 'portrait', 'landscape']:
        # 임시로 cluster_type을 설정 (실제로는 cluster_id로 자동 감지)
        print(f"\n{cluster_type}:")
        eval_noise = manager.evaluate_quality('noise', noise_value, cluster_id=0 if cluster_type == 'closeup' else 5)
        print(f"   {eval_noise['message']}")
        print(f"   레벨: {eval_noise['level']}")
