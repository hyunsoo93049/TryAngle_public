# ============================================================
# 💡 Reference Recommender
# Phase 3.1: AI 기반 레퍼런스 추천 시스템
# ============================================================

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import sys

# Project paths
UTILS_DIR = Path(__file__).resolve().parent
VERSION3_DIR = UTILS_DIR.parent
PROJECT_ROOT = VERSION3_DIR
while PROJECT_ROOT != PROJECT_ROOT.parent and not ((PROJECT_ROOT / "data").exists() and (PROJECT_ROOT / "src").exists()):
    PROJECT_ROOT = PROJECT_ROOT.parent

if str(VERSION3_DIR) not in sys.path:
    sys.path.append(str(VERSION3_DIR))


class ReferenceRecommender:
    """
    사용자 이미지와 비슷하면서 더 나은 레퍼런스 추천

    사용자 관점:
    - "어떤 사진을 목표로 해야 할지 모르겠어요"
    - "내 사진과 비슷한 좋은 예시 보여주세요"
    - "이 스타일로 더 잘 찍는 법?"
    """

    def __init__(
        self,
        clustered_images_dir: Optional[str] = None,
        cluster_info_path: Optional[str] = None
    ):
        """
        Args:
            clustered_images_dir: data/clustered_images 경로
            cluster_info_path: cluster_interpretation.json 경로
        """
        if clustered_images_dir is None:
            clustered_images_dir = PROJECT_ROOT / "data" / "clustered_images"
        if cluster_info_path is None:
            cluster_info_path = PROJECT_ROOT / "features" / "cluster_interpretation.json"

        self.clustered_dir = Path(clustered_images_dir)
        self.cluster_info_path = Path(cluster_info_path)

        # 클러스터 정보 로드
        with open(self.cluster_info_path, 'r', encoding='utf-8') as f:
            self.cluster_info = json.load(f)

    def recommend(
        self,
        user_image_path: str,
        user_cluster_id: int,
        user_embedding: np.ndarray,
        top_k: int = 3,
        quality_threshold: float = 0.7
    ) -> List[Dict]:
        """
        레퍼런스 추천

        Args:
            user_image_path: 사용자 이미지 경로
            user_cluster_id: 사용자 이미지의 클러스터 ID
            user_embedding: 사용자 이미지의 embedding (128D)
            top_k: 추천할 개수
            quality_threshold: 품질 필터 (0-1)

        Returns:
            [
                {
                    'image_path': 추천 이미지 경로,
                    'cluster_id': 클러스터 ID,
                    'similarity': 유사도 (0-1),
                    'reason': 추천 이유
                },
                ...
            ]
        """
        # 같은 클러스터의 이미지들 수집
        cluster_folder = self.clustered_dir / f"cluster_{user_cluster_id}"

        if not cluster_folder.exists():
            return []

        # 클러스터 내 이미지 목록
        image_files = list(cluster_folder.glob("*.jpg")) + list(cluster_folder.glob("*.png")) + list(cluster_folder.glob("*.jpeg"))

        if len(image_files) == 0:
            return []

        # 사용자 이미지 제외
        user_path = Path(user_image_path).resolve()
        candidates = [img for img in image_files if img.resolve() != user_path]

        if len(candidates) == 0:
            return []

        # 간단한 휴리스틱으로 품질 추정
        # (실제로는 quality_analyzer를 사용할 수 있지만, 여기선 파일 크기 & 개수로 추정)
        quality_scores = self._estimate_quality(candidates)

        # 품질 필터링
        high_quality = [
            (img, score) for img, score in zip(candidates, quality_scores)
            if score >= quality_threshold
        ]

        if len(high_quality) == 0:
            # 품질 기준 낮추기
            high_quality = [(img, score) for img, score in zip(candidates, quality_scores)]

        # 유사도 계산 (여기서는 랜덤, 실제로는 embedding 거리)
        # 실제 구현에서는 각 이미지의 embedding을 로드해서 cosine similarity 계산
        recommendations = []

        for img_path, quality_score in high_quality[:top_k * 2]:  # 여유있게 2배
            # 유사도 (0-1, 랜덤 시뮬레이션)
            # 실제로는: cosine_similarity(user_embedding, image_embedding)
            similarity = np.random.uniform(0.7, 0.95)

            recommendations.append({
                'image_path': str(img_path),
                'cluster_id': user_cluster_id,
                'similarity': similarity,
                'quality_score': quality_score,
                'reason': self._generate_reason(similarity, quality_score)
            })

        # 유사도 순 정렬
        recommendations.sort(key=lambda x: x['similarity'], reverse=True)

        return recommendations[:top_k]

    def _estimate_quality(self, image_paths: List[Path]) -> List[float]:
        """
        간단한 품질 추정 (휴리스틱)

        실제로는 quality_analyzer 사용 권장
        """
        quality_scores = []

        for img_path in image_paths:
            # 파일 크기 기반 (큰 파일 = 고품질)
            file_size = img_path.stat().st_size
            size_score = min(file_size / (5 * 1024 * 1024), 1.0)  # 5MB 기준

            # 랜덤 노이즈
            quality_scores.append(size_score * np.random.uniform(0.8, 1.0))

        return quality_scores

    def _generate_reason(self, similarity: float, quality_score: float) -> str:
        """
        추천 이유 생성
        """
        if similarity > 0.9 and quality_score > 0.8:
            return "매우 유사하면서 고품질이에요!"
        elif similarity > 0.85:
            return "비슷한 스타일이에요"
        elif quality_score > 0.8:
            return "고품질 참고 이미지예요"
        else:
            return "같은 스타일이에요"

    def format_recommendations(self, recommendations: List[Dict]) -> str:
        """
        추천 결과를 사용자 친화적으로 표시
        """
        if not recommendations:
            return "추천할 이미지가 없어요. 다른 스타일을 시도해보세요!"

        cluster_id = recommendations[0]['cluster_id']
        cluster_label = self.cluster_info[str(cluster_id)]['auto_label']

        lines = []
        lines.append("="*60)
        lines.append(f"💡 추천 레퍼런스 ({cluster_label})")
        lines.append("="*60)

        for i, rec in enumerate(recommendations, 1):
            lines.append(f"\n{i}. 📸 유사도: {rec['similarity']:.0%}")
            lines.append(f"   {rec['reason']}")
            lines.append(f"   파일: {Path(rec['image_path']).name}")

        lines.append("\n💡 이 사진들을 참고해서 촬영해보세요!")

        return "\n".join(lines)


# ============================================================
# 사용 예시
# ============================================================

if __name__ == "__main__":
    print("="*60)
    print("Phase 3.1: AI 레퍼런스 추천 시스템 테스트")
    print("="*60)

    # 실제 데이터로 테스트 (clustered_images가 있다면)
    clustered_dir = PROJECT_ROOT / "data" / "clustered_images"

    if not clustered_dir.exists():
        print(f"\n⚠️  {clustered_dir} 없음")
        print("실제 클러스터링 데이터가 필요합니다")
        print("\n시뮬레이션 모드:")

        # 시뮬레이션
        print("\n사용자 이미지: test.jpg")
        print("클러스터: 5 (실외/멀리/웜톤/반신)")
        print("Embedding: [128D vector]")
        print("\n추천 결과:")
        print("1. 📸 유사도: 92%")
        print("   매우 유사하면서 고품질이에요!")
        print("   파일: IMG_1234.jpg")
        print("\n2. 📸 유사도: 88%")
        print("   비슷한 스타일이에요")
        print("   파일: IMG_5678.jpg")
        print("\n3. 📸 유사도: 85%")
        print("   같은 스타일이에요")
        print("   파일: IMG_9012.jpg")
    else:
        # 실제 데이터로 테스트
        recommender = ReferenceRecommender()

        # 임의의 사용자 데이터
        user_cluster = 5
        user_embedding = np.random.rand(128)
        user_image = "test_user.jpg"

        print(f"\n사용자 이미지: {user_image}")
        print(f"클러스터: {user_cluster}")

        recommendations = recommender.recommend(
            user_image_path=user_image,
            user_cluster_id=user_cluster,
            user_embedding=user_embedding,
            top_k=3
        )

        print(recommender.format_recommendations(recommendations))

    # 사용자 관점 설명
    print("\n" + "="*60)
    print("💬 사용자에게 이렇게 보입니다:")
    print("="*60)
    print('"어떤 사진을 찍어야 할지 모르겠어요"')
    print('→ AI가 비슷한 스타일의 좋은 예시 3개 추천')
    print('→ "이 사진들을 참고해서 촬영해보세요!"')
