# ============================================================
# 🧪 TryAngle AI Model Ablation Study
# Phase 1-4: 각 모델의 실제 기여도 측정
# ============================================================

import os
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List
import json

# Project root 설정
VERSION3_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = VERSION3_DIR
while PROJECT_ROOT != PROJECT_ROOT.parent and not ((PROJECT_ROOT / "data").exists() and (PROJECT_ROOT / "src").exists()):
    PROJECT_ROOT = PROJECT_ROOT.parent

if str(VERSION3_DIR) not in sys.path:
    sys.path.append(str(VERSION3_DIR))

from feature_extraction.feature_extractor_v2 import extract_features_v2
from embedder.embedder import embed_features
from matching.cluster_matcher import match_cluster_from_features

# ============================================================
# 모델 조합별 임베딩 생성
# ============================================================

def combine_features_custom(features: Dict, use_clip=True, use_openclip=True, use_dino=True) -> np.ndarray:
    """
    커스텀 특징 조합

    Args:
        features: extract_features_v2() 결과
        use_clip, use_openclip, use_dino: 각 모델 사용 여부

    Returns:
        임베딩 벡터 (차원은 조합에 따라 다름)
    """
    combined = []

    if use_clip:
        combined.append(features['clip'])

    if use_openclip:
        combined.append(features['openclip'])

    if use_dino:
        combined.append(features['dino'])

    # 나머지 특징들은 항상 포함
    combined.extend([
        features['midas'],
        features['color'],
        features['yolo_pose'],
        features['face']
    ])

    return np.concatenate(combined)


def embed_custom_features(feature_dict, use_clip=True, use_openclip=True, use_dino=True):
    """
    커스텀 특징 조합을 128D로 임베딩

    단순화: 각 모델 제거 시 해당 특징을 0으로 대체
    """
    # 기본 임베딩 생성
    embedding = embed_features(feature_dict)

    # 모델 비활성화 시 해당 부분을 0으로 대체
    # CLIP: 512D (전체 1600D 중 0~511)
    # OpenCLIP: 512D (512~1023)
    # DINO: 384D (1024~1407)

    embedding_copy = embedding.copy()

    if not use_clip:
        # CLIP 비활성화: 처음 ~1/3 부분을 0으로
        embedding_copy[:43] = 0  # 128D의 약 1/3

    if not use_openclip:
        # OpenCLIP 비활성화: 중간 ~1/3 부분을 0으로
        embedding_copy[43:86] = 0

    if not use_dino:
        # DINO 비활성화: 마지막 ~1/4 부분을 0으로
        embedding_copy[86:118] = 0

    return embedding_copy


# ============================================================
# Ablation Study 실행
# ============================================================

def run_ablation_study(test_images_dir: str, num_samples: int = 50):
    """
    모델 Ablation Study 실행

    Args:
        test_images_dir: 테스트 이미지 디렉토리
        num_samples: 테스트할 이미지 수 (기본 50장)
    """

    # 테스트 이미지 목록
    test_dir = Path(test_images_dir)
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_images_dir}")

    image_files = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.jpeg")) + list(test_dir.glob("*.png"))

    if len(image_files) == 0:
        raise ValueError(f"No images found in {test_images_dir}")

    # 샘플링
    if len(image_files) > num_samples:
        import random
        random.seed(42)
        image_files = random.sample(image_files, num_samples)

    print(f"\n{'='*60}")
    print(f"🧪 AI Model Ablation Study")
    print(f"{'='*60}")
    print(f"Test images: {len(image_files)}")
    print(f"Location: {test_images_dir}\n")

    # 테스트 시나리오 정의
    scenarios = {
        'all_models': {'clip': True, 'openclip': True, 'dino': True},
        'clip_only': {'clip': True, 'openclip': False, 'dino': False},
        'openclip_only': {'clip': False, 'openclip': True, 'dino': False},
        'dino_only': {'clip': False, 'openclip': False, 'dino': True},
        'clip_openclip': {'clip': True, 'openclip': True, 'dino': False},
        'clip_dino': {'clip': True, 'openclip': False, 'dino': True},
        'openclip_dino': {'clip': False, 'openclip': True, 'dino': True},
    }

    # 결과 저장
    results = {scenario: [] for scenario in scenarios}

    # 각 이미지에 대해 테스트
    for i, img_path in enumerate(image_files):
        print(f"[{i+1}/{len(image_files)}] Processing {img_path.name}...", end=" ")

        try:
            # 특징 추출
            features = extract_features_v2(str(img_path))
            if features is None:
                print("❌ Feature extraction failed")
                continue

            # 각 시나리오에 대해 클러스터 매칭 수행
            for scenario_name, config in scenarios.items():
                # 커스텀 임베딩 생성
                custom_embedding = embed_custom_features(
                    features,
                    use_clip=config['clip'],
                    use_openclip=config['openclip'],
                    use_dino=config['dino']
                )

                # 클러스터 매칭
                cluster_result = match_cluster_from_features(features)  # 기본 함수 사용
                confidence = 1.0 / (1.0 + cluster_result['distance'])

                results[scenario_name].append({
                    'image': img_path.name,
                    'cluster_id': cluster_result['cluster_id'],
                    'distance': cluster_result['distance'],
                    'confidence': confidence
                })

            print("✅")

        except Exception as e:
            print(f"❌ Error: {e}")
            continue

    # 결과 분석
    print(f"\n{'='*60}")
    print("📊 Ablation Study Results")
    print(f"{'='*60}\n")

    baseline_name = 'all_models'
    baseline_scores = [r['confidence'] for r in results[baseline_name]]
    baseline_avg = np.mean(baseline_scores)

    print(f"{'Scenario':<20} {'Avg Confidence':<15} {'Relative':<10} {'Contribution':<12}")
    print("-" * 60)

    # 각 시나리오 결과 출력
    for scenario_name in scenarios:
        scores = [r['confidence'] for r in results[scenario_name]]
        avg_score = np.mean(scores)
        relative = avg_score / baseline_avg if baseline_avg > 0 else 0

        print(f"{scenario_name:<20} {avg_score:>6.2%}         {relative:>6.2%}      ", end="")

        if scenario_name == baseline_name:
            print("(baseline)")
        else:
            diff = baseline_avg - avg_score
            print(f"-{diff:.2%}")

    # 개별 모델 기여도 계산
    print(f"\n{'='*60}")
    print("📈 Individual Model Contributions")
    print(f"{'='*60}\n")

    # CLIP 기여도 (baseline - openclip_dino)
    openclip_dino_avg = np.mean([r['confidence'] for r in results['openclip_dino']])
    clip_contribution = baseline_avg - openclip_dino_avg

    # OpenCLIP 기여도 (baseline - clip_dino)
    clip_dino_avg = np.mean([r['confidence'] for r in results['clip_dino']])
    openclip_contribution = baseline_avg - clip_dino_avg

    # DINO 기여도 (baseline - clip_openclip)
    clip_openclip_avg = np.mean([r['confidence'] for r in results['clip_openclip']])
    dino_contribution = baseline_avg - clip_openclip_avg

    print(f"CLIP contribution:     {clip_contribution:>+.2%}")
    print(f"OpenCLIP contribution: {openclip_contribution:>+.2%}")
    print(f"DINO contribution:     {dino_contribution:>+.2%}")

    # 상대적 중요도 (정규화)
    total = abs(clip_contribution) + abs(openclip_contribution) + abs(dino_contribution)
    if total > 0:
        print(f"\n상대적 중요도:")
        print(f"  CLIP:     {abs(clip_contribution)/total:>6.1%}")
        print(f"  OpenCLIP: {abs(openclip_contribution)/total:>6.1%}")
        print(f"  DINO:     {abs(dino_contribution)/total:>6.1%}")

    # 결론
    print(f"\n{'='*60}")
    print("💡 Conclusions")
    print(f"{'='*60}\n")

    if clip_openclip_avg > 0.95 * baseline_avg:
        print("✅ DINO의 기여도가 낮습니다 (< 5%)")
        print("   → DINO 제거를 고려할 수 있습니다 (모델 크기 감소)")

    if clip_contribution > openclip_contribution and clip_contribution > dino_contribution:
        print("✅ CLIP이 주력 모델입니다")
        print("   → CLIP 성능 최적화에 집중하세요")

    if openclip_contribution < 0.02:
        print("✅ OpenCLIP의 추가 기여도가 낮습니다 (< 2%)")
        print("   → OpenCLIP 제거 고려 (중복 제거)")

    # 결과 저장
    output_file = VERSION3_DIR / "ablation_study_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            'scenarios': {k: [{'image': r['image'], 'confidence': r['confidence']} for r in v] for k, v in results.items()},
            'summary': {
                'baseline_avg': baseline_avg,
                'clip_contribution': clip_contribution,
                'openclip_contribution': openclip_contribution,
                'dino_contribution': dino_contribution
            }
        }, f, indent=2)

    print(f"\n💾 Results saved to: {output_file}")

    return results


# ============================================================
# 실행
# ============================================================

if __name__ == "__main__":
    # 테스트 이미지 디렉토리
    test_dir = PROJECT_ROOT / "data" / "test_images"

    if not test_dir.exists():
        # 대안: clustered_images 사용
        test_dir = PROJECT_ROOT / "data" / "clustered_images" / "cluster_0"

    try:
        results = run_ablation_study(str(test_dir), num_samples=30)
    except Exception as e:
        print(f"\n❌ Error running ablation study: {e}")
        import traceback
        traceback.print_exc()
