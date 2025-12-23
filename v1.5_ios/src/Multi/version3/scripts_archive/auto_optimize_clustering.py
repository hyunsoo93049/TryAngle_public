# ============================================================
# 🤖 TryAngle Auto Clustering Optimizer
# 최적 K와 가중치를 자동으로 찾기
# ============================================================

import os
import numpy as np
import polars as pl
import joblib
import json
from sklearn.preprocessing import RobustScaler
from umap import UMAP
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from datetime import datetime
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
while PROJECT_ROOT != PROJECT_ROOT.parent and not ((PROJECT_ROOT / "data").exists() and (PROJECT_ROOT / "src").exists()):
    PROJECT_ROOT = PROJECT_ROOT.parent

# ============================================================
# 설정
# ============================================================
INPUT_PARQUET = PROJECT_ROOT / "feature_models" / "features" / "fusion_features_v2.parquet"
OUTPUT_DIR = PROJECT_ROOT / "feature_models"
RESULTS_FILE = OUTPUT_DIR / "optimization_results.json"

# ============================================================
# 탐색 공간
# ============================================================

# 클러스터 개수
K_RANGE = [10, 12, 15, 18, 20, 25]

# 가중치 스타일들
WEIGHT_STYLES = {
    "balanced": {
        "clip": 0.20,
        "openclip": 0.20,
        "dino": 0.15,
        "color": 0.20,
        "midas": 0.10,
        "pose": 0.15,
    },
    "clip_focused": {
        "clip": 0.32,
        "openclip": 0.32,
        "dino": 0.24,
        "color": 0.10,
        "midas": 0.01,
        "pose": 0.01,
    },
    "original_style": {
        "clip": 0.30,
        "openclip": 0.30,
        "dino": 0.25,
        "color": 0.12,
        "midas": 0.03,
        "pose": 0.00,
    },
    "compromise": {
        "clip": 0.26,
        "openclip": 0.26,
        "dino": 0.22,
        "color": 0.15,
        "midas": 0.06,
        "pose": 0.05,
    },
    "aggressive": {
        "clip": 0.35,
        "openclip": 0.35,
        "dino": 0.25,
        "color": 0.05,
        "midas": 0.00,
        "pose": 0.00,
    },
}

# UMAP 차원
UMAP_DIMS = [128, 192]


# ============================================================
# 평가 함수
# ============================================================
def evaluate_clustering(data, labels):
    """
    클러스터링 품질 평가

    Returns:
        dict: 여러 평가 지표
    """
    silhouette = silhouette_score(data, labels, metric="euclidean")
    davies_bouldin = davies_bouldin_score(data, labels)
    calinski_harabasz = calinski_harabasz_score(data, labels)

    return {
        "silhouette": float(silhouette),  # 높을수록 좋음 (0~1)
        "davies_bouldin": float(davies_bouldin),  # 낮을수록 좋음
        "calinski_harabasz": float(calinski_harabasz),  # 높을수록 좋음
    }


# ============================================================
# 단일 조합 테스트
# ============================================================
def test_single_combination(
    scaled_features, K, weights, umap_dim, show_progress=True
):
    """
    하나의 조합 테스트
    """
    try:
        # --------------------------------------------------------
        # 1) 가중치 융합 (가중치가 0인 특징은 아예 제외)
        # --------------------------------------------------------
        feature_list = []

        # CLIP
        if weights["clip"] > 0:
            feature_list.append(
                scaled_features["clip"] * weights["clip"]
            )

        # OpenCLIP
        if weights["openclip"] > 0:
            feature_list.append(
                scaled_features["openclip"] * weights["openclip"]
            )

        # DINO
        if weights["dino"] > 0:
            feature_list.append(
                scaled_features["dino"] * weights["dino"]
            )

        # Color
        if weights["color"] > 0:
            feature_list.append(
                scaled_features["color"] * weights["color"]
            )

        # MiDaS
        if weights["midas"] > 0:
            feature_list.append(
                scaled_features["midas"] * weights["midas"]
            )

        # Pose + Face (pose 가중치가 0이면 아예 포함 X)
        if weights["pose"] > 0:
            pose_combined = np.concatenate(
                [
                    scaled_features["yolo_pose"],
                    scaled_features["face"],
                ],
                axis=1,
            )
            feature_list.append(
                pose_combined * weights["pose"]
            )

        # 최종 융합 벡터
        if len(feature_list) == 0:
            raise ValueError(
                "No feature used in fusion (all weights are zero)."
            )

        fusion = np.concatenate(feature_list, axis=1)

        # --------------------------------------------------------
        # 2) UMAP
        # --------------------------------------------------------
        umap_model = UMAP(
            n_components=umap_dim,
            n_neighbors=15,
            min_dist=0.1,
            metric="cosine",
            random_state=42,
            verbose=show_progress,  # 🔥 진행상황 표시
        )

        fusion_low = umap_model.fit_transform(fusion)

        # --------------------------------------------------------
        # 3) KMeans
        # --------------------------------------------------------
        kmeans = KMeans(
            n_clusters=K,
            random_state=42,
            n_init=5,  # 빠르게 (10 → 5)
            max_iter=200,  # 빠르게 (300 → 200)
            verbose=0,
        )

        clusters = kmeans.fit_predict(fusion_low)

        # --------------------------------------------------------
        # 4) 평가
        # --------------------------------------------------------
        scores = evaluate_clustering(fusion_low, clusters)

        return {
            "success": True,
            "scores": scores,
            "n_samples": len(fusion_low),
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


# ============================================================
# 메인 최적화
# ============================================================
def main():
    print("=" * 60)
    print("🤖 TryAngle Auto Clustering Optimizer")
    print("=" * 60)

    start_time = time.time()

    # --------------------------------------------------------
    # Step 0: 파일 존재 확인 🔥
    # --------------------------------------------------------
    print("\n🔍 Checking files...")

    if not os.path.exists(INPUT_PARQUET):
        print("❌ ERROR: Input file not found!")
        print(f"   Expected: {INPUT_PARQUET}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("✅ Files OK")

    # --------------------------------------------------------
    # Step 1: 데이터 로드
    # --------------------------------------------------------
    print("\n📂 Loading features...")
    df = pl.read_parquet(INPUT_PARQUET)
    print(f"✅ Loaded {len(df)} samples")

    # --------------------------------------------------------
    # Step 2: Scaler 학습 (한 번만)
    # --------------------------------------------------------
    print("\n🔧 Training Scalers (once)...")

    scalers = {}
    scaled_features = {}

    for feat_name in [
        "clip",
        "openclip",
        "dino",
        "color",
        "midas",
        "yolo_pose",
        "face",
    ]:
        feat_array = np.vstack(df[feat_name].to_list())
        scaler = RobustScaler()
        scaled = scaler.fit_transform(feat_array)

        scalers[feat_name] = scaler
        scaled_features[feat_name] = scaled

    print("✅ Scalers ready")

    # --------------------------------------------------------
    # Step 3: Grid Search
    # --------------------------------------------------------
    print("\n🔍 Starting Grid Search...")
    print(f"   K values: {K_RANGE}")
    print(f"   Weight styles: {list(WEIGHT_STYLES.keys())}")
    print(f"   UMAP dims: {UMAP_DIMS}")

    total_combinations = (
        len(K_RANGE) * len(WEIGHT_STYLES) * len(UMAP_DIMS)
    )
    print(f"\n   Total combinations: {total_combinations}")
    print(f"   Estimated time: {total_combinations * 2:.0f} minutes\n")

    results = []
    best_score = -1
    best_config = None

    combo_idx = 0
    combo_times = []  # 🔥 시간 추적

    for K in K_RANGE:
        for style_name, weights in WEIGHT_STYLES.items():
            for umap_dim in UMAP_DIMS:
                combo_idx += 1

                print("\n" + "=" * 60)
                print(f"[{combo_idx}/{total_combinations}] Testing:")
                print(
                    f"   K={K}, style={style_name}, umap_dim={umap_dim}"
                )

                # 🔥 예상 남은 시간 계산
                if combo_times:
                    avg_time = np.mean(combo_times)
                    remaining = (
                        total_combinations - combo_idx
                    ) * avg_time
                    print(
                        f"   ⏱ Estimated remaining: {remaining/60:.1f} minutes"
                    )

                combo_start = time.time()

                # 테스트 (첫 번째만 verbose=True)
                result = test_single_combination(
                    scaled_features,
                    K,
                    weights,
                    umap_dim,
                    show_progress=(combo_idx == 1),
                )

                combo_time = time.time() - combo_start
                combo_times.append(combo_time)

                if result["success"]:
                    scores = result["scores"]
                    silhouette = scores["silhouette"]

                    print(f"   ✅ Silhouette: {silhouette:.4f}")
                    print(
                        f"   Davies-Bouldin: {scores['davies_bouldin']:.4f}"
                    )
                    print(
                        f"   Calinski-Harabasz: {scores['calinski_harabasz']:.2f}"
                    )
                    print(f"   Time: {combo_time:.1f}s")

                    # 결과 저장
                    results.append(
                        {
                            "K": K,
                            "style": style_name,
                            "weights": weights,
                            "umap_dim": umap_dim,
                            "silhouette": silhouette,
                            "davies_bouldin": scores[
                                "davies_bouldin"
                            ],
                            "calinski_harabasz": scores[
                                "calinski_harabasz"
                            ],
                            "time_seconds": combo_time,
                        }
                    )

                    # 최고 점수 업데이트
                    if silhouette > best_score:
                        best_score = silhouette
                        best_config = {
                            "K": K,
                            "style": style_name,
                            "weights": weights,
                            "umap_dim": umap_dim,
                            "scores": scores,
                        }
                        print(
                            f"   🔥 NEW BEST! Silhouette: {silhouette:.4f}"
                        )
                else:
                    print(f"   ❌ Failed: {result['error']}")

    # --------------------------------------------------------
    # Step 4: 결과 저장
    # --------------------------------------------------------
    total_time = time.time() - start_time

    print("\n" + "=" * 60)
    print("🎉 Optimization Complete!")
    print("=" * 60)
    
    if not results or best_config is None:
        print("\n❌ 모든 조합에서 클러스터링에 실패했습니다.")
        print(" - INPUT_PARQUET 스키마 또는 feature 이름을 확인하세요.")
        print(" - UMAP / KMeans 파라미터도 한번 점검해보세요.")
        print(f"\n⏱ Total time: {total_time/60:.1f} minutes")
        return

    # 상위 5개
    results_sorted = sorted(
        results, key=lambda x: x["silhouette"], reverse=True
    )

    print("\n🏆 Top 5 Configurations:")
    for i, r in enumerate(results_sorted[:5], 1):
        print(f"\n{i}. Silhouette: {r['silhouette']:.4f}")
        print(
            f"   K={r['K']}, style={r['style']}, umap_dim={r['umap_dim']}"
        )
        print(f"   Davies-Bouldin: {r['davies_bouldin']:.4f}")

    print(f"\n🔥 Best Configuration:")
    print(f"   K: {best_config['K']}")
    print(f"   Style: {best_config['style']}")
    print(f"   UMAP Dim: {best_config['umap_dim']}")
    print(
        f"   Silhouette: {best_config['scores']['silhouette']:.4f}"
    )
    print(
        f"   Davies-Bouldin: {best_config['scores']['davies_bouldin']:.4f}"
    )
    print(
        f"   Calinski-Harabasz: {best_config['scores']['calinski_harabasz']:.2f}"
    )

    print(f"\n⏱ Total time: {total_time/60:.1f} minutes")

    # JSON 저장
    output = {
        "timestamp": datetime.now().isoformat(),
        "total_time_seconds": total_time,
        "total_combinations": total_combinations,
        "best_config": best_config,
        "top_5": results_sorted[:5],
        "all_results": results,
    }

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Results saved to: {RESULTS_FILE}")

    # --------------------------------------------------------
    # Step 5: 최적 설정으로 최종 학습 제안
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("📝 Next Steps:")
    print("=" * 60)
    print("\n최적 설정으로 retrain_clustering.py를 수정하세요:")
    print(f"\nK = {best_config['K']}")
    print("\nWEIGHTS = {")
    for key, val in best_config["weights"].items():
        print(f'    "{key}": {val},')
    print("}")
    print(f"\nn_components = {best_config['umap_dim']}")
    print("\n그 다음:")
    print("  python retrain_clustering.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
