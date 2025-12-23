import os
import json
import numpy as np
import polars as pl
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
while PROJECT_ROOT != PROJECT_ROOT.parent and not ((PROJECT_ROOT / "data").exists() and (PROJECT_ROOT / "src").exists()):
    PROJECT_ROOT = PROJECT_ROOT.parent

# ============================================================
# 📌 1) 경로 설정
# ============================================================
PARQUET_PATH = PROJECT_ROOT / "features" / "clustered_umap_v2_result.parquet"
INTERPRET_PATH = PROJECT_ROOT / "features" / "cluster_interpretation.json"
CENTROID_PATH = PROJECT_ROOT / "features" / "cluster_centroids.json"


# ============================================================
# 📌 2) feature 컬럼 추출 (clip + dino + midas + openclip)
# ============================================================
def get_feature_columns(df):
    return [
        c for c in df.columns if c.startswith("clip_")
        or c.startswith("dino_")
        or c.startswith("openclip_")
        or c.startswith("midas_")
    ]


# ============================================================
# 📌 3) 클러스터 중심(centroid) 계산 후 JSON으로 저장
# ============================================================
def compute_and_save_centroids():
    df = pl.read_parquet(PARQUET_PATH)
    feature_cols = get_feature_columns(df)

    centroids = {}
    unique_clusters = sorted(df["cluster"].unique().to_list())

    print(f"✔ Found {len(unique_clusters)} clusters")
    print(f"✔ Feature dim: {len(feature_cols)}")

    for cid in unique_clusters:
        subset = df.filter(pl.col("cluster") == cid).select(feature_cols)
        center = subset.to_numpy().mean(axis=0).tolist()
        centroids[int(cid)] = center

    with open(CENTROID_PATH, "w") as f:
        json.dump(centroids, f, indent=4)

    print("✔ Cluster centroids saved →", CENTROID_PATH)


# ============================================================
# 📌 4) CLIP + DINO + MiDaS + OpenCLIP feature 추출
#     (👉 너의 프로젝트에 맞게 아래 함수만 연결하면 됨)
# ============================================================

# ---- 여기에 너의 기존 feature extractor 함수 연결하면 됨 ----
# 예시: 반드시 네 실 환경에 맞게 수정해줘.

def extract_features(image_path):
    """
    실제 프로젝트용 feature extractor.
    네가 기존에 사용하던 CLIP + DINO + MiDaS + OpenCLIP 기반으로
    아래 부분만 자신의 코드로 교체해서 붙여넣으면 된다.
    """
    clip_vec = get_clip_feature(image_path)        # (512,)
    dino_vec = get_dino_feature(image_path)        # (384 or 768,)
    depth_mean, depth_std = get_midas_depth(image_path)  # (2,)
    openclip_vec = get_openclip_feature(image_path)      # (768,)

    # 하나의 벡터로 합치기
    return np.concatenate([
        clip_vec,
        dino_vec,
        np.array([depth_mean, depth_std]),
        openclip_vec
    ])

# -------------------------------------------------------------
# ⚠️ extract_features() 안에 들어가는 4가지 함수는
# 네가 이미 만든 feature 코드 그대로 가져오면 됨.
# -------------------------------------------------------------


# ============================================================
# 📌 5) 새 이미지 → 가장 가까운 클러스터 찾기
# ============================================================
def load_centroids():
    with open(CENTROID_PATH, "r") as f:
        data = json.load(f)
    return {int(k): np.array(v) for k, v in data.items()}


def load_interpretation():
    with open(INTERPRET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def predict_cluster(image_path):
    feat = extract_features(image_path)
    centroids = load_centroids()
    interps = load_interpretation()

    distances = {
        cid: np.linalg.norm(feat - center)
        for cid, center in centroids.items()
    }

    best_cluster = min(distances, key=distances.get)
    best_distance = distances[best_cluster]
    best_interpret = interps[str(best_cluster)]

    return best_cluster, best_distance, best_interpret


# ============================================================
# 📌 6) 실행 예시
# ============================================================
if __name__ == "__main__":
    print("========================================")
    print(" PHASE 2 — NEW IMAGE CLUSTER MATCHING")
    print("========================================\n")

    # (1) 먼저 한번만 centroid 생성
    if not os.path.exists(CENTROID_PATH):
        print("✔ Computing centroids...")
        compute_and_save_centroids()

    # (2) 테스트 이미지로 실행
    test_image = PROJECT_ROOT / "test_images" / "sample.jpg"  # ← 너가 넣고 싶은 이미지 경로
    print("✔ Running cluster prediction for:", test_image)

    cid, dist, info = predict_cluster(str(test_image))

    print("\n📸 Prediction Result")
    print("----------------------------------------")
    print(f"Cluster ID: {cid}")
    print(f"Distance: {dist:.4f}")
    print("Style Interpretation:")
    print(json.dumps(info, indent=4, ensure_ascii=False))
    print("----------------------------------------")
