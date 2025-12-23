# ============================================================
# 🎯 TryAngle Clustering Re-training v3 (Final)
# 적용: K=20, Best Weights(original_style), pose 제거, labels 저장
# ============================================================

import os
import numpy as np
import polars as pl
import joblib
import json
from pathlib import Path
from sklearn.preprocessing import RobustScaler
from umap import UMAP
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ============================================================
# 경로 설정
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent
while PROJECT_ROOT != PROJECT_ROOT.parent and not ((PROJECT_ROOT / "data").exists() and (PROJECT_ROOT / "src").exists()):
    PROJECT_ROOT = PROJECT_ROOT.parent

INPUT_PARQUET = PROJECT_ROOT / "feature_models" / "features" / "fusion_features_v2.parquet"
OUTPUT_DIR = PROJECT_ROOT / "feature_models" / "feature_models_v3"

# 🔥 최적 K 설정
K = 20   # <-- Auto Optimizer 결과

# 🔥 최적 가중치(original_style)
WEIGHTS = {
    "clip": 0.30,
    "openclip": 0.30,
    "dino": 0.25,
    "color": 0.12,
    "midas": 0.03,
    "pose": 0.00,   # pose 제거 (성능 향상)
}

# ============================================================
# Main
# ============================================================
def main():
    print("="*60)
    print("🎯 TryAngle Clustering Re-training v3 (K=20)")
    print("="*60)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # --------------------------------------------------------
    # Step 1: Parquet 로드
    # --------------------------------------------------------
    print("\n📂 Loading features from parquet...")
    df = pl.read_parquet(INPUT_PARQUET)
    
    print(f"✅ Loaded {len(df)} samples")
    
    # --------------------------------------------------------
    # Step 2: Scaler 학습
    # --------------------------------------------------------
    print("\n🔧 Training Scalers...")
    
    scalers = {}
    scaled_features = {}
    
    for feat_name in ["clip", "openclip", "dino", "color", "midas"]:
        print(f"   - Scaling {feat_name}...")
        feat_array = np.vstack(df[feat_name].to_list())
        
        scaler = RobustScaler()
        scaled = scaler.fit_transform(feat_array)
        
        scalers[feat_name] = scaler
        scaled_features[feat_name] = scaled
    
    # pose 제거(가중치=0이므로 포함 안 함)
    scaled_yolo_pose = np.zeros((len(df), 15))
    scaled_face = np.zeros((len(df), 7))
    
    # --------------------------------------------------------
    # Step 3: Weighted Fusion
    # --------------------------------------------------------
    print("\n🔗 Weighted Fusion (applying best weights)...")
    print(json.dumps(WEIGHTS, indent=2))
    
    pose_combined = np.concatenate([scaled_yolo_pose, scaled_face], axis=1)
    
    fusion_1600d = np.concatenate([
        scaled_features["clip"] * WEIGHTS["clip"],
        scaled_features["openclip"] * WEIGHTS["openclip"],
        scaled_features["dino"] * WEIGHTS["dino"],
        scaled_features["color"] * WEIGHTS["color"],
        scaled_features["midas"] * WEIGHTS["midas"],
        pose_combined * WEIGHTS["pose"],     # 어차피 0
    ], axis=1)
    
    print(f"✅ Fusion shape: {fusion_1600d.shape}")
    
    # --------------------------------------------------------
    # Step 4: UMAP 128D
    # --------------------------------------------------------
    print("\n📉 UMAP 1600D → 128D...")
    
    umap_model = UMAP(
        n_components=128,
        n_neighbors=15,
        min_dist=0.1,
        metric='cosine',
        random_state=42,
        verbose=True
    )
    
    fusion_128d = umap_model.fit_transform(fusion_1600d)
    print(f"✅ UMAP shape: {fusion_128d.shape}")
    
    # --------------------------------------------------------
    # Step 5: KMeans (K=20)
    # --------------------------------------------------------
    print(f"\n🎯 KMeans clustering (K={K})...")
    
    kmeans = KMeans(
        n_clusters=K,
        random_state=42,
        n_init=10,
        max_iter=300,
        verbose=1
    )
    
    clusters = kmeans.fit_predict(fusion_128d)
    centroids = kmeans.cluster_centers_
    
    print("✅ KMeans complete")
    
    # --------------------------------------------------------
    # Step 6: Silhouette
    # --------------------------------------------------------
    print("\n📊 Silhouette Score 계산 중...")
    silhouette = silhouette_score(fusion_128d, clusters)
    print(f"   🔥 Silhouette: {silhouette:.4f}")
    
    # --------------------------------------------------------
    # Step 7: cluster_labels.npy 저장
    # --------------------------------------------------------
    label_path = os.path.join(OUTPUT_DIR, "cluster_labels.npy")
    np.save(label_path, clusters)
    print(f"   💾 Saved cluster labels → {label_path}")
    
    # --------------------------------------------------------
    # Step 8: 기타 모델 저장
    # --------------------------------------------------------
    print("\n💾 Saving models...")
    
    for feat_name, scaler in scalers.items():
        joblib.dump(scaler, os.path.join(OUTPUT_DIR, f"scaler_{feat_name}.joblib"))
    
    joblib.dump(umap_model, os.path.join(OUTPUT_DIR, "umap_128d_model.joblib"))
    joblib.dump(kmeans, os.path.join(OUTPUT_DIR, "kmeans_model.pkl"))
    
    np.save(os.path.join(OUTPUT_DIR, "kmeans_centroids.npy"), centroids)
    np.save(os.path.join(OUTPUT_DIR, "fusion_128d.npy"), fusion_128d)
    
    # cluster info json
    cluster_info = {
        int(cid): {
            "count": int(np.sum(clusters == cid))
        }
        for cid in range(K)
    }
    with open(os.path.join(OUTPUT_DIR, "cluster_info.json"), "w", encoding="utf-8") as f:
        json.dump(cluster_info, f, indent=2, ensure_ascii=False)
    
    # weight 저장
    with open(os.path.join(OUTPUT_DIR, "weights.json"), "w") as f:
        json.dump(WEIGHTS, f, indent=2)
    
    print("\n🎉 완료!")

if __name__ == "__main__":
    main()
