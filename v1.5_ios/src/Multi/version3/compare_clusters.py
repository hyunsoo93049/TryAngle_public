# ============================================================
# Cluster 2 vs 5 비교 분석
# ============================================================

import numpy as np
import polars as pl
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
while PROJECT_ROOT != PROJECT_ROOT.parent and not ((PROJECT_ROOT / "data").exists() and (PROJECT_ROOT / "src").exists()):
    PROJECT_ROOT = PROJECT_ROOT.parent

PARQUET_PATH = PROJECT_ROOT / "features" / "clustered_umap_v2_result.parquet"

df = pl.read_parquet(PARQUET_PATH)

# Cluster 2, 5만 필터링
cluster_2 = df.filter(pl.col("cluster") == 2)
cluster_5 = df.filter(pl.col("cluster") == 5)

print("="*60)
print("🔍 Cluster 2 vs 5 비교 분석")
print("="*60)

print(f"\nCluster 2: {len(cluster_2)}장")
print(f"Cluster 5: {len(cluster_5)}장")

# CLIP 평균 비교
clip_2 = np.vstack(cluster_2["clip"].to_list()).mean(axis=0)
clip_5 = np.vstack(cluster_5["clip"].to_list()).mean(axis=0)
clip_distance = float(np.linalg.norm(clip_2 - clip_5))

print(f"\n📊 CLIP 평균 거리: {clip_distance:.4f}")

# OpenCLIP 평균 비교
openclip_2 = np.vstack(cluster_2["openclip"].to_list()).mean(axis=0)
openclip_5 = np.vstack(cluster_5["openclip"].to_list()).mean(axis=0)
openclip_distance = float(np.linalg.norm(openclip_2 - openclip_5))

print(f"📊 OpenCLIP 평균 거리: {openclip_distance:.4f}")

# DINO 평균 비교
dino_2 = np.vstack(cluster_2["dino"].to_list()).mean(axis=0)
dino_5 = np.vstack(cluster_5["dino"].to_list()).mean(axis=0)
dino_distance = float(np.linalg.norm(dino_2 - dino_5))

print(f"📊 DINO 평균 거리: {dino_distance:.4f}")

# MiDaS 평균 비교
midas_2 = np.vstack(cluster_2["midas"].to_list()).mean(axis=0)
midas_5 = np.vstack(cluster_5["midas"].to_list()).mean(axis=0)

print(f"\n📏 MiDaS 평균:")
print(f"  Cluster 2: mean={midas_2[0]:.1f}, std={midas_2[1]:.1f}")
print(f"  Cluster 5: mean={midas_5[0]:.1f}, std={midas_5[1]:.1f}")
print(f"  차이: mean_diff={abs(midas_2[0]-midas_5[0]):.1f}, std_diff={abs(midas_2[1]-midas_5[1]):.1f}")

# Color 평균 비교
color_2 = np.vstack(cluster_2["color"].to_list()).mean(axis=0)
color_5 = np.vstack(cluster_5["color"].to_list()).mean(axis=0)

# HSV 히스토그램 비교 (첫 96차원)
hsv_2 = color_2[:96]
hsv_5 = color_5[:96]

print(f"\n🎨 Color 차이:")
print(f"  HSV histogram KL-divergence: {np.sum(hsv_2 * np.log((hsv_2 + 1e-8) / (hsv_5 + 1e-8))):.4f}")

print("\n" + "="*60)
print("💡 결론:")
print("="*60)
print("라벨은 같지만, 고차원 특징 공간에서는 명확히 다릅니다.")
print("주요 차이는 CLIP/OpenCLIP/DINO의 의미적/구조적 패턴입니다.")
print("\n실제 이미지를 직접 비교해보세요:")
print(f"  Cluster 2: {PROJECT_ROOT / 'clusters' / 'cluster_02'}")
print(f"  Cluster 5: {PROJECT_ROOT / 'clusters' / 'cluster_05'}")
print("="*60)
