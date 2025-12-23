# ============================================================
# cluster_summary.json → cluster_interpretation.json 변환
# ============================================================

import json
import polars as pl
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
while PROJECT_ROOT != PROJECT_ROOT.parent and not ((PROJECT_ROOT / "data").exists() and (PROJECT_ROOT / "src").exists()):
    PROJECT_ROOT = PROJECT_ROOT.parent

# 경로
CLUSTER_SUMMARY_PATH = PROJECT_ROOT / "clusters" / "cluster_summary.json"
PARQUET_PATH = PROJECT_ROOT / "features" / "clustered_umap_v2_result.parquet"
OUTPUT_PATH = PROJECT_ROOT / "features" / "cluster_interpretation.json"

print("📂 Loading data...")

# cluster_summary.json 로드
with open(CLUSTER_SUMMARY_PATH, "r", encoding="utf-8") as f:
    summary = json.load(f)

# parquet 데이터 로드 (특징 정보 포함)
df = pl.read_parquet(PARQUET_PATH)

print(f"✅ Loaded {len(df)} samples")
print(f"✅ Clusters in summary: {len(summary['clusters'])}")

# ============================================================
# 각 클러스터별로 상세 정보 계산
# ============================================================

cluster_interpretation = {}

for cluster_id_str, cluster_data in summary["clusters"].items():
    cluster_id = int(cluster_id_str)

    # 해당 클러스터의 이미지들 필터링
    cluster_df = df.filter(pl.col("cluster") == cluster_id)

    if len(cluster_df) == 0:
        print(f"⚠️ Cluster {cluster_id}: No samples found")
        continue

    # MiDaS depth 평균 계산
    midas_features = np.vstack(cluster_df["midas"].to_list())
    depth_mean = float(midas_features[:, 0].mean())

    # depth 라벨
    if depth_mean > 1050:
        depth_label = "실외 / 멀리"
    elif depth_mean > 950:
        depth_label = "실외 / 중간"
    else:
        depth_label = "실내 / 가까이"

    # Pose 정보 (있으면)
    if "yolo_pose" in cluster_df.columns:
        pose_features = np.vstack(cluster_df["yolo_pose"].to_list())
        person_ratio = float(np.mean(np.sum(pose_features, axis=1) > 0.1))

        if person_ratio > 0.7:
            dominant_pose = "전신"
        elif person_ratio > 0.4:
            dominant_pose = "반신"
        else:
            dominant_pose = "클로즈업"
    else:
        person_ratio = 0.0
        dominant_pose = "반신"

    # Color 정보
    color_features = np.vstack(cluster_df["color"].to_list())

    # HSV 히스토그램에서 색조 분석 (첫 32차원)
    h_hist = color_features[:, :32].mean(axis=0)
    dominant_h = np.argmax(h_hist)

    if dominant_h < 5 or dominant_h > 28:
        tone = "웜톤"
    elif 10 <= dominant_h <= 20:
        tone = "쿨톤"
    else:
        tone = "뉴트럴"

    # 밝기 (LAB의 L 채널 mean - 96~108번째 위치)
    if len(color_features[0]) > 100:
        brightness_values = color_features[:, 96].mean()
        if brightness_values > 140:
            brightness = "밝음"
        elif brightness_values > 100:
            brightness = "중간"
        else:
            brightness = "어두움"
    else:
        brightness = "중간"

    # 채도 (HSV의 S 채널 - 32~64번째)
    saturation_values = color_features[:, 32:64].mean()
    if saturation_values > 0.15:
        saturation = "채도높음"
    else:
        saturation = "채도낮음"

    # auto_label 생성
    auto_label = f"{depth_label}, {tone}, {brightness}, {dominant_pose}"

    cluster_interpretation[str(cluster_id)] = {
        "cluster_id": cluster_id,
        "depth_mean": depth_mean,
        "depth_label": depth_label,
        "dominant_pose": dominant_pose,
        "person_ratio_mean": person_ratio,
        "tone": tone,
        "brightness": brightness,
        "saturation": saturation,
        "auto_label": auto_label,
        "sample_count": cluster_data["count"]
    }

    print(f"✅ Cluster {cluster_id:2d}: {auto_label} ({cluster_data['count']} samples)")

# ============================================================
# 저장
# ============================================================
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(cluster_interpretation, f, indent=4, ensure_ascii=False)

print(f"\n🎉 Cluster interpretation saved to: {OUTPUT_PATH}")
print(f"📊 Total clusters: {len(cluster_interpretation)}")
