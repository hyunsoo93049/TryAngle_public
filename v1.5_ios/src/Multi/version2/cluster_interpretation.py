# ============================================================
# 🎨 TryAngle - Cluster Interpreter (완전 자동 해석)
# ============================================================

import os
import cv2
import json
import numpy as np
import polars as pl
from tqdm import tqdm
from ultralytics import YOLO
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
while PROJECT_ROOT != PROJECT_ROOT.parent and not ((PROJECT_ROOT / "data").exists() and (PROJECT_ROOT / "src").exists()):
    PROJECT_ROOT = PROJECT_ROOT.parent

# -----------------------------
# [1] 경로 설정
# -----------------------------
PARQUET_PATH = PROJECT_ROOT / "features" / "clustered_umap_v2_result.parquet"
IMG_DIR = PROJECT_ROOT / "data" / "train_images"
OUTPUT_JSON = PROJECT_ROOT / "features" / "cluster_interpretation.json"
YOLO_WEIGHTS = "yolov8s-pose.pt"

# -----------------------------
# [2] 데이터 로드
# -----------------------------
print("📂 Parquet 로드 중…")
df = pl.read_parquet(PARQUET_PATH)

filenames = df["filename"].to_list()
clusters = df["cluster"].to_list()
depth_mean = df["midas_depth_mean"].to_list() if "midas_depth_mean" in df.columns else None
depth_raw = df.select([c for c in df.columns if "midas" in c]).to_numpy()

# YOLO 로드
pose_model = YOLO(YOLO_WEIGHTS)

# -----------------------------
# [3] 보조 함수들
# -----------------------------

def analyze_color(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h_mean = hsv[:, :, 0].mean()
    s_mean = hsv[:, :, 1].mean()
    v_mean = hsv[:, :, 2].mean()

    tone = "웜톤" if h_mean < 20 or h_mean > 150 else "쿨톤"
    brightness = "밝음" if v_mean > 130 else "어두움"
    saturation = "채도높음" if s_mean > 80 else "채도낮음"

    return {
        "h_mean": float(h_mean),
        "s_mean": float(s_mean),
        "v_mean": float(v_mean),
        "tone": tone,
        "brightness": brightness,
        "saturation": saturation,
    }


def analyze_pose(image):
    res = pose_model(image, verbose=False)
    if len(res[0].keypoints) == 0:
        return {"pose": "없음", "ratio": 0}

    pts = res[0].keypoints.xy[0].cpu().numpy()
    x_min, y_min = pts[:, 0].min(), pts[:, 1].min()
    x_max, y_max = pts[:, 0].max(), pts[:, 1].max()

    h, w = image.shape[:2]
    ratio = (y_max - y_min) / h

    if ratio > 0.7:
        pose_type = "전신"
    elif ratio > 0.4:
        pose_type = "반신"
    else:
        pose_type = "클로즈업"

    return {"pose": pose_type, "ratio": float(ratio)}


def depth_to_scene(depth_val):
    if depth_val > 1000:
        return "실외 / 멀리"
    elif depth_val > 700:
        return "실외 / 중간"
    elif depth_val > 400:
        return "실내 / 가까움"
    else:
        return "실내 / 매우가까움"


# -----------------------------
# [4] 클러스터별 해석
# -----------------------------
cluster_info = {}

unique_clusters = sorted(set(clusters))
print(f"🔍 총 {len(unique_clusters)}개 클러스터 분석 시작…")

for cid in unique_clusters:
    print(f"\n=== Cluster {cid} 분석 중 ===")
    cluster_df = df.filter(pl.col("cluster") == cid)

    file_list = cluster_df["filename"].to_list()
    depth_vals = cluster_df.select([c for c in df.columns if "midas" in c]).to_numpy()

    color_list = []
    pose_list = []
    ratios = []
    
    # 50장 샘플만 분석
    sample_files = file_list[:50]

    for fname in tqdm(sample_files):
        path = IMG_DIR / os.path.basename(fname)
        img = cv2.imread(str(path))
        if img is None:
            continue

        # 색감 분석
        color = analyze_color(img)
        color_list.append(color)

        # 포즈 분석
        pose = analyze_pose(img)
        pose_list.append(pose["pose"])
        ratios.append(pose["ratio"])

    # 평균 계산
    mean_ratio = float(np.mean(ratios)) if ratios else 0
    dominant_pose = max(set(pose_list), key=pose_list.count) if pose_list else "없음"

    # depth
    depth_mean = float(depth_vals[:, 0].mean())
    depth_label = depth_to_scene(depth_mean)

    # 대표 색감
    tone = max(set([c["tone"] for c in color_list]), key=[c["tone"] for c in color_list].count)
    brightness = max(set([c["brightness"] for c in color_list]), key=[c["brightness"] for c in color_list].count)
    saturation = max(set([c["saturation"] for c in color_list]), key=[c["saturation"] for c in color_list].count)

    # 최종 클러스터 자동 라벨
    auto_label = f"{depth_label}, {tone}, {brightness}, {dominant_pose}"

    cluster_info[cid] = {
        "cluster_id": cid,
        "depth_mean": depth_mean,
        "depth_label": depth_label,
        "dominant_pose": dominant_pose,
        "person_ratio_mean": mean_ratio,
        "tone": tone,
        "brightness": brightness,
        "saturation": saturation,
        "auto_label": auto_label,
        "sample_count": len(file_list),
    }

# -----------------------------
# [5] JSON 저장
# -----------------------------
print("\n💾 JSON 저장:", OUTPUT_JSON)
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(cluster_info, f, indent=4, ensure_ascii=False)

print("🎉 클러스터 해석 완료!")
