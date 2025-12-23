# ============================================================
# cluster_summary.json → cluster_interpretation.json 변환 (Simple)
# ============================================================

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
while PROJECT_ROOT != PROJECT_ROOT.parent and not ((PROJECT_ROOT / "data").exists() and (PROJECT_ROOT / "src").exists()):
    PROJECT_ROOT = PROJECT_ROOT.parent

# 경로
CLUSTER_SUMMARY_PATH = PROJECT_ROOT / "clusters" / "cluster_summary.json"
OUTPUT_PATH = PROJECT_ROOT / "features" / "cluster_interpretation.json"

print("Loading cluster_summary.json...")

with open(CLUSTER_SUMMARY_PATH, "r", encoding="utf-8") as f:
    summary = json.load(f)

print(f"Found {len(summary['clusters'])} clusters")

# ============================================================
# cluster_summary의 label을 파싱해서 상세 정보 생성
# ============================================================

cluster_interpretation = {}

for cluster_id_str, cluster_data in summary["clusters"].items():
    cluster_id = int(cluster_id_str)
    label = cluster_data["label"]
    count = cluster_data["count"]

    # 라벨 파싱 (예: "어두운 무채색 / 중간 밝기 / 중앙 집중 / 단순 구도 / 인물 중심 / 하이앵글")
    parts = [p.strip() for p in label.split("/")]

    # 기본값
    depth_mean = 1000.0  # 임시 기본값
    depth_label = "실외 / 멀리"
    dominant_pose = "반신"
    person_ratio = 0.6
    tone = "쿨톤"
    brightness = "중간"
    saturation = "채도낮음"

    # 라벨에서 정보 추출
    for part in parts:
        part_lower = part.lower()

        # 색감
        if "어두운" in part or "무채색" in part:
            tone = "쿨톤"
            brightness = "어두움"
        elif "생기있는" in part or "따뜻" in part:
            tone = "웜톤"
            brightness = "밝음"
        elif "뉴트럴" in part:
            tone = "뉴트럴"

        # 밝기
        if "밝은" in part:
            brightness = "밝음"
        elif "어두운" in part:
            brightness = "어두움"
        elif "중간" in part and "밝기" in part:
            brightness = "중간"

        # 인물
        if "인물" in part:
            dominant_pose = "반신"
            person_ratio = 0.6

        # 앵글
        if "하이앵글" in part:
            depth_mean = 1050.0
            depth_label = "실외 / 멀리"
        elif "로우앵글" in part:
            depth_mean = 950.0
            depth_label = "실외 / 중간"

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
        "sample_count": count,
        "original_label": label
    }

    print(f"Cluster {cluster_id:2d}: {auto_label} ({count} samples)")

# ============================================================
# 저장
# ============================================================
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(cluster_interpretation, f, indent=4, ensure_ascii=False)

print(f"\nCluster interpretation saved to: {OUTPUT_PATH}")
print(f"Total clusters: {len(cluster_interpretation)}")
