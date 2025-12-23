# ======================================================
# 🧠 TryAngle Cluster Interpreter
# 각 클러스터의 의미를 자동으로 분석하고 라벨링
# ======================================================

import os
import numpy as np
import polars as pl
import cv2
from collections import Counter
from ultralytics import YOLO
from tqdm import tqdm
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
while PROJECT_ROOT != PROJECT_ROOT.parent and not ((PROJECT_ROOT / "data").exists() and (PROJECT_ROOT / "src").exists()):
    PROJECT_ROOT = PROJECT_ROOT.parent

# ------------------------------------------------------
# [1] 경로 설정
# ------------------------------------------------------
PARQUET_PATH = PROJECT_ROOT / "features" / "clustered_umap_v2_result.parquet"
IMG_DIR = PROJECT_ROOT / "data" / "train_images"
YOLO_WEIGHTS = "yolov8s-pose.pt"

# ------------------------------------------------------
# [2] 데이터 로드
# ------------------------------------------------------
print("📂 데이터 로드 중...")
df = pl.read_parquet(PARQUET_PATH)
print(f"✅ 로드 완료: {len(df)}장")

# YOLO 모델 로드
print("🔧 YOLO Pose 모델 로드 중...")
yolo = YOLO(YOLO_WEIGHTS)

# ------------------------------------------------------
# [3] 색감 분석 함수
# ------------------------------------------------------
def analyze_color_tone(image_paths, max_samples=50):
    """이미지들의 평균 색감 분석"""
    hues, sats, vals = [], [], []
    temps = []  # 색온도
    
    for path in image_paths[:max_samples]:
        img = cv2.imread(path)
        if img is None:
            continue
        
        # HSV 분석
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hues.append(hsv[:,:,0].mean())
        sats.append(hsv[:,:,1].mean())
        vals.append(hsv[:,:,2].mean())
        
        # 색온도 (R-B 비율)
        b, g, r = cv2.split(img)
        temp = (r.mean() - b.mean()) / 255.0
        temps.append(temp)
    
    if not hues:
        return None
    
    avg_hue = np.mean(hues)
    avg_sat = np.mean(sats)
    avg_val = np.mean(vals)
    avg_temp = np.mean(temps)
    
    # 색온도 판단
    if avg_temp > 0.15:
        tone = "웜톤"
    elif avg_temp < -0.15:
        tone = "쿨톤"
    else:
        tone = "중간톤"
    
    # 밝기 판단
    if avg_val > 160:
        brightness = "밝음"
    elif avg_val > 100:
        brightness = "보통"
    else:
        brightness = "어두움"
    
    # 채도 판단
    if avg_sat > 120:
        saturation = "높음"
    elif avg_sat > 60:
        saturation = "보통"
    else:
        saturation = "낮음"
    
    return {
        "tone": tone,
        "brightness": brightness,
        "saturation": saturation,
        "avg_hue": avg_hue,
        "avg_sat": avg_sat,
        "avg_val": avg_val,
        "avg_temp": avg_temp
    }

# ------------------------------------------------------
# [4] 포즈 분석 함수
# ------------------------------------------------------
def analyze_pose_patterns(image_paths, yolo_model, max_samples=30):
    """포즈 패턴 분석 (전신/반신/클로즈업)"""
    pose_types = []
    pose_angles = []  # 상하 각도
    
    for path in image_paths[:max_samples]:
        img = cv2.imread(path)
        if img is None:
            continue
        
        try:
            results = yolo_model(img, verbose=False)
            if not results or results[0].boxes is None or len(results[0].boxes) == 0:
                continue
            
            bbox = results[0].boxes.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = bbox
            
            # 인물 세로 비율
            h_ratio = (y2 - y1) / img.shape[0]
            
            if h_ratio > 0.65:
                pose_types.append("전신")
            elif h_ratio > 0.35:
                pose_types.append("반신")
            else:
                pose_types.append("클로즈업")
            
            # 인물 위치 (상단/중단/하단)
            center_y = (y1 + y2) / 2 / img.shape[0]
            pose_angles.append(center_y)
            
        except Exception as e:
            continue
    
    if not pose_types:
        return None
    
    # 최빈값
    pose_type = Counter(pose_types).most_common(1)[0][0]
    
    # 평균 위치
    avg_position = np.mean(pose_angles)
    if avg_position < 0.4:
        position = "상단"
    elif avg_position < 0.6:
        position = "중앙"
    else:
        position = "하단"
    
    return {
        "pose_type": pose_type,
        "position": position,
        "avg_position": avg_position,
        "distribution": Counter(pose_types)
    }

# ------------------------------------------------------
# [5] 구도 분석 함수
# ------------------------------------------------------
def analyze_composition(image_paths, max_samples=30):
    """구도 특성 분석"""
    aspect_ratios = []
    
    for path in image_paths[:max_samples]:
        img = cv2.imread(path)
        if img is None:
            continue
        
        h, w = img.shape[:2]
        aspect_ratios.append(w / h)
    
    if not aspect_ratios:
        return None
    
    avg_aspect = np.mean(aspect_ratios)
    
    # 종횡비 판단
    if avg_aspect > 1.4:
        aspect = "가로형"
    elif avg_aspect < 0.8:
        aspect = "세로형"
    else:
        aspect = "정방형"
    
    return {
        "aspect": aspect,
        "avg_aspect_ratio": avg_aspect
    }

# ------------------------------------------------------
# [6] 클러스터 종합 분석
# ------------------------------------------------------
def analyze_cluster_comprehensive(cluster_id):
    """클러스터 종합 분석"""
    print(f"\n{'='*70}")
    print(f"🔍 Cluster {cluster_id} 분석 중...")
    print(f"{'='*70}")
    
    # 클러스터 데이터 필터링
    cluster_df = df.filter(pl.col("cluster") == cluster_id)
    cluster_size = len(cluster_df)
    
    print(f"📊 총 {cluster_size}장")
    
    # 파일 경로 리스트
    image_paths = [
        str(IMG_DIR / os.path.basename(fname))
        for fname in cluster_df["filename"].to_list()
    ]
    
    # 1. MiDaS Depth 분석
    avg_depth = cluster_df["midas_mean"].mean()
    avg_depth_std = cluster_df["midas_std"].mean()
    
    if avg_depth > 1050:
        depth_class = "실외/원거리"
    elif avg_depth > 950:
        depth_class = "혼합"
    else:
        depth_class = "실내/근접"
    
    print(f"\n📐 [Depth 분석]")
    print(f"  평균 Depth: {avg_depth:.1f}")
    print(f"  분산: {avg_depth_std:.1f}")
    print(f"  판단: {depth_class}")
    
    # 2. 색감 분석
    print(f"\n🎨 [색감 분석 중...]")
    color_result = analyze_color_tone(image_paths, max_samples=50)
    
    if color_result:
        print(f"  색온도: {color_result['tone']}")
        print(f"  밝기: {color_result['brightness']}")
        print(f"  채도: {color_result['saturation']}")
        print(f"  세부: H={color_result['avg_hue']:.1f}, S={color_result['avg_sat']:.1f}, V={color_result['avg_val']:.1f}")
    else:
        print("  ⚠️ 색감 분석 실패")
        color_result = {"tone": "알 수 없음", "brightness": "알 수 없음", "saturation": "알 수 없음"}
    
    # 3. 포즈 분석
    print(f"\n🕺 [포즈 분석 중...]")
    pose_result = analyze_pose_patterns(image_paths, yolo, max_samples=30)
    
    if pose_result:
        print(f"  포즈 유형: {pose_result['pose_type']}")
        print(f"  인물 위치: {pose_result['position']}")
        print(f"  분포: {dict(pose_result['distribution'])}")
    else:
        print("  ⚠️ 포즈 분석 실패")
        pose_result = {"pose_type": "알 수 없음", "position": "알 수 없음"}
    
    # 4. 구도 분석
    print(f"\n📸 [구도 분석 중...]")
    comp_result = analyze_composition(image_paths, max_samples=30)
    
    if comp_result:
        print(f"  화면 비율: {comp_result['aspect']} (평균 {comp_result['avg_aspect_ratio']:.2f})")
    else:
        print("  ⚠️ 구도 분석 실패")
        comp_result = {"aspect": "알 수 없음"}
    
    # 5. 자동 라벨 생성
    label_parts = []
    
    if depth_class != "혼합":
        label_parts.append(depth_class.split("/")[0])
    
    if color_result["tone"] != "중간톤":
        label_parts.append(color_result["tone"])
    
    if color_result["brightness"] in ["밝음", "어두움"]:
        label_parts.append(color_result["brightness"])
    
    if pose_result["pose_type"] != "알 수 없음":
        label_parts.append(pose_result["pose_type"])
    
    auto_label = "_".join(label_parts) if label_parts else "일반_스타일"
    
    print(f"\n💡 [자동 라벨]")
    print(f"  → {auto_label}")
    
    # 6. 대표 이미지 추천
    print(f"\n🖼️ [대표 이미지 3장]")
    sample_files = cluster_df["filename"][:3].to_list()
    for i, fname in enumerate(sample_files, 1):
        print(f"  {i}. {fname}")
    
    return {
        "cluster_id": cluster_id,
        "size": cluster_size,
        "depth_class": depth_class,
        "avg_depth": avg_depth,
        "color": color_result,
        "pose": pose_result,
        "composition": comp_result,
        "auto_label": auto_label,
        "representative_images": sample_files
    }

# ------------------------------------------------------
# [7] 전체 클러스터 분석 실행
# ------------------------------------------------------
def main():
    print("\n" + "="*70)
    print("🧠 TryAngle Cluster Interpreter")
    print("="*70)
    
    n_clusters = df["cluster"].n_unique()
    print(f"\n총 {n_clusters}개 클러스터 분석 시작\n")
    
    all_results = []
    
    for c in range(n_clusters):
        result = analyze_cluster_comprehensive(c)
        all_results.append(result)
    
    # 최종 요약
    print("\n" + "="*70)
    print("📋 전체 클러스터 요약")
    print("="*70)
    
    for result in all_results:
        print(f"\nCluster {result['cluster_id']}: {result['auto_label']} ({result['size']}장)")
        print(f"  - {result['depth_class']}, {result['color']['tone']}, {result['pose']['pose_type']}")
    
    print("\n✅ 클러스터 해석 완료!")
    
    # 결과 저장 (선택사항)
    import json
    output_path = PROJECT_ROOT / "features" / "cluster_interpretation.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n💾 결과 저장: {output_path}")

# ------------------------------------------------------
# [8] 실행
# ------------------------------------------------------
if __name__ == "__main__":
    main()


## 💾 저장 방법

#파일명: cluster_interpreter.py
#위치: <프로젝트 루트>/src/Multi/version2/
