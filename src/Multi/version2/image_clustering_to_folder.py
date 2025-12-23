# ======================================================
# 🧠 TryAngle Deep Cluster Analysis
# CLIP/OpenCLIP/DINO/MiDaS 관점별 심층 분석
# ======================================================

import os
import numpy as np
import polars as pl
import cv2
import json
from collections import Counter
from ultralytics import YOLO
from tqdm import tqdm
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
while PROJECT_ROOT != PROJECT_ROOT.parent and not ((PROJECT_ROOT / "data").exists() and (PROJECT_ROOT / "src").exists()):
    PROJECT_ROOT = PROJECT_ROOT.parent

# ------------------------------------------------------
# [1] 경로 설정
# ------------------------------------------------------
PARQUET_PATH = PROJECT_ROOT / "features" / "clustered_umap_v2_result.parquet"
IMG_DIR = PROJECT_ROOT / "data" / "train_images"
CLUSTER_DIR = PROJECT_ROOT / "data" / "clustered_images"
YOLO_WEIGHTS = "yolov8s-pose.pt"

# ------------------------------------------------------
# [2] 데이터 로드
# ------------------------------------------------------
print("📂 데이터 로드 중...")
df = pl.read_parquet(PARQUET_PATH)
print(f"✅ 총 {len(df)}장 로드")

print("🔧 YOLO 모델 로드 중...")
yolo = YOLO(YOLO_WEIGHTS)

# ------------------------------------------------------
# [3] 모델별 특징 분석 함수
# ------------------------------------------------------

def analyze_clip_features(cluster_df):
    """CLIP 특징 분석 (감성/전체적 분위기)"""
    clip_cols = [c for c in cluster_df.columns if c.startswith("clip_")]
    if not clip_cols:
        return None
    
    clip_feats = cluster_df.select(clip_cols).to_numpy()
    
    # 클러스터 중심 (평균)
    clip_center = clip_feats.mean(axis=0)
    
    # 클러스터 내 분산
    clip_std = clip_feats.std(axis=0).mean()
    
    # 특징 활성화 패턴 (상위 10개 차원)
    top_dims = np.argsort(np.abs(clip_center))[-10:][::-1]
    top_values = clip_center[top_dims]
    
    return {
        "model": "CLIP (ViT-B/32)",
        "purpose": "전체적 감성/분위기 인식",
        "dimensions": len(clip_cols),
        "cluster_cohesion": float(1.0 / (clip_std + 1e-6)),  # 높을수록 응집도 높음
        "top_activated_dims": top_dims.tolist()[:5],
        "top_values": top_values.tolist()[:5],
        "interpretation": interpret_clip_pattern(clip_center)
    }

def interpret_clip_pattern(clip_center):
    """CLIP 패턴 해석 (휴리스틱)"""
    # 절대값 상위 차원 분석
    strong_features = np.abs(clip_center) > 0.1
    activation_rate = strong_features.sum() / len(clip_center)
    
    if activation_rate > 0.3:
        return "복잡하고 다양한 시각적 요소 포함"
    elif activation_rate > 0.15:
        return "중간 복잡도의 구성"
    else:
        return "단순하고 명확한 구성"

def analyze_openclip_features(cluster_df):
    """OpenCLIP 특징 분석 (의미적 이해)"""
    openclip_cols = [c for c in cluster_df.columns if c.startswith("openclip_")]
    if not openclip_cols:
        return None
    
    openclip_feats = cluster_df.select(openclip_cols).to_numpy()
    
    openclip_center = openclip_feats.mean(axis=0)
    openclip_std = openclip_feats.std(axis=0).mean()
    
    top_dims = np.argsort(np.abs(openclip_center))[-10:][::-1]
    top_values = openclip_center[top_dims]
    
    return {
        "model": "OpenCLIP (ViT-B/32, LAION-2B)",
        "purpose": "의미적 개념 이해",
        "dimensions": len(openclip_cols),
        "cluster_cohesion": float(1.0 / (openclip_std + 1e-6)),
        "top_activated_dims": top_dims.tolist()[:5],
        "top_values": top_values.tolist()[:5],
        "interpretation": interpret_openclip_pattern(openclip_center)
    }

def interpret_openclip_pattern(openclip_center):
    """OpenCLIP 패턴 해석"""
    activation_strength = np.abs(openclip_center).mean()
    
    if activation_strength > 0.08:
        return "강한 개념적 특징 (특정 장소/사물/스타일)"
    elif activation_strength > 0.04:
        return "중간 강도의 의미적 특징"
    else:
        return "일반적이고 보편적인 특징"

def analyze_dino_features(cluster_df):
    """DINO 특징 분석 (구조/구도)"""
    dino_cols = [c for c in cluster_df.columns if c.startswith("dino_")]
    if not dino_cols:
        return None
    
    dino_feats = cluster_df.select(dino_cols).to_numpy()
    
    dino_center = dino_feats.mean(axis=0)
    dino_std = dino_feats.std(axis=0).mean()
    
    top_dims = np.argsort(np.abs(dino_center))[-10:][::-1]
    top_values = dino_center[top_dims]
    
    # DINO는 구조 정보 중심
    spatial_consistency = 1.0 / (dino_std + 1e-6)
    
    return {
        "model": "DINOv2 (ViT-S/14)",
        "purpose": "구조적 패턴/구도 인식",
        "dimensions": len(dino_cols),
        "spatial_consistency": float(spatial_consistency),
        "cluster_cohesion": float(spatial_consistency),
        "top_activated_dims": top_dims.tolist()[:5],
        "top_values": top_values.tolist()[:5],
        "interpretation": interpret_dino_pattern(dino_center, spatial_consistency)
    }

def interpret_dino_pattern(dino_center, consistency):
    """DINO 패턴 해석"""
    if consistency > 20:
        return "매우 일관된 구도/레이아웃 (유사한 프레이밍)"
    elif consistency > 10:
        return "중간 정도의 구도 일관성"
    else:
        return "다양한 구도 패턴 혼재"

def analyze_midas_features(cluster_df):
    """MiDaS 특징 분석 (깊이/공간)"""
    midas_mean = cluster_df["midas_mean"].to_numpy()
    midas_std = cluster_df["midas_std"].to_numpy()
    
    avg_depth = midas_mean.mean()
    depth_variance = midas_mean.std()
    avg_complexity = midas_std.mean()
    complexity_variance = midas_std.std()
    
    # 깊이 분류
    if avg_depth > 1050:
        depth_class = "실외/원거리"
        space_type = "개방된 공간"
    elif avg_depth > 950:
        depth_class = "경계 공간"
        space_type = "실내외 경계 (카페 창가, 복도 등)"
    else:
        depth_class = "실내/근접"
        space_type = "폐쇄된 공간"
    
    # 복잡도 분류
    if avg_complexity > 850:
        complexity = "높음 (복잡한 배경)"
    elif avg_complexity > 700:
        complexity = "중간"
    else:
        complexity = "낮음 (단순한 배경)"
    
    # 일관성
    consistency_score = 1.0 / (depth_variance + 1e-6)
    
    return {
        "model": "MiDaS (DPT-Hybrid)",
        "purpose": "깊이/공간 구조 인식",
        "avg_depth": float(avg_depth),
        "depth_variance": float(depth_variance),
        "avg_complexity": float(avg_complexity),
        "depth_class": depth_class,
        "space_type": space_type,
        "background_complexity": complexity,
        "depth_consistency": float(consistency_score),
        "interpretation": f"{space_type}, {complexity} 배경 복잡도"
    }

# ------------------------------------------------------
# [4] 색감 분석 (기존)
# ------------------------------------------------------
def analyze_color_detailed(image_paths, max_samples=50):
    """상세 색감 분석"""
    hues, sats, vals, temps = [], [], [], []
    contrasts, edges = [], []
    
    for path in image_paths[:max_samples]:
        img = cv2.imread(path)
        if img is None:
            continue
        
        # HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hues.append(hsv[:,:,0].mean())
        sats.append(hsv[:,:,1].mean())
        vals.append(hsv[:,:,2].mean())
        
        # 색온도
        b, g, r = cv2.split(img)
        temp = (r.mean() - b.mean()) / 255.0
        temps.append(temp)
        
        # 대비
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        contrasts.append(gray.std())
        
        # 엣지 밀도
        edge = cv2.Canny(gray, 50, 150)
        edges.append(edge.sum() / edge.size)
    
    if not hues:
        return None
    
    avg_hue = np.mean(hues)
    avg_sat = np.mean(sats)
    avg_val = np.mean(vals)
    avg_temp = np.mean(temps)
    avg_contrast = np.mean(contrasts)
    avg_edge = np.mean(edges)
    
    # 분류
    tone = "웜톤" if avg_temp > 0.15 else ("쿨톤" if avg_temp < -0.15 else "중간톤")
    brightness = "밝음" if avg_val > 160 else ("보통" if avg_val > 100 else "어두움")
    saturation = "높음" if avg_sat > 120 else ("보통" if avg_sat > 60 else "낮음")
    contrast_level = "높음" if avg_contrast > 50 else ("보통" if avg_contrast > 30 else "낮음")
    edge_density = "복잡" if avg_edge > 0.15 else ("보통" if avg_edge > 0.08 else "단순")
    
    return {
        "tone": tone,
        "brightness": brightness,
        "saturation": saturation,
        "contrast": contrast_level,
        "edge_density": edge_density,
        "avg_hue": float(avg_hue),
        "avg_sat": float(avg_sat),
        "avg_val": float(avg_val),
        "avg_temp": float(avg_temp),
        "avg_contrast": float(avg_contrast),
        "avg_edge_density": float(avg_edge)
    }

# ------------------------------------------------------
# [5] 포즈 분석 (기존)
# ------------------------------------------------------
def analyze_pose_detailed(image_paths, yolo_model, max_samples=30):
    """상세 포즈 분석"""
    pose_types = []
    pose_positions = []
    angles = []
    
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
            
            h_ratio = (y2 - y1) / img.shape[0]
            w_ratio = (x2 - x1) / img.shape[1]
            
            # 포즈 타입
            if h_ratio > 0.65:
                pose_types.append("전신")
            elif h_ratio > 0.35:
                pose_types.append("반신")
            else:
                pose_types.append("클로즈업")
            
            # 위치
            center_y = (y1 + y2) / 2 / img.shape[0]
            pose_positions.append(center_y)
            
            # 가로 중심
            center_x = (x1 + x2) / 2 / img.shape[1]
            angles.append(center_x)
            
        except Exception:
            continue
    
    if not pose_types:
        return None
    
    pose_type = Counter(pose_types).most_common(1)[0][0]
    avg_y = np.mean(pose_positions)
    avg_x = np.mean(angles)
    
    position_y = "상단" if avg_y < 0.4 else ("중앙" if avg_y < 0.6 else "하단")
    position_x = "좌측" if avg_x < 0.4 else ("중앙" if avg_x < 0.6 else "우측")
    
    return {
        "pose_type": pose_type,
        "position_vertical": position_y,
        "position_horizontal": position_x,
        "avg_vertical_pos": float(avg_y),
        "avg_horizontal_pos": float(avg_x),
        "distribution": dict(Counter(pose_types))
    }

# ------------------------------------------------------
# [6] 종합 분석
# ------------------------------------------------------
def analyze_cluster_deep(cluster_id):
    """클러스터 심층 분석"""
    print(f"\n{'='*70}")
    print(f"🔍 Cluster {cluster_id} 심층 분석")
    print(f"{'='*70}")
    
    cluster_df = df.filter(pl.col("cluster") == cluster_id)
    cluster_size = len(cluster_df)
    
    image_paths = [
        str(IMG_DIR / os.path.basename(fname))
        for fname in cluster_df["filename"].to_list()
    ]
    
    # 각 모델별 분석
    print("  📊 CLIP 분석...")
    clip_analysis = analyze_clip_features(cluster_df)
    
    print("  📊 OpenCLIP 분석...")
    openclip_analysis = analyze_openclip_features(cluster_df)
    
    print("  📊 DINO 분석...")
    dino_analysis = analyze_dino_features(cluster_df)
    
    print("  📊 MiDaS 분석...")
    midas_analysis = analyze_midas_features(cluster_df)
    
    print("  🎨 색감 분석...")
    color_analysis = analyze_color_detailed(image_paths, max_samples=50)
    
    print("  🕺 포즈 분석...")
    pose_analysis = analyze_pose_detailed(image_paths, yolo, max_samples=30)
    
    # 대표 이미지
    sample_files = cluster_df["filename"][:5].to_list()
    
    return {
        "cluster_id": cluster_id,
        "size": cluster_size,
        "timestamp": datetime.now().isoformat(),
        "model_analyses": {
            "clip": clip_analysis,
            "openclip": openclip_analysis,
            "dino": dino_analysis,
            "midas": midas_analysis
        },
        "visual_analyses": {
            "color": color_analysis,
            "pose": pose_analysis
        },
        "representative_images": [os.path.basename(f) for f in sample_files]
    }

# ------------------------------------------------------
# [7] TXT 리포트 생성
# ------------------------------------------------------
def generate_txt_report(analysis, output_path):
    """분석 결과를 읽기 쉬운 TXT로 저장"""
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("="*70 + "\n")
        f.write(f"TryAngle 클러스터 심층 분석 리포트\n")
        f.write(f"Cluster {analysis['cluster_id']}\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"📊 기본 정보\n")
        f.write(f"  총 이미지 수: {analysis['size']}장\n")
        f.write(f"  분석 시각: {analysis['timestamp']}\n\n")
        
        # AI 모델별 분석
        f.write("="*70 + "\n")
        f.write("🤖 AI 모델별 특징 분석\n")
        f.write("="*70 + "\n\n")
        
        models = analysis['model_analyses']
        
        # CLIP
        if models['clip']:
            f.write("📌 CLIP (ViT-B/32)\n")
            f.write(f"  역할: {models['clip']['purpose']}\n")
            f.write(f"  특징 차원: {models['clip']['dimensions']}차원\n")
            f.write(f"  클러스터 응집도: {models['clip']['cluster_cohesion']:.2f}\n")
            f.write(f"  해석: {models['clip']['interpretation']}\n\n")
        
        # OpenCLIP
        if models['openclip']:
            f.write("📌 OpenCLIP (ViT-B/32, LAION-2B)\n")
            f.write(f"  역할: {models['openclip']['purpose']}\n")
            f.write(f"  특징 차원: {models['openclip']['dimensions']}차원\n")
            f.write(f"  클러스터 응집도: {models['openclip']['cluster_cohesion']:.2f}\n")
            f.write(f"  해석: {models['openclip']['interpretation']}\n\n")
        
        # DINO
        if models['dino']:
            f.write("📌 DINOv2 (ViT-S/14)\n")
            f.write(f"  역할: {models['dino']['purpose']}\n")
            f.write(f"  특징 차원: {models['dino']['dimensions']}차원\n")
            f.write(f"  공간 일관성: {models['dino']['spatial_consistency']:.2f}\n")
            f.write(f"  해석: {models['dino']['interpretation']}\n\n")
        
        # MiDaS
        if models['midas']:
            f.write("📌 MiDaS (DPT-Hybrid)\n")
            f.write(f"  역할: {models['midas']['purpose']}\n")
            f.write(f"  평균 깊이: {models['midas']['avg_depth']:.1f}\n")
            f.write(f"  깊이 분산: {models['midas']['depth_variance']:.1f}\n")
            f.write(f"  공간 유형: {models['midas']['space_type']}\n")
            f.write(f"  배경 복잡도: {models['midas']['background_complexity']}\n")
            f.write(f"  깊이 일관성: {models['midas']['depth_consistency']:.2f}\n")
            f.write(f"  해석: {models['midas']['interpretation']}\n\n")
        
        # 시각적 분석
        f.write("="*70 + "\n")
        f.write("🎨 시각적 특징 분석\n")
        f.write("="*70 + "\n\n")
        
        visual = analysis['visual_analyses']
        
        # 색감
        if visual['color']:
            c = visual['color']
            f.write("📌 색감 분석\n")
            f.write(f"  색온도: {c['tone']}\n")
            f.write(f"  밝기: {c['brightness']}\n")
            f.write(f"  채도: {c['saturation']}\n")
            f.write(f"  대비: {c['contrast']}\n")
            f.write(f"  엣지 밀도: {c['edge_density']}\n")
            f.write(f"  \n")
            f.write(f"  세부 수치:\n")
            f.write(f"    평균 색상(H): {c['avg_hue']:.1f}\n")
            f.write(f"    평균 채도(S): {c['avg_sat']:.1f}\n")
            f.write(f"    평균 밝기(V): {c['avg_val']:.1f}\n")
            f.write(f"    색온도 지수: {c['avg_temp']:.3f}\n\n")
        
        # 포즈
        if visual['pose']:
            p = visual['pose']
            f.write("📌 포즈/구도 분석\n")
            f.write(f"  주요 포즈: {p['pose_type']}\n")
            f.write(f"  수직 위치: {p['position_vertical']}\n")
            f.write(f"  수평 위치: {p['position_horizontal']}\n")
            f.write(f"  포즈 분포: {p['distribution']}\n\n")
        
        # 종합 해석
        f.write("="*70 + "\n")
        f.write("💡 종합 해석\n")
        f.write("="*70 + "\n\n")
        
        f.write(generate_comprehensive_interpretation(analysis))
        
        # 대표 이미지
        f.write("\n" + "="*70 + "\n")
        f.write("🖼️ 대표 이미지\n")
        f.write("="*70 + "\n\n")
        for i, fname in enumerate(analysis['representative_images'], 1):
            f.write(f"  {i}. {fname}\n")
        
        f.write("\n" + "="*70 + "\n")

def generate_comprehensive_interpretation(analysis):  # ← 함수명 변경!
    """종합 해석 생성"""
    text = ""
    
    models = analysis['model_analyses']
    visual = analysis['visual_analyses']
    
    # MiDaS 기반
    if models['midas']:
        text += f"이 클러스터는 {models['midas']['space_type']}에서 촬영된 이미지들입니다.\n"
        text += f"배경 복잡도는 {models['midas']['background_complexity']}입니다.\n\n"
    
    # 포즈
    if visual['pose']:
        text += f"주로 {visual['pose']['pose_type']} 포즈로, "
        text += f"인물은 화면 {visual['pose']['position_vertical']}에 위치합니다.\n\n"
    
    # 색감
    if visual['color']:
        text += f"색감은 {visual['color']['tone']}이며, "
        text += f"{visual['color']['brightness']} 밝기와 "
        text += f"{visual['color']['saturation']} 채도를 보입니다.\n\n"
    
    # DINO
    if models['dino']:
        if models['dino']['spatial_consistency'] > 15:
            text += "DINO 분석 결과, 이 클러스터는 매우 일관된 구도 패턴을 보입니다.\n"
        else:
            text += "DINO 분석 결과, 다양한 구도가 혼재되어 있습니다.\n"
    
    # CLIP/OpenCLIP
    if models['clip'] and models['clip']['cluster_cohesion'] > 15:
        text += "CLIP 분석 결과, 감성적으로 매우 유사한 이미지들로 구성되어 있습니다.\n"
    
    return text

# ------------------------------------------------------
# [8] 메인 실행
# ------------------------------------------------------
def main():
    print("\n" + "="*70)
    print("🧠 TryAngle 클러스터 심층 분석")
    print("="*70)
    
    n_clusters = df["cluster"].n_unique()
    
    for c in range(n_clusters):
        # 분석 수행
        analysis = analyze_cluster_deep(c)
        
        # JSON 저장
        json_path = CLUSTER_DIR / f"cluster_{c}" / "analysis.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2, default=str)
        
        # TXT 리포트 저장
        txt_path = CLUSTER_DIR / f"cluster_{c}" / "README.txt"
        generate_txt_report(analysis, txt_path)
        
        print(f"✅ Cluster {c} 분석 완료")
        print(f"   📄 {txt_path}")
    
    print("\n" + "="*70)
    print("✅ 전체 분석 완료!")
    print(f"📁 결과 위치: {CLUSTER_DIR}")
    print("="*70)

if __name__ == "__main__":
    main()
