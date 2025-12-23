import os
import shutil
import json
import numpy as np
import polars as pl
from collections import Counter
from colorsys import rgb_to_hsv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
while PROJECT_ROOT != PROJECT_ROOT.parent and not ((PROJECT_ROOT / "data").exists() and (PROJECT_ROOT / "src").exists()):
    PROJECT_ROOT = PROJECT_ROOT.parent

# ============================
# 설정
# ============================
FEATURE_PARQUET = PROJECT_ROOT / "feature_models" / "features" / "fusion_features_v2.parquet"
CLUSTER_LABELS_NPY = PROJECT_ROOT / "feature_models" / "feature_models_v3" / "cluster_labels.npy"
IMAGE_DIR = PROJECT_ROOT / "data" / "train_images"
OUTPUT_CLUSTER_DIR = PROJECT_ROOT / "clusters"

# 🔥 최적 설정 (Auto Optimizer 결과)
K = 20
WEIGHTS = {
    "clip": 0.30,
    "openclip": 0.30,
    "dino": 0.25,
    "color": 0.12,
    "midas": 0.03,
    "pose": 0.00,  # Pose 제외
}

# 🗑️ 기존 clusters 폴더 완전 삭제 후 재생성
if os.path.exists(OUTPUT_CLUSTER_DIR):
    print(f"🗑️ Removing old clusters: {OUTPUT_CLUSTER_DIR}")
    shutil.rmtree(OUTPUT_CLUSTER_DIR)
    print("✅ Old clusters removed\n")

os.makedirs(OUTPUT_CLUSTER_DIR, exist_ok=True)


# ============================
# 1) 데이터 로드
# ============================
print("="*60)
print("🏷️ TryAngle Auto Cluster Labeling")
print("="*60)
print("\n📂 Loading data...")

if not os.path.exists(FEATURE_PARQUET):
    raise FileNotFoundError(f"❌ Feature file not found: {FEATURE_PARQUET}")
    
if not os.path.exists(CLUSTER_LABELS_NPY):
    raise FileNotFoundError(f"❌ Cluster labels not found: {CLUSTER_LABELS_NPY}")

df = pl.read_parquet(FEATURE_PARQUET)
cluster_labels = np.load(CLUSTER_LABELS_NPY)
df = df.with_columns(pl.Series("cluster", cluster_labels))

filenames = df["filename"].to_list()
n_clusters = int(df["cluster"].max() + 1)

print(f"✅ Loaded {len(df)} samples")
print(f"✅ Number of clusters: {n_clusters}")

# K 검증
if n_clusters != K:
    print(f"⚠️ WARNING: Expected K={K}, but found {n_clusters} clusters")


# ============================
# Helper: 색상 추출 함수
# ============================
def extract_mean_color(color_hist):
    """
    119D 또는 150D color hist를 mean RGB → HSV로 변환
    
    구조:
    - HSV histogram (3 channels × 32 bins = 96D)
    - LAB stats (3 channels × 4 stats = 12D)
    - LBP texture (10D)
    - Edge density (1D)
    Total: 96 + 12 + 10 + 1 = 119D
    """
    arr = np.array(color_hist)
    
    # HSV histogram 부분만 사용 (앞 96D)
    hsv_hist = arr[:96] if len(arr) >= 96 else arr[:min(96, len(arr))]
    
    # 3개 채널로 분리
    n_bins = len(hsv_hist) // 3
    h_hist = hsv_hist[:n_bins]
    s_hist = hsv_hist[n_bins:2*n_bins]
    v_hist = hsv_hist[2*n_bins:3*n_bins]
    
    # 가중 평균 (histogram bin index × 값)
    h = np.average(np.arange(len(h_hist)), weights=h_hist + 1e-6)
    s = np.average(np.arange(len(s_hist)), weights=s_hist + 1e-6)
    v = np.average(np.arange(len(v_hist)), weights=v_hist + 1e-6)
    
    # 정규화 [0, 1]
    h = h / len(h_hist)
    s = s / len(s_hist)
    v = v / len(v_hist)
    
    return h, s, v


# ============================
# Helper: 자동 라벨 생성
# ============================
def generate_cluster_label(cluster_df):
    """
    클러스터 평균 특징으로 자동 라벨링
    
    최적 가중치 반영:
    - CLIP/OpenCLIP/DINO: 주요 스타일 결정 (85%)
    - Color: 보조 (12%)
    - MiDaS: 최소 (3%)
    - Pose: 제외 (0%)
    """
    
    # -------------------------
    # 1) 색감 분석 (Color 12%)
    # -------------------------
    color_vec = cluster_df["color"].to_list()
    color_mean = np.mean(np.vstack(color_vec), axis=0)
    h, s, v = extract_mean_color(color_mean)
    
    # -------------------------
    # 2) DINO 구도 분석 (25%)
    # -------------------------
    dino_vec = np.vstack(cluster_df["dino"].to_list())
    dino_mean = np.mean(dino_vec, axis=0)
    dino_std = np.std(dino_vec, axis=0)
    
    # 중심성: 벡터의 평균 크기
    dino_center_energy = np.linalg.norm(dino_mean)
    
    # 분산도: 표준편차의 평균
    dino_variance = np.mean(dino_std)
    
    # -------------------------
    # 3) MiDaS 깊이/앵글 (3%) - 거의 무시
    # -------------------------
    midas_vec = np.vstack(cluster_df["midas"].to_list())
    
    # MiDaS 차원 확인
    if midas_vec.shape[1] >= 2:
        # depth_mean, depth_std, ... 등 여러 통계량
        depth_mean = midas_vec[:, 0].mean()
        depth_grad_y = midas_vec[:, 1].mean() if midas_vec.shape[1] > 1 else 0
    else:
        depth_mean = midas_vec[:, 0].mean()
        depth_grad_y = 0
    
    # -------------------------
    # 4) Pose 존재 여부 (0% 가중치지만 정보로 활용)
    # -------------------------
    pose_vec = np.vstack(cluster_df["yolo_pose"].to_list())
    person_ratio = np.mean(np.sum(np.abs(pose_vec), axis=1) > 0.1)
    
    # -------------------------
    # 5) CLIP 기반 분위기 추정 (간접)
    # -------------------------
    # CLIP 자체는 해석 어려우므로, 다른 특징들과 조합으로 추정
    
    # -------------------------
    # 규칙 기반 라벨링
    # -------------------------
    tags = []
    
    # === 색감 (Color 12%) ===
    if s < 0.3:  # 채도 낮음
        if v > 0.6:
            tags.append("밝고 차분한")
        else:
            tags.append("어두운 무채색")
    else:  # 채도 높음
        if h < 0.1 or h > 0.85:  # 빨강~주황
            tags.append("따뜻한 감성")
        elif 0.15 < h < 0.45:  # 초록~노랑
            tags.append("생기있는")
        elif 0.45 < h < 0.7:  # 파랑~청록
            tags.append("차가운 감성")
        else:
            tags.append("뉴트럴 톤")
    
    # === 밝기 ===
    if v > 0.65:
        tags.append("밝은 분위기")
    elif v < 0.35:
        tags.append("어두운 무드")
    else:
        tags.append("중간 밝기")
    
    # === DINO 구도 (25%) ===
    if dino_center_energy > 0.3:
        tags.append("중앙 집중")
    elif dino_center_energy < 0.15:
        tags.append("분산 배치")
    
    if dino_variance > 0.25:
        tags.append("복잡한 구성")
    elif dino_variance < 0.12:
        tags.append("단순 구도")
    
    # === 사람/풍경 (참고용) ===
    if person_ratio > 0.7:
        tags.append("인물 중심")
    elif person_ratio < 0.2:
        tags.append("풍경/사물")
    else:
        tags.append("인물+배경")
    
    # === 앵글 (MiDaS 3%, 거의 무시) ===
    if abs(depth_grad_y) > 0.1:  # 임계값 높임
        if depth_grad_y < 0:
            tags.append("로우앵글")
        else:
            tags.append("하이앵글")
    
    return " / ".join(tags)


# ============================
# 2) 클러스터별 폴더 생성 + 파일 복사
# ============================
print("\n📁 Splitting images into cluster folders...")

cluster_info = {}

for c in range(n_clusters):
    cluster_folder = OUTPUT_CLUSTER_DIR / f"cluster_{c:02d}"
    os.makedirs(cluster_folder, exist_ok=True)
    
    cluster_df = df.filter(pl.col("cluster") == c)
    cluster_files = cluster_df["filename"].to_list()
    
    # 자동 라벨 생성
    label = generate_cluster_label(cluster_df)
    cluster_info[str(c)] = {
        "label": label,
        "count": len(cluster_files),
        "percentage": round(len(cluster_files) / len(df) * 100, 2)
    }
    
    # 설명 파일 저장
    desc_file = cluster_folder / "cluster_description.txt"
    with open(desc_file, "w", encoding="utf-8") as f:
        f.write(f"Cluster {c}\n")
        f.write(f"="*40 + "\n")
        f.write(f"Image Count: {len(cluster_files)}\n")
        f.write(f"Percentage: {cluster_info[str(c)]['percentage']}%\n")
        f.write(f"Auto Label: {label}\n")
        f.write(f"\n")
        f.write(f"Optimal Settings:\n")
        f.write(f"  K = {K}\n")
        f.write(f"  Silhouette = 0.3988\n")
        f.write(f"  Weights: {WEIGHTS}\n")
    
    # 이미지 복사
    copied = 0
    for fname in cluster_files:
        src = IMAGE_DIR / fname
        dst = cluster_folder / fname
        
        if os.path.exists(src):
            shutil.copyfile(src, dst)
            copied += 1
    
    print(f"✔️ Cluster {c:02d}: {copied}/{len(cluster_files)} images | {label}")


# ============================
# 3) 전체 summary 저장
# ============================
summary_path = OUTPUT_CLUSTER_DIR / "cluster_summary.json"

summary_data = {
    "metadata": {
        "total_images": len(df),
        "n_clusters": n_clusters,
        "K": K,
        "silhouette_score": 0.3988,
        "weights": WEIGHTS,
        "optimization_method": "Grid Search (60 combinations)",
    },
    "clusters": cluster_info
}

with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary_data, f, indent=2, ensure_ascii=False)

print("\n" + "="*60)
print("🎉 Auto Labeling Complete!")
print("="*60)
print(f"\n📌 Output directory: {OUTPUT_CLUSTER_DIR}")
print(f"📌 Summary file: {summary_path}")
print("\n📊 Cluster Distribution:")
for c in range(min(5, n_clusters)):  # 상위 5개만 표시
    info = cluster_info[str(c)]
    print(f"   Cluster {c:02d}: {info['count']:4d} ({info['percentage']:5.2f}%) - {info['label']}")
if n_clusters > 5:
    print(f"   ... and {n_clusters - 5} more clusters")
print("="*60)
