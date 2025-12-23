# ======================================================
# 🧠 TryAngle Ultra-Advanced Clustering v2
# (PCA 제거 + UMAP 차원 축소 + KMeans)
#  - CLIP + OpenCLIP + DINO + MiDaS + Color/Texture
#  - UMAP 128D 임베딩 저장
#  - KMeans 모델 + 센트로이드 + 스케일러 저장
# ======================================================

import os, gc, warnings
import numpy as np
import polars as pl
import cv2
from scipy.stats import skew, kurtosis
from skimage.feature import local_binary_pattern

from sklearn.preprocessing import RobustScaler, normalize
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from umap import UMAP
import joblib
from pathlib import Path

warnings.filterwarnings("ignore")

# ------------------------------------------------------
# [1] 경로 설정
# ------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
while PROJECT_ROOT != PROJECT_ROOT.parent and not ((PROJECT_ROOT / "data").exists() and (PROJECT_ROOT / "src").exists()):
    PROJECT_ROOT = PROJECT_ROOT.parent

PARQUET_PATH = PROJECT_ROOT / "features" / "fusion_final_with_openclip.parquet"
IMG_DIR      = PROJECT_ROOT / "data" / "train_images"
SAVE_DIR     = PROJECT_ROOT / "models_v2"

os.makedirs(SAVE_DIR, exist_ok=True)

# ------------------------------------------------------
# [2] 색상/텍스처 특징 추출 함수
# ------------------------------------------------------
def extract_color_texture_features(img_path: str):
    """
    색상 히스토그램(HSV) + LAB 통계 + LBP 텍스처 + 엣지 밀도
    총 119차원 정도의 hand-crafted feature
    """
    try:
        img = cv2.imread(img_path)
        if img is None:
            return None

        img = cv2.resize(img, (256, 256))
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        features = []

        # 1) HSV 히스토그램 (각 채널 32bin)
        for channel in cv2.split(img_hsv):
            hist = cv2.calcHist([channel], [0], None, [32], [0, 256]).flatten()
            hist = hist / (hist.sum() + 1e-7)
            features.extend(hist)

        # 2) LAB 통계량 (mean, std, skew, kurtosis)
        for channel in cv2.split(img_lab):
            flat = channel.flatten()
            features.extend([
                flat.mean(),
                flat.std(),
                skew(flat),
                kurtosis(flat)
            ])

        # 3) LBP 텍스처 히스토그램
        lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10))
        lbp_hist = lbp_hist / (lbp_hist.sum() + 1e-7)
        features.extend(lbp_hist)

        # 4) 엣지 밀도
        edges = cv2.Canny(gray, 50, 150)
        edge_density = edges.sum() / edges.size
        features.append(edge_density)

        return np.array(features, dtype=np.float32)

    except Exception:
        # 이미지 깨졌거나 로드 실패한 경우
        return None


# ------------------------------------------------------
# [3] 데이터 로드
# ------------------------------------------------------
print("📂 데이터 로드 중...")
df = pl.read_parquet(PARQUET_PATH)

clip_feats     = df.select([c for c in df.columns if c.startswith("clip_")]).to_numpy()
openclip_feats = df.select([c for c in df.columns if c.startswith("openclip_")]).to_numpy()
dino_feats     = df.select([c for c in df.columns if c.startswith("dino_")]).to_numpy()
midas_mean     = df.select("midas_mean").to_numpy().flatten()
midas_std      = df.select("midas_std").to_numpy().flatten()
filenames      = df["filename"].to_list()

print("✅ 기존 특징 로드 완료")
print(f"   CLIP      : {clip_feats.shape}")
print(f"   OpenCLIP  : {openclip_feats.shape}")
print(f"   DINO      : {dino_feats.shape}")
print(f"   MiDaS     : mean/std → ({midas_mean.shape[0]}, 2)")
print(f"   파일 개수 : {len(filenames)}")

# ------------------------------------------------------
# [4] 색상/텍스처 특징 추출
# ------------------------------------------------------
print("\n🎨 색상/텍스처 특징 추출 중...")
color_texture_list = []
for fname in filenames:
    fpath = os.path.join(IMG_DIR, os.path.basename(fname))
    feat = extract_color_texture_features(fpath)
    if feat is None:
        # 실패 시 0벡터 대체 (차원 맞추기용)
        feat = np.zeros(119, dtype=np.float32)
    color_texture_list.append(feat)

color_texture_feats = np.array(color_texture_list)
print("✅ 색상/텍스처 특징:", color_texture_feats.shape)

# ------------------------------------------------------
# [5] 정규화 + 스케일러 저장
# ------------------------------------------------------
print("\n🔹 특징 정규화 중...")

clip_scaler     = RobustScaler()
openclip_scaler = RobustScaler()
dino_scaler     = RobustScaler()
color_scaler    = RobustScaler()
midas_scaler    = RobustScaler()

clip_z     = clip_scaler.fit_transform(clip_feats)
open_z     = openclip_scaler.fit_transform(openclip_feats)
dino_z     = dino_scaler.fit_transform(dino_feats)
color_z    = color_scaler.fit_transform(color_texture_feats)

midas_feats = np.column_stack([midas_mean, midas_std])
midas_z     = midas_scaler.fit_transform(midas_feats)

# 💾 스케일러 저장
joblib.dump(clip_scaler,     os.path.join(SAVE_DIR, "scaler_clip.joblib"))
joblib.dump(openclip_scaler, os.path.join(SAVE_DIR, "scaler_openclip.joblib"))
joblib.dump(dino_scaler,     os.path.join(SAVE_DIR, "scaler_dino.joblib"))
joblib.dump(color_scaler,    os.path.join(SAVE_DIR, "scaler_color.joblib"))
joblib.dump(midas_scaler,    os.path.join(SAVE_DIR, "scaler_midas.joblib"))

print("💾 스케일러 저장 완료:")
print(f"   {os.path.join(SAVE_DIR, 'scaler_*.joblib')}")

# 가중치
w_clip, w_op, w_dino, w_color, w_midas = 0.30, 0.30, 0.25, 0.12, 0.03

fusion_raw = np.concatenate(
    [
        clip_z * w_clip,
        open_z * w_op,
        dino_z * w_dino,
        color_z * w_color,
        midas_z * w_midas,
    ],
    axis=1,
)

print("✅ 원본 융합 특징:", fusion_raw.shape)

# ------------------------------------------------------
# [6] UMAP 128D 학습 + 저장
# ------------------------------------------------------
print("\n🔹 UMAP 차원 축소 중 (128D)...")
umap_high = UMAP(
    n_components=128,
    n_neighbors=30,
    min_dist=0.0,
    metric="cosine",
    random_state=42,
    verbose=True,
)
fusion_128 = umap_high.fit_transform(fusion_raw)
fusion_128 = normalize(fusion_128)

print("✅ UMAP 128D 완료:", fusion_128.shape)

# 💾 UMAP + 128D 임베딩 저장
joblib.dump(umap_high, os.path.join(SAVE_DIR, "umap_128d_model.joblib"))
np.save(os.path.join(SAVE_DIR, "fusion_128d.npy"), fusion_128)
print("💾 UMAP 모델 및 임베딩 저장 완료")

# ------------------------------------------------------
# [7] KMeans 학습 + 저장
# ------------------------------------------------------
print("\n🔹 KMeans 학습 중...")
k = 10
km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
labels_km = km.fit_predict(fusion_128)

joblib.dump(km, os.path.join(SAVE_DIR, "kmeans_model.pkl"))
np.save(os.path.join(SAVE_DIR, "kmeans_centroids.npy"), km.cluster_centers_)

print("💾 KMeans 모델/센트로이드 저장 완료")

# ------------------------------------------------------
# [8] 성능 평가 + 결과 parquet 저장
# ------------------------------------------------------
sil = silhouette_score(fusion_128, labels_km, sample_size=min(5000, len(fusion_128)))
print(f"\n🏆 Silhouette = {sil:.4f}")

df = df.with_columns(pl.Series("cluster", labels_km))
clustered_output = PROJECT_ROOT / "features" / "clustered_umap_v2_result.parquet"
df.write_parquet(clustered_output)

print("\n🎉 클러스터링 완료!")
print(f"   → {clustered_output.name} 저장됨")
print(f"   → 모델 디렉토리: {SAVE_DIR}")
