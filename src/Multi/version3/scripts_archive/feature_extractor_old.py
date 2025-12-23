# ============================================================
# 🔥 TryAngle Unified Feature Extractor v3
#  - CLIP + OpenCLIP + DINO(384D) + MiDaS + ColorTexture
#  - train_embedding_clusters.py / clustering_umap_enhanced_v2.py
#    에서 사용된 feature 차원과 정확히 맞추는 버전
# ============================================================

import os
import cv2
import numpy as np
import torch
from PIL import Image
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
while PROJECT_ROOT != PROJECT_ROOT.parent and not ((PROJECT_ROOT / "data").exists() and (PROJECT_ROOT / "src").exists()):
    PROJECT_ROOT = PROJECT_ROOT.parent

# CLIP
import clip
# OpenCLIP
import open_clip

# DINOv2
from timm import create_model
from timm.data import resolve_model_data_config
from timm.data.transforms_factory import create_transform

# MiDaS
from transformers import DPTImageProcessor, DPTForDepthEstimation

# Texture
from scipy.stats import skew, kurtosis
from skimage.feature import local_binary_pattern

# ------------------------------------------------------------
# Device
# ------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------------------------
# Lazy-loaded models
# ------------------------------------------------------------
_models = {}


def load_models():
    """
    모든 모델을 한 번만 로드해서 _models에 캐시.
    (CLIP / OpenCLIP / DINOv2 / MiDaS)
    """
    global _models
    if _models:
        return _models

    print("🔧 Loading AI backbone models...")

    # ---- CLIP ----
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

    # ---- OpenCLIP ----
    openclip_model, _, openclip_preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="laion2b_s34b_b79k",
        device=device,
    )

    # ---- DINOv2 (vit_small_patch14_dinov2.lvd142m) ----
    dino_model = create_model(
        "vit_small_patch14_dinov2.lvd142m",
        pretrained=True
    ).to(device).eval()
    dino_cfg = resolve_model_data_config(dino_model)
    dino_tf = create_transform(**dino_cfg)

    # ---- MiDaS ----
    midas_processor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")
    midas_model = DPTForDepthEstimation.from_pretrained(
        "Intel/dpt-hybrid-midas"
    ).to(device).eval()

    _models = {
        "clip_model": clip_model,
        "clip_preprocess": clip_preprocess,
        "openclip_model": openclip_model,
        "openclip_preprocess": openclip_preprocess,
        "dino_model": dino_model,
        "dino_tf": dino_tf,
        "midas_processor": midas_processor,
        "midas_model": midas_model,
    }

    print("✅ Backbone models loaded")
    return _models


# ------------------------------------------------------------
# Color / Texture features  (150D)
# ------------------------------------------------------------
def extract_color_texture(img_bgr: np.ndarray) -> np.ndarray:
    """
    색상 히스토그램(HSV) + LAB 통계 + LBP + Edge density + RGB histogram
    → 총 150차원 (확장 버전)
    """
    img = cv2.resize(img_bgr, (256, 256))
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    feats = []

    # HSV histograms: 32 bins × 3채널 = 96
    for channel in cv2.split(img_hsv):
        hist = cv2.calcHist([channel], [0], None, [32], [0, 256]).flatten()
        hist = hist / (hist.sum() + 1e-7)
        feats.extend(hist)

    # LAB stats: (mean, std, skew, kurt) × 3 = 12
    for channel in cv2.split(img_lab):
        flat = channel.flatten()
        feats.extend([
            flat.mean(),
            flat.std(),
            skew(flat),
            kurtosis(flat),
        ])

    # LBP texture: 10 bins
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10))
    lbp_hist = lbp_hist / (lbp_hist.sum() + 1e-7)
    feats.extend(lbp_hist)

    # Edge density: 1
    edges = cv2.Canny(gray, 50, 150)
    edge_density = edges.sum() / edges.size
    feats.append(edge_density)

    # RGB histogram: 10 bins × 3채널 = 30
    for channel in cv2.split(img_bgr):
        hist = cv2.calcHist([channel], [0], None, [10], [0, 256]).flatten()
        hist = hist / (hist.sum() + 1e-7)
        feats.extend(hist)

    # Gradient magnitude: 1
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2).mean()
    feats.append(gradient_magnitude)

    feats = np.array(feats, dtype=np.float32)
    assert feats.shape == (150,), f"Color/Texture shape mismatch: {feats.shape}, expected (150,)"
    return feats


# ------------------------------------------------------------
# DINO helper  (must be 384D)
# ------------------------------------------------------------
def extract_dino_feature(models, img_pil: Image.Image) -> np.ndarray:
    """
    DINOv2에서 **반드시 384차원**짜리 토큰만 뽑아서 반환.
    train_embedding_clusters.py 에서 사용된 dino_feats 형식과 맞춰야 함.
    """
    dino_model = models["dino_model"]
    dino_tf = models["dino_tf"]

    x = dino_tf(img_pil).unsqueeze(0).to(device)  # (1, C, H, W)

    with torch.no_grad():
        feats = dino_model.forward_features(x)

    # 1) dict 출력인 경우
    if isinstance(feats, dict):
        token = None
        # 가장 먼저 cls token 후보부터 찾기
        for key in ["x_norm_clstoken", "cls_token", "pool"]:
            if key in feats:
                token = feats[key]
                break

        # 그래도 못 찾으면 다른 키들 시도
        if token is None:
            for key in ["x_norm", "head", "features"]:
                if key in feats:
                    token = feats[key]
                    break

        if token is None:
            raise ValueError(f"❌ DINO dict에서 사용할 수 있는 토큰을 찾지 못했습니다. keys={list(feats.keys())}")

    # 2) 텐서 직접 나오는 경우
    elif isinstance(feats, torch.Tensor):
        token = feats
    else:
        raise ValueError(f"❌ 알 수 없는 DINO 출력 타입: {type(feats)}")

    # ----- shape 정리 -----
    # 가능한 케이스:
    # (1, 384)
    # (1, 384, 1, 1)
    # (1, seq_len, 384)
    # 등등
    if token.ndim == 2:
        # (1, 384) 같은 케이스
        pass
    elif token.ndim == 3:
        # (1, seq, 384) → CLS 토큰으로 가정하고 첫 토큰만 사용
        token = token[:, 0, :]  # (1, 384)
    elif token.ndim == 4:
        # (1, 384, H, W) → spatial 평균 풀링
        token = token.mean(dim=[-2, -1])  # (1, 384)
    else:
        raise ValueError(f"❌ DINO token 차원 수가 이상함: {token.shape}")

    # 이제 (1, 384) 이어야 함
    if token.shape != (1, 384):
        raise ValueError(f"❌ DINO 최종 토큰 shape 오류: {token.shape}, expected (1, 384)")

    # 정규화 + numpy 변환
    token = token / (token.norm(dim=-1, keepdim=True) + 1e-8)
    vec = token.cpu().numpy().astype(np.float32).flatten()  # (384,)
    assert vec.shape == (384,), f"❌ DINO feature shape mismatch: {vec.shape}"
    return vec


# ------------------------------------------------------------
# 🎯 Main Feature Extractor
# ------------------------------------------------------------
def extract_features(image_path: str):
    """
    이미지 1장을 입력 → 모든 모델 특징을 dict로 반환.

    반환 형식 (train_embedding_clusters.py / clustering_umap_enhanced_v2.py 와 호환):
    {
      "clip":    (512,),
      "openclip":(512,),
      "dino":    (384,),
      "midas":   (20,),
      "color":   (150,)
    }
    """
    models = load_models()

    # ----- 이미지 로드 -----
    try:
        img_pil = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"[❌] 이미지 로드 실패: {image_path} : {e}")
        return None

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"[❌] cv2.imread 실패: {image_path}")
        return None

    # --------------------------------------------------------
    # 1) CLIP (512D)
    # --------------------------------------------------------
    clip_in = models["clip_preprocess"](img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        clip_feat = models["clip_model"].encode_image(clip_in)
        clip_feat = clip_feat / clip_feat.norm(dim=-1, keepdim=True)
    clip_vec = clip_feat.cpu().numpy().astype(np.float32).flatten()
    assert clip_vec.shape == (512,), f"CLIP shape mismatch: {clip_vec.shape}"

    # --------------------------------------------------------
    # 2) OpenCLIP (512D)
    # --------------------------------------------------------
    openclip_in = models["openclip_preprocess"](img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        oc_feat = models["openclip_model"].encode_image(openclip_in)
        oc_feat = oc_feat / oc_feat.norm(dim=-1, keepdim=True)
    openclip_vec = oc_feat.cpu().numpy().astype(np.float32).flatten()
    assert openclip_vec.shape == (512,), f"OpenCLIP shape mismatch: {openclip_vec.shape}"

    # --------------------------------------------------------
    # 3) DINOv2 (384D, 정확히 맞춰야 함)
    # --------------------------------------------------------
    dino_vec = extract_dino_feature(models, img_pil)  # (384,)

    # --------------------------------------------------------
    # 4) MiDaS Depth (20D: global stats + spatial regions + histogram)
    # --------------------------------------------------------
    inputs = models["midas_processor"](images=img_pil, return_tensors="pt").to(device)
    with torch.no_grad():
        depth = models["midas_model"](inputs["pixel_values"]).predicted_depth

    # Depth map을 numpy로 변환
    depth_map = depth.squeeze().cpu().numpy()

    midas_features = []

    # 1. Global statistics (2D)
    midas_features.append(float(depth_map.mean()))
    midas_features.append(float(depth_map.std()))

    # 2. Spatial grid statistics (2x2 grid, 각 region mean/std = 8D)
    h, w = depth_map.shape
    grid_h, grid_w = h // 2, w // 2

    for i in range(2):
        for j in range(2):
            region = depth_map[i*grid_h:(i+1)*grid_h, j*grid_w:(j+1)*grid_w]
            midas_features.append(float(region.mean()))
            midas_features.append(float(region.std()))

    # 3. Depth histogram (5 bins = 5D)
    hist, _ = np.histogram(depth_map.flatten(), bins=5)
    hist = hist / (hist.sum() + 1e-7)
    midas_features.extend(hist.tolist())

    # 4. Gradient statistics (5D)
    grad_y = np.gradient(depth_map, axis=0)
    grad_x = np.gradient(depth_map, axis=1)
    gradient_mag = np.sqrt(grad_y**2 + grad_x**2)

    midas_features.append(float(grad_y.mean()))
    midas_features.append(float(grad_x.mean()))
    midas_features.append(float(gradient_mag.mean()))
    midas_features.append(float(gradient_mag.std()))
    midas_features.append(float(gradient_mag.max()))

    midas_vec = np.array(midas_features, dtype=np.float32)
    assert midas_vec.shape == (20,), f"MiDaS shape mismatch: {midas_vec.shape}, expected (20,)"

    # --------------------------------------------------------
    # 5) Color + Texture (150D)
    # --------------------------------------------------------
    color_vec = extract_color_texture(img_bgr)  # (150,)

    return {
        "clip": clip_vec,
        "openclip": openclip_vec,
        "dino": dino_vec,
        "midas": midas_vec,
        "color": color_vec,
    }


# ------------------------------------------------------------
# ✅ Backward-compatible wrapper (for old code)
# ------------------------------------------------------------
def extract_features_full(image_path: str):
    """
    version2에서 쓰던 extract_features_full 이름을 그대로 유지하기 위한 래퍼.
    내부적으로는 extract_features()를 그대로 호출.
    """
    return extract_features(image_path)


# 디버그용
if __name__ == "__main__":
    test_img = PROJECT_ROOT / "data" / "test_images" / "test1.jpg"
    feats = extract_features_full(str(test_img))
    if feats is None:
        print("❌ Feature extraction failed")
    else:
        print("✅ Feature shapes:")
        for k, v in feats.items():
            print(f"  {k:8s} -> {v.shape}")
