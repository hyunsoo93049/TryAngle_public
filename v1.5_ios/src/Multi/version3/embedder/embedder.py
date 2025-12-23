# ============================================================
# 🔧 Load Embedded Models (Scaler + UMAP) — version2 compatible
# ============================================================

import os
import joblib
import numpy as np
import sys
from pathlib import Path

# Model cache
current_dir = Path(__file__).parent
sys.path.append(str(current_dir.parent))
from utils.model_cache import model_cache

# 모델 저장 경로 (상대 경로)
# embedder.py -> version3 -> Multi -> src -> Try_Angle
FEATURE_MODEL_DIR = current_dir.parent.parent.parent.parent / "feature_models"

SCALER_CLIP_PATH      = os.path.join(FEATURE_MODEL_DIR, "scaler_clip.joblib")
SCALER_OPENCLIP_PATH  = os.path.join(FEATURE_MODEL_DIR, "scaler_openclip.joblib")
SCALER_DINO_PATH      = os.path.join(FEATURE_MODEL_DIR, "scaler_dino.joblib")
SCALER_COLOR_PATH     = os.path.join(FEATURE_MODEL_DIR, "scaler_color.joblib")
SCALER_MIDAS_PATH     = os.path.join(FEATURE_MODEL_DIR, "scaler_midas.joblib")

UMAP_MODEL_PATH       = os.path.join(FEATURE_MODEL_DIR, "umap_128d_model.joblib")

# -----------------------------
# Load models (싱글톤)
# -----------------------------
def _load_embedder_models():
    """Embedder 모델 로드 (한 번만)"""
    print("🔧 Loading embedder models...")

    scaler_clip      = joblib.load(SCALER_CLIP_PATH)
    scaler_openclip  = joblib.load(SCALER_OPENCLIP_PATH)
    scaler_dino      = joblib.load(SCALER_DINO_PATH)
    scaler_color     = joblib.load(SCALER_COLOR_PATH)
    scaler_midas     = joblib.load(SCALER_MIDAS_PATH)

    umap_model       = joblib.load(UMAP_MODEL_PATH)

    print("✅ Embedder models loaded successfully")

    return {
        "scaler_clip": scaler_clip,
        "scaler_openclip": scaler_openclip,
        "scaler_dino": scaler_dino,
        "scaler_color": scaler_color,
        "scaler_midas": scaler_midas,
        "umap_model": umap_model
    }

def get_embedder_models():
    """Embedder 모델 가져오기 (싱글톤)"""
    return model_cache.get_or_load("embedder_models", _load_embedder_models)

def embed_features(feature_dict: dict):
    """
    feature_extractor_v2.py → extract_features_v2() 결과(dict) 를 입력으로 받는다.
    {
        "clip": (512,),
        "openclip": (512,),
        "dino": (384,),
        "color": (150,),
        "midas": (20,),
        "yolo_pose": (15,),  # v2에서 추가
        "face": (7,)         # v2에서 추가
    }

    Note: yolo_pose와 face는 학습 시 가중치=0이므로 실제로는 사용 안 됨
    """
    # -----------------------------
    # 입력이 path 면 → ❌ 잘못된 호출
    # -----------------------------
    if isinstance(feature_dict, str):
        raise ValueError(
            "❌ embed_features()는 image_path가 아니라 feature_dict를 입력해야 합니다.\n"
            "먼저 extract_features_full(image_path)로 feature를 추출하세요."
        )

    # 모델 가져오기 (캐시됨)
    models = get_embedder_models()
    scaler_clip = models["scaler_clip"]
    scaler_openclip = models["scaler_openclip"]
    scaler_dino = models["scaler_dino"]
    scaler_color = models["scaler_color"]
    scaler_midas = models["scaler_midas"]
    umap_model = models["umap_model"]

    clip_vec  = feature_dict["clip"].reshape(1, -1)
    open_vec  = feature_dict["openclip"].reshape(1, -1)
    dino_vec  = feature_dict["dino"].reshape(1, -1)   # 반드시 384
    color_vec = feature_dict["color"].reshape(1, -1)
    midas_vec = feature_dict["midas"].reshape(1, -1)

    # yolo_pose, face (v2에서 추가, 하지만 가중치=0)
    yolo_pose_vec = feature_dict["yolo_pose"].reshape(1, -1)  # (1, 15)
    face_vec = feature_dict["face"].reshape(1, -1)            # (1, 7)

    # -----------------------------
    # feature_models에서 불러온 scaler 적용
    # -----------------------------
    clip_scaled      = scaler_clip.transform(clip_vec)
    openclip_scaled  = scaler_openclip.transform(open_vec)
    dino_scaled      = scaler_dino.transform(dino_vec)
    color_scaled     = scaler_color.transform(color_vec)
    midas_scaled     = scaler_midas.transform(midas_vec)

    # pose, face는 scaling 없이 그냥 사용 (어차피 가중치=0)
    # 학습 시와 동일하게 처리
    pose_combined = np.concatenate([yolo_pose_vec, face_vec], axis=1)  # (1, 22)

    # -----------------------------
    # 1600D 융합 (512+512+384+150+20+22)
    # 학습 시 가중치: clip=0.3, openclip=0.3, dino=0.25, color=0.12, midas=0.03, pose=0.0
    # -----------------------------
    fusion = np.concatenate([
        clip_scaled * 0.30,
        openclip_scaled * 0.30,
        dino_scaled * 0.25,
        color_scaled * 0.12,
        midas_scaled * 0.03,
        pose_combined * 0.00  # 가중치 0이므로 실제로는 무시됨
    ], axis=1)

    # -----------------------------
    # UMAP 128D 축소
    # -----------------------------
    vec128 = umap_model.transform(fusion)[0]
    return vec128
