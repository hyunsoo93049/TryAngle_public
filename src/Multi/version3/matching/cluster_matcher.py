# =========================================================
# 🔮 TryAngle Cluster Matcher v3
# =========================================================

import os
import numpy as np
import joblib
import json
import sys
from pathlib import Path

# Model cache
VERSION3_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = VERSION3_DIR
while PROJECT_ROOT != PROJECT_ROOT.parent and not ((PROJECT_ROOT / "data").exists() and (PROJECT_ROOT / "src").exists()):
    PROJECT_ROOT = PROJECT_ROOT.parent

if str(VERSION3_DIR) not in sys.path:
    sys.path.append(str(VERSION3_DIR))

from utils.model_cache import model_cache

# 🔥 절대 import로만 구성 (가장 안정적)
from feature_extraction.feature_extractor_v2 import extract_features_v2 as extract_features_full
from embedder.embedder import embed_features

# =============================================
# 1) 모델 경로 설정
# =============================================
FEATURE_MODEL_DIR = PROJECT_ROOT / "feature_models"

KMEANS_MODEL_PATH   = FEATURE_MODEL_DIR / "kmeans_model.pkl"
CENTROIDS_PATH      = FEATURE_MODEL_DIR / "kmeans_centroids.npy"
CLUSTER_INFO_PATH   = FEATURE_MODEL_DIR / "cluster_info.json"

# ---------------------------------------------------------
# [2] 모델 로딩 (싱글톤)
# ---------------------------------------------------------
def _load_cluster_models():
    """Cluster 모델 로드 (한 번만)"""
    print("🔧 Loading cluster matcher models...")

    kmeans_model = joblib.load(KMEANS_MODEL_PATH)
    centroids    = np.load(CENTROIDS_PATH)

    # interpretation(라벨)이 있으면 불러오고 없으면 None
    if os.path.exists(CLUSTER_INFO_PATH):
        with open(CLUSTER_INFO_PATH, "r", encoding="utf-8") as f:
            cluster_info = json.load(f)
    else:
        cluster_info = None

    print("✅ Cluster matcher models loaded successfully")

    return {
        "kmeans_model": kmeans_model,
        "centroids": centroids,
        "cluster_info": cluster_info
    }

# 싱글톤으로 로드
_cluster_models = None

def get_cluster_models():
    """Cluster 모델 가져오기 (싱글톤)"""
    return model_cache.get_or_load("cluster_matcher_models", _load_cluster_models)


# ---------------------------------------------------------
# [3] 클러스터 예측 함수
# ---------------------------------------------------------
def match_cluster_from_features(feature_dict):
    # 모델 가져오기 (캐시됨)
    models = get_cluster_models()
    kmeans_model = models["kmeans_model"]
    centroids = models["centroids"]
    cluster_info = models["cluster_info"]

    vec_128 = embed_features(feature_dict).reshape(1, -1)

    cluster_id = int(kmeans_model.predict(vec_128)[0])

    center = centroids[cluster_id]
    dist = float(np.linalg.norm(vec_128 - center))

    if cluster_info and str(cluster_id) in cluster_info:
        label = cluster_info[str(cluster_id)]
    else:
        label = f"cluster_{cluster_id}"

    return {
        "cluster_id": cluster_id,
        "distance": dist,
        "label": label,
        "raw_embedding": vec_128.flatten()
    }


# ---------------------------------------------------------
# [3.5] Phase 1-2: 클러스터 폴백 로직 추가
# ---------------------------------------------------------
def match_with_fallback(feature_dict, confidence_threshold=0.6):
    """
    클러스터 매칭 + 폴백 로직

    Args:
        feature_dict: 특징 딕셔너리 (CLIP, OpenCLIP, DINO 등)
        confidence_threshold: 클러스터 신뢰도 임계값 (기본 0.6)

    Returns:
        {
            'cluster_id': int or -1 (폴백 시),
            'distance': float,
            'confidence': float (0~1),
            'method': 'cluster' or 'fallback',
            'label': str,
            'raw_embedding': numpy array (128D),
            'fallback_reason': str (폴백 시에만)
        }
    """
    # 모델 가져오기
    models = get_cluster_models()
    kmeans_model = models["kmeans_model"]
    centroids = models["centroids"]
    cluster_info = models["cluster_info"]

    # 128D 임베딩 생성
    vec_128 = embed_features(feature_dict).reshape(1, -1)

    # 모든 클러스터 중심까지의 거리 계산
    distances = np.linalg.norm(centroids - vec_128, axis=1)
    nearest_cluster = np.argmin(distances)
    min_distance = float(distances[nearest_cluster])

    # Confidence 계산 (거리 기반, 0~1 범위)
    # 거리가 0이면 confidence 1.0, 거리가 클수록 confidence 감소
    confidence = 1.0 / (1.0 + min_distance)

    # 라벨 가져오기
    if cluster_info and str(nearest_cluster) in cluster_info:
        label = cluster_info[str(nearest_cluster)]
    else:
        label = f"cluster_{nearest_cluster}"

    # Threshold 체크
    if confidence >= confidence_threshold:
        # 클러스터 매칭 성공
        return {
            'cluster_id': int(nearest_cluster),
            'distance': min_distance,
            'confidence': confidence,
            'method': 'cluster',
            'label': label,
            'raw_embedding': vec_128.flatten()
        }
    else:
        # 폴백 모드: 클러스터 없음 (직접 유사도 비교로 전환 필요)
        print(f"⚠️ Cluster confidence low ({confidence:.3f} < {confidence_threshold}), using fallback mode...")

        return {
            'cluster_id': -1,  # 클러스터 없음
            'distance': min_distance,
            'confidence': confidence,
            'method': 'fallback',
            'label': 'unknown_style',
            'raw_embedding': vec_128.flatten(),
            'fallback_reason': f'low_confidence ({confidence:.3f})'
        }


# ---------------------------------------------------------
# [4] 이미지 파일 입력 전용
# ---------------------------------------------------------
def match_cluster_from_image(image_path):

    feat = extract_features_full(image_path)
    if feat is None:
        raise ValueError(f"❌ Feature extraction failed: {image_path}")

    return match_cluster_from_features(feat)


# ---------------------------------------------------------
# [5] 테스트
# ---------------------------------------------------------
if __name__ == "__main__":
    test_img = PROJECT_ROOT / "data" / "test_images" / "test1.jpg"
    result = match_cluster_from_image(str(test_img))

    print("\n🎯 Prediction Result")
    print(result)
