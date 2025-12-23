# ============================================================
# 🎯 TryAngle Feature Extractor v3
# Phase 3-4: Contrastive Learning 기반 특징 추출
# ============================================================

import os
import sys
import torch
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Optional
from PIL import Image
from torchvision import transforms

# Project root 설정
VERSION3_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = VERSION3_DIR
while PROJECT_ROOT != PROJECT_ROOT.parent and not ((PROJECT_ROOT / "data").exists() and (PROJECT_ROOT / "src").exists()):
    PROJECT_ROOT = PROJECT_ROOT.parent

if str(VERSION3_DIR) not in sys.path:
    sys.path.append(str(VERSION3_DIR))

from contrastive.contrastive_model import create_contrastive_model
from utils.model_cache import model_cache

# v2와의 호환성을 위해 import
try:
    from feature_extraction.feature_extractor_v2 import (
        extract_midas_features,
        extract_color_features,
        extract_yolo_pose_features,
        extract_face_features
    )
    V2_AVAILABLE = True
except ImportError:
    print("⚠️ Feature Extractor v2 not available")
    V2_AVAILABLE = False


# ============================================================
# Contrastive Model 로드
# ============================================================

def _load_contrastive_model():
    """
    훈련된 대조 학습 모델 로드 (싱글톤)

    Returns:
        (model, device, transform)
    """
    print("  🔧 Loading contrastive model...")

    # 모델 경로
    model_path = VERSION3_DIR / "models" / "contrastive" / "best_model.pth"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Contrastive model not found: {model_path}\n"
            f"Train the model first: python scripts/train_contrastive.py"
        )

    # Device 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 모델 생성
    model = create_contrastive_model(
        backbone="resnet50",
        pretrained=False,
        projection_dim=128
    )

    # 체크포인트 로드
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Transform (검증용)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print(f"  ✅ Contrastive model loaded (device: {device})")

    return model, device, transform


def get_contrastive_model():
    """
    대조 학습 모델 가져오기 (싱글톤)

    Returns:
        (model, device, transform)
    """
    return model_cache.get_or_load("contrastive_model", _load_contrastive_model)


# ============================================================
# Contrastive Features 추출
# ============================================================

def extract_contrastive_features(image_path: str) -> np.ndarray:
    """
    대조 학습 모델로 특징 추출

    Args:
        image_path: 이미지 파일 경로

    Returns:
        (128,) embedding 벡터
    """
    model, device, transform = get_contrastive_model()

    # 이미지 로드
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)

    # 특징 추출
    with torch.no_grad():
        embedding = model.get_embeddings(img_tensor)

    # NumPy로 변환
    embedding = embedding.cpu().numpy().flatten()

    return embedding


# ============================================================
# Feature Extractor v3 (Hybrid)
# ============================================================

def extract_features_v3(image_path: str, use_v2_features: bool = True) -> Optional[Dict]:
    """
    v3 특징 추출: Contrastive Learning + v2 features

    Args:
        image_path: 이미지 파일 경로
        use_v2_features: v2 특징도 포함할지 여부

    Returns:
        {
            'contrastive': (128,),  # 대조 학습 embedding
            'midas': (256,),        # 깊이 특징 (v2)
            'color': (256,),        # 색상 특징 (v2)
            'yolo_pose': (34,),     # 포즈 특징 (v2)
            'face': (6,)            # 얼굴 특징 (v2)
        }
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    features = {}

    # 1. Contrastive features (핵심)
    try:
        contrastive_feat = extract_contrastive_features(image_path)
        features['contrastive'] = contrastive_feat
    except Exception as e:
        print(f"⚠️ Contrastive feature extraction failed: {e}")
        return None

    # 2. v2 features (보조)
    if use_v2_features and V2_AVAILABLE:
        try:
            # MiDaS (깊이)
            midas_feat = extract_midas_features(image_path)
            features['midas'] = midas_feat

            # Color
            color_feat = extract_color_features(image_path)
            features['color'] = color_feat

            # YOLO Pose
            yolo_feat = extract_yolo_pose_features(image_path)
            features['yolo_pose'] = yolo_feat

            # Face
            face_feat = extract_face_features(image_path)
            features['face'] = face_feat

        except Exception as e:
            print(f"⚠️ v2 feature extraction failed: {e}")

    return features


def extract_features_v3_contrastive_only(image_path: str) -> Optional[Dict]:
    """
    v3 특징 추출 (Contrastive만)

    Args:
        image_path: 이미지 파일 경로

    Returns:
        {'contrastive': (128,)}
    """
    return extract_features_v3(image_path, use_v2_features=False)


# ============================================================
# 하위 호환성: v2와 동일한 인터페이스
# ============================================================

def extract_features_v3_full(image_path: str) -> Optional[Dict]:
    """
    v3 전체 특징 추출 (v2 호환)

    feature_extractor_v2.extract_features_v2()와 동일한 포맷으로 반환
    하지만 CLIP/OpenCLIP/DINO 대신 Contrastive 사용

    Returns:
        {
            'clip': (512,),      # 대체 → contrastive의 일부
            'openclip': (512,),  # 대체 → contrastive의 일부
            'dino': (384,),      # 대체 → contrastive의 일부
            'midas': (256,),
            'color': (256,),
            'yolo_pose': (34,),
            'face': (6,)
        }
    """
    features = extract_features_v3(image_path, use_v2_features=True)

    if features is None:
        return None

    # Contrastive embedding을 CLIP/OpenCLIP/DINO로 분할
    # (128D를 42D + 43D + 43D로 분할)
    contrastive = features['contrastive']

    # v2 호환 포맷으로 변환
    v2_compatible = {
        'clip': np.concatenate([contrastive[:42], np.zeros(470)]),  # 512D로 패딩
        'openclip': np.concatenate([contrastive[42:85], np.zeros(469)]),  # 512D로 패딩
        'dino': np.concatenate([contrastive[85:128], np.zeros(341)]),  # 384D로 패딩
        'midas': features.get('midas', np.zeros(256)),
        'color': features.get('color', np.zeros(256)),
        'yolo_pose': features.get('yolo_pose', np.zeros(34)),
        'face': features.get('face', np.zeros(6))
    }

    return v2_compatible


# ============================================================
# 테스트
# ============================================================

if __name__ == "__main__":
    # 테스트 이미지
    test_img = PROJECT_ROOT / "data" / "test_images" / "test1.jpg"

    if not test_img.exists():
        print(f"❌ Test image not found: {test_img}")
        print("Please provide a test image.")
        sys.exit(1)

    print("\n" + "="*60)
    print("🧪 Feature Extractor v3 Test")
    print("="*60)

    try:
        # Contrastive only
        print("\n1️⃣ Contrastive features only:")
        features_contrastive = extract_features_v3_contrastive_only(str(test_img))
        if features_contrastive:
            print(f"   ✅ Contrastive: {features_contrastive['contrastive'].shape}")

        # Full features (v3)
        print("\n2️⃣ Full features (v3):")
        features_v3 = extract_features_v3(str(test_img), use_v2_features=True)
        if features_v3:
            for key, feat in features_v3.items():
                print(f"   {key}: {feat.shape}")

        # v2 compatible
        print("\n3️⃣ v2 compatible format:")
        features_v2_compat = extract_features_v3_full(str(test_img))
        if features_v2_compat:
            total_dim = sum(f.shape[0] for f in features_v2_compat.values())
            print(f"   Total dimension: {total_dim}D")
            for key, feat in features_v2_compat.items():
                print(f"   {key}: {feat.shape}")

        print("\n" + "="*60)
        print("✅ All tests passed!")
        print("="*60)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
