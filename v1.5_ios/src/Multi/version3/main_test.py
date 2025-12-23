import os
from pathlib import Path
from feature_extraction.feature_extractor import extract_features_full
from embedder.embedder import embed_features
from matching.cluster_matcher import match_cluster_from_features

PROJECT_ROOT = Path(__file__).resolve().parent
while PROJECT_ROOT != PROJECT_ROOT.parent and not ((PROJECT_ROOT / "data").exists() and (PROJECT_ROOT / "src").exists()):
    PROJECT_ROOT = PROJECT_ROOT.parent


def main():
    # -------------------------------------------------------
    # ① 자동 테스트 이미지 경로 (원하는대로 변경 가능)
    # -------------------------------------------------------
    img_path = PROJECT_ROOT / "data" / "test_images" / "test1.jpg"

    print(f"\n📌 자동 테스트 이미지: {img_path}")

    if not img_path.exists():
        raise FileNotFoundError(f"❌ Image not found: {img_path}")

    # -------------------------------------------------------
    # ② Feature 추출
    # -------------------------------------------------------
    print("\n🔧 Step 1: Extracting features...")
    feat = extract_features_full(str(img_path))
    if feat is None:
        raise RuntimeError("❌ Feature extraction failed!")

    # -------------------------------------------------------
    # ③ 128D 임베딩 생성
    # -------------------------------------------------------
    print("\n🔧 Step 2: Embedding to 128D...")
    vec128 = embed_features(feat)
    print("   → shape:", vec128.shape)

    # -------------------------------------------------------
    # ④ 클러스터 예측
    # -------------------------------------------------------
    print("\n🔍 Step 3: Predict cluster...")
    result = match_cluster_from_features(feat)

    # -------------------------------------------------------
    # ⑤ 출력
    # -------------------------------------------------------
    print("\n==============================")
    print(f"🎯 Predicted Cluster : {result['cluster_id']}")
    print(f"📏 Distance          : {result['distance']:.4f}")
    print(f"🏷 Label             : {result['label']}")
    print("==============================\n")

    print("RAW Embedding (first 10 dims):")
    print(result["raw_embedding"][:10])


if __name__ == "__main__":
    main()
