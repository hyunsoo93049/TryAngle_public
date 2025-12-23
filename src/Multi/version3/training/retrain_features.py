# ============================================================
# 🔄 TryAngle Feature Re-extraction v2
# 2700장 이미지에서 version2 특징 추출
# ============================================================

import os
import sys
import numpy as np
import polars as pl
from tqdm import tqdm
from pathlib import Path

# feature_extractor_v2 import
# training 폴더에서 상위로 가서 feature_extraction으로
VERSION3_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = VERSION3_DIR
while PROJECT_ROOT != PROJECT_ROOT.parent and not ((PROJECT_ROOT / "data").exists() and (PROJECT_ROOT / "src").exists()):
    PROJECT_ROOT = PROJECT_ROOT.parent

if str(VERSION3_DIR) not in sys.path:
    sys.path.append(str(VERSION3_DIR))

from feature_extraction.feature_extractor_v2 import extract_features_v2

# ============================================================
# 경로 설정
# ============================================================
IMG_DIR = PROJECT_ROOT / "data" / "train_images"
EXISTING_PARQUET = PROJECT_ROOT / "feature_models" / "features" / "fusion_final_with_openclip.parquet"
OUTPUT_PARQUET = PROJECT_ROOT / "feature_models" / "features" / "fusion_features_v2.parquet"

# ============================================================
# Main
# ============================================================
def main():
    print("="*60)
    print("🔄 TryAngle Feature Re-extraction v2")
    print("="*60)
    
    # --------------------------------------------------------
    # Step 1: 기존 parquet에서 파일명 리스트 로드
    # --------------------------------------------------------
    print("\n📂 Loading existing parquet for filename list...")
    df_old = pl.read_parquet(EXISTING_PARQUET)
    filenames = df_old["filename"].to_list()
    
    print(f"✅ Found {len(filenames)} images")
    
    # --------------------------------------------------------
    # Step 2: 각 이미지에서 v2 특징 추출
    # --------------------------------------------------------
    print("\n🔧 Extracting features v2...")
    
    results = []
    failed_count = 0
    
    for filename in tqdm(filenames, desc="Processing"):
        img_path = IMG_DIR / filename
        
        if not img_path.exists():
            print(f"\n⚠️ Image not found: {filename}")
            failed_count += 1
            continue
        
        try:
            feat = extract_features_v2(str(img_path))
            
            if feat is None:
                print(f"\n❌ Feature extraction failed: {filename}")
                failed_count += 1
                continue
            
            # 1D로 flatten
            results.append({
                "filename": filename,
                "clip": feat["clip"],
                "openclip": feat["openclip"],
                "dino": feat["dino"],
                "midas": feat["midas"],
                "color": feat["color"],
                "yolo_pose": feat["yolo_pose"],
                "face": feat["face"],
            })
            
        except Exception as e:
            print(f"\n❌ Error processing {filename}: {e}")
            failed_count += 1
            continue
    
    print(f"\n✅ Extracted {len(results)} / {len(filenames)} images")
    print(f"❌ Failed: {failed_count}")
    
    if len(results) == 0:
        print("❌ No features extracted! Exiting...")
        return
    
    # --------------------------------------------------------
    # Step 3: Polars DataFrame 생성
    # --------------------------------------------------------
    print("\n📊 Creating Polars DataFrame...")
    
    df_data = {
        "filename": [r["filename"] for r in results],
        "clip": [r["clip"] for r in results],
        "openclip": [r["openclip"] for r in results],
        "dino": [r["dino"] for r in results],
        "midas": [r["midas"] for r in results],
        "color": [r["color"] for r in results],
        "yolo_pose": [r["yolo_pose"] for r in results],
        "face": [r["face"] for r in results],
    }
    
    df = pl.DataFrame(df_data)
    
    print(f"✅ DataFrame shape: {df.shape}")
    print(f"   Columns: {df.columns}")
    
    # --------------------------------------------------------
    # Step 4: Parquet 저장
    # --------------------------------------------------------
    print(f"\n💾 Saving to: {OUTPUT_PARQUET}")
    
    os.makedirs(OUTPUT_PARQUET.parent, exist_ok=True)
    df.write_parquet(OUTPUT_PARQUET)
    
    print("✅ Parquet saved successfully!")
    
    # --------------------------------------------------------
    # Step 5: 검증
    # --------------------------------------------------------
    print("\n🔍 Verification...")
    df_verify = pl.read_parquet(OUTPUT_PARQUET)
    
    print(f"   Loaded shape: {df_verify.shape}")
    print(f"   Sample filename: {df_verify['filename'][0]}")
    print(f"   CLIP shape: {df_verify['clip'][0].shape}")
    print(f"   OpenCLIP shape: {df_verify['openclip'][0].shape}")
    print(f"   DINO shape: {df_verify['dino'][0].shape}")
    print(f"   MiDaS shape: {df_verify['midas'][0].shape}")
    print(f"   Color shape: {df_verify['color'][0].shape}")
    print(f"   YOLO-Pose shape: {df_verify['yolo_pose'][0].shape}")
    print(f"   Face shape: {df_verify['face'][0].shape}")
    
    total_dim = (
        df_verify['clip'][0].shape[0] +
        df_verify['openclip'][0].shape[0] +
        df_verify['dino'][0].shape[0] +
        df_verify['midas'][0].shape[0] +
        df_verify['color'][0].shape[0] +
        df_verify['yolo_pose'][0].shape[0] +
        df_verify['face'][0].shape[0]
    )
    
    print(f"\n📊 Total Feature Dimensions: {total_dim}D")
    print(f"   Expected: 1600D (512+512+384+20+150+15+7)")
    
    if total_dim == 1600:
        print("✅ Dimension check passed!")
    else:
        print(f"⚠️ Dimension mismatch! Expected 1600, got {total_dim}")
    
    print("\n" + "="*60)
    print("🎉 Feature Re-extraction Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
