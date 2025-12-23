# ============================================================
# 🎯 Contrastive Learning Data Preparation
# Phase 3-1: 대조 학습을 위한 데이터셋 준비
# ============================================================

import os
import sys
import json
import random
import shutil
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np

# Project root 설정
VERSION3_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = VERSION3_DIR
while PROJECT_ROOT != PROJECT_ROOT.parent and not ((PROJECT_ROOT / "data").exists() and (PROJECT_ROOT / "src").exists()):
    PROJECT_ROOT = PROJECT_ROOT.parent

if str(VERSION3_DIR) not in sys.path:
    sys.path.append(str(VERSION3_DIR))


# ============================================================
# 데이터 수집
# ============================================================

def collect_clustered_images() -> Dict[int, List[Path]]:
    """
    클러스터링된 이미지 수집

    Returns:
        {cluster_id: [image_path, ...], ...}
    """
    clustered_dir = PROJECT_ROOT / "data" / "clustered_images"

    if not clustered_dir.exists():
        raise FileNotFoundError(f"Clustered images not found: {clustered_dir}")

    cluster_images = {}

    # 각 클러스터 폴더에서 이미지 수집
    for cluster_folder in sorted(clustered_dir.iterdir()):
        if not cluster_folder.is_dir():
            continue

        # cluster_0, cluster_1, ... 형식
        cluster_id = int(cluster_folder.name.split("_")[1])

        # 이미지 파일 수집
        images = list(cluster_folder.glob("*.jpg")) + list(cluster_folder.glob("*.jpeg")) + list(cluster_folder.glob("*.png"))

        if len(images) > 0:
            cluster_images[cluster_id] = images

    return cluster_images


# ============================================================
# Positive/Negative Pair 생성
# ============================================================

def create_positive_pairs(cluster_images: Dict[int, List[Path]], num_pairs_per_cluster: int = 50) -> List[Tuple[Path, Path, int]]:
    """
    Positive pairs 생성 (같은 클러스터 내에서)

    Args:
        cluster_images: {cluster_id: [image_paths]}
        num_pairs_per_cluster: 각 클러스터에서 생성할 pair 수

    Returns:
        [(img1_path, img2_path, label=1), ...]
    """
    positive_pairs = []

    for cluster_id, images in cluster_images.items():
        if len(images) < 2:
            print(f"⚠️ Cluster {cluster_id} has only {len(images)} images, skipping...")
            continue

        # 같은 클러스터 내에서 랜덤 페어링
        for _ in range(num_pairs_per_cluster):
            img1, img2 = random.sample(images, 2)
            positive_pairs.append((img1, img2, 1))  # label=1 (positive)

    return positive_pairs


def create_negative_pairs(cluster_images: Dict[int, List[Path]], num_pairs_per_cluster: int = 50) -> List[Tuple[Path, Path, int]]:
    """
    Negative pairs 생성 (다른 클러스터 간)

    Args:
        cluster_images: {cluster_id: [image_paths]}
        num_pairs_per_cluster: 각 클러스터에서 생성할 pair 수

    Returns:
        [(img1_path, img2_path, label=0), ...]
    """
    negative_pairs = []
    cluster_ids = list(cluster_images.keys())

    for cluster_id in cluster_ids:
        images_from_cluster = cluster_images[cluster_id]

        # 다른 클러스터 선택
        other_clusters = [cid for cid in cluster_ids if cid != cluster_id]

        for _ in range(num_pairs_per_cluster):
            # 현재 클러스터에서 1개
            img1 = random.choice(images_from_cluster)

            # 다른 클러스터에서 1개
            other_cluster = random.choice(other_clusters)
            img2 = random.choice(cluster_images[other_cluster])

            negative_pairs.append((img1, img2, 0))  # label=0 (negative)

    return negative_pairs


# ============================================================
# 데이터셋 저장
# ============================================================

def save_dataset(pairs: List[Tuple[Path, Path, int]], output_dir: Path, split_name: str):
    """
    데이터셋을 디스크에 저장

    Args:
        pairs: [(img1, img2, label), ...]
        output_dir: 저장 디렉토리
        split_name: "train" or "val"
    """
    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    # pairs.json 저장
    pairs_data = []
    for i, (img1, img2, label) in enumerate(pairs):
        pairs_data.append({
            'id': i,
            'img1': str(img1),
            'img2': str(img2),
            'label': label
        })

    pairs_file = split_dir / "pairs.json"
    with open(pairs_file, 'w') as f:
        json.dump(pairs_data, f, indent=2)

    print(f"  ✅ Saved {len(pairs)} pairs to {pairs_file}")


# ============================================================
# 메인 실행
# ============================================================

def prepare_contrastive_dataset(
    output_dir: str = None,
    num_positive_per_cluster: int = 100,
    num_negative_per_cluster: int = 100,
    train_ratio: float = 0.8,
    seed: int = 42
):
    """
    대조 학습 데이터셋 준비

    Args:
        output_dir: 출력 디렉토리 (기본: PROJECT_ROOT/data/contrastive_dataset)
        num_positive_per_cluster: 클러스터당 positive pair 수
        num_negative_per_cluster: 클러스터당 negative pair 수
        train_ratio: 학습/검증 비율
        seed: 랜덤 시드
    """
    random.seed(seed)
    np.random.seed(seed)

    if output_dir is None:
        output_dir = PROJECT_ROOT / "data" / "contrastive_dataset"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"🎯 Contrastive Learning Dataset Preparation")
    print(f"{'='*60}")
    print(f"Output: {output_dir}\n")

    # Step 1: 클러스터 이미지 수집
    print("📂 Step 1: Collecting clustered images...")
    cluster_images = collect_clustered_images()

    total_images = sum(len(imgs) for imgs in cluster_images.values())
    print(f"  ✅ Found {len(cluster_images)} clusters, {total_images} images")

    for cluster_id, images in cluster_images.items():
        print(f"     Cluster {cluster_id}: {len(images)} images")

    # Step 2: Positive pairs 생성
    print(f"\n📌 Step 2: Creating positive pairs (same cluster)...")
    positive_pairs = create_positive_pairs(cluster_images, num_positive_per_cluster)
    print(f"  ✅ Created {len(positive_pairs)} positive pairs")

    # Step 3: Negative pairs 생성
    print(f"\n📌 Step 3: Creating negative pairs (different clusters)...")
    negative_pairs = create_negative_pairs(cluster_images, num_negative_per_cluster)
    print(f"  ✅ Created {len(negative_pairs)} negative pairs")

    # Step 4: Train/Val split
    print(f"\n📊 Step 4: Splitting into train/val ({train_ratio:.0%}/{1-train_ratio:.0%})...")

    all_pairs = positive_pairs + negative_pairs
    random.shuffle(all_pairs)

    split_idx = int(len(all_pairs) * train_ratio)
    train_pairs = all_pairs[:split_idx]
    val_pairs = all_pairs[split_idx:]

    print(f"  ✅ Train: {len(train_pairs)} pairs")
    print(f"  ✅ Val: {len(val_pairs)} pairs")

    # 통계 계산
    train_pos = sum(1 for _, _, label in train_pairs if label == 1)
    train_neg = len(train_pairs) - train_pos
    val_pos = sum(1 for _, _, label in val_pairs if label == 1)
    val_neg = len(val_pairs) - val_pos

    print(f"\n     Train: {train_pos} positive, {train_neg} negative")
    print(f"     Val:   {val_pos} positive, {val_neg} negative")

    # Step 5: 저장
    print(f"\n💾 Step 5: Saving dataset...")
    save_dataset(train_pairs, output_dir, "train")
    save_dataset(val_pairs, output_dir, "val")

    # 메타데이터 저장
    metadata = {
        'num_clusters': len(cluster_images),
        'total_images': total_images,
        'num_positive_per_cluster': num_positive_per_cluster,
        'num_negative_per_cluster': num_negative_per_cluster,
        'train_ratio': train_ratio,
        'seed': seed,
        'train_size': len(train_pairs),
        'val_size': len(val_pairs),
        'train_positive': train_pos,
        'train_negative': train_neg,
        'val_positive': val_pos,
        'val_negative': val_neg
    }

    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"  ✅ Saved metadata to {metadata_file}")

    # 완료
    print(f"\n{'='*60}")
    print(f"✅ Dataset Preparation Complete!")
    print(f"{'='*60}")
    print(f"\n📂 Dataset location: {output_dir}")
    print(f"   train/pairs.json: {len(train_pairs)} pairs")
    print(f"   val/pairs.json: {len(val_pairs)} pairs")
    print(f"\n💡 Next steps:")
    print(f"   1. Train contrastive model: python scripts/train_contrastive.py")
    print(f"   2. Evaluate model performance")
    print(f"   3. Use trained model for feature extraction (v3)")

    return {
        'train_pairs': train_pairs,
        'val_pairs': val_pairs,
        'metadata': metadata
    }


# ============================================================
# 실행
# ============================================================

if __name__ == "__main__":
    try:
        result = prepare_contrastive_dataset(
            num_positive_per_cluster=100,
            num_negative_per_cluster=100,
            train_ratio=0.8
        )
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
