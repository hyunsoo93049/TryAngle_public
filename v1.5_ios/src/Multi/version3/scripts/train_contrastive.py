# ============================================================
# 🎓 Contrastive Learning Training Script
# Phase 3-3: 대조 학습 모델 훈련
# ============================================================

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Tuple
import numpy as np

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# Project root 설정
VERSION3_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = VERSION3_DIR
while PROJECT_ROOT != PROJECT_ROOT.parent and not ((PROJECT_ROOT / "data").exists() and (PROJECT_ROOT / "src").exists()):
    PROJECT_ROOT = PROJECT_ROOT.parent

if str(VERSION3_DIR) not in sys.path:
    sys.path.append(str(VERSION3_DIR))

from contrastive.contrastive_model import create_contrastive_model, BinaryContrastiveLoss


# ============================================================
# Dataset
# ============================================================

class ContrastiveDataset(Dataset):
    """
    대조 학습 데이터셋

    pairs.json에서 이미지 쌍을 로드하고 augmentation 적용
    """

    def __init__(self, pairs_json_path: str, transform=None):
        """
        Args:
            pairs_json_path: pairs.json 파일 경로
            transform: 이미지 변환 (augmentation)
        """
        with open(pairs_json_path, 'r') as f:
            self.pairs = json.load(f)

        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]

        # 이미지 로드
        img1 = Image.open(pair['img1']).convert('RGB')
        img2 = Image.open(pair['img2']).convert('RGB')

        # Transform 적용
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        label = pair['label']

        return img1, img2, label


# ============================================================
# Data Augmentation
# ============================================================

def get_train_transform(image_size: int = 224):
    """
    학습용 이미지 augmentation

    Args:
        image_size: 입력 이미지 크기

    Returns:
        torchvision transforms
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_val_transform(image_size: int = 224):
    """
    검증용 이미지 transform (augmentation 없음)

    Args:
        image_size: 입력 이미지 크기

    Returns:
        torchvision transforms
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


# ============================================================
# Training Loop
# ============================================================

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int
) -> Dict:
    """
    1 epoch 학습

    Args:
        model: 대조 학습 모델
        dataloader: 학습 데이터로더
        criterion: 손실 함수
        optimizer: 옵티마이저
        device: 디바이스 (cuda/cpu)
        epoch: 현재 epoch

    Returns:
        학습 통계
    """
    model.train()

    total_loss = 0.0
    num_batches = 0

    start_time = time.time()

    for batch_idx, (img1, img2, labels) in enumerate(dataloader):
        # Move to device
        img1 = img1.to(device)
        img2 = img2.to(device)
        labels = labels.float().to(device)

        # Forward pass
        _, emb1 = model(img1)
        _, emb2 = model(img2)

        # Compute loss
        loss = criterion(emb1, emb2, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Statistics
        total_loss += loss.item()
        num_batches += 1

        # Progress
        if (batch_idx + 1) % 10 == 0:
            print(f"   [{batch_idx + 1}/{len(dataloader)}] Loss: {loss.item():.4f}")

    elapsed = time.time() - start_time
    avg_loss = total_loss / num_batches

    return {
        'loss': avg_loss,
        'time': elapsed
    }


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict:
    """
    검증

    Args:
        model: 대조 학습 모델
        dataloader: 검증 데이터로더
        criterion: 손실 함수
        device: 디바이스

    Returns:
        검증 통계
    """
    model.eval()

    total_loss = 0.0
    num_batches = 0

    # 정확도 계산을 위한 변수
    correct_positive = 0
    total_positive = 0
    correct_negative = 0
    total_negative = 0

    with torch.no_grad():
        for img1, img2, labels in dataloader:
            # Move to device
            img1 = img1.to(device)
            img2 = img2.to(device)
            labels = labels.float().to(device)

            # Forward pass
            _, emb1 = model(img1)
            _, emb2 = model(img2)

            # Compute loss
            loss = criterion(emb1, emb2, labels)

            total_loss += loss.item()
            num_batches += 1

            # Accuracy: 거리 기반 분류
            distances = torch.pairwise_distance(emb1, emb2, p=2)
            predictions = (distances < 0.5).float()  # 0.5 threshold

            # Positive/Negative 정확도
            positive_mask = (labels == 1)
            negative_mask = (labels == 0)

            if positive_mask.sum() > 0:
                correct_positive += (predictions[positive_mask] == 1).sum().item()
                total_positive += positive_mask.sum().item()

            if negative_mask.sum() > 0:
                correct_negative += (predictions[negative_mask] == 0).sum().item()
                total_negative += negative_mask.sum().item()

    avg_loss = total_loss / num_batches
    positive_acc = correct_positive / total_positive if total_positive > 0 else 0
    negative_acc = correct_negative / total_negative if total_negative > 0 else 0
    overall_acc = (correct_positive + correct_negative) / (total_positive + total_negative)

    return {
        'loss': avg_loss,
        'positive_acc': positive_acc,
        'negative_acc': negative_acc,
        'overall_acc': overall_acc
    }


# ============================================================
# Main Training Function
# ============================================================

def train_contrastive_model(
    dataset_dir: str,
    output_dir: str,
    backbone: str = "resnet50",
    batch_size: int = 32,
    num_epochs: int = 50,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    freeze_backbone: bool = False,
    device: str = "cuda"
):
    """
    대조 학습 모델 훈련

    Args:
        dataset_dir: 데이터셋 디렉토리 (train/pairs.json, val/pairs.json 포함)
        output_dir: 모델 체크포인트 저장 디렉토리
        backbone: 백본 모델 ("resnet50", "resnet101", "efficientnet_b0")
        batch_size: 배치 사이즈
        num_epochs: 학습 epoch 수
        learning_rate: 학습률
        weight_decay: Weight decay (L2 regularization)
        freeze_backbone: 백본 고정 여부
        device: 디바이스 ("cuda" or "cpu")
    """
    print(f"\n{'='*60}")
    print(f"🎓 Contrastive Learning Training")
    print(f"{'='*60}")
    print(f"Dataset: {dataset_dir}")
    print(f"Output: {output_dir}")
    print(f"Backbone: {backbone}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {num_epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Device: {device}\n")

    # Device 설정
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # 출력 디렉토리 생성
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Dataset 로드
    print("📂 Loading datasets...")
    train_dataset = ContrastiveDataset(
        pairs_json_path=Path(dataset_dir) / "train" / "pairs.json",
        transform=get_train_transform()
    )
    val_dataset = ContrastiveDataset(
        pairs_json_path=Path(dataset_dir) / "val" / "pairs.json",
        transform=get_val_transform()
    )

    print(f"  ✅ Train: {len(train_dataset)} pairs")
    print(f"  ✅ Val: {len(val_dataset)} pairs")

    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Windows compatibility
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Windows compatibility
        pin_memory=True
    )

    # 모델 생성
    print(f"\n🔧 Creating model...")
    model = create_contrastive_model(
        backbone=backbone,
        pretrained=True,
        freeze_backbone=freeze_backbone,
        projection_dim=128
    )
    model = model.to(device)
    print(f"  ✅ Model created: {backbone}")

    # Loss & Optimizer
    criterion = BinaryContrastiveLoss(margin=1.0)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_positive_acc': [],
        'val_negative_acc': [],
        'val_overall_acc': []
    }

    best_val_loss = float('inf')

    # Training loop
    print(f"\n{'='*60}")
    print(f"🚀 Training started...")
    print(f"{'='*60}\n")

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 60)

        # Train
        train_stats = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        print(f"  Train Loss: {train_stats['loss']:.4f} ({train_stats['time']:.1f}s)")

        # Validate
        val_stats = validate(model, val_loader, criterion, device)
        print(f"  Val Loss: {val_stats['loss']:.4f}")
        print(f"  Val Accuracy: {val_stats['overall_acc']:.2%} (Pos: {val_stats['positive_acc']:.2%}, Neg: {val_stats['negative_acc']:.2%})")

        # Update scheduler
        scheduler.step()

        # Save history
        history['train_loss'].append(train_stats['loss'])
        history['val_loss'].append(val_stats['loss'])
        history['val_positive_acc'].append(val_stats['positive_acc'])
        history['val_negative_acc'].append(val_stats['negative_acc'])
        history['val_overall_acc'].append(val_stats['overall_acc'])

        # Save best model
        if val_stats['loss'] < best_val_loss:
            best_val_loss = val_stats['loss']
            checkpoint_path = output_dir / "best_model.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_stats['loss'],
                'val_acc': val_stats['overall_acc']
            }, checkpoint_path)
            print(f"  💾 Best model saved! (val_loss: {best_val_loss:.4f})")

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch + 1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)
            print(f"  💾 Checkpoint saved: epoch {epoch + 1}")

        print()

    # Save final model
    final_path = output_dir / "final_model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'history': history
    }, final_path)

    # Save history
    history_path = output_dir / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*60}")
    print(f"✅ Training Complete!")
    print(f"{'='*60}")
    print(f"\n💾 Models saved to: {output_dir}")
    print(f"   best_model.pth: Best validation loss ({best_val_loss:.4f})")
    print(f"   final_model.pth: Final model")
    print(f"   training_history.json: Training metrics")

    return model, history


# ============================================================
# 실행
# ============================================================

if __name__ == "__main__":
    # 설정
    dataset_dir = PROJECT_ROOT / "data" / "contrastive_dataset"
    output_dir = VERSION3_DIR / "models" / "contrastive"

    if not dataset_dir.exists():
        print(f"❌ Dataset not found: {dataset_dir}")
        print("Run prepare_contrastive_data.py first!")
        sys.exit(1)

    try:
        model, history = train_contrastive_model(
            dataset_dir=str(dataset_dir),
            output_dir=str(output_dir),
            backbone="resnet50",
            batch_size=32,
            num_epochs=50,
            learning_rate=1e-4,
            weight_decay=1e-4,
            freeze_backbone=False,
            device="cuda"
        )

    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
