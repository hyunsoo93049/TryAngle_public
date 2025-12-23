# ============================================================
# 🎯 Contrastive Learning Model
# Phase 3-2: PyTorch 기반 대조 학습 모델
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import torchvision.models as models


# ============================================================
# Encoder Network
# ============================================================

class ImageEncoder(nn.Module):
    """
    이미지 인코더 (ResNet50 기반)

    사진 스타일을 인식하기 위한 feature extraction
    """

    def __init__(self, backbone: str = "resnet50", pretrained: bool = True, freeze_backbone: bool = False):
        """
        Args:
            backbone: 백본 모델 ("resnet50", "resnet101", "efficientnet_b0")
            pretrained: ImageNet pretrained weights 사용
            freeze_backbone: 백본 레이어 고정 (transfer learning)
        """
        super().__init__()

        self.backbone_name = backbone

        # 백본 선택
        if backbone == "resnet50":
            resnet = models.resnet50(pretrained=pretrained)
            self.encoder = nn.Sequential(*list(resnet.children())[:-1])  # FC 레이어 제거
            self.feature_dim = 2048

        elif backbone == "resnet101":
            resnet = models.resnet101(pretrained=pretrained)
            self.encoder = nn.Sequential(*list(resnet.children())[:-1])
            self.feature_dim = 2048

        elif backbone == "efficientnet_b0":
            efficientnet = models.efficientnet_b0(pretrained=pretrained)
            self.encoder = nn.Sequential(*list(efficientnet.children())[:-1])
            self.feature_dim = 1280

        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # 백본 고정 옵션
        if freeze_backbone:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) 이미지 텐서

        Returns:
            (B, feature_dim) feature 벡터
        """
        features = self.encoder(x)  # (B, feature_dim, 1, 1)
        features = features.flatten(1)  # (B, feature_dim)
        return features


# ============================================================
# Projection Head
# ============================================================

class ProjectionHead(nn.Module):
    """
    Projection Head (MLP)

    Feature space를 contrastive learning에 적합한 embedding space로 변환
    SimCLR 논문 참고: 2-layer MLP with ReLU
    """

    def __init__(self, input_dim: int = 2048, hidden_dim: int = 512, output_dim: int = 128):
        """
        Args:
            input_dim: 인코더 출력 차원
            hidden_dim: 히든 레이어 차원
            output_dim: 최종 embedding 차원
        """
        super().__init__()

        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_dim) feature 벡터

        Returns:
            (B, output_dim) embedding 벡터
        """
        return self.projection(x)


# ============================================================
# Complete Contrastive Model
# ============================================================

class ContrastiveModel(nn.Module):
    """
    완전한 대조 학습 모델

    Encoder + Projection Head
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        projection_dim: int = 128
    ):
        """
        Args:
            backbone: 백본 모델 이름
            pretrained: ImageNet pretrained weights 사용
            freeze_backbone: 백본 고정 여부
            projection_dim: 최종 embedding 차원
        """
        super().__init__()

        # Encoder
        self.encoder = ImageEncoder(backbone, pretrained, freeze_backbone)

        # Projection Head
        self.projection_head = ProjectionHead(
            input_dim=self.encoder.feature_dim,
            hidden_dim=512,
            output_dim=projection_dim
        )

        self.projection_dim = projection_dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, 3, H, W) 이미지 텐서

        Returns:
            features: (B, feature_dim) encoder 출력
            embeddings: (B, projection_dim) projection head 출력
        """
        features = self.encoder(x)
        embeddings = self.projection_head(features)

        # L2 정규화 (cosine similarity 사용을 위해)
        embeddings = F.normalize(embeddings, dim=1)

        return features, embeddings

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        추론 시 사용: embedding만 반환

        Args:
            x: (B, 3, H, W) 이미지 텐서

        Returns:
            (B, projection_dim) 정규화된 embedding
        """
        _, embeddings = self.forward(x)
        return embeddings


# ============================================================
# Contrastive Loss (InfoNCE / NT-Xent)
# ============================================================

class ContrastiveLoss(nn.Module):
    """
    InfoNCE Loss (Normalized Temperature-scaled Cross Entropy)

    SimCLR, MoCo 등에서 사용되는 대조 학습 손실 함수
    """

    def __init__(self, temperature: float = 0.07):
        """
        Args:
            temperature: 온도 파라미터 (작을수록 hard negative mining)
        """
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings1: torch.Tensor, embeddings2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings1: (B, D) 첫 번째 이미지의 embedding
            embeddings2: (B, D) 두 번째 이미지의 embedding

        Returns:
            loss: scalar 손실 값
        """
        batch_size = embeddings1.shape[0]
        device = embeddings1.device

        # Concatenate embeddings: (2B, D)
        embeddings = torch.cat([embeddings1, embeddings2], dim=0)

        # Compute similarity matrix: (2B, 2B)
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature

        # Create labels (positive pairs)
        # [0, 1], [1, 0], [2, 3], [3, 2], ...
        labels = torch.arange(batch_size, device=device)
        labels = torch.cat([labels + batch_size, labels])  # (2B,)

        # Mask out self-similarity
        mask = torch.eye(2 * batch_size, device=device, dtype=torch.bool)
        similarity_matrix = similarity_matrix.masked_fill(mask, -1e9)

        # Cross-entropy loss
        loss = F.cross_entropy(similarity_matrix, labels)

        return loss


class BinaryContrastiveLoss(nn.Module):
    """
    Binary Contrastive Loss

    Positive pair (label=1): 거리 최소화
    Negative pair (label=0): 거리 최대화
    """

    def __init__(self, margin: float = 1.0):
        """
        Args:
            margin: negative pair의 최소 거리
        """
        super().__init__()
        self.margin = margin

    def forward(self, embeddings1: torch.Tensor, embeddings2: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings1: (B, D) 첫 번째 이미지의 embedding
            embeddings2: (B, D) 두 번째 이미지의 embedding
            labels: (B,) 0 (negative) or 1 (positive)

        Returns:
            loss: scalar 손실 값
        """
        # Euclidean distance
        distances = F.pairwise_distance(embeddings1, embeddings2)

        # Positive loss: minimize distance
        positive_loss = labels * torch.pow(distances, 2)

        # Negative loss: maximize distance (up to margin)
        negative_loss = (1 - labels) * torch.pow(torch.clamp(self.margin - distances, min=0.0), 2)

        # Total loss
        loss = torch.mean(positive_loss + negative_loss)

        return loss


# ============================================================
# 모델 생성 헬퍼 함수
# ============================================================

def create_contrastive_model(
    backbone: str = "resnet50",
    pretrained: bool = True,
    freeze_backbone: bool = False,
    projection_dim: int = 128
) -> ContrastiveModel:
    """
    대조 학습 모델 생성

    Args:
        backbone: 백본 모델 ("resnet50", "resnet101", "efficientnet_b0")
        pretrained: ImageNet pretrained weights 사용
        freeze_backbone: 백본 고정 (transfer learning)
        projection_dim: 최종 embedding 차원

    Returns:
        ContrastiveModel 인스턴스
    """
    model = ContrastiveModel(
        backbone=backbone,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        projection_dim=projection_dim
    )

    return model


# ============================================================
# 테스트
# ============================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🧪 Contrastive Model Test")
    print("="*60)

    # 모델 생성
    model = create_contrastive_model(backbone="resnet50", pretrained=False)

    # 더미 입력
    batch_size = 4
    images = torch.randn(batch_size, 3, 224, 224)

    # Forward pass
    features, embeddings = model(images)

    print(f"\n✅ Model created successfully!")
    print(f"   Backbone: ResNet50")
    print(f"   Features shape: {features.shape}")
    print(f"   Embeddings shape: {embeddings.shape}")

    # Loss 테스트
    loss_fn = ContrastiveLoss(temperature=0.07)

    images1 = torch.randn(batch_size, 3, 224, 224)
    images2 = torch.randn(batch_size, 3, 224, 224)

    _, emb1 = model(images1)
    _, emb2 = model(images2)

    loss = loss_fn(emb1, emb2)

    print(f"\n✅ Loss computed successfully!")
    print(f"   Loss: {loss.item():.4f}")

    # Binary loss 테스트
    binary_loss_fn = BinaryContrastiveLoss(margin=1.0)
    labels = torch.tensor([1, 1, 0, 0])  # positive, positive, negative, negative
    binary_loss = binary_loss_fn(emb1, emb2, labels)

    print(f"\n✅ Binary loss computed successfully!")
    print(f"   Binary Loss: {binary_loss.item():.4f}")

    print("\n" + "="*60)
    print("✅ All tests passed!")
    print("="*60)
