# dino_module.py
# ---------------------------------------------------------------------
# DINOv2 Tri-Path Feature Extractor
#   - Global: 전체 이미지 특징
#   - Person: YOLO bbox 인물 crop 특징
#   - Background: 인물 영역을 마스크로 제거한 배경 특징
#   - 최종 유사도 = α*Global + β*Person + γ*Background
# ---------------------------------------------------------------------

import torch
from PIL import Image, ImageDraw
import torchvision.transforms as T
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
while PROJECT_ROOT != PROJECT_ROOT.parent and not ((PROJECT_ROOT / "data").exists() and (PROJECT_ROOT / "src").exists()):
    PROJECT_ROOT = PROJECT_ROOT.parent

print("🔹 Loading DINOv2 model (ViT-S/14) via torch.hub...")
model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", pretrained=True)
model.eval()

# 이미지 전처리 파이프라인
transform = T.Compose([
    T.Resize((518, 518)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

# ---------------------------------------------------
# 유틸: 인물 영역 마스크 제거 (배경용)
# ---------------------------------------------------
def remove_person_region(img: Image.Image, bbox):
    bg = img.copy()
    draw = ImageDraw.Draw(bg)
    if bbox is not None:
        # 인물 영역 검게 덮기
        x1, y1, x2, y2 = [float(x) for x in bbox]
        draw.rectangle((x1, y1, x2, y2), fill=(0, 0, 0))
    return bg

# ---------------------------------------------------
# Feature 추출 함수 (Global / Person / Background)
# ---------------------------------------------------
@torch.no_grad()
def extract_dino_features(image_path: str, bbox=None):
    img = Image.open(image_path).convert("RGB")

    # Global Feature
    x_global = transform(img).unsqueeze(0)
    feat_global = model(x_global)
    feat_global = feat_global / feat_global.norm(dim=-1, keepdim=True)

    # Person Feature
    if bbox is not None:
        x1, y1, x2, y2 = [float(x) for x in bbox]
        img_crop = img.crop((x1, y1, x2, y2))
        x_person = transform(img_crop).unsqueeze(0)
        feat_person = model(x_person)
        feat_person = feat_person / feat_person.norm(dim=-1, keepdim=True)
    else:
        feat_person = feat_global

    # Background Feature (인물 영역 제거 후)
    img_bg = remove_person_region(img, bbox)
    x_bg = transform(img_bg).unsqueeze(0)
    feat_bg = model(x_bg)
    feat_bg = feat_bg / feat_bg.norm(dim=-1, keepdim=True)

    return feat_global, feat_person, feat_bg


# ---------------------------------------------------
# Tri-Path 유사도 계산
# ---------------------------------------------------
@torch.no_grad()
def dino_similarity(path1: str, path2: str,
                    bbox1=None, bbox2=None,
                    alpha: float = 0.2,  # Global weight
                    beta: float = 0.6,   # Person weight
                    gamma: float = 0.2) -> float:
    """
    DINO Tri-Path Similarity
    - path1: 레퍼런스 이미지
    - path2: 비교 이미지
    - bbox1, bbox2: YOLO 인물 bbox
    - alpha,beta,gamma: 가중치 (합=1)
    """

    # feature 추출
    f1_g, f1_p, f1_b = extract_dino_features(path1, bbox1)
    f2_g, f2_p, f2_b = extract_dino_features(path2, bbox2)

    # cosine similarity
    sim_global = torch.cosine_similarity(f1_g, f2_g).item()
    sim_person = torch.cosine_similarity(f1_p, f2_p).item()
    sim_bg = torch.cosine_similarity(f1_b, f2_b).item()

    # weighted sum
    final_sim = alpha * sim_global + beta * sim_person + gamma * sim_bg

    print(f"   DINO(Global): {sim_global:.3f} | Person: {sim_person:.3f} | BG: {sim_bg:.3f}")

    return float(final_sim)


# ---------------------------------------------------
# 단독 테스트용
# ---------------------------------------------------
if __name__ == "__main__":
    ref = PROJECT_ROOT / "data" / "sample_images" / "cafe1.jpg"
    tgt = PROJECT_ROOT / "data" / "sample_images" / "cafe5.jpg"
    sim = dino_similarity(str(ref), str(tgt))
    print(f"DINO Tri-Path Similarity: {sim:.3f}")
