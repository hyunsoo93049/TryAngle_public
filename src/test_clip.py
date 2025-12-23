# clip_reference_comparator.py
# ✅ 레퍼런스 이미지 1장을 CLIP으로 임베딩해 저장하고,
#    실시간 카메라 프레임과 스타일/구도 유사도를 비교하는 코드

import cv2
import torch
import clip
import numpy as np
from PIL import Image
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
while PROJECT_ROOT != PROJECT_ROOT.parent and not ((PROJECT_ROOT / "data").exists() and (PROJECT_ROOT / "src").exists()):
    PROJECT_ROOT = PROJECT_ROOT.parent

# 🔸 CLIP 모델 로드 (ViT-B/32 사용)
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# 🔸 레퍼런스 이미지 로딩 및 전처리
REFERENCE_PATH = PROJECT_ROOT / "data" / "sample_images" / "test1.jpg"
ref_img_pil = Image.open(REFERENCE_PATH).convert("RGB")
ref_preprocessed = preprocess(ref_img_pil).unsqueeze(0).to(device)

# 🔸 레퍼런스 임베딩 벡터 추출 (한 번만 수행)
with torch.no_grad():
    reference_embedding = model.encode_image(ref_preprocessed)
    reference_embedding /= reference_embedding.norm(dim=-1, keepdim=True)

print("✅ 레퍼런스 이미지 임베딩 완료. 실시간 유사도 비교 시작.")

# 🔸 웹캠 실행
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 🔸 현재 프레임 → PIL → CLIP 입력 이미지 전처리
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_pil = Image.fromarray(frame_rgb)
    frame_tensor = preprocess(frame_pil).unsqueeze(0).to(device)

    # 🔸 현재 프레임 임베딩 계산
    with torch.no_grad():
        live_embedding = model.encode_image(frame_tensor)
        live_embedding /= live_embedding.norm(dim=-1, keepdim=True)

    # 🔸 코사인 유사도 계산
    similarity = (live_embedding @ reference_embedding.T).item()
    similarity_percent = int(similarity * 100)

    # 🔸 유사도 점수 표시
    cv2.putText(frame, f"CLIP Similarity: {similarity_percent}%", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    # 🔸 화면 출력
    cv2.imshow("CLIP Style Similarity", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
