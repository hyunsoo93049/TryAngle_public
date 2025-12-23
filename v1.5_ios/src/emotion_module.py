# emotion_module.py
# -----------------------------------------------------------------------------
# CLIP을 이용해 "참조 이미지 vs 현재 프레임"의 시각적 임베딩 유사도를 계산
# - ref 이미지 1장 기준 cosine similarity(0~100 스케일) 반환
# - 사전학습 모델이므로 "정확히 어떤 감성"을 보장하진 않음
#   → 내부 기준(레퍼런스 셋 확장, 텍스트 프롬프트 결합 등)으로 점차 고도화
# -----------------------------------------------------------------------------

from typing import Optional
import torch
from PIL import Image
import numpy as np

try:
    import clip  # openai/CLIP
except ImportError:
    clip = None  # 선택 모듈이므로 없을 수도 있음

class EmotionAnalyzer:
    """
    CLIP 임베딩 기반의 유사도 계산기.
    - compare_to_reference(ref_path, target_image): 0~100 유사도 점수
    """

    def __init__(self, device: Optional[str] = None):
        if clip is None:
            raise ImportError("clip 패키지가 설치되어 있지 않습니다. pip install git+https://github.com/openai/CLIP.git")

        # 디바이스 자동 선택
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # ViT-B/32 모델 로드(속도/정확도 균형)
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

    def _encode_image(self, img_pil: Image.Image) -> torch.Tensor:
        """PIL 이미지를 받아 CLIP 이미지 임베딩 반환."""
        with torch.no_grad():
            x = self.preprocess(img_pil).unsqueeze(0).to(self.device)
            feat = self.model.encode_image(x)
            feat = feat / feat.norm(dim=-1, keepdim=True)  # 정규화
        return feat

    def compare_to_reference(self, ref_path: str, target_image: np.ndarray) -> float:
        """
        ref 이미지 파일 경로와 현재 프레임(NumPy HxWxC) 간 임베딩 유사도(%) 계산.
        - 반환: 0~100 스케일(소수점 2자리 반올림)
        """
        # 입력 변환
        ref_pil = Image.open(ref_path).convert("RGB")
        tgt_pil = Image.fromarray(target_image[:, :, ::-1]).convert("RGB") \
            if target_image.shape[2] == 3 else Image.fromarray(target_image)

        # 임베딩 추출
        ref_feat = self._encode_image(ref_pil)
        tgt_feat = self._encode_image(tgt_pil)

        # cosine similarity → [0, 1] → [%]
        sim = torch.cosine_similarity(ref_feat, tgt_feat).item()
        return round(sim * 100.0, 2)
