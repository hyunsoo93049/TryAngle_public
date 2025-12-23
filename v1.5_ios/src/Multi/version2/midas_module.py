# midas_module.py
# -------------------------------------------------------------------
# MiDaS Depth Estimator - Windows/Inference 버전
# 설치 없이 torch.hub로 가중치 로드 → 깊이맵/시점 차이 추정
# -------------------------------------------------------------------

from typing import Optional, Tuple
import torch
import numpy as np
import cv2

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("🔹 Loading MiDaS (DPT_Large) via torch.hub...")
_midas: Optional[torch.nn.Module] = None
_transform = None  # lazy load (hub에서 transforms 객체를 로드해야 함)

def _ensure_midas() -> Tuple[torch.nn.Module, callable]:
    global _midas, _transform
    if _midas is None:
        _midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
        _midas.eval().to(_DEVICE)
    if _transform is None:
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        _transform = transforms.dpt_transform
    return _midas, _transform

@torch.no_grad()
def estimate_depth(image_path: str) -> np.ndarray:
    """이미지 경로 입력 → 깊이 맵(depth map, float32 ndarray) 반환."""
    midas, transform = _ensure_midas()
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    input_batch = transform(img_rgb).to(_DEVICE)  # 1x3xHxW, normalized
    pred = midas(input_batch)
    depth = pred.squeeze().detach().cpu().numpy().astype("float32")
    # scale-invariant 특성 → 시각/비교용으로 min-max 정규화(0~1)
    dmin, dmax = float(depth.min()), float(depth.max())
    if dmax > dmin:
        depth_norm = (depth - dmin) / (dmax - dmin)
    else:
        depth_norm = np.zeros_like(depth, dtype="float32")
    return depth_norm

@torch.no_grad()
def camera_height_diff(ref_path: str, tgt_path: str) -> float:
    """
    카메라 '상대적 높낮이' 차이를 추정하는 간단 지표 [-1, 1].
    - 아이디어: 프레임 상단/하단의 평균 깊이 분포를 비교해 카메라 고저 판단
      (하이앵글일수록 상단이 더 가깝게(얕게) 예측되는 경향 등)
    """
    d1 = estimate_depth(ref_path)
    d2 = estimate_depth(tgt_path)

    # 위/아래 반띵 영역 평균
    h = d1.shape[0]
    top1, bot1 = d1[: h//2, :].mean(), d1[h//2 :, :].mean()
    top2, bot2 = d2[: h//2, :].mean(), d2[h//2 :, :].mean()

    # '상-하' 대비(참조)와 '상-하' 대비(타겟)의 차이
    # 값이 +면 타겟 카메라가 더 '하이앵글' 경향, -면 '로우앵글' 경향
    ref_contrast = top1 - bot1
    tgt_contrast = top2 - bot2
    diff = tgt_contrast - ref_contrast

    # 안전 정규화 (경험적 스케일)
    # 대체로 -0.5~0.5 안에 들어옴 → [-1,1]로 클램프
    score = max(-1.0, min(1.0, diff * 2.0))
    return float(score)
