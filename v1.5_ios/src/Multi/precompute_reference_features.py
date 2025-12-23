"""
precompute_reference_features.py
- 레퍼런스 이미지의 특성을 '한 번만' 계산해 저장하는 스크립트
- 저장물: .npz (CLIP/DINO 임베딩, MiDaS depth 요약, 선택: 구도요약)
- 추후 실시간 비교 단계에서는 이 .npz를 불러 '비교만' 수행 → 프레임 끊김 최소화

[권장 설치]
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install opencv-python pillow numpy
pip install git+https://github.com/openai/CLIP.git
pip install timm
# (선택) YOLO 포즈 & 구도요약용
pip install ultralytics
# (선택) MiDaS(깊이) 모델 가중치 자동 다운로드
# torch.hub가 인터넷에서 weights를 받습니다.
"""

import os
import json
import time
import argparse
from typing import Optional, Dict, Any, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

# --------- CLIP ----------
import clip

# --------- DINO(v2 via timm pooling 대체) ----------
import timm
import torch.nn.functional as F

# --------- (선택) YOLO pose + 구도요약 ----------
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False

# --------- (선택) MiDaS depth ----------
# torch.hub에서 모델을 받아옵니다(최초 1회 인터넷 필요)
def _load_midas(device: str):
    midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")  # 속도/정확도 균형
    midas.to(device)
    midas.eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.dpt_transform
    return midas, transform

# --------- 유틸 ----------
def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _to_tensor(img_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

def _l2norm(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, dim=-1)

# --------- CLIP ----------
def compute_clip_embedding(model, preprocess, device, img_bgr: np.ndarray) -> np.ndarray:
    pil = _to_tensor(img_bgr)
    with torch.no_grad():
        image = preprocess(pil).unsqueeze(0).to(device)
        feat = model.encode_image(image)
        feat = _l2norm(feat)
    return feat.squeeze(0).float().cpu().numpy()

# --------- DINO ----------
def load_dino_model(device: str):
    """
    timm의 DINO 계열 모델을 로드하고 GAP로 전역 임베딩을 만듭니다.
    - 대안: facebookresearch/dinov2 hub 사용 가능(인터넷 필요)
    """
    model_name = "vit_small_patch14_dinov2"  # 가벼운 편
    model = timm.create_model(model_name, pretrained=True)
    model.eval().to(device)
    # 특징 추출용: 마지막 분류기 전에 global average pooling 사용
    return model, model_name

def compute_dino_embedding(dino_model, device, img_bgr: np.ndarray) -> np.ndarray:
    # timm 기본 전처리
    cfg = timm.data.resolve_model_data_config(dino_model)
    tfm = timm.data.create_transform(**cfg, is_training=False)
    img = _to_tensor(img_bgr)
    x = tfm(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feats = dino_model.forward_features(x)
        # timm vit forward_features 결과에 따라 키가 다를 수 있음 → 풀링 처리
        if isinstance(feats, dict) and "x_norm_clstoken" in feats:
            vec = feats["x_norm_clstoken"]  # [B, C]
        elif isinstance(feats, dict) and "pool" in feats:
            vec = feats["pool"]             # [B, C]
        else:
            # 최후의 보루: GAP
            if isinstance(feats, (list, tuple)):
                feats = feats[-1]
            vec = feats.mean(dim=(2, 3)) if feats.dim() == 4 else feats
    vec = _l2norm(vec)
    return vec.squeeze(0).float().cpu().numpy()

# --------- YOLO Pose & 구도요약(선택) ----------
def summarize_composition_with_yolo(image_bgr: np.ndarray) -> Dict[str, Any]:
    """
    YOLOv8 pose로 사람 keypoints와 bbox를 얻고,
    간단한 구도 지표를 요약해서 반환합니다.
    (분석 속도를 위해 최소한만 계산)
    """
    if not YOLO_AVAILABLE:
        return {"enabled": False, "reason": "ultralytics(YOLO) 미설치"}

    pose = YOLO("yolov8s-pose.pt")
    res = pose(image_bgr)
    r0 = res[0]

    if r0.keypoints is None or r0.boxes is None or len(r0.keypoints) == 0:
        return {"enabled": True, "detected": False}

    kpts = r0.keypoints.xy[0].cpu().numpy()  # (K,2)
    bbox = r0.boxes.xyxy[0].cpu().numpy()    # (x1,y1,x2,y2)
    h, w = image_bgr.shape[:2]

    cx, cy = np.mean(kpts, axis=0)
    cx = float(np.clip(cx, 0, w - 1))
    cy = float(np.clip(cy, 0, h - 1))

    # 삼분할선 근접 여부
    tol = 0.06
    thirds_x = [w / 3, 2 * w / 3]
    thirds_y = [h / 3, 2 * h / 3]
    on_thirds = any(abs(cx - tx) <= w * tol for tx in thirds_x) and \
                any(abs(cy - ty) <= h * tol for ty in thirds_y)

    # 인물 비율, 헤드룸 요약
    x1, y1, x2, y2 = bbox
    bw, bh = (x2 - x1), (y2 - y1)
    size_ratio = float((bw * bh) / (w * h))
    head_y = float(np.min(kpts[:, 1]))
    headroom_ratio = float(np.clip(head_y / h, 0, 1))

    return {
        "enabled": True,
        "detected": True,
        "center": (float(cx), float(cy)),
        "on_rule_of_thirds": bool(on_thirds),
        "size_ratio": size_ratio,
        "headroom_ratio": headroom_ratio,
        "bbox": [float(x1), float(y1), float(x2), float(y2)],
        "image_wh": [int(w), int(h)]
    }

# --------- MiDaS depth 요약(선택) ----------
def summarize_depth(midas, midas_tf, device: str, img_bgr: np.ndarray) -> Dict[str, Any]:
    """
    깊이 전체 분포를 간단히 요약(히스토그램)하고,
    중앙 40%영역의 평균/분산을 계산해 저장(피사체 근접 가정)합니다.
    """
    img = _to_tensor(img_bgr)
    x = midas_tf(img).to(device)
    with torch.no_grad():
        pred = midas(x)
        depth = torch.nn.functional.interpolate(
            pred.unsqueeze(1),
            size=img_bgr.shape[:2],
            mode="bicubic",
            align_corners=False
        ).squeeze().cpu().numpy()

    d = depth.astype(np.float32)
    d = (d - d.min()) / max(1e-6, (d.max() - d.min()))  # 0~1 정규화

    h, w = d.shape
    x0, x1 = int(w * 0.3), int(w * 0.7)
    y0, y1 = int(h * 0.3), int(h * 0.7)
    center_crop = d[y0:y1, x0:x1]

    hist, _ = np.histogram(d, bins=16, range=(0, 1), density=True)
    return {
        "hist16": hist.astype(np.float32),
        "global_mean": float(d.mean()),
        "global_std": float(d.std()),
        "center_mean": float(center_crop.mean()),
        "center_std": float(center_crop.std())
    }

# --------- 메인 ---------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="레퍼런스 이미지 경로")
    ap.add_argument("--out_dir", default="./ref_cache", help="저장 폴더(.npz)")
    ap.add_argument("--name", default=None, help="저장 파일명(확장자 제외). 기본: 이미지파일명 기반")
    ap.add_argument("--no_clip", action="store_true", help="CLIP 생략")
    ap.add_argument("--no_dino", action="store_true", help="DINO 생략")
    ap.add_argument("--no_depth", action="store_true", help="MiDaS 생략")
    ap.add_argument("--no_comp", action="store_true", help="YOLO 구도요약 생략")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _ensure_dir(args.out_dir)

    # 이미지 로드
    img = cv2.imread(args.ref)
    if img is None:
        raise FileNotFoundError(f"이미지를 열 수 없습니다: {args.ref}")

    # 메타
    meta: Dict[str, Any] = {
        "source_path": os.path.abspath(args.ref),
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": device,
        "versions": {}
    }

    # ---- CLIP ----
    clip_feat = None
    if not args.no_clip:
        clip_model, preprocess = clip.load("ViT-B/32", device=device)
        clip_feat = compute_clip_embedding(clip_model, preprocess, device, img)
        meta["versions"]["clip"] = "ViT-B/32"

    # ---- DINO ----
    dino_feat = None
    if not args.no_dino:
        dino_model, dino_name = load_dino_model(device)
        dino_feat = compute_dino_embedding(dino_model, device, img)
        meta["versions"]["dino"] = dino_name

    # ---- YOLO 구도요약 ----
    comp = None
    if not args.no_comp:
        comp = summarize_composition_with_yolo(img)
        meta["versions"]["yolo_pose"] = "yolov8s-pose.pt" if YOLO_AVAILABLE else "not_available"

    # ---- MiDaS ----
    depth = None
    if not args.no_depth:
        midas, midas_tf = _load_midas(device)
        depth = summarize_depth(midas, midas_tf, device, img)
        meta["versions"]["midas"] = "DPT_Hybrid"

    # ---- 저장 ----
    base = args.name or os.path.splitext(os.path.basename(args.ref))[0]
    out_path = os.path.join(args.out_dir, f"{base}.npz")

    np.savez_compressed(
        out_path,
        clip_feat=clip_feat if clip_feat is not None else np.array([]),
        dino_feat=dino_feat if dino_feat is not None else np.array([]),
        depth_hist=depth["hist16"] if depth else np.array([]),
        depth_stats=np.array([
            depth["global_mean"], depth["global_std"],
            depth["center_mean"], depth["center_std"]
        ]) if depth else np.array([]),
        # comp는 dict → json 별도 저장
    )
    # composition 요약/메타는 json으로 저장
    with open(out_path.replace(".npz", ".json"), "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "composition": comp}, f, ensure_ascii=False, indent=2)

    print(f"✅ 저장 완료: {out_path}")
    print(f"📝 메타/구도요약: {out_path.replace('.npz', '.json')}")


if __name__ == "__main__":
    main()
