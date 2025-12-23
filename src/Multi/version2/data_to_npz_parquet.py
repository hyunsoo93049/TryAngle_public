"""
TryAngle Feature Extractor (NPZ + Parquet ver.)
------------------------------------------
원본 CSV 로직 유지 + 저장만 NPZ/Parquet으로 변경
"""

import os
import numpy as np
import torch
import pandas as pd
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
import gc
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
while PROJECT_ROOT != PROJECT_ROOT.parent and not ((PROJECT_ROOT / "data").exists() and (PROJECT_ROOT / "src").exists()):
    PROJECT_ROOT = PROJECT_ROOT.parent

# -------- 모델 임포트 --------
import clip
from transformers import DPTImageProcessor, DPTForDepthEstimation  # ✅ Deprecated 경고 해결
from timm import create_model
from timm.data import resolve_model_data_config
from timm.data.transforms_factory import create_transform

# -------- 디바이스 설정 --------
device = "cuda" if torch.cuda.is_available() else "cpu"

# -------- 모델 로드 --------
print("[🔹] 모델 로드 중...")
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
dino_model = create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True).eval().to(device)
dino_cfg = resolve_model_data_config(dino_model)
dino_tf = create_transform(**dino_cfg)
midas_processor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")  # ✅ 변경
midas_model = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to(device).eval()

# -------- 특징 추출 함수 --------
def extract_features(img_path: str):
    """CLIP + DINO + MiDaS feature 추출"""
    try:
        img_pil = Image.open(img_path).convert("RGB")
    except (UnidentifiedImageError, FileNotFoundError, OSError):
        print(f"[⚠️] 이미지 로드 실패: {img_path}")
        return None

    try:
        # --- CLIP ---
        with torch.no_grad():
            clip_in = clip_preprocess(img_pil).unsqueeze(0).to(device)
            clip_feat = clip_model.encode_image(clip_in)
            clip_feat = clip_feat / clip_feat.norm(dim=-1, keepdim=True)
            clip_feat = clip_feat.cpu().numpy().astype(np.float16).flatten()

        # --- DINO (timm) ---
        with torch.no_grad():
            dino_in = dino_tf(img_pil).unsqueeze(0).to(device)
            feats = dino_model.forward_features(dino_in)
            if isinstance(feats, dict):
                if "x_norm_clstoken" in feats:
                    dino_vec = feats["x_norm_clstoken"]
                elif "pool" in feats:
                    dino_vec = feats["pool"]
                else:
                    dino_vec = torch.flatten(feats, 1)
            else:
                dino_vec = torch.flatten(feats, 1)
            dino_vec = dino_vec / (dino_vec.norm(dim=-1, keepdim=True) + 1e-12)
            dino_feat = dino_vec.cpu().numpy().astype(np.float16).flatten()

        # --- MiDaS ---
        pixel_values = midas_processor(images=img_pil, return_tensors="pt").pixel_values.to(device)
        with torch.no_grad():
            depth_pred = midas_model(pixel_values).predicted_depth
            depth_mean = float(depth_pred.mean().item())
            depth_std = float(depth_pred.std().item())

        return {
            "filename": os.path.basename(img_path),
            "clip_emb": clip_feat,
            "dino_emb": dino_feat,
            "midas_mean": depth_mean,
            "midas_std": depth_std,
        }

    except Exception as e:
        print(f"[❌] 처리 중 오류 ({img_path}): {e}")
        return None
    finally:
        torch.cuda.empty_cache()
        gc.collect()


def get_image_list(data_dir):
    """jpg/jpeg/png 확장자만 수집"""
    valid_ext = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
    files = []
    for root, _, f_names in os.walk(data_dir):
        for f in f_names:
            if f.endswith(valid_ext):
                files.append(os.path.join(root, f))
    return sorted(files)


# -------- 청크 저장 함수 --------
def save_chunk(records, chunk_idx, parq_dir, npz_dir):
    """500장 단위로 Parquet + NPZ 저장"""
    if not records:
        return
    
    # ✅ Parquet 저장 (메타데이터만)
    df = pd.DataFrame({
        "filename": [r["filename"] for r in records],
        "midas_mean": [r["midas_mean"] for r in records],
        "midas_std": [r["midas_std"] for r in records],
    })
    parq_path = os.path.join(parq_dir, f"chunk_{chunk_idx:03d}.parquet")
    df.to_parquet(parq_path, engine="pyarrow", compression="snappy", index=False)
    
    # ✅ NPZ 저장 (임베딩)
    npz_path = os.path.join(npz_dir, f"chunk_{chunk_idx:03d}.npz")
    np.savez_compressed(
        npz_path,
        clip_emb=np.stack([r["clip_emb"] for r in records]),
        dino_emb=np.stack([r["dino_emb"] for r in records]),
        filenames=np.array([r["filename"] for r in records])
    )
    
    print(f"[💾] Chunk {chunk_idx:03d} 저장 완료 ({len(records)}개)")


# -------- 메인 파이프라인 --------
def main():
    data_dir = PROJECT_ROOT / "data" / "train_images"  # ✅ 실제 경로
    save_dir = PROJECT_ROOT / "features"
    os.makedirs(save_dir, exist_ok=True)

    parq_dir = save_dir / "parquet_shards"
    npz_dir = save_dir / "npz_shards"
    os.makedirs(parq_dir, exist_ok=True)
    os.makedirs(npz_dir, exist_ok=True)

    image_paths = get_image_list(data_dir)
    print(f"[🔹] 총 {len(image_paths)}장의 이미지 감지됨.")

    records = []
    chunk_size = 500
    chunk_idx = 0

    for idx, img_path in enumerate(tqdm(image_paths, desc="특징 추출 중")):
        feat = extract_features(img_path)
        if feat is None:
            continue

        records.append(feat)

        # 💾 500장마다 저장
        if len(records) >= chunk_size:
            save_chunk(records, chunk_idx, parq_dir, npz_dir)
            chunk_idx += 1
            records = []
            gc.collect()
            torch.cuda.empty_cache()

    # 남은 데이터 저장
    if records:
        save_chunk(records, chunk_idx, parq_dir, npz_dir)
        print(f"[💾] 남은 {len(records)}장 추가 저장 완료.")

    print(f"[✅] 전체 특징 추출 완료")
    print(f"  📁 Parquet: {parq_dir}")
    print(f"  📁 NPZ: {npz_dir}")


# -------- 실행 --------
if __name__ == "__main__":
    main()
