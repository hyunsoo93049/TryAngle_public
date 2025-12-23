"""
TryAngle Feature Extractor (Memory-Safe ver.)
------------------------------------------
한 번의 실행으로 모든 이미지(jpg/jpeg/png 등)에 대해
CLIP + DINO + MiDaS 특징을 추출해 CSV로 저장합니다.

- 메모리 안정: 500장마다 중간 저장
- float16 사용으로 feature 크기 절반으로 감소

출력: features/clip_dino_midas_features.csv
"""

import os
import cv2
import numpy as np
import torch
import pandas as pd
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
import gc

# -------- 1️⃣ 모델 임포트 --------
import clip
from transformers import DPTFeatureExtractor, DPTForDepthEstimation
from timm import create_model
from timm.data import resolve_model_data_config
from timm.data.transforms_factory import create_transform

# -------- 2️⃣ 디바이스 설정 --------
device = "cuda" if torch.cuda.is_available() else "cpu"

# -------- 3️⃣ 모델 로드 --------
print("[🔹] 모델 로드 중...")
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
dino_model = create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True).eval().to(device)
dino_cfg = resolve_model_data_config(dino_model)
dino_tf = create_transform(**dino_cfg)
midas_feat = DPTFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas")
midas_model = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to(device).eval()

# -------- 4️⃣ 특징 추출 함수 --------
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
        pixel_values = midas_feat(images=img_pil, return_tensors="pt").pixel_values.to(device)
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


# -------- 5️⃣ 메인 파이프라인 --------
def main():
    data_dir = "data/train_images"
    save_dir = "features"
    os.makedirs(save_dir, exist_ok=True)
    out_csv = os.path.join(save_dir, "clip_dino_midas_features.csv")

    image_paths = get_image_list(data_dir)
    print(f"[🔹] 총 {len(image_paths)}장의 이미지 감지됨.")

    records = []
    chunk_size = 500  # 500장마다 저장

    for idx, img_path in enumerate(tqdm(image_paths, desc="특징 추출 중")):
        feat = extract_features(img_path)
        if feat is None:
            continue

        record = {
            "filename": feat["filename"],
            "midas_mean": feat["midas_mean"],
            "midas_std": feat["midas_std"],
        }

        clip_vec = {f"clip_{i}": v for i, v in enumerate(feat["clip_emb"])}
        dino_vec = {f"dino_{i}": v for i, v in enumerate(feat["dino_emb"])}
        record.update(clip_vec)
        record.update(dino_vec)
        records.append(record)

        # 💾 500장마다 CSV에 append 후 메모리 비움
        if (idx + 1) % chunk_size == 0:
            df = pd.DataFrame(records)
            mode = "a" if os.path.exists(out_csv) else "w"
            header = not os.path.exists(out_csv)
            df.to_csv(out_csv, mode=mode, header=header, index=False)
            print(f"[💾] {idx + 1}장까지 저장 완료.")
            records = []
            gc.collect()
            torch.cuda.empty_cache()

    # 남은 데이터 저장
    if records:
        df = pd.DataFrame(records)
        df.to_csv(out_csv, mode="a", header=not os.path.exists(out_csv), index=False)
        print(f"[💾] 남은 {len(records)}장 추가 저장 완료.")

    print(f"[✅] 전체 특징 추출 완료: {out_csv}")


# -------- 6️⃣ 실행 --------
if __name__ == "__main__":
    main()
