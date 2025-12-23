# emotion_module_v2.py
# ---------------------------------------------------------------------
# Part A: CLIP 감성 유사도 (EmotionAnalyzer)
# Part B: 감성 요인 비교 (EmotionFactors)
# ---------------------------------------------------------------------

import cv2, torch, numpy as np
from PIL import Image
import clip
from typing import Dict, Optional

# -------------------------------
# Part A. CLIP 기반 감성 유사도
# -------------------------------
class EmotionAnalyzer:
    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

    def _encode(self, img_pil: Image.Image):
        with torch.no_grad():
            x = self.preprocess(img_pil).unsqueeze(0).to(self.device)
            feat = self.model.encode_image(x)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat

    def compare_to_reference(self, ref_path: str, target_image: np.ndarray) -> float:
        ref_pil = Image.open(ref_path).convert("RGB")
        tgt_pil = Image.fromarray(target_image[:, :, ::-1]).convert("RGB")
        ref_feat = self._encode(ref_pil)
        tgt_feat = self._encode(tgt_pil)
        sim = torch.cosine_similarity(ref_feat, tgt_feat).item()
        return round(sim * 100.0, 2)


# -------------------------------
# Part B. 감성 요인 비교
# -------------------------------
class EmotionFactors:
    """색온도, 조명, 채도, 밝기, 질감 비교"""
    
    @staticmethod
    def compare_color_temperature(img1, img2) -> float:
        b1,g1,r1 = cv2.split(img1); b2,g2,r2 = cv2.split(img2)
        t1 = (r1.mean() + g1.mean()) / (b1.mean() + 1e-6)
        t2 = (r2.mean() + g2.mean()) / (b2.mean() + 1e-6)
        return 1.0 - min(abs(t1 - t2) / max(t1, t2), 1.0)
    
    @staticmethod
    def compare_brightness(img1, img2) -> float:
        y1 = cv2.cvtColor(img1, cv2.COLOR_BGR2YUV)[:,:,0].mean()
        y2 = cv2.cvtColor(img2, cv2.COLOR_BGR2YUV)[:,:,0].mean()
        return 1.0 - min(abs(y1 - y2)/255.0, 1.0)
    
    @staticmethod
    def compare_saturation(img1, img2) -> float:
        s1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)[:,:,1].mean()
        s2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)[:,:,1].mean()
        return 1.0 - min(abs(s1 - s2)/255.0, 1.0)

    @staticmethod
    def compare_texture(img1, img2) -> float:
        def edge_strength(img):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)
            return np.mean(np.abs(sobel))
        e1, e2 = edge_strength(img1), edge_strength(img2)
        return 1.0 - min(abs(e1 - e2) / max(e1, e2 + 1e-6), 1.0)
    
    @staticmethod
    def compare_lighting_direction(img1, img2) -> float:
        def bright_side(img):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            left = gray[:,:gray.shape[1]//2].mean()
            right = gray[:,gray.shape[1]//2:].mean()
            return (right - left) / (right + left + 1e-6)
        d1, d2 = bright_side(img1), bright_side(img2)
        return 1.0 - min(abs(d1 - d2)*2, 1.0)

    @staticmethod
    def analyze_emotion_factors(ref_path: str, tgt_img: np.ndarray) -> Dict[str, float]:
        ref_img = cv2.imread(ref_path)
        f = EmotionFactors
        return {
            "lighting_direction": f.compare_lighting_direction(ref_img, tgt_img),
            "color_temperature": f.compare_color_temperature(ref_img, tgt_img),
            "saturation_sim": f.compare_saturation(ref_img, tgt_img),
            "brightness_sim": f.compare_brightness(ref_img, tgt_img),
            "texture_sim": f.compare_texture(ref_img, tgt_img),
        }
