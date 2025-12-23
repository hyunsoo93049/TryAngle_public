# ============================================================
# 🎯 Phase 1: Image Interpretation Layer
# 각 모델의 출력을 구체적인 정보로 해석하는 계층
# ============================================================

import os
import cv2
import numpy as np
import torch
from PIL import Image
from typing import Dict, List, Optional
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
while PROJECT_ROOT != PROJECT_ROOT.parent and not ((PROJECT_ROOT / "data").exists() and (PROJECT_ROOT / "src").exists()):
    PROJECT_ROOT = PROJECT_ROOT.parent

# 기존 feature extractor
from feature_extraction.feature_extractor import extract_features_full
import clip
import open_clip


# ============================================================
# 1. DINO 기반 구성 분석
# ============================================================

class CompositionAnalyzer:
    """
    DINO 벡터를 통해 이미지 구성을 분석
    - 객체 위치
    - 대칭성
    - 밸런스
    """
    
    def __init__(self, image_path: str, dino_feature: np.ndarray):
        self.image_path = image_path
        self.dino_feature = dino_feature  # (384,)
        self.image = cv2.imread(image_path)
        self.height, self.width = self.image.shape[:2]
        
    def analyze(self) -> Dict:
        """
        DINO 벡터로부터 구성 특성을 추출
        (현재는 벡터 통계, 향후 DINO 공간 해석 고도화)
        """
        
        # DINO 벡터의 통계적 특성
        dino_mean = float(np.mean(self.dino_feature))
        dino_std = float(np.std(self.dino_feature))
        dino_entropy = self._entropy(self.dino_feature)
        
        # 이미지 중심 밀집도 추정 (DINO는 spatial attention이 있음)
        # 실제로는 DINO의 attention map을 추출하면 더 정확
        composition_score = self._estimate_composition_score()
        
        # 대칭성 점수 (이미지 픽셀 기반)
        symmetry_score = self._estimate_symmetry()
        
        return {
            "dino_mean": dino_mean,
            "dino_std": dino_std,
            "dino_entropy": dino_entropy,
            "composition_score": composition_score,  # 0-100
            "symmetry_score": symmetry_score,  # 0-100
            "is_balanced": composition_score > 60,
            "is_symmetric": symmetry_score > 60,
        }
    
    def _entropy(self, arr: np.ndarray) -> float:
        """특성 벡터의 엔트로피 (얼마나 많은 정보를 담고 있는가?)"""
        hist, _ = np.histogram(arr, bins=32, range=(arr.min(), arr.max()))
        hist = hist / (hist.sum() + 1e-8)
        entropy = -np.sum(hist * np.log(hist + 1e-8))
        return float(entropy)
    
    def _estimate_composition_score(self) -> float:
        """
        이미지 픽셀 기반 구성 평가
        (향후: DINO attention map으로 대체)
        """
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        # 가우시안 블러로 부드러운 밝기 분포 생성
        blurred = cv2.GaussianBlur(gray, (51, 51), 0)
        
        # 중심부와 전체의 밝기 비율
        h, w = gray.shape
        center = blurred[h//4:3*h//4, w//4:3*w//4]
        overall = blurred
        
        center_brightness = np.mean(center)
        overall_brightness = np.mean(overall)
        
        # 중심이 밝으면 good composition
        score = min(100, max(0, (center_brightness / (overall_brightness + 1e-8)) * 80 + 20))
        return float(score)
    
    def _estimate_symmetry(self) -> float:
        """좌우 대칭성 점수"""
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (256, 256))
        
        flipped = cv2.flip(gray, 1)
        
        # MSE 기반 유사도
        mse = np.mean((gray.astype(float) - flipped.astype(float)) ** 2)
        max_mse = 255 ** 2
        
        symmetry = max(0, 100 - (mse / max_mse) * 100)
        return float(symmetry)


# ============================================================
# 2. MiDaS 기반 카메라 분석
# ============================================================

class CameraAnalyzer:
    """
    MiDaS 깊이 맵으로부터 카메라 정보 추출
    - 촬영 거리
    - 촬영 각도 (위/가운데/아래)
    - 초점 깊이
    """
    
    def __init__(self, image_path: str, midas_stats: np.ndarray):
        self.image_path = image_path
        self.midas_mean = float(midas_stats[0])
        self.midas_std = float(midas_stats[1])
        self.image = cv2.imread(image_path)
        
    def analyze(self) -> Dict:
        """
        MiDaS 통계로부터 카메라 특성 추출
        """
        
        # 깊이 분포로부터 촬영 거리 추정
        # (실제 MiDaS 출력값은 상대적 깊이)
        distance_estimate = self._estimate_distance()
        
        # 깊이 표준편차로부터 촬영 각도 추정
        # std가 크면 = 위/아래에서 촬영한 느낌
        angle_estimate = self._estimate_angle()
        
        # 초점 깊이
        focus_depth = self._estimate_focus_depth()
        
        return {
            "mean_depth": self.midas_mean,
            "depth_std": self.midas_std,
            "estimated_distance": distance_estimate,  # cm (상대값)
            "estimated_angle": angle_estimate,  # -45~+45 (음수=위, 양수=아래)
            "focus_depth": focus_depth,  # 0-100 (깊이감)
            "is_wide_angle": distance_estimate > 150,
            "is_close_up": distance_estimate < 50,
        }
    
    def _estimate_distance(self) -> float:
        """
        깊이 평균값으로부터 거리 추정 (정규화)
        """
        # MiDaS 출력은 대대로 0~1 범위
        # 이를 cm 단위로 변환 (상대값)
        distance = self.midas_mean * 200  # 임의 스케일
        return float(distance)
    
    def _estimate_angle(self) -> float:
        """
        깊이 표준편차로부터 촬영 각도 추정
        높은 std = 위/아래에서 촬영한 느낌
        
        return: -45 (위에서) ~ 0 (가운데) ~ +45 (아래에서)
        """
        # std가 작으면 정면, 크면 각진 각도
        angle_score = (self.midas_std - 0.05) * 100  # 정규화
        angle = np.clip(angle_score * 45 - 22.5, -45, 45)
        return float(angle)
    
    def _estimate_focus_depth(self) -> float:
        """초점 깊이감 (0-100)"""
        # std가 크면 깊이감 있음
        focus = min(100, self.midas_std * 500)
        return float(focus)


# ============================================================
# 3. CLIP 기반 심미성 분석
# ============================================================

class AestheticAnalyzer:
    """
    CLIP 임베딩과 프롬프트를 이용해 심미성 분석
    - 밝기 레벨
    - 색감 온도
    - 포화도
    """
    
    def __init__(self, image_path: str, clip_feature: np.ndarray):
        self.image_path = image_path
        self.clip_feature = clip_feature
        self.image = cv2.imread(image_path)
        
        # CLIP 모델 로드
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        
    def analyze(self) -> Dict:
        """
        심미 속성 분석
        """
        
        # 픽셀 기반 분석
        brightness = self._analyze_brightness()
        color_temp = self._analyze_color_temperature()
        saturation = self._analyze_saturation()
        
        # CLIP 기반 감정 분석 (프롬프트 사용)
        emotions = self._analyze_emotions_with_prompts()
        
        return {
            "brightness": brightness,  # 0-100
            "color_temperature": color_temp,  # 0-100 (0=차가움, 100=따뜻함)
            "saturation": saturation,  # 0-100
            "emotions": emotions,  # dict
            "overall_aesthetic_score": self._calculate_aesthetic_score(
                brightness, color_temp, saturation, emotions
            ),
        }
    
    def _analyze_brightness(self) -> float:
        """밝기 분석"""
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray) / 255.0 * 100
        return float(brightness)
    
    def _analyze_color_temperature(self) -> float:
        """
        색온도 분석 (따뜻함/차가움)
        따뜻함: 빨강/노랑 채널이 높음
        차가움: 파랑 채널이 높음
        """
        b, g, r = cv2.split(self.image)
        
        warm_score = (np.mean(r) + np.mean(g)) / 2
        cool_score = np.mean(b)
        
        # 0 = 차가움, 50 = 중립, 100 = 따뜻함
        if warm_score + cool_score > 0:
            temperature = (warm_score / (warm_score + cool_score)) * 100
        else:
            temperature = 50
            
        return float(temperature)
    
    def _analyze_saturation(self) -> float:
        """포화도 분석"""
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        s_channel = hsv[:, :, 1]
        saturation = np.mean(s_channel) / 255.0 * 100
        return float(saturation)
    
    def _analyze_emotions_with_prompts(self) -> Dict:
        """
        텍스트 프롬프트로 감정 분석
        느좋 스타일의 감정들을 프롬프트로 입력
        """
        emotions_prompts = {
            "bright": "a bright and vivid photo",
            "moody": "a moody and dark photo",
            "warm": "a warm and cozy photo",
            "cool": "a cool and calm photo",
            "vibrant": "a vibrant and saturated photo",
            "soft": "a soft and muted photo",
        }
        
        img_pil = Image.open(self.image_path).convert("RGB")
        img_tensor = self.clip_preprocess(img_pil).unsqueeze(0).to(self.device)
        
        emotions = {}
        with torch.no_grad():
            image_features = self.clip_model.encode_image(img_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            for emotion, prompt in emotions_prompts.items():
                text = clip.tokenize(prompt).to(self.device)
                text_features = self.clip_model.encode_text(text)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                similarity = float((image_features @ text_features.T).item())
                # -1~1을 0~100으로 변환
                score = ((similarity + 1) / 2) * 100
                emotions[emotion] = score
        
        return emotions
    
    def _calculate_aesthetic_score(self, brightness, color_temp, saturation, emotions) -> float:
        """종합 심미 점수"""
        # 단순 평균 (향후 가중치 적용 가능)
        emotion_avg = np.mean(list(emotions.values()))
        overall = (brightness + color_temp + saturation + emotion_avg) / 4
        return float(overall)


# ============================================================
# 4. OpenCLIP 기반 조명 분석
# ============================================================

class LightingAnalyzer:
    """
    OpenCLIP으로 조명 방향/유형 분석
    - 정면/측면/역광
    - 자연광/인공광
    - 소프트/하드 라이트
    """
    
    def __init__(self, image_path: str, openclip_feature: np.ndarray):
        self.image_path = image_path
        self.openclip_feature = openclip_feature
        
        # OpenCLIP 모델 로드
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k", device=self.device
        )
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
        
    def analyze(self) -> Dict:
        """조명 특성 분석"""
        
        lighting_type = self._analyze_lighting_type()
        lighting_direction = self._analyze_lighting_direction()
        lighting_softness = self._analyze_lighting_softness()
        
        return {
            "lighting_type": lighting_type,  # "natural", "artificial", "mixed"
            "lighting_direction": lighting_direction,  # "front", "side", "backlit"
            "lighting_softness": lighting_softness,  # 0-100 (0=hard, 100=soft)
            "is_natural_light": lighting_type == "natural",
            "is_backlit": lighting_direction == "backlit",
        }
    
    def _analyze_lighting_type(self) -> str:
        """조명 유형 판정 (자연광/인공광)"""
        prompts = {
            "natural": "natural light from outside, sunlight",
            "artificial": "artificial light, indoor light, lamp",
        }
        
        img_pil = Image.open(self.image_path).convert("RGB")
        img_tensor = self.preprocess(img_pil).unsqueeze(0).to(self.device)
        
        scores = {}
        with torch.no_grad():
            image_features = self.model.encode_image(img_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            for key, prompt in prompts.items():
                text = self.tokenizer(prompt).to(self.device)
                text_features = self.model.encode_text(text)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                similarity = float((image_features @ text_features.T).item())
                scores[key] = similarity
        
        # 점수가 높은 것 선택
        return max(scores, key=scores.get)
    
    def _analyze_lighting_direction(self) -> str:
        """조명 방향 판정"""
        prompts = {
            "front": "front lighting, face is well lit",
            "side": "side lighting, one side is lit",
            "backlit": "backlit, light from behind, silhouette",
        }
        
        img_pil = Image.open(self.image_path).convert("RGB")
        img_tensor = self.preprocess(img_pil).unsqueeze(0).to(self.device)
        
        scores = {}
        with torch.no_grad():
            image_features = self.model.encode_image(img_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            for key, prompt in prompts.items():
                text = self.tokenizer(prompt).to(self.device)
                text_features = self.model.encode_text(text)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                similarity = float((image_features @ text_features.T).item())
                scores[key] = similarity
        
        return max(scores, key=scores.get)
    
    def _analyze_lighting_softness(self) -> float:
        """조명의 부드러움"""
        prompts = {
            "soft": "soft light, diffused light, no harsh shadows",
            "hard": "hard light, sharp shadows, high contrast",
        }
        
        img_pil = Image.open(self.image_path).convert("RGB")
        img_tensor = self.preprocess(img_pil).unsqueeze(0).to(self.device)
        
        scores = {}
        with torch.no_grad():
            image_features = self.model.encode_image(img_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            for key, prompt in prompts.items():
                text = self.tokenizer(prompt).to(self.device)
                text_features = self.model.encode_text(text)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                similarity = float((image_features @ text_features.T).item())
                scores[key] = similarity
        
        # soft와 hard의 상대 점수로 0-100 변환
        soft_score = scores.get("soft", 0)
        hard_score = scores.get("hard", 0)
        
        softness = ((soft_score - hard_score + 2) / 4) * 100  # -1~1을 0~100으로
        return float(np.clip(softness, 0, 100))


# ============================================================
# 5. Color/Texture 기반 기술 분석
# ============================================================

class TechnicalAnalyzer:
    """
    색상/텍스처로부터 기술적 속성 분석
    - 명도 분포 (노출)
    - 색상 분포
    - 디테일 수준
    - ISO/노출 추천
    """
    
    def __init__(self, image_path: str, color_feature: np.ndarray):
        self.image_path = image_path
        self.color_feature = color_feature
        self.image = cv2.imread(image_path)
        
    def analyze(self) -> Dict:
        """기술적 분석"""
        
        exposure = self._analyze_exposure()
        detail_level = self._analyze_detail_level()
        noise_level = self._estimate_noise()
        white_balance = self._analyze_white_balance()
        
        return {
            "exposure": exposure,  # 0-100 (0=underexposed, 50=correct, 100=overexposed)
            "detail_level": detail_level,  # 0-100
            "noise_level": noise_level,  # 0-100 (추정)
            "white_balance": white_balance,  # "cool", "neutral", "warm"
            "recommended_iso": self._recommend_iso(noise_level),
            "recommended_ev_adjustment": self._recommend_ev(exposure),
        }
    
    def _analyze_exposure(self) -> float:
        """노출 분석"""
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        
        # 0-255 범위를 0-100으로 (128=50)
        exposure = (mean_brightness / 255.0) * 100
        return float(exposure)
    
    def _analyze_detail_level(self) -> float:
        """디테일 수준 (엣지 밀도)"""
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        detail = (np.sum(edges) / edges.size) * 100
        return float(np.clip(detail, 0, 100))
    
    def _estimate_noise(self) -> float:
        """노이즈 수준 추정"""
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        # Laplacian 적용 후 분산으로 노이즈 추정
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        noise = np.var(laplacian) / 1000  # 정규화
        
        return float(np.clip(noise, 0, 100))
    
    def _analyze_white_balance(self) -> str:
        """화이트 밸런스"""
        b, g, r = cv2.split(self.image)
        
        b_mean = np.mean(b)
        g_mean = np.mean(g)
        r_mean = np.mean(r)
        
        # R/G 비율로 판정
        rg_ratio = r_mean / (g_mean + 1e-8)
        
        if rg_ratio > 1.05:
            return "warm"
        elif rg_ratio < 0.95:
            return "cool"
        else:
            return "neutral"
    
    def _recommend_iso(self, noise_level: float) -> int:
        """ISO 추천"""
        if noise_level < 20:
            return 100
        elif noise_level < 40:
            return 200
        elif noise_level < 60:
            return 400
        else:
            return 800
    
    def _recommend_ev(self, exposure: float) -> float:
        """EV 값 조정 추천 (-2.0 ~ +2.0)"""
        # 50이 정상이라고 가정
        target = 50
        diff = target - exposure
        
        # 한 단계 = 약 10
        ev_adjustment = diff / 10 * 0.3  # 1 단계 = 0.3 EV
        return float(np.clip(ev_adjustment, -2.0, 2.0))


# ============================================================
# 6. 통합 해석 클래스
# ============================================================

class ImageInterpretation:
    """
    한 장의 이미지를 모든 모델로 분석해서 종합 정보 생성
    """
    
    def __init__(self, image_path: str):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"❌ Image not found: {image_path}")
        
        self.image_path = image_path
        print(f"📸 Loading image: {image_path}")
        
        # Feature 추출
        print("🔧 Extracting features...")
        features = extract_features_full(image_path)
        if features is None:
            raise RuntimeError("❌ Feature extraction failed!")
        
        self.features = features
        
        # 각 분석기 초기화
        self.composition_analyzer = CompositionAnalyzer(
            image_path, features["dino"]
        )
        self.camera_analyzer = CameraAnalyzer(
            image_path, features["midas"]
        )
        self.aesthetic_analyzer = AestheticAnalyzer(
            image_path, features["clip"]
        )
        self.lighting_analyzer = LightingAnalyzer(
            image_path, features["openclip"]
        )
        self.technical_analyzer = TechnicalAnalyzer(
            image_path, features["color"]
        )
        
    def analyze(self) -> Dict:
        """모든 분석 실행"""
        print("\n🎯 Analyzing image...")
        
        composition = self.composition_analyzer.analyze()
        camera = self.camera_analyzer.analyze()
        aesthetic = self.aesthetic_analyzer.analyze()
        lighting = self.lighting_analyzer.analyze()
        technical = self.technical_analyzer.analyze()
        
        return {
            "composition": composition,
            "camera": camera,
            "aesthetic": aesthetic,
            "lighting": lighting,
            "technical": technical,
        }
    
    def get_summary(self) -> Dict:
        """간단한 요약"""
        analysis = self.analyze()
        
        return {
            "composition_score": analysis["composition"]["composition_score"],
            "balance_status": "balanced" if analysis["composition"]["is_balanced"] else "unbalanced",
            "camera_distance": analysis["camera"]["estimated_distance"],
            "camera_angle": analysis["camera"]["estimated_angle"],
            "brightness": analysis["aesthetic"]["brightness"],
            "color_temperature": analysis["aesthetic"]["color_temperature"],
            "lighting_type": analysis["lighting"]["lighting_type"],
            "lighting_direction": analysis["lighting"]["lighting_direction"],
            "recommended_iso": analysis["technical"]["recommended_iso"],
            "recommended_ev": analysis["technical"]["recommended_ev_adjustment"],
        }


# ============================================================
# 테스트
# ============================================================

if __name__ == "__main__":
    test_img = PROJECT_ROOT / "data" / "test_images" / "test1.jpg"
    
    try:
        interp = ImageInterpretation(str(test_img))
        result = interp.analyze()
        
        print("\n" + "="*50)
        print("📊 FULL ANALYSIS RESULT")
        print("="*50)
        
        import json
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        print("\n" + "="*50)
        print("📋 SUMMARY")
        print("="*50)
        summary = interp.get_summary()
        for key, value in summary.items():
            print(f"{key:25s}: {value}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
