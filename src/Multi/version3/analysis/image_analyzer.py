# ============================================================
# 🎯 TryAngle - Image Analyzer
# 단일 이미지 분석: 클러스터 예측 + 측정 가능한 값 추출
# ============================================================

import os
import json
import cv2
import numpy as np
from pathlib import Path

# 기존 시스템 import
import sys

VERSION3_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = VERSION3_DIR
while PROJECT_ROOT != PROJECT_ROOT.parent and not ((PROJECT_ROOT / "data").exists() and (PROJECT_ROOT / "src").exists()):
    PROJECT_ROOT = PROJECT_ROOT.parent

if str(VERSION3_DIR) not in sys.path:
    sys.path.append(str(VERSION3_DIR))

from feature_extraction.feature_extractor_v2 import extract_features_v2 as extract_features_full
from matching.cluster_matcher import match_cluster_from_features

# Phase 1.3: Feature Cache
try:
    utils_dir = VERSION3_DIR / "utils"
    if str(utils_dir) not in sys.path:
        sys.path.append(str(utils_dir))
    from feature_cache import CachedFeatureExtractor
    FEATURE_CACHE_AVAILABLE = True
except ImportError:
    FEATURE_CACHE_AVAILABLE = False
    print("⚠️ Feature Cache not available (Phase 1.3)")

# 포즈 분석
try:
    from analysis.pose_analyzer import PoseAnalyzer
    POSE_AVAILABLE = True
except ImportError:
    print("⚠️ PoseAnalyzer not available. Install: pip install ultralytics mediapipe")
    POSE_AVAILABLE = False

# EXIF 분석
try:
    from analysis.exif_analyzer import ExifAnalyzer
    EXIF_AVAILABLE = True
except ImportError:
    print("⚠️ ExifAnalyzer not available")
    EXIF_AVAILABLE = False

# 품질 분석
try:
    from analysis.quality_analyzer import QualityAnalyzer
    QUALITY_AVAILABLE = True
except ImportError:
    print("⚠️ QualityAnalyzer not available")
    QUALITY_AVAILABLE = False

# 조명 분석
try:
    from analysis.lighting_analyzer import LightingAnalyzer
    LIGHTING_AVAILABLE = True
except ImportError:
    print("⚠️ LightingAnalyzer not available")
    LIGHTING_AVAILABLE = False


class ImageAnalyzer:
    """
    한 장의 이미지를 분석해서:
    1) 클러스터 예측 (스타일 DNA)
    2) 측정 가능한 값들 추출 (비교용)
    """
    
    def __init__(self, image_path: str, enable_pose: bool = True, enable_exif: bool = True, enable_quality: bool = True, enable_lighting: bool = True, use_movenet: bool = False):
        """
        Args:
            image_path: 이미지 파일 경로
            enable_pose: 포즈 분석 활성화
            enable_exif: EXIF 분석 활성화
            enable_quality: 품질 분석 활성화
            enable_lighting: 조명 분석 활성화
            use_movenet: True면 MoveNet 사용, False면 YOLO11 사용 (Phase 2-4)
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"❌ Image not found: {image_path}")

        self.image_path = image_path
        self.enable_pose = enable_pose and POSE_AVAILABLE
        self.enable_exif = enable_exif and EXIF_AVAILABLE
        self.enable_quality = enable_quality and QUALITY_AVAILABLE
        self.enable_lighting = enable_lighting and LIGHTING_AVAILABLE
        self.use_movenet = use_movenet  # Phase 2-4: MoveNet 옵션

        # ==========================================
        # Step 1: Feature 추출 (모든 모델 사용)
        # ==========================================
        print(f"  🔧 Extracting features from {os.path.basename(image_path)}...")

        # Phase 1.3: Feature Cache 사용
        if FEATURE_CACHE_AVAILABLE:
            cache_dir = VERSION3_DIR / "cache" / "features"
            cached_extractor = CachedFeatureExtractor(cache_dir=str(cache_dir))
            self.features = cached_extractor.extract(image_path)
        else:
            # Fallback: 직접 추출
            self.features = extract_features_full(image_path)

        if self.features is None:
            raise RuntimeError("❌ Feature extraction failed!")

        # ==========================================
        # Step 2: 클러스터 예측 (스타일 DNA 찾기)
        # ==========================================
        self.cluster_result = match_cluster_from_features(self.features)

        # ==========================================
        # Step 3: 클러스터 특성 로드 (집단지성)
        # ==========================================
        cluster_info_path = PROJECT_ROOT / "features" / "cluster_interpretation.json"
        with open(cluster_info_path, "r", encoding="utf-8") as f:
            cluster_info = json.load(f)

        self.cluster_data = cluster_info[str(self.cluster_result["cluster_id"])]

        print(f"  ✅ Cluster {self.cluster_result['cluster_id']}: {self.cluster_data['auto_label']}")

        # ==========================================
        # Step 4: PoseAnalyzer 초기화 (lazy loading)
        # ==========================================
        self.pose_analyzer = None
        if self.enable_pose:
            try:
                # Phase 2-4: MoveNet 옵션 전달
                self.pose_analyzer = PoseAnalyzer(use_movenet=self.use_movenet)
                model_name = "MoveNet" if self.use_movenet else "YOLO11"
                print(f"  ✅ PoseAnalyzer ready (using {model_name})")
            except Exception as e:
                print(f"  ⚠️ PoseAnalyzer initialization failed: {e}")
                self.enable_pose = False

        # ==========================================
        # Step 5: EXIF Analyzer 초기화
        # ==========================================
        self.exif_analyzer = None
        if self.enable_exif:
            try:
                self.exif_analyzer = ExifAnalyzer(image_path)
                if self.exif_analyzer.has_exif():
                    print(f"  ✅ EXIF: {len(self.exif_analyzer.exif_data)} fields")
                else:
                    print(f"  ⚠️ No EXIF data")
            except Exception as e:
                print(f"  ⚠️ EXIF extraction failed: {e}")
                self.enable_exif = False

        # ==========================================
        # Step 6: Quality Analyzer 초기화
        # ==========================================
        self.quality_analyzer = None
        if self.enable_quality:
            try:
                self.quality_analyzer = QualityAnalyzer(image_path)
                print(f"  ✅ Quality analysis ready")
            except Exception as e:
                print(f"  ⚠️ Quality analysis failed: {e}")
                self.enable_quality = False

        # ==========================================
        # Step 7: Lighting Analyzer 초기화
        # ==========================================
        self.lighting_analyzer = None
        if self.enable_lighting:
            try:
                # pose_data와 depth_data는 나중에 analyze()에서 전달
                self.lighting_analyzer = LightingAnalyzer(image_path)
                print(f"  ✅ Lighting analysis ready")
            except Exception as e:
                print(f"  ⚠️ Lighting analysis failed: {e}")
                self.enable_lighting = False
    
    def analyze(self) -> dict:
        """
        비교 가능한 모든 정보 반환
        """
        
        # ==========================================
        # 1) 클러스터 정보 (스타일 DNA)
        # ==========================================
        cluster_info = {
            "cluster_id": self.cluster_result["cluster_id"],
            "cluster_label": self.cluster_data["auto_label"],
            "cluster_distance": self.cluster_result["distance"],
            "sample_count": self.cluster_data["sample_count"],
            "embedding_128d": self.cluster_result["raw_embedding"]
        }
        
        # ==========================================
        # 2) MiDaS Depth (상대적 거리)
        # ==========================================
        # MiDaS feature는 20D지만, depth_mean은 첫 번째 값 (global mean)
        depth_info = {
            "depth_mean": float(self.features["midas"][0]),  # global mean
            "depth_std": float(self.features["midas"][1]),   # global std
            "cluster_typical_depth": self.cluster_data["depth_mean"],
            "depth_deviation": float(self.features["midas"][0]) - self.cluster_data["depth_mean"]
        }
        
        # ==========================================
        # 3) 픽셀 기반 분석 (직접 측정)
        # ==========================================
        pixel_analysis = self._analyze_pixels()
        
        # ==========================================
        # 4) 구도 분석
        # ==========================================
        composition_info = self._analyze_composition()

        # ==========================================
        # 5) 포즈 분석 (YOLO + MediaPipe)
        # ==========================================
        pose_info = None
        if self.enable_pose and self.pose_analyzer is not None:
            try:
                pose_info = self.pose_analyzer.analyze(self.image_path)
                print(f"  ✅ Pose: {pose_info['scenario']} (conf={pose_info['confidence']:.2f})")
            except Exception as e:
                print(f"  ⚠️ Pose analysis failed: {e}")
                pose_info = None

        # ==========================================
        # 6) EXIF 분석 (카메라 설정)
        # ==========================================
        exif_info = None
        if self.enable_exif and self.exif_analyzer is not None and self.exif_analyzer.has_exif():
            exif_info = {
                "camera_settings": self.exif_analyzer.get_camera_settings(),
                "shooting_info": self.exif_analyzer.get_shooting_info()
            }

        # ==========================================
        # 7) Quality 분석 (노이즈, 블러, 선명도, 대비)
        # ==========================================
        quality_info = None
        if self.enable_quality and self.quality_analyzer is not None:
            try:
                quality_info = self.quality_analyzer.analyze_all()
                print(f"  ✅ Quality: blur={quality_info['blur']['blur_score']:.1f}, noise={quality_info['noise']['noise_level']:.2f}")
            except Exception as e:
                print(f"  ⚠️ Quality analysis failed: {e}")
                quality_info = None

        # ==========================================
        # 8) Lighting 분석 (조명 방향, 역광, HDR)
        # ==========================================
        lighting_info = None
        if self.enable_lighting and self.lighting_analyzer is not None:
            try:
                # pose_data와 depth_data 전달 (있으면)
                if pose_info is not None:
                    self.lighting_analyzer.pose_data = pose_info
                if depth_info is not None:
                    # depth_mean을 depth map으로 사용할 수는 없으므로, 일단 None
                    # 실제로는 MiDaS로 depth map을 생성해야 함
                    pass

                lighting_info = self.lighting_analyzer.analyze_all()
                light_dir = lighting_info['light_direction']['direction']
                backlight = '있음' if lighting_info['backlight']['is_backlight'] else '없음'
                hdr = '있음' if lighting_info['hdr']['is_hdr'] else '없음'
                print(f"  ✅ Lighting: {light_dir} 조명, 역광={backlight}, HDR={hdr}")
            except Exception as e:
                print(f"  ⚠️ Lighting analysis failed: {e}")
                lighting_info = None

        return {
            "cluster": cluster_info,
            "depth": depth_info,
            "pixels": pixel_analysis,
            "composition": composition_info,
            "pose": pose_info,
            "exif": exif_info,
            "quality": quality_info,
            "lighting": lighting_info,
            "raw_features": self.features
        }
    
    def _analyze_pixels(self) -> dict:
        """픽셀 직접 분석"""
        img = cv2.imread(self.image_path)
        
        # 밝기
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
        
        # 채도
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        saturation = float(np.mean(hsv[:,:,1]) / 255.0)
        
        # 콘트라스트
        contrast = float(np.std(gray) / 128.0)
        
        # 색온도
        b, g, r = cv2.split(img)
        r_mean, g_mean, b_mean = np.mean(r), np.mean(g), np.mean(b)
        warm_score = (r_mean + g_mean) / 2
        cool_score = b_mean
        
        if warm_score > cool_score * 1.05:
            color_temp = "warm"
        elif cool_score > warm_score * 1.05:
            color_temp = "cool"
        else:
            color_temp = "neutral"
        
        # 히스토그램 분석 (클리핑 검사)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        total_pixels = gray.shape[0] * gray.shape[1]
        
        highlight_clipping = float(np.sum(hist[250:]) / total_pixels)
        shadow_clipping = float(np.sum(hist[:5]) / total_pixels)
        
        return {
            "brightness": brightness,  # 0~255
            "saturation": saturation,  # 0~1
            "contrast": contrast,      # 0~1
            "color_temperature": color_temp,
            "rgb_ratio": {
                "r": float(r_mean / (g_mean + 1e-8)),
                "g": 1.0,
                "b": float(b_mean / (g_mean + 1e-8))
            },
            "histogram": {
                "highlight_clipping": highlight_clipping,
                "shadow_clipping": shadow_clipping
            }
        }
    
    def _analyze_composition(self) -> dict:
        """구도 분석"""
        img = cv2.imread(self.image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 기울기 (간단한 Hough 변환)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
        
        if lines is not None and len(lines) > 0:
            angles = []
            for line in lines[:10]:  # 상위 10개만
                rho, theta = line[0]
                angle = np.degrees(theta) - 90
                if -45 <= angle <= 45:  # 유효한 범위만
                    angles.append(angle)
            
            if angles:
                tilt_angle = float(np.median(angles))
            else:
                tilt_angle = 0.0
        else:
            tilt_angle = 0.0
        
        # 대칭성
        h, w = gray.shape
        left = gray[:, :w//2]
        right = cv2.flip(gray[:, w//2:], 1)
        
        # 크기 맞추기
        min_w = min(left.shape[1], right.shape[1])
        left = cv2.resize(left, (min_w, h))
        right = cv2.resize(right, (min_w, h))
        
        mse = np.mean((left.astype(float) - right.astype(float)) ** 2)
        symmetry = float(max(0, 1 - mse / (255**2)))
        
        # 무게중심
        M = cv2.moments(gray)
        if M["m00"] != 0:
            cx = M["m10"] / M["m00"] / w
            cy = M["m01"] / M["m00"] / h
        else:
            cx, cy = 0.5, 0.5
        
        return {
            "tilt_angle": tilt_angle,
            "symmetry": symmetry,
            "center_of_mass": {"x": float(cx), "y": float(cy)}
        }


# ============================================================
# 테스트
# ============================================================
if __name__ == "__main__":
    test_img = PROJECT_ROOT / "data" / "test_images" / "test1.jpg"
    
    try:
        analyzer = ImageAnalyzer(str(test_img))
        result = analyzer.analyze()
        
        print("\n" + "="*60)
        print("📊 IMAGE ANALYSIS RESULT")
        print("="*60)
        
        print(f"\n🎯 Cluster: {result['cluster']['cluster_label']}")
        print(f"   └ ID: {result['cluster']['cluster_id']}")
        print(f"   └ Distance: {result['cluster']['cluster_distance']:.4f}")
        print(f"   └ Sample count: {result['cluster']['sample_count']} 장")
        
        print(f"\n📏 Depth:")
        print(f"   └ Current: {result['depth']['depth_mean']:.1f}")
        print(f"   └ Cluster typical: {result['depth']['cluster_typical_depth']:.1f}")
        print(f"   └ Deviation: {result['depth']['depth_deviation']:.1f}")
        
        print(f"\n🎨 Pixels:")
        print(f"   └ Brightness: {result['pixels']['brightness']:.1f}")
        print(f"   └ Saturation: {result['pixels']['saturation']:.2f}")
        print(f"   └ Color temp: {result['pixels']['color_temperature']}")
        
        print(f"\n📐 Composition:")
        print(f"   └ Tilt angle: {result['composition']['tilt_angle']:.1f}°")
        print(f"   └ Symmetry: {result['composition']['symmetry']:.2f}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
