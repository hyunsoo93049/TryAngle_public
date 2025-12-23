#!/usr/bin/env python3
# ============================================================
# TryAngle v1.5 - Enhanced Feature Extraction Pipeline
# 풍부한 특징 추출 + 종횡비 인식 + 신뢰도 레벨
# ============================================================

import os
import sys
import argparse
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PIL import Image
from tqdm import tqdm
from datetime import datetime
import json
import cv2

# 모듈 경로 추가
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from models import GroundingDINOWrapper, DepthAnythingWrapper, RTMPoseWrapper
from utils import PoseClassifier, CompositionAnalyzer, PatternStatistics


def load_config(config_path: str) -> Dict:
    """설정 파일 로드"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def get_image_files(folder: Path, extensions: tuple = ('.jpg', '.jpeg', '.png', '.webp')) -> List[Path]:
    """이미지 파일 목록 반환"""
    files = []
    for ext in extensions:
        files.extend(folder.glob(f'*{ext}'))
        files.extend(folder.glob(f'*{ext.upper()}'))
    return sorted(files)


def resize_image(image: Image.Image, max_size: int) -> Image.Image:
    """이미지 리사이즈 (처리 속도용)"""
    w, h = image.size
    if max(w, h) <= max_size:
        return image

    if w > h:
        new_w = max_size
        new_h = int(h * max_size / w)
    else:
        new_h = max_size
        new_w = int(w * max_size / h)

    return image.resize((new_w, new_h), Image.Resampling.LANCZOS)


def detect_aspect_ratio(image: Image.Image) -> Tuple[str, float]:
    """이미지 종횡비 감지"""
    w, h = image.size
    ratio = w / h

    # 공통 비율 매칭 (tolerance 포함)
    if abs(ratio - 1.0) < 0.05:
        return "1:1", ratio
    elif abs(ratio - 1.333) < 0.05:
        return "4:3", ratio
    elif abs(ratio - 0.75) < 0.05:
        return "3:4", ratio
    elif abs(ratio - 1.5) < 0.05:
        return "3:2", ratio
    elif abs(ratio - 1.777) < 0.05:
        return "16:9", ratio
    elif abs(ratio - 0.5625) < 0.05:
        return "9:16", ratio
    else:
        return f"custom_{ratio:.2f}", ratio


def measure_background_blur(image_np: np.ndarray, person_bbox: List[float]) -> float:
    """배경 블러 정도 측정"""
    h, w = image_np.shape[:2]

    # 바운딩 박스를 픽셀 좌표로 변환
    x1 = int(person_bbox[0] * w)
    y1 = int(person_bbox[1] * h)
    x2 = int(person_bbox[2] * w)
    y2 = int(person_bbox[3] * h)

    # 마스크 생성 (사람 영역 제외)
    mask = np.ones((h, w), dtype=np.uint8) * 255
    mask[y1:y2, x1:x2] = 0

    # 라플라시안으로 선명도 측정
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    # 배경 영역의 평균 블러
    background_laplacian = laplacian[mask > 0]
    if len(background_laplacian) > 0:
        blur_score = 1.0 / (1.0 + np.var(background_laplacian))
    else:
        blur_score = 0.5

    return blur_score


class EnhancedFeatureExtractor:
    """
    향상된 특징 추출 파이프라인

    - 종횡비 인식
    - 풍부한 특징 추출
    - 신뢰도 레벨 구분
    - Theme × ShotType 구조 지원
    """

    def __init__(self, config: Dict):
        self.config = config
        self.device = config.get("models", {}).get("grounding_dino", {}).get("device", "cuda")

        # 모델 초기화
        self.grounding_dino: Optional[GroundingDINOWrapper] = None
        self.depth_anything: Optional[DepthAnythingWrapper] = None
        self.rtmpose: Optional[RTMPoseWrapper] = None

        # 유틸리티
        self.pose_classifier = PoseClassifier(
            confidence_threshold=config.get("pose_classification", {}).get("confidence_threshold", 0.3)
        )
        self.composition_analyzer = CompositionAnalyzer()
        self.statistics = PatternStatistics(
            min_samples=config.get("statistics", {}).get("min_samples", 10),
            remove_outliers=config.get("statistics", {}).get("remove_outliers", True)
        )

        # 추출된 전체 데이터 저장
        self.all_features = []

    def load_models(self):
        """모델 로드"""
        print("\n" + "=" * 60)
        print("Loading Models...")
        print("=" * 60)

        # Grounding DINO
        gd_config = self.config.get("models", {}).get("grounding_dino", {})
        self.grounding_dino = GroundingDINOWrapper(
            model_id=gd_config.get("model_id", "IDEA-Research/grounding-dino-base"),
            box_threshold=gd_config.get("box_threshold", 0.35),
            text_threshold=gd_config.get("text_threshold", 0.25),
            device=self.device
        )
        self.grounding_dino.load()

        # Depth Anything V2
        da_config = self.config.get("models", {}).get("depth_anything", {})
        self.depth_anything = DepthAnythingWrapper(
            model_id=da_config.get("model_id", "depth-anything/Depth-Anything-V2-Large"),
            device=self.device
        )
        self.depth_anything.load()

        # RTMPose
        rtm_config = self.config.get("models", {}).get("rtmpose", {})
        self.rtmpose = RTMPoseWrapper(
            config=rtm_config.get("config", "rtmpose-l_8xb256-420e_body8-256x192"),
            checkpoint=rtm_config.get("checkpoint"),
            device=self.device,
            confidence_threshold=self.config.get("pose_classification", {}).get("confidence_threshold", 0.3)
        )
        self.rtmpose.load()

        print("\n[OK] All models loaded!")

    def extract_comprehensive_features(
        self,
        image_path: Path,
        theme: str,
        shot_type: str,
        bg_prompts: List[str]
    ) -> Optional[Dict]:
        """
        종합적인 특징 추출

        Returns:
            {
                "certain": {...},      # 신뢰도 95%+
                "useful": {...},       # 신뢰도 70-95%
                "experimental": {...}, # 신뢰도 50-70%
                "metadata": {...}      # 메타정보
            }
        """
        try:
            # 이미지 로드
            image = Image.open(image_path).convert("RGB")
            original_size = image.size
            max_size = self.config.get("extraction", {}).get("resize_max", 1024)
            image_resized = resize_image(image, max_size)

            # 종횡비 감지
            aspect_ratio_label, aspect_ratio_value = detect_aspect_ratio(image)

            # numpy 변환
            image_np = np.array(image_resized)

            # ========== Level 1: 확실한 정보 (95%+ 신뢰도) ==========

            # Grounding DINO: Person detection
            gd_result = self.grounding_dino.detect(
                image_resized,
                person_prompt=self.config.get("extraction", {}).get("person_prompt", "person"),
                bg_prompts=bg_prompts
            )

            if gd_result.person_bbox is None:
                return None  # Person 없으면 스킵

            person_bbox = gd_result.person_bbox  # [x1, y1, x2, y2] normalized
            h, w = image_resized.size[1], image_resized.size[0]

            # 정확한 여백 계산
            margins = {
                "top": person_bbox[1],
                "bottom": 1.0 - person_bbox[3],
                "left": person_bbox[0],
                "right": 1.0 - person_bbox[2],
                "top_pixels": int(person_bbox[1] * h),
                "bottom_pixels": int((1.0 - person_bbox[3]) * h),
                "left_pixels": int(person_bbox[0] * w),
                "right_pixels": int((1.0 - person_bbox[2]) * w)
            }

            # 위치 계산
            position = {
                "center_x": (person_bbox[0] + person_bbox[2]) / 2,
                "center_y": (person_bbox[1] + person_bbox[3]) / 2,
                "bbox": list(person_bbox)
            }

            # 크기 계산
            size_info = {
                "width_ratio": person_bbox[2] - person_bbox[0],
                "height_ratio": person_bbox[3] - person_bbox[1],
                "area_ratio": (person_bbox[2] - person_bbox[0]) * (person_bbox[3] - person_bbox[1]),
                "aspect_ratio_person": (person_bbox[3] - person_bbox[1]) / max(person_bbox[2] - person_bbox[0], 0.001)
            }

            # 여백 밸런스
            margin_balance = {
                "horizontal_balance": margins["left"] / max(margins["left"] + margins["right"], 0.001),
                "vertical_balance": margins["top"] / max(margins["top"] + margins["bottom"], 0.001),
                "is_centered_x": 0.45 < position["center_x"] < 0.55,
                "is_centered_y": 0.45 < position["center_y"] < 0.55
            }

            certain_features = {
                "aspect_ratio": aspect_ratio_label,
                "aspect_ratio_value": aspect_ratio_value,
                "original_size": original_size,
                "margins": margins,
                "position": position,
                "size": size_info,
                "margin_balance": margin_balance,
                "background_objects": [
                    {"label": obj.label, "confidence": obj.confidence}
                    for obj in gd_result.background_objects
                ]
            }

            # ========== Level 2: 유용한 정보 (70-95% 신뢰도) ==========

            # Depth Anything: 압축감 분석
            depth_result = self.depth_anything.analyze(
                image_resized,
                person_bbox=gd_result.person_bbox
            )

            # RTMPose: Pose 분석
            pose_result = self.rtmpose.predict(image_resized)

            # Pose 분류
            if pose_result.keypoints:
                keypoints_dict = [
                    {"name": kp.name, "x": kp.x, "y": kp.y, "confidence": kp.confidence}
                    for kp in pose_result.keypoints
                ]
                pose_class = self.pose_classifier.classify(keypoints_dict, gd_result.person_bbox)
            else:
                pose_class = self.pose_classifier.classify([], gd_result.person_bbox)

            # 배경 블러 측정
            blur_score = measure_background_blur(image_np, person_bbox)

            # 압축감 추정 (여러 신호 조합)
            compression_hints = {
                "depth_compression": depth_result.compression_index,
                "depth_camera_type": depth_result.camera_type,
                "background_blur": blur_score,
                "estimated_type": self._estimate_compression_type(
                    depth_result.compression_index,
                    blur_score,
                    size_info["area_ratio"]
                )
            }

            useful_features = {
                "shot_type_detected": pose_class.pose_type,
                "sitting": pose_class.sitting,
                "visible_joints": pose_class.visible_joints,
                "pose_confidence": pose_class.confidence,
                "compression": compression_hints,
                "keypoints_count": len(pose_result.keypoints) if pose_result.keypoints else 0
            }

            # ========== Level 3: 실험적 정보 (50-70% 신뢰도) ==========

            experimental_features = {
                "estimated_angle": pose_result.angle_estimation.get("estimated_angle", 0),
                "angle_confidence": pose_result.angle_estimation.get("confidence", 0),
                "rule_of_thirds_score": self.composition_analyzer.analyze(gd_result.person_bbox).rule_of_thirds_score,
                "golden_ratio_fit": self._check_golden_ratio(position["center_x"], position["center_y"])
            }

            # ========== Metadata ==========

            metadata = {
                "filename": image_path.name,
                "theme": theme,
                "shot_type": shot_type,
                "extraction_timestamp": datetime.now().isoformat(),
                "confidence_levels": {
                    "certain": 0.95,
                    "useful": 0.80,
                    "experimental": 0.60
                }
            }

            return {
                "certain": certain_features,
                "useful": useful_features,
                "experimental": experimental_features,
                "metadata": metadata
            }

        except Exception as e:
            print(f"\n[Error] {image_path.name}: {e}")
            return None

    def _estimate_compression_type(self, depth_index: float, blur: float, size: float) -> str:
        """압축감 타입 추정"""
        if depth_index < 0.3 and blur < 0.3:
            return "wide"
        elif depth_index > 0.7 and blur > 0.6:
            return "tele"
        else:
            return "normal"

    def _check_golden_ratio(self, x: float, y: float) -> float:
        """황금비 적합도 체크"""
        golden = 0.618
        score = 0.0

        # 가로 황금비
        if abs(x - golden) < 0.1 or abs(x - (1 - golden)) < 0.1:
            score += 0.5

        # 세로 황금비
        if abs(y - golden) < 0.1 or abs(y - (1 - golden)) < 0.1:
            score += 0.5

        return score

    def run(self):
        """전체 파이프라인 실행 (Theme/ShotType 구조 지원)"""
        print("\n" + "=" * 60)
        print("TryAngle v1.5 Enhanced Feature Extraction")
        print("=" * 60)

        # 경로 설정
        input_dir = Path(self.config["paths"]["input_dir"])
        output_dir = Path(self.config["paths"]["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        # 모델 로드
        self.load_models()

        # 테마별 처리
        themes = self.config.get("themes", [])
        extraction_config = self.config.get("extraction", {})

        total_processed = 0
        total_failed = 0

        for theme_config in themes:
            theme_name = theme_config["name"]
            theme_folder = theme_config["folder"]
            target_samples = theme_config.get("target_samples", 100)

            theme_dir = input_dir / theme_folder

            if not theme_dir.exists():
                print(f"\n[Warning] Theme folder not found: {theme_dir}")
                continue

            print(f"\n{'='*60}")
            print(f"Processing Theme: {theme_name}")
            print(f"{'='*60}")

            # 배경 프롬프트
            bg_prompts = extraction_config.get("background_prompts", {}).get(theme_name, [])

            # Shot type별 처리 (v1.5 구조)
            shot_types = ["closeup", "medium_shot", "knee_shot", "full_shot"]

            for shot_type in shot_types:
                shot_dir = theme_dir / shot_type

                if not shot_dir.exists():
                    print(f"  [Skip] {shot_type} folder not found")
                    continue

                # 이미지 파일 목록
                image_files = get_image_files(shot_dir)[:target_samples]

                if not image_files:
                    print(f"  [Skip] No images in {shot_type}")
                    continue

                print(f"\n  Processing: {theme_name}/{shot_type}")
                print(f"    Found: {len(image_files)} images")

                # 특징 추출
                for image_path in tqdm(image_files, desc=f"    {shot_type}"):
                    features = self.extract_comprehensive_features(
                        image_path, theme_name, shot_type, bg_prompts
                    )

                    if features:
                        # 전체 데이터 저장
                        self.all_features.append(features)

                        # 통계에 추가 (Theme_ShotType 키로)
                        pattern_key = f"{theme_name}_{shot_type}"
                        self.statistics.add_sample(
                            theme=pattern_key,
                            pose_type=shot_type,
                            features=features,
                            filename=features["metadata"]["filename"]
                        )
                        total_processed += 1
                    else:
                        total_failed += 1

        # 통계 계산
        print("\n" + "=" * 60)
        print("Calculating Statistics...")
        print("=" * 60)

        self.statistics.calculate_all()
        print(self.statistics.summary())

        # JSON 저장 (풀 데이터)
        pattern_json_path = Path(self.config["paths"]["pattern_json"])
        pattern_json_path.parent.mkdir(parents=True, exist_ok=True)

        # 풀 패턴 저장
        full_patterns = self._create_full_patterns()
        with open(pattern_json_path, 'w', encoding='utf-8') as f:
            json.dump(full_patterns, f, indent=2, ensure_ascii=False)

        # 앱용 경량 패턴 저장
        app_json_path = pattern_json_path.parent / "patterns_app_v1.json"
        app_patterns = self._create_app_patterns(full_patterns)
        with open(app_json_path, 'w', encoding='utf-8') as f:
            json.dump(app_patterns, f, indent=2, ensure_ascii=False)

        # 원시 데이터 저장 (연구용)
        if self.config.get("processing", {}).get("save_intermediate", True):
            raw_path = output_dir / "raw_features.json"
            self._save_raw_features(raw_path)

        # 결과 요약
        print("\n" + "=" * 60)
        print("Extraction Complete!")
        print("=" * 60)
        print(f"  Total processed: {total_processed}")
        print(f"  Total failed: {total_failed}")
        print(f"  Patterns generated: {len(self.statistics.patterns)}")
        print(f"  Full patterns: {pattern_json_path}")
        print(f"  App patterns: {app_json_path}")

        return self.statistics.patterns

    def _create_full_patterns(self) -> Dict:
        """풀 패턴 생성 (모든 정보 포함)"""
        patterns = {}

        for (theme_shot, _), samples in self.statistics.samples.items():
            if len(samples) < 5:  # 최소 샘플
                continue

            # 종횡비별 분리
            aspect_groups = {}
            for sample in samples:
                aspect = sample["certain"]["aspect_ratio"]
                if aspect not in aspect_groups:
                    aspect_groups[aspect] = []
                aspect_groups[aspect].append(sample)

            pattern = {
                "sample_count": len(samples),
                "patterns_by_aspect": {},
                "common_features": {}
            }

            # 종횡비별 패턴
            for aspect, aspect_samples in aspect_groups.items():
                if len(aspect_samples) < 3:
                    continue

                # 통계 계산
                margins_top = [s["certain"]["margins"]["top"] for s in aspect_samples]
                margins_bottom = [s["certain"]["margins"]["bottom"] for s in aspect_samples]
                margins_left = [s["certain"]["margins"]["left"] for s in aspect_samples]
                margins_right = [s["certain"]["margins"]["right"] for s in aspect_samples]

                position_x = [s["certain"]["position"]["center_x"] for s in aspect_samples]
                position_y = [s["certain"]["position"]["center_y"] for s in aspect_samples]

                compression_indices = [s["useful"]["compression"]["depth_compression"] for s in aspect_samples]

                pattern["patterns_by_aspect"][aspect] = {
                    "sample_count": len(aspect_samples),
                    "margins": {
                        "top": {"mean": np.mean(margins_top), "std": np.std(margins_top)},
                        "bottom": {"mean": np.mean(margins_bottom), "std": np.std(margins_bottom)},
                        "left": {"mean": np.mean(margins_left), "std": np.std(margins_left)},
                        "right": {"mean": np.mean(margins_right), "std": np.std(margins_right)}
                    },
                    "position": {
                        "x": {"mean": np.mean(position_x), "std": np.std(position_x)},
                        "y": {"mean": np.mean(position_y), "std": np.std(position_y)}
                    },
                    "compression": {
                        "mean": np.mean(compression_indices),
                        "std": np.std(compression_indices),
                        "type_distribution": self._get_compression_distribution(aspect_samples)
                    }
                }

            patterns[theme_shot] = pattern

        return {
            "version": "1.0",
            "extraction_date": datetime.now().isoformat(),
            "total_images": len(self.all_features),
            "patterns": patterns
        }

    def _create_app_patterns(self, full_patterns: Dict) -> Dict:
        """앱용 경량 패턴 (필수 정보만)"""
        app_patterns = {
            "version": "1.0",
            "patterns": {}
        }

        for pattern_key, pattern_data in full_patterns["patterns"].items():
            app_pattern = {}

            for aspect, aspect_data in pattern_data["patterns_by_aspect"].items():
                app_pattern[f"{pattern_key}_{aspect}"] = {
                    "m": [  # margins (축약)
                        round(aspect_data["margins"]["top"]["mean"], 3),
                        round(aspect_data["margins"]["bottom"]["mean"], 3),
                        round(aspect_data["margins"]["left"]["mean"], 3),
                        round(aspect_data["margins"]["right"]["mean"], 3)
                    ],
                    "p": [  # position
                        round(aspect_data["position"]["x"]["mean"], 3),
                        round(aspect_data["position"]["y"]["mean"], 3)
                    ],
                    "c": round(aspect_data["compression"]["mean"], 2)  # compression
                }

            app_patterns["patterns"].update(app_pattern)

        return app_patterns

    def _get_compression_distribution(self, samples: List[Dict]) -> Dict:
        """압축감 분포 계산"""
        types = [s["useful"]["compression"]["estimated_type"] for s in samples]
        total = len(types)

        return {
            "wide": types.count("wide") / total if total > 0 else 0,
            "normal": types.count("normal") / total if total > 0 else 0,
            "tele": types.count("tele") / total if total > 0 else 0
        }

    def _save_raw_features(self, path: Path):
        """원시 특징 데이터 저장 (연구용)"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "total_samples": len(self.all_features),
                "features": self.all_features[:100]  # 처음 100개만 (용량 때문에)
            }, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="TryAngle v1.5 Enhanced Feature Extraction")
    parser.add_argument(
        "--config",
        type=str,
        default=str(SCRIPT_DIR / "config.yaml"),
        help="Path to config.yaml"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use"
    )

    args = parser.parse_args()

    # 설정 로드
    config = load_config(args.config)

    # 디바이스 설정
    if args.device:
        config["models"]["grounding_dino"]["device"] = args.device
        config["models"]["depth_anything"]["device"] = args.device
        config["models"]["rtmpose"]["device"] = args.device

    # 실행
    extractor = EnhancedFeatureExtractor(config)
    extractor.run()


if __name__ == "__main__":
    main()