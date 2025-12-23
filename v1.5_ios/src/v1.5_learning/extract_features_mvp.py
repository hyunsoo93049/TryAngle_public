#!/usr/bin/env python3
# ============================================================
# TryAngle v1.5 - MVP Feature Extraction Pipeline
# 오프라인 학습: 분류된 이미지에서 패턴 추출
# ============================================================

import os
import sys
import argparse
import yaml
from pathlib import Path
from typing import Dict, List, Optional
from PIL import Image
from tqdm import tqdm
from datetime import datetime
import json

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


class MVPExtractor:
    """
    MVP 특징 추출 파이프라인

    1. Grounding DINO: person bbox + 배경 객체
    2. Depth Anything V2: 압축감 분석
    3. RTMPose: pose type + angle 추정
    4. Statistics: 패턴 통계 계산
    """

    def __init__(self, config: Dict):
        self.config = config
        self.device = config.get("models", {}).get("grounding_dino", {}).get("device", "cuda")

        # 모델 초기화 (lazy loading)
        self.grounding_dino: Optional[GroundingDINOWrapper] = None
        self.depth_anything: Optional[DepthAnythingWrapper] = None
        self.rtmpose: Optional[RTMPoseWrapper] = None

        # 유틸리티
        self.pose_classifier = PoseClassifier(
            confidence_threshold=config.get("pose_classification", {}).get("confidence_threshold", 0.3)
        )
        self.composition_analyzer = CompositionAnalyzer()
        self.statistics = PatternStatistics(
            min_samples=config.get("statistics", {}).get("min_samples", 20),
            remove_outliers=config.get("statistics", {}).get("remove_outliers", True)
        )

    def load_models(self):
        """모델 로드"""
        print("\n" + "=" * 60)
        print("Loading Models...")
        print("=" * 60)

        gd_config = self.config.get("models", {}).get("grounding_dino", {})
        self.grounding_dino = GroundingDINOWrapper(
            model_id=gd_config.get("model_id", "IDEA-Research/grounding-dino-base"),
            box_threshold=gd_config.get("box_threshold", 0.35),
            text_threshold=gd_config.get("text_threshold", 0.25),
            device=self.device
        )
        self.grounding_dino.load()

        da_config = self.config.get("models", {}).get("depth_anything", {})
        self.depth_anything = DepthAnythingWrapper(
            model_id=da_config.get("model_id", "depth-anything/Depth-Anything-V2-Large"),
            device=self.device
        )
        self.depth_anything.load()

        rtm_config = self.config.get("models", {}).get("rtmpose", {})
        self.rtmpose = RTMPoseWrapper(
            config=rtm_config.get("config", "rtmpose-l_8xb256-420e_body8-256x192"),
            checkpoint=rtm_config.get("checkpoint"),
            device=self.device,
            confidence_threshold=self.config.get("pose_classification", {}).get("confidence_threshold", 0.3)
        )
        self.rtmpose.load()

        print("\n[OK] All models loaded!")

    def extract_single_image(
        self,
        image_path: Path,
        theme: str,
        bg_prompts: List[str]
    ) -> Optional[Dict]:
        """
        단일 이미지에서 특징 추출

        Returns:
            {
                "filename": str,
                "theme": str,
                "pose_type": str,
                "position": [x, y],
                "size_ratio": float,
                "margins": {...},
                "compression_index": float,
                "estimated_angle": float,
                "sitting": bool,
                "visible_joints": [...],
                "background_objects": [...]
            }
        """
        try:
            # 이미지 로드
            image = Image.open(image_path).convert("RGB")
            max_size = self.config.get("extraction", {}).get("resize_max", 1024)
            image = resize_image(image, max_size)

            # 1. Grounding DINO: Person bbox + 배경 객체
            gd_result = self.grounding_dino.detect(
                image,
                person_prompt=self.config.get("extraction", {}).get("person_prompt", "person"),
                bg_prompts=bg_prompts
            )

            if gd_result.person_bbox is None:
                return None  # Person 없으면 스킵

            # 2. Depth Anything: 압축감 분석
            depth_result = self.depth_anything.analyze(
                image,
                person_bbox=gd_result.person_bbox
            )

            # 3. RTMPose: Pose 분석
            pose_result = self.rtmpose.predict(image)

            # 4. Pose Type 분류 (RTMPose 결과 또는 재분류)
            if pose_result.keypoints:
                keypoints_dict = [
                    {"name": kp.name, "x": kp.x, "y": kp.y, "confidence": kp.confidence}
                    for kp in pose_result.keypoints
                ]
                pose_class = self.pose_classifier.classify(keypoints_dict, gd_result.person_bbox)
            else:
                # Keypoint 없으면 bbox 기반으로 추정
                pose_class = self.pose_classifier.classify([], gd_result.person_bbox)

            # 5. 구도 분석
            composition = self.composition_analyzer.analyze(gd_result.person_bbox)

            # 결과 조합
            features = {
                "filename": image_path.name,
                "theme": theme,
                "pose_type": pose_class.pose_type,

                # 구도
                "position": list(composition.position),
                "size_ratio": composition.size_ratio,
                "margins": composition.margins,
                "rule_of_thirds_score": composition.rule_of_thirds_score,
                "horizontal_asymmetry": composition.horizontal_asymmetry,

                # 카메라
                "compression_index": depth_result.compression_index,
                "camera_type": depth_result.camera_type,
                "estimated_angle": pose_result.angle_estimation.get("estimated_angle", 0),
                "angle_confidence": pose_result.angle_estimation.get("confidence", 0),

                # 포즈
                "sitting": pose_class.sitting,
                "visible_joints": pose_class.visible_joints,
                "pose_confidence": pose_class.confidence,

                # 배경
                "background_objects": [
                    {"label": obj.label, "direction": self._get_direction(gd_result.person_bbox, obj.bbox)}
                    for obj in gd_result.background_objects
                ]
            }

            return features

        except Exception as e:
            print(f"\n[Error] {image_path.name}: {e}")
            return None

    def _get_direction(self, person_bbox, obj_bbox) -> str:
        """배경 객체의 person 대비 방향"""
        person_center_x = (person_bbox[0] + person_bbox[2]) / 2
        obj_center_x = (obj_bbox[0] + obj_bbox[2]) / 2

        if obj_center_x < person_center_x - 0.1:
            return "left"
        elif obj_center_x > person_center_x + 0.1:
            return "right"
        else:
            return "center"

    def run(self):
        """전체 파이프라인 실행"""
        print("\n" + "=" * 60)
        print("TryAngle v1.5 MVP Feature Extraction")
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
            print(f"Processing: {theme_name}")
            print(f"  Folder: {theme_dir}")
            print(f"  Target: {target_samples} samples")
            print(f"{'='*60}")

            # 배경 프롬프트
            bg_prompts = extraction_config.get("background_prompts", {}).get(theme_name, [])

            # 이미지 파일 목록
            image_files = get_image_files(theme_dir)[:target_samples]
            print(f"  Found: {len(image_files)} images")

            # 특징 추출
            for image_path in tqdm(image_files, desc=f"  {theme_name}"):
                features = self.extract_single_image(image_path, theme_name, bg_prompts)

                if features:
                    self.statistics.add_sample(
                        theme=theme_name,
                        pose_type=features["pose_type"],
                        features=features,
                        filename=features["filename"]
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

        # JSON 저장
        pattern_json_path = Path(self.config["paths"]["pattern_json"])
        pattern_json_path.parent.mkdir(parents=True, exist_ok=True)
        self.statistics.save_json(str(pattern_json_path))

        # 결과 요약
        print("\n" + "=" * 60)
        print("Extraction Complete!")
        print("=" * 60)
        print(f"  Total processed: {total_processed}")
        print(f"  Total failed: {total_failed}")
        print(f"  Patterns generated: {len(self.statistics.patterns)}")
        print(f"  Output: {pattern_json_path}")

        # 중간 결과 저장 (선택적)
        if self.config.get("processing", {}).get("save_intermediate", True):
            intermediate_path = output_dir / "intermediate_features.json"
            self._save_intermediate(intermediate_path)
            print(f"  Intermediate: {intermediate_path}")

        return self.statistics.patterns

    def _save_intermediate(self, path: Path):
        """중간 결과 저장 (디버깅용)"""
        data = {
            "timestamp": datetime.now().isoformat(),
            "samples": {
                f"{theme}_{pose}": [
                    {k: v for k, v in s.items() if not isinstance(v, (list, dict)) or k in ["position", "margins"]}
                    for s in samples[:10]  # 처음 10개만
                ]
                for (theme, pose), samples in self.statistics.samples.items()
            }
        }

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="TryAngle v1.5 MVP Feature Extraction")
    parser.add_argument(
        "--config",
        type=str,
        default=str(SCRIPT_DIR / "config.yaml"),
        help="Path to config.yaml"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        help="Override input directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Override output directory"
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

    # 오버라이드
    if args.input_dir:
        config["paths"]["input_dir"] = args.input_dir
    if args.output_dir:
        config["paths"]["output_dir"] = args.output_dir
        config["paths"]["pattern_json"] = str(Path(args.output_dir) / "patterns_mvp_v1.json")
    if args.device:
        config["models"]["grounding_dino"]["device"] = args.device
        config["models"]["depth_anything"]["device"] = args.device
        config["models"]["rtmpose"]["device"] = args.device

    # 실행
    extractor = MVPExtractor(config)
    extractor.run()


if __name__ == "__main__":
    main()
