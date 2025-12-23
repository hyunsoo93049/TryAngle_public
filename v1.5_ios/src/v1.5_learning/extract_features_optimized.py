#!/usr/bin/env python3
# ============================================================
# TryAngle v1.5 - Optimized Feature Extraction Pipeline
# 메모리 최적화 + 체크포인트 + GPU 가속
# ============================================================

import os
import sys
import argparse
import yaml
import numpy as np
import torch
import gc
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PIL import Image
from tqdm import tqdm
from datetime import datetime
import json
import cv2
import pickle
import hashlib

# 모듈 경로 추가
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from models import GroundingDINOWrapper, DepthAnythingWrapper  # RTMPoseWrapper 제거
from utils import PoseClassifier, CompositionAnalyzer, PatternStatistics

# GPU 메모리 관리
def clear_gpu_cache():
    """GPU 메모리 캐시 정리"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def set_gpu_memory_fraction(fraction=0.8):
    """GPU 메모리 사용량 제한"""
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(fraction)


# ============================================================
# Aspect Ratio Analysis for v1.5
# ============================================================

def get_aspect_ratio_info(width: int, height: int) -> Tuple[str, str]:
    """
    종횡비와 방향을 구분해서 반환

    Returns:
        (aspect_ratio, orientation)
        예: ("16:9", "landscape") 또는 ("16:9", "portrait")
    """
    # 1. 방향 판단
    if width > height:
        orientation = "landscape"
        ratio = width / height
    elif height > width:
        orientation = "portrait"
        ratio = height / width
    else:
        orientation = "square"
        ratio = 1.0

    # 2. 종횡비 판단 (항상 큰수:작은수 형식)
    if abs(ratio - 1.0) < 0.1:
        aspect_ratio = "1:1"
    elif abs(ratio - 1.33) < 0.1:  # 4:3
        aspect_ratio = "4:3"
    elif abs(ratio - 1.78) < 0.1:  # 16:9
        aspect_ratio = "16:9"
    else:
        # 기타 비율은 가장 가까운 것으로
        if ratio < 1.2:
            aspect_ratio = "1:1"
        elif ratio < 1.55:
            aspect_ratio = "4:3"
        else:
            aspect_ratio = "16:9"

    return aspect_ratio, orientation

# 종횡비 패턴 타입 (방향과 무관)
ASPECT_PATTERNS = {
    "1:1": "square",     # 정방형
    "4:3": "standard",   # 전통적 사진 비율
    "16:9": "wide"       # 와이드스크린
}

def apply_orientation_adjustment(margins: dict, source_orientation: str, target_orientation: str) -> dict:
    """
    방향이 다를 때 여백 조정
    예: landscape 패턴을 portrait 촬영에 적용
    """
    if source_orientation == target_orientation:
        return margins  # 같은 방향이면 조정 불필요

    adjusted = {}

    if source_orientation == "landscape" and target_orientation == "portrait":
        # 가로 → 세로: 좌우 여백을 상하로, 상하를 좌우로
        adjusted["left"] = margins["top"]
        adjusted["right"] = margins["bottom"]
        adjusted["top"] = margins["left"]
        adjusted["bottom"] = margins["right"]

    elif source_orientation == "portrait" and target_orientation == "landscape":
        # 세로 → 가로: 상하 여백을 좌우로, 좌우를 상하로
        adjusted["left"] = margins["top"]
        adjusted["right"] = margins["bottom"]
        adjusted["top"] = margins["left"]
        adjusted["bottom"] = margins["right"]

    else:
        # square는 조정 불필요
        adjusted = margins

    return adjusted


class CheckpointManager:
    """체크포인트 관리 (중단/재개 기능)"""

    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.progress_file = self.checkpoint_dir / "progress.json"
        self.features_dir = self.checkpoint_dir / "features"
        self.features_dir.mkdir(exist_ok=True)

        self.processed_files = set()
        self.load_progress()

    def load_progress(self):
        """이전 진행 상황 로드"""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                data = json.load(f)
                self.processed_files = set(data.get("processed_files", []))
            print(f"[Checkpoint] Resuming from {len(self.processed_files)} processed files")

    def save_progress(self, new_file: str):
        """진행 상황 저장"""
        self.processed_files.add(new_file)

        with open(self.progress_file, 'w') as f:
            json.dump({
                "processed_files": list(self.processed_files),
                "last_update": datetime.now().isoformat()
            }, f)

    def is_processed(self, file_path: str) -> bool:
        """이미 처리된 파일인지 확인"""
        return str(file_path) in self.processed_files

    def save_batch_features(self, features: List[Dict], batch_id: int):
        """배치 단위로 특징 저장"""
        batch_file = self.features_dir / f"batch_{batch_id:04d}.pkl"
        with open(batch_file, 'wb') as f:
            pickle.dump(features, f)
        return batch_file

    def load_all_features(self) -> List[Dict]:
        """저장된 모든 특징 로드"""
        all_features = []

        batch_files = sorted(self.features_dir.glob("batch_*.pkl"))
        for batch_file in batch_files:
            with open(batch_file, 'rb') as f:
                batch_features = pickle.load(f)
                all_features.extend(batch_features)

        return all_features


class MemoryEfficientStatistics:
    """메모리 효율적인 통계 계산"""

    def __init__(self, checkpoint_dir: Path):
        self.stats_file = checkpoint_dir / "running_stats.json"
        self.stats = {}
        self.load_stats()

    def load_stats(self):
        """기존 통계 로드"""
        if self.stats_file.exists():
            with open(self.stats_file, 'r') as f:
                self.stats = json.load(f)

    def update_incremental(self, pattern_key: str, features: Dict):
        """증분 통계 업데이트 (Welford's algorithm)"""
        if pattern_key not in self.stats:
            self.stats[pattern_key] = {
                "count": 0,
                "margins_sum": {"top": 0, "bottom": 0, "left": 0, "right": 0},
                "margins_sum_sq": {"top": 0, "bottom": 0, "left": 0, "right": 0},
                "position_sum": {"x": 0, "y": 0},
                "position_sum_sq": {"x": 0, "y": 0},
                "compression_sum": 0,
                "compression_sum_sq": 0,
                "aspect_ratios": {}
            }

        s = self.stats[pattern_key]
        s["count"] += 1

        # 여백 통계
        margins = features["certain"]["margins"]
        for key in ["top", "bottom", "left", "right"]:
            val = margins[key]
            s["margins_sum"][key] += val
            s["margins_sum_sq"][key] += val ** 2

        # 위치 통계
        pos = features["certain"]["position"]
        s["position_sum"]["x"] += pos["center_x"]
        s["position_sum_sq"]["x"] += pos["center_x"] ** 2
        s["position_sum"]["y"] += pos["center_y"]
        s["position_sum_sq"]["y"] += pos["center_y"] ** 2

        # 압축감 통계
        comp = features["useful"]["compression"]["depth_compression"]
        s["compression_sum"] += comp
        s["compression_sum_sq"] += comp ** 2

        # 종횡비별 카운트
        aspect = features["certain"]["aspect_ratio"]
        if aspect not in s["aspect_ratios"]:
            s["aspect_ratios"][aspect] = 0
        s["aspect_ratios"][aspect] += 1

        # 주기적으로 디스크에 저장
        if s["count"] % 10 == 0:
            self.save_stats()

    def save_stats(self):
        """통계 저장"""
        with open(self.stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)

    def calculate_final_stats(self) -> Dict:
        """최종 통계 계산"""
        final_stats = {}

        for pattern_key, s in self.stats.items():
            if s["count"] < 5:
                continue

            n = s["count"]

            # 평균과 표준편차 계산
            final = {
                "sample_count": n,
                "margins": {},
                "position": {},
                "compression": {}
            }

            # 여백
            for key in ["top", "bottom", "left", "right"]:
                mean = s["margins_sum"][key] / n
                variance = (s["margins_sum_sq"][key] / n) - (mean ** 2)
                std = np.sqrt(max(0, variance))
                final["margins"][key] = {"mean": mean, "std": std}

            # 위치
            for key in ["x", "y"]:
                mean = s["position_sum"][key] / n
                variance = (s["position_sum_sq"][key] / n) - (mean ** 2)
                std = np.sqrt(max(0, variance))
                final["position"][key] = {"mean": mean, "std": std}

            # 압축감
            mean = s["compression_sum"] / n
            variance = (s["compression_sum_sq"] / n) - (mean ** 2)
            std = np.sqrt(max(0, variance))
            final["compression"] = {"mean": mean, "std": std}

            # 종횡비 분포
            final["aspect_distribution"] = s["aspect_ratios"]

            final_stats[pattern_key] = final

        return final_stats


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
    """이미지 리사이즈 (메모리 절약)"""
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
    """배경 블러 정도 측정 (GPU 가속)"""
    h, w = image_np.shape[:2]

    x1 = int(person_bbox[0] * w)
    y1 = int(person_bbox[1] * h)
    x2 = int(person_bbox[2] * w)
    y2 = int(person_bbox[3] * h)

    # 마스크 생성
    mask = np.ones((h, w), dtype=np.uint8) * 255
    mask[y1:y2, x1:x2] = 0

    # GPU 사용 가능하면 GPU로
    if torch.cuda.is_available():
        image_tensor = torch.from_numpy(image_np).cuda()
        gray = torch.mean(image_tensor.float(), dim=2)

        # Laplacian (간단한 커널 연산)
        kernel = torch.tensor([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=torch.float32).cuda()

        # Conv2d로 라플라시안
        gray_unsqueezed = gray.unsqueeze(0).unsqueeze(0)
        kernel_unsqueezed = kernel.unsqueeze(0).unsqueeze(0)

        laplacian = torch.nn.functional.conv2d(
            gray_unsqueezed,
            kernel_unsqueezed,
            padding=1
        )

        laplacian_np = laplacian.squeeze().cpu().numpy()
    else:
        # CPU 폴백
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        laplacian_np = cv2.Laplacian(gray, cv2.CV_64F)

    # 배경 영역의 평균 블러
    background_laplacian = laplacian_np[mask > 0]
    if len(background_laplacian) > 0:
        blur_score = 1.0 / (1.0 + np.var(background_laplacian))
    else:
        blur_score = 0.5

    return float(blur_score)


class OptimizedFeatureExtractor:
    """
    메모리 최적화된 특징 추출 파이프라인

    - 배치 처리
    - 체크포인트 지원
    - GPU 가속
    - 메모리 효율적 통계
    """

    def __init__(self, config: Dict):
        self.config = config
        self.device = config.get("models", {}).get("grounding_dino", {}).get("device", "cuda")

        # GPU 설정
        if self.device == "cuda" and torch.cuda.is_available():
            set_gpu_memory_fraction(0.8)
            print(f"[GPU] Using {torch.cuda.get_device_name(0)}")

        # 배치 크기
        self.batch_size = config.get("processing", {}).get("batch_size", 10)

        # 체크포인트 매니저
        output_dir = Path(config["paths"]["output_dir"])
        self.checkpoint_mgr = CheckpointManager(output_dir / "checkpoints")

        # 통계 매니저
        self.stats_mgr = MemoryEfficientStatistics(output_dir / "checkpoints")

        # 모델 (lazy loading)
        self.grounding_dino: Optional[GroundingDINOWrapper] = None
        self.depth_anything: Optional[DepthAnythingWrapper] = None
        self.rtmpose: Optional[RTMPoseWrapper] = None

        # 유틸리티
        self.pose_classifier = PoseClassifier(
            confidence_threshold=config.get("pose_classification", {}).get("confidence_threshold", 0.3)
        )
        self.composition_analyzer = CompositionAnalyzer()

    def load_models(self):
        """모델 로드 (GPU 메모리 효율적)"""
        print("\n" + "=" * 60)
        print("Loading Models with GPU Optimization...")
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
        clear_gpu_cache()

        # Depth Anything V2
        da_config = self.config.get("models", {}).get("depth_anything", {})
        self.depth_anything = DepthAnythingWrapper(
            model_id=da_config.get("model_id", "depth-anything/Depth-Anything-V2-Large"),
            device=self.device
        )
        self.depth_anything.load()
        clear_gpu_cache()

        # RTMPose - SKIP (use folder structure for pose_type)
        print("[INFO] RTMPose skipped - using folder structure for pose_type")
        self.rtmpose = None
        clear_gpu_cache()

        print("\n[OK] All models loaded with GPU optimization!")

    def extract_single_feature(
        self,
        image_path: Path,
        theme: str,
        shot_type: str,
        bg_prompts: List[str]
    ) -> Optional[Dict]:
        """단일 이미지 특징 추출"""
        try:
            # 이미지 로드
            image = Image.open(image_path).convert("RGB")
            original_size = image.size
            max_size = self.config.get("extraction", {}).get("resize_max", 1024)
            image_resized = resize_image(image, max_size)

            # 종횡비 및 방향 감지
            width, height = image.size
            aspect_ratio, orientation = get_aspect_ratio_info(width, height)
            aspect_type = ASPECT_PATTERNS.get(aspect_ratio, "unknown")

            # numpy 변환
            image_np = np.array(image_resized)

            # ========== Level 1: 확실한 정보 ==========

            # Grounding DINO
            gd_result = self.grounding_dino.detect(
                image_resized,
                person_prompt=self.config.get("extraction", {}).get("person_prompt", "person"),
                bg_prompts=bg_prompts
            )

            if gd_result.person_bbox is None:
                return None

            person_bbox = gd_result.person_bbox
            h, w = image_resized.size[1], image_resized.size[0]

            # 여백 계산
            margins = {
                "top": float(person_bbox[1]),
                "bottom": float(1.0 - person_bbox[3]),
                "left": float(person_bbox[0]),
                "right": float(1.0 - person_bbox[2])
            }

            # 위치 계산
            position = {
                "center_x": float((person_bbox[0] + person_bbox[2]) / 2),
                "center_y": float((person_bbox[1] + person_bbox[3]) / 2)
            }

            # 크기 계산
            size_info = {
                "width_ratio": float(person_bbox[2] - person_bbox[0]),
                "height_ratio": float(person_bbox[3] - person_bbox[1]),
                "area_ratio": float((person_bbox[2] - person_bbox[0]) * (person_bbox[3] - person_bbox[1]))
            }

            certain_features = {
                "aspect_ratio": aspect_ratio,      # "16:9", "4:3", "1:1"
                "orientation": orientation,         # "landscape", "portrait", "square"
                "aspect_type": aspect_type,        # "wide", "standard", "square"
                "margins": margins,
                "position": position,
                "size": size_info
            }

            # ========== Level 2: 유용한 정보 ==========

            # Depth Anything
            depth_result = self.depth_anything.analyze(
                image_resized,
                person_bbox=gd_result.person_bbox
            )

            # 배경 블러
            blur_score = measure_background_blur(image_np, person_bbox)

            # 압축감 타입 추정
            if depth_result.compression_index < 0.3:
                comp_type = "wide"
            elif depth_result.compression_index > 0.7:
                comp_type = "tele"
            else:
                comp_type = "normal"

            useful_features = {
                "compression": {
                    "depth_compression": float(depth_result.compression_index),
                    "background_blur": float(blur_score),
                    "estimated_type": comp_type
                }
            }

            # ========== Level 3: 실험적 정보 ==========

            experimental_features = {
                # RTMPose 제거로 angle estimation 불가
                # 실시간에서는 gyroscope 사용 예정
                "estimated_angle": 0.0
            }

            # ========== Metadata ==========

            metadata = {
                "filename": image_path.name,
                "theme": theme,
                "shot_type": shot_type
            }

            result = {
                "certain": certain_features,
                "useful": useful_features,
                "experimental": experimental_features,
                "metadata": metadata
            }

            # GPU 캐시 정리
            if self.device == "cuda":
                clear_gpu_cache()

            return result

        except Exception as e:
            print(f"\n[Error] {image_path.name}: {e}")
            if self.device == "cuda":
                clear_gpu_cache()
            return None

    def process_batch(
        self,
        image_files: List[Path],
        theme: str,
        shot_type: str,
        bg_prompts: List[str],
        batch_id: int
    ) -> List[Dict]:
        """배치 단위 처리"""
        batch_features = []

        for image_path in image_files:
            # 이미 처리된 파일 스킵
            if self.checkpoint_mgr.is_processed(str(image_path)):
                continue

            # 특징 추출
            features = self.extract_single_feature(
                image_path, theme, shot_type, bg_prompts
            )

            if features:
                batch_features.append(features)

                # 통계 업데이트
                pattern_key = f"{theme}_{shot_type}"
                self.stats_mgr.update_incremental(pattern_key, features)

                # 진행 상황 저장
                self.checkpoint_mgr.save_progress(str(image_path))

        # 배치 저장
        if batch_features:
            self.checkpoint_mgr.save_batch_features(batch_features, batch_id)

        # 메모리 정리
        del batch_features
        gc.collect()

        return []  # 메모리 절약을 위해 반환하지 않음

    def run(self):
        """메모리 최적화된 파이프라인 실행"""
        print("\n" + "=" * 60)
        print("TryAngle v1.5 Optimized Feature Extraction")
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
        total_skipped = 0
        batch_counter = 0

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

            # Shot type별 처리
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

                # 배치 처리
                for i in range(0, len(image_files), self.batch_size):
                    batch = image_files[i:i+self.batch_size]

                    # 이미 처리된 파일 확인
                    unprocessed = [f for f in batch if not self.checkpoint_mgr.is_processed(str(f))]

                    if not unprocessed:
                        total_skipped += len(batch)
                        continue

                    # 배치 처리
                    print(f"    Batch {i//self.batch_size + 1}/{(len(image_files)-1)//self.batch_size + 1}")

                    self.process_batch(
                        unprocessed,
                        theme_name,
                        shot_type,
                        bg_prompts,
                        batch_counter
                    )

                    batch_counter += 1
                    total_processed += len(unprocessed)
                    total_skipped += len(batch) - len(unprocessed)

                    # 주기적으로 통계 저장
                    if batch_counter % 5 == 0:
                        self.stats_mgr.save_stats()
                        gc.collect()

        # 최종 통계 계산 및 저장
        print("\n" + "=" * 60)
        print("Finalizing Statistics...")
        print("=" * 60)

        # 모든 배치 특징 로드
        print("  Loading all features...")
        all_features = self.checkpoint_mgr.load_all_features()

        # 최종 통계 계산
        final_stats = self.stats_mgr.calculate_final_stats()

        # 패턴 JSON 생성
        pattern_json_path = Path(self.config["paths"]["pattern_json"])
        pattern_json_path.parent.mkdir(parents=True, exist_ok=True)

        # 풀 패턴 저장
        full_patterns = self._create_patterns(final_stats, all_features)
        with open(pattern_json_path, 'w', encoding='utf-8') as f:
            json.dump(full_patterns, f, indent=2, ensure_ascii=False)

        # 앱용 경량 패턴 저장
        app_json_path = pattern_json_path.parent / "patterns_app_v1.json"
        app_patterns = self._create_app_patterns(final_stats)
        with open(app_json_path, 'w', encoding='utf-8') as f:
            json.dump(app_patterns, f, indent=2, ensure_ascii=False)

        # 결과 요약
        print("\n" + "=" * 60)
        print("Extraction Complete!")
        print("=" * 60)
        print(f"  Total processed: {total_processed}")
        print(f"  Total skipped: {total_skipped}")
        print(f"  Patterns generated: {len(final_stats)}")
        print(f"  Full patterns: {pattern_json_path}")
        print(f"  App patterns: {app_json_path}")

        # 체크포인트 정리 옵션 (백그라운드 실행 시 자동으로 유지)
        # if input("Clean checkpoint files? (y/n): ").lower() == 'y':
        #     import shutil
        #     shutil.rmtree(self.checkpoint_mgr.checkpoint_dir)
        #     print("  Checkpoint files cleaned.")
        print("  Checkpoint files kept for incremental processing.")

    def _create_patterns(self, stats: Dict, features: List[Dict]) -> Dict:
        """패턴 생성"""
        patterns = {}

        for pattern_key, pattern_stats in stats.items():
            # 종횡비별 분리
            aspect_patterns = {}

            for aspect, count in pattern_stats.get("aspect_distribution", {}).items():
                if count < 3:
                    continue

                # 해당 종횡비 샘플만 필터
                aspect_features = [
                    f for f in features
                    if f["metadata"].get("theme") + "_" + f["metadata"].get("shot_type") == pattern_key
                    and f["certain"]["aspect_ratio"] == aspect
                ]

                if len(aspect_features) >= 3:
                    aspect_patterns[aspect] = {
                        "sample_count": len(aspect_features),
                        "margins": {
                            k: pattern_stats["margins"][k]
                            for k in ["top", "bottom", "left", "right"]
                        },
                        "position": pattern_stats["position"],
                        "compression": pattern_stats["compression"]
                    }

            if aspect_patterns:
                patterns[pattern_key] = {
                    "sample_count": pattern_stats["sample_count"],
                    "patterns_by_aspect": aspect_patterns
                }

        return {
            "version": "1.0",
            "extraction_date": datetime.now().isoformat(),
            "patterns": patterns
        }

    def _create_app_patterns(self, stats: Dict) -> Dict:
        """앱용 경량 패턴"""
        app_patterns = {"version": "1.0", "patterns": {}}

        for pattern_key, pattern_stats in stats.items():
            for aspect in pattern_stats.get("aspect_distribution", {}).keys():
                key = f"{pattern_key}_{aspect}"

                app_patterns["patterns"][key] = {
                    "m": [  # margins
                        round(pattern_stats["margins"]["top"]["mean"], 3),
                        round(pattern_stats["margins"]["bottom"]["mean"], 3),
                        round(pattern_stats["margins"]["left"]["mean"], 3),
                        round(pattern_stats["margins"]["right"]["mean"], 3)
                    ],
                    "p": [  # position
                        round(pattern_stats["position"]["x"]["mean"], 3),
                        round(pattern_stats["position"]["y"]["mean"], 3)
                    ],
                    "c": round(pattern_stats["compression"]["mean"], 2)  # compression
                }

        return app_patterns


def main():
    parser = argparse.ArgumentParser(description="TryAngle v1.5 Optimized Feature Extraction")
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
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Batch size for processing"
    )

    args = parser.parse_args()

    # 설정 로드
    config = load_config(args.config)

    # 오버라이드
    if args.device:
        config["models"]["grounding_dino"]["device"] = args.device
        config["models"]["depth_anything"]["device"] = args.device
        config["models"]["rtmpose"]["device"] = args.device

    if args.batch_size:
        config["processing"]["batch_size"] = args.batch_size

    # 실행
    extractor = OptimizedFeatureExtractor(config)
    extractor.run()


if __name__ == "__main__":
    main()