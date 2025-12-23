#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TryAngle v1.5 - 자동 이미지 분류 스크립트
현재 clustering 폴더의 이미지를 Theme × Pose Type으로 재분류
"""

import os
import sys
import json
import shutil
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from tqdm import tqdm
import argparse
import numpy as np
import cv2

# 설정
USE_MODELS = False  # True로 변경하면 실제 모델 사용
COPY_MODE = True    # True: 복사, False: 이동

# 테마 분류 규칙
THEME_RULES = {
    'cafe_indoor': ['table', 'chair', 'coffee', 'window', 'cup', 'cafe'],
    'street_urban': ['building', 'car', 'road', 'sidewalk', 'sign', 'street'],
    'park_nature': ['tree', 'grass', 'bench', 'sky', 'flower', 'park'],
    'beach': ['ocean', 'sand', 'wave', 'umbrella', 'beach'],
    'winter': ['snow', 'ice', 'coat', 'scarf', 'winter'],
    'indoor_home': ['sofa', 'tv', 'bed', 'door', 'room']
}

# Pose Type 정의 (새로운 4-type 분류 체계)
POSE_TYPES = ['closeup', 'medium_shot', 'knee_shot', 'full_shot']


class ImageClassifier:
    """이미지 자동 분류기"""

    def __init__(self, source_dir: str, target_dir: str, use_models: bool = False):
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.use_models = use_models
        self.stats = {}

        # 모델 사용 시
        if self.use_models:
            print("[AI] 모델 로딩...")
            self._load_models()
        else:
            print("[Simple Mode] 폴더명 기반 분류")

    def _load_models(self):
        """실제 모델 로드 - rtmlib 사용 (ONNX 기반)"""
        try:
            from rtmlib import Wholebody

            print("[RTMPose] rtmlib 모델 로딩 (ONNX)...")

            # rtmlib Wholebody 모델 (YOLOX + RTMPose Wholebody)
            # mode: 'lightweight', 'balanced', 'performance'
            self.wholebody = Wholebody(
                mode='balanced',  # balanced가 정확도/속도 균형
                backend='onnxruntime'
            )

            print("[OK] rtmlib RTMPose 로드 완료")
        except Exception as e:
            print(f"[WARNING] 모델 로드 실패: {e}")
            print("Simple Mode로 전환합니다.")
            self.use_models = False

    def classify_theme(self, image_path: Path) -> str:
        """테마 분류"""
        if self.use_models:
            return self._classify_theme_with_model(image_path)
        else:
            return self._classify_theme_simple(image_path)

    def _classify_theme_simple(self, image_path: Path) -> str:
        """Simple Mode: 현재 폴더명 기반 분류"""
        path_str = str(image_path).lower()

        # 현재 폴더 구조 기반
        if 'cafe' in path_str or 'cluster_02' in path_str or 'cluster_05' in path_str:
            return 'cafe_indoor'
        elif 'park' in path_str or 'nature' in path_str or 'cluster_03' in path_str or 'cluster_07' in path_str:
            return 'park_nature'
        elif 'street' in path_str or 'urban' in path_str or 'cluster_12' in path_str or 'cluster_14' in path_str or 'cluster_09' in path_str or 'cluster_11' in path_str:
            return 'street_urban'
        elif 'winter' in path_str or 'cluster_06' in path_str:
            return 'winter'
        elif 'indoor' in path_str or 'home' in path_str or 'highangle' in path_str or 'cluster_08' in path_str:
            return 'indoor_home'
        elif 'person' in path_str:
            # person 폴더 내 클러스터별 분류 (레거시)
            if 'cluster_03' in path_str or 'cluster_07' in path_str:
                return 'park_nature'
            elif 'cluster_09' in path_str or 'cluster_11' in path_str:
                return 'street_urban'
            else:
                return 'park_nature'  # 기본값
        else:
            return 'unknown'

    def _classify_theme_with_model(self, image_path: Path) -> str:
        """Model Mode: Grounding DINO로 테마 분류"""
        # TODO: 실제 구현
        # objects = self.grounding_dino.detect(image_path)
        # scores = {}
        # for theme, keywords in THEME_RULES.items():
        #     score = sum(1 for obj in objects if any(kw in obj for kw in keywords))
        #     scores[theme] = score
        # return max(scores, key=scores.get)

        return self._classify_theme_simple(image_path)  # 임시

    def classify_pose_type(self, image_path: Path) -> str:
        """Pose Type 분류"""
        if self.use_models:
            return self._classify_pose_with_model(image_path)
        else:
            return self._classify_pose_simple(image_path)

    def _classify_pose_simple(self, image_path: Path) -> str:
        """Simple Mode: 랜덤 분포 (실제는 RTMPose 사용)"""
        # 새로운 4-type 분류 체계
        weights = {
            'closeup': 0.1,
            'medium_shot': 0.35,
            'knee_shot': 0.35,
            'full_shot': 0.2
        }
        return random.choices(
            list(weights.keys()),
            weights=list(weights.values())
        )[0]

    def _classify_pose_with_model(self, image_path: Path) -> str:
        """
        Model Mode: rtmlib RTMPose로 pose type 분류

        새로운 4-type 분류 체계:
        - closeup: 얼굴~어깨 (셀카, 클로즈업)
        - medium_shot: 가슴~허리 (버스트샷, 웨이스트샷 통합)
        - knee_shot: 허벅지~무릎
        - full_shot: 전신

        + 앉은 자세 감지 기능 포함
        """
        try:
            # 이미지 로드 (BGR)
            img = cv2.imread(str(image_path))
            if img is None:
                return self._classify_pose_simple(image_path)

            img_height = img.shape[0]
            img_width = img.shape[1]

            # rtmlib로 keypoint 추출
            keypoints, scores = self.wholebody(img)

            if keypoints is None or len(keypoints) == 0:
                return 'unknown'

            # 첫 번째 person의 keypoints 사용
            kpts = keypoints[0]  # (133, 2) - x, y coordinates
            scores_kpt = scores[0]  # (133,) - confidence scores

            # 주요 관절 인덱스 (COCO-WholeBody format)
            NOSE = 0
            LEFT_EYE = 1
            RIGHT_EYE = 2
            LEFT_SHOULDER = 5
            RIGHT_SHOULDER = 6
            LEFT_ELBOW = 7
            RIGHT_ELBOW = 8
            LEFT_HIP = 11
            RIGHT_HIP = 12
            LEFT_KNEE = 13
            RIGHT_KNEE = 14
            LEFT_ANKLE = 15
            RIGHT_ANKLE = 16

            # confidence threshold
            conf_threshold = 0.5

            def is_valid(idx):
                """keypoint가 유효한지 확인 (confidence 체크)"""
                return scores_kpt[idx] >= conf_threshold

            def is_in_frame(idx):
                """keypoint가 프레임 내에 있는지 확인"""
                if not is_valid(idx):
                    return False
                x, y = kpts[idx][0], kpts[idx][1]
                margin_x = img_width * 0.03
                margin_y = img_height * 0.03
                return (margin_x < x < img_width - margin_x and
                        margin_y < y < img_height - margin_y)

            def get_avg_y(indices):
                """유효한 keypoint들의 y좌표 평균 반환"""
                valid_ys = [kpts[idx][1] for idx in indices if is_valid(idx)]
                return np.mean(valid_ys) if valid_ys else None

            def get_visible_y(indices):
                """프레임 내 유효한 keypoint들의 y좌표 평균 반환"""
                valid_ys = [kpts[idx][1] for idx in indices if is_in_frame(idx)]
                return np.mean(valid_ys) if valid_ys else None

            # ============================================
            # 앉은 자세 감지
            # ============================================
            def detect_sitting():
                """
                앉은 자세인지 감지
                - 서 있을 때: (knee_y - hip_y) / (hip_y - shoulder_y) ≈ 1.0~1.5
                - 앉아 있을 때: (knee_y - hip_y) / (hip_y - shoulder_y) ≈ 0.2~0.5
                """
                shoulder_y = get_avg_y([LEFT_SHOULDER, RIGHT_SHOULDER])
                hip_y = get_avg_y([LEFT_HIP, RIGHT_HIP])
                knee_y = get_avg_y([LEFT_KNEE, RIGHT_KNEE])

                if shoulder_y is None or hip_y is None or knee_y is None:
                    return False, 0.0

                torso_length = hip_y - shoulder_y
                if torso_length <= 0:
                    return False, 0.0

                hip_to_knee = knee_y - hip_y
                ratio = hip_to_knee / torso_length

                # ratio < 0.6 이면 앉은 자세로 판단
                is_sitting = ratio < 0.6
                return is_sitting, ratio

            is_sitting, sit_ratio = detect_sitting()

            # ============================================
            # 각 부위별 Y좌표 계산 (프레임 내 가시 부위만)
            # ============================================
            head_y = get_visible_y([NOSE, LEFT_EYE, RIGHT_EYE])
            shoulder_y = get_visible_y([LEFT_SHOULDER, RIGHT_SHOULDER])
            elbow_y = get_visible_y([LEFT_ELBOW, RIGHT_ELBOW])
            hip_y = get_visible_y([LEFT_HIP, RIGHT_HIP])
            knee_y = get_visible_y([LEFT_KNEE, RIGHT_KNEE])
            ankle_y = get_visible_y([LEFT_ANKLE, RIGHT_ANKLE])

            # ============================================
            # Shot Type 결정 로직
            # ============================================

            # 1. 발목이 보이면 → full_shot 후보
            if ankle_y is not None:
                # 발목이 프레임 하단 85% 아래에 있으면 full_shot
                if ankle_y > img_height * 0.85:
                    return 'full_shot'
                # 앉은 자세에서 발목이 보이는 경우 → knee_shot으로 분류
                elif is_sitting:
                    return 'knee_shot'
                # 발목이 중간에 있으면 (잘린 전신) → knee_shot
                elif ankle_y > img_height * 0.7:
                    return 'knee_shot'
                else:
                    return 'medium_shot'

            # 2. 무릎이 보이면 → knee_shot
            if knee_y is not None:
                # 무릎이 프레임 하단 75% 아래면 확실한 knee_shot
                if knee_y > img_height * 0.75:
                    return 'knee_shot'
                # 앉은 자세에서 무릎이 보이면 → medium_shot
                elif is_sitting:
                    return 'medium_shot'
                # 무릎이 중간에 있으면 → knee_shot
                elif knee_y > img_height * 0.5:
                    return 'knee_shot'
                else:
                    return 'medium_shot'

            # 3. 골반(hip) 또는 팔꿈치(elbow)가 보이면 → medium_shot
            if hip_y is not None or elbow_y is not None:
                return 'medium_shot'

            # 4. 어깨까지만 보이면 → closeup vs medium_shot
            if shoulder_y is not None:
                # 어깨가 프레임 하단 70% 아래면 medium_shot
                if shoulder_y > img_height * 0.7:
                    return 'medium_shot'
                # 어깨가 중간에 있으면 closeup
                else:
                    return 'closeup'

            # 5. 머리만 보이면 → closeup
            if head_y is not None:
                return 'closeup'

            return 'unknown'

        except Exception as e:
            print(f"\n[WARNING] RTMPose 에러 ({image_path.name}): {e}")
            return self._classify_pose_simple(image_path)

    def process_all(self):
        """전체 이미지 처리"""
        print("\n[SEARCH] 이미지 검색 중...")

        # 모든 이미지 찾기 (Windows는 대소문자 구분 안함)
        image_extensions = ['.jpg', '.jpeg', '.png', '.webp']  # .JPG 제거
        all_images = []
        seen_paths = set()  # 중복 제거용

        for ext in image_extensions:
            for img_path in self.source_dir.rglob(f'*{ext}'):
                # 경로를 소문자로 변환하여 중복 체크
                normalized_path = str(img_path).lower()
                if normalized_path not in seen_paths:
                    all_images.append(img_path)
                    seen_paths.add(normalized_path)
            # 대문자 확장자도 체크 (Linux/Mac 호환성)
            for img_path in self.source_dir.rglob(f'*{ext.upper()}'):
                normalized_path = str(img_path).lower()
                if normalized_path not in seen_paths:
                    all_images.append(img_path)
                    seen_paths.add(normalized_path)

        total_images = len(all_images)
        print(f"[IMAGES] 총 {total_images}장의 이미지 발견\n")

        # 진행 상황 표시
        with tqdm(total=total_images, desc="분류 중") as pbar:
            for img_path in all_images:
                # 1. Theme 분류
                theme = self.classify_theme(img_path)

                # 2. Pose Type 분류
                pose_type = self.classify_pose_type(img_path)

                # 3. 새 경로 생성
                new_dir = self.target_dir / theme / pose_type
                new_dir.mkdir(parents=True, exist_ok=True)

                # 4. 파일 복사 또는 이동
                new_path = new_dir / img_path.name

                # 중복 파일명 처리
                if new_path.exists():
                    stem = img_path.stem
                    suffix = img_path.suffix
                    counter = 1
                    while new_path.exists():
                        new_path = new_dir / f"{stem}_{counter}{suffix}"
                        counter += 1

                if COPY_MODE:
                    shutil.copy2(img_path, new_path)
                else:
                    shutil.move(str(img_path), str(new_path))

                # 5. 통계 업데이트
                key = f"{theme}/{pose_type}"
                self.stats[key] = self.stats.get(key, 0) + 1

                pbar.update(1)

        print("\n[OK] 분류 완료!")
        self.print_statistics()
        self.save_statistics()

    def print_statistics(self):
        """분류 결과 통계 출력"""
        print("\n[STATS] 분류 결과:")
        print("=" * 50)

        # Theme별 집계
        theme_totals = {}
        for key, count in sorted(self.stats.items()):
            theme = key.split('/')[0]
            theme_totals[theme] = theme_totals.get(theme, 0) + count
            print(f"  {key}: {count}장")

        print("\n[SUMMARY] 테마별 총계:")
        print("-" * 50)
        for theme, total in sorted(theme_totals.items()):
            print(f"  {theme}: {total}장")

        print(f"\n  총합: {sum(self.stats.values())}장")

    def save_statistics(self):
        """통계를 JSON으로 저장"""
        stats_file = self.target_dir / 'classification_stats.json'

        # Theme별로 정리
        organized_stats = {}
        for key, count in self.stats.items():
            theme, pose_type = key.split('/')

            if theme not in organized_stats:
                organized_stats[theme] = {
                    'total': 0,
                    'pose_types': {}
                }

            organized_stats[theme]['pose_types'][pose_type] = count
            organized_stats[theme]['total'] += count

        # 메타데이터 추가
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'source_dir': str(self.source_dir),
            'target_dir': str(self.target_dir),
            'total_images': sum(self.stats.values()),
            'mode': 'model' if self.use_models else 'simple',
            'statistics': organized_stats
        }

        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print(f"\n[FILE] 통계 저장: {stats_file}")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description='TryAngle v1.5 - 이미지 자동 분류'
    )

    parser.add_argument(
        '--source',
        default='C:/try_angle/v1.5_ios/data/clustering',
        help='원본 이미지 폴더 경로'
    )

    parser.add_argument(
        '--target',
        default='C:/try_angle/v1.5_ios/data/classified_full',
        help='분류된 이미지 저장 폴더'
    )

    parser.add_argument(
        '--use-models',
        action='store_true',
        help='AI 모델 사용 (기본: Simple Mode)'
    )

    parser.add_argument(
        '--move',
        action='store_true',
        help='파일 이동 (기본: 복사)'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='실제 파일 작업 없이 시뮬레이션만'
    )

    args = parser.parse_args()

    # 설정 업데이트
    global USE_MODELS, COPY_MODE
    USE_MODELS = args.use_models
    COPY_MODE = not args.move

    print("=" * 60)
    print("[TARGET] TryAngle v1.5 - 자동 이미지 분류")
    print("=" * 60)
    print(f"[FOLDER] 원본 폴더: {args.source}")
    print(f"[FOLDER] 대상 폴더: {args.target}")
    print(f"[CONFIG] 모드: {'AI 모델' if USE_MODELS else 'Simple (폴더명 기반)'}")
    print(f"[MODE] 작업: {'복사' if COPY_MODE else '이동'}")
    print("=" * 60)

    if args.dry_run:
        print("\n[WARNING] DRY RUN 모드: 실제 파일 작업은 수행되지 않습니다.")
        return

    # 확인 (자동 진행)
    print("\n자동 진행 모드로 시작합니다...")
    # response = input("\n계속 진행하시겠습니까? (y/n): ")
    # if response.lower() != 'y':
    #     print("취소되었습니다.")
    #     return

    # 분류기 실행
    classifier = ImageClassifier(
        source_dir=args.source,
        target_dir=args.target,
        use_models=USE_MODELS
    )

    classifier.process_all()

    print("\n[DONE] 완료!")
    print(f"[FOLDER] 분류된 이미지 확인: {args.target}")


if __name__ == "__main__":
    main()