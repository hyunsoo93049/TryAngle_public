# ============================================================
# 🏃 TryAngle MoveNet Analyzer
# Phase 2-2: MoveNet Thunder 기반 포즈 분석
# ============================================================

import os
import sys
import cv2
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path

# Project root 설정
VERSION3_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = VERSION3_DIR
while PROJECT_ROOT != PROJECT_ROOT.parent and not ((PROJECT_ROOT / "data").exists() and (PROJECT_ROOT / "src").exists()):
    PROJECT_ROOT = PROJECT_ROOT.parent

if str(VERSION3_DIR) not in sys.path:
    sys.path.append(str(VERSION3_DIR))

from utils.model_cache import model_cache

# TensorFlow Lite 가져오기
try:
    import tensorflow as tf
    TFLITE_AVAILABLE = True
except ImportError:
    print("⚠️ TensorFlow not installed. Install: pip install tensorflow==2.15.0")
    TFLITE_AVAILABLE = False


class MoveNetAnalyzer:
    """
    MoveNet Thunder 포즈 분석기

    YOLO11 대체용 - 포즈 전문 모델
    - 정확도: 77.6% mAP (YOLO11: 62.5%)
    - 속도: 30fps (YOLO11: 45fps)
    - 크기: 12MB (YOLO11: 22MB)
    """

    # MoveNet 17개 키포인트 (COCO format, YOLO11과 동일)
    KEYPOINTS = [
        'nose',           # 0
        'left_eye',       # 1
        'right_eye',      # 2
        'left_ear',       # 3
        'right_ear',      # 4
        'left_shoulder',  # 5
        'right_shoulder', # 6
        'left_elbow',     # 7
        'right_elbow',    # 8
        'left_wrist',     # 9
        'right_wrist',    # 10
        'left_hip',       # 11
        'right_hip',      # 12
        'left_knee',      # 13
        'right_knee',     # 14
        'left_ankle',     # 15
        'right_ankle'     # 16
    ]

    def __init__(self, model_path: Optional[str] = None):
        """
        MoveNet 모델 초기화

        Args:
            model_path: TFLite 모델 경로. None이면 기본 경로 사용
        """
        if not TFLITE_AVAILABLE:
            raise ImportError("TensorFlow required. Install: pip install tensorflow==2.15.0")

        # 모델 경로 설정
        if model_path is None:
            model_path = VERSION3_DIR / "models" / "movenet_thunder.tflite"

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"MoveNet model not found: {model_path}\n"
                f"Run: python scripts/download_movenet.py"
            )

        self.model_path = str(model_path)

        # Singleton 패턴으로 모델 로드
        def load_interpreter():
            print(f"  🔧 Loading MoveNet from {os.path.basename(self.model_path)}...")
            interpreter = tf.lite.Interpreter(model_path=self.model_path)
            interpreter.allocate_tensors()
            return interpreter

        self.interpreter = model_cache.get_or_load("movenet_interpreter", load_interpreter)

        # Input/Output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Input size (MoveNet: 256x256)
        self.input_size = self.input_details[0]['shape'][1]

        print(f"  ✅ MoveNet loaded (input size: {self.input_size}x{self.input_size})")

    def analyze(self, image_path: str) -> Dict:
        """
        이미지에서 포즈 추출

        Args:
            image_path: 이미지 파일 경로

        Returns:
            {
                'keypoints': [{name, x, y, confidence}, ...],
                'confidence': float (전체 평균),
                'bbox': [x1, y1, x2, y2] (정규화 좌표)
            }
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # 이미지 로드
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        # 전처리: 256x256 리사이즈
        img_resized = cv2.resize(img_rgb, (self.input_size, self.input_size))

        # UINT8로 변환 (0~255, MoveNet은 정규화 필요 없음)
        img_input = np.expand_dims(img_resized, axis=0).astype(np.uint8)

        # 추론
        self.interpreter.set_tensor(self.input_details[0]['index'], img_input)
        self.interpreter.invoke()

        # 결과: [1, 1, 17, 3] (batch, person, keypoints, [y, x, confidence])
        keypoints_with_scores = self.interpreter.get_tensor(
            self.output_details[0]['index']
        )[0, 0]  # (17, 3)

        # 키포인트 파싱
        keypoints = []
        for i, kp_name in enumerate(self.KEYPOINTS):
            y, x, conf = keypoints_with_scores[i]
            keypoints.append({
                'name': kp_name,
                'x': float(x),  # 이미 정규화됨 (0~1)
                'y': float(y),
                'confidence': float(conf)
            })

        # 전체 confidence (평균)
        avg_confidence = float(np.mean([kp['confidence'] for kp in keypoints]))

        # BBox 계산 (confidence > 0.3인 키포인트만)
        valid_kps = [kp for kp in keypoints if kp['confidence'] > 0.3]
        if valid_kps:
            xs = [kp['x'] for kp in valid_kps]
            ys = [kp['y'] for kp in valid_kps]
            bbox = [min(xs), min(ys), max(xs), max(ys)]
        else:
            bbox = None

        return {
            'keypoints': keypoints,
            'confidence': avg_confidence,
            'bbox': bbox
        }

    def analyze_batch(self, image_paths: List[str]) -> List[Dict]:
        """
        여러 이미지를 배치로 분석

        Args:
            image_paths: 이미지 경로 리스트

        Returns:
            분석 결과 리스트
        """
        results = []
        for img_path in image_paths:
            try:
                result = self.analyze(img_path)
                results.append(result)
            except Exception as e:
                print(f"⚠️ Failed to analyze {img_path}: {e}")
                results.append(None)

        return results


# ============================================================
# YOLO11과 호환되는 래퍼 함수
# ============================================================

def analyze_pose_movenet(image_path: str) -> Dict:
    """
    MoveNet으로 포즈 분석 (YOLO11과 동일한 인터페이스)

    Returns:
        YOLO11 analyze()와 동일한 포맷
        {
            'scenario': str,
            'yolo_keypoints': List[Dict],  # 호환성 위해 이름 유지
            'merged_keypoints': Dict,
            'confidence': float,
            'bbox': List[float]
        }
    """
    analyzer = MoveNetAnalyzer()
    result = analyzer.analyze(image_path)

    # 시나리오 판단 (간단 버전)
    scenario = _detect_scenario_simple(result)

    # YOLO11 포맷으로 변환
    return {
        'scenario': scenario,
        'yolo_keypoints': result['keypoints'],  # 호환성 위해 이름 유지
        'merged_keypoints': {
            'base': {
                kp['name']: {
                    'x': kp['x'],
                    'y': kp['y'],
                    'confidence': kp['confidence']
                }
                for kp in result['keypoints']
            }
        },
        'confidence': result['confidence'],
        'bbox': result['bbox']
    }


def _detect_scenario_simple(result: Dict) -> str:
    """
    간단한 시나리오 판단

    Args:
        result: MoveNet analyze() 결과

    Returns:
        'full_body' | 'upper_body' | 'face_closeup' | 'no_person'
    """
    if result['confidence'] < 0.15:
        return 'no_person'

    keypoints = result['keypoints']
    kp_dict = {kp['name']: kp for kp in keypoints}

    # 하체 키포인트 확인
    lower_body_conf = np.mean([
        kp_dict['left_knee']['confidence'],
        kp_dict['right_knee']['confidence'],
        kp_dict['left_ankle']['confidence'],
        kp_dict['right_ankle']['confidence']
    ])

    # 얼굴 키포인트 확인
    face_conf = np.mean([
        kp_dict['nose']['confidence'],
        kp_dict['left_eye']['confidence'],
        kp_dict['right_eye']['confidence']
    ])

    # BBox 크기 확인
    if result['bbox']:
        bbox_width = result['bbox'][2] - result['bbox'][0]
        bbox_height = result['bbox'][3] - result['bbox'][1]
        bbox_area = bbox_width * bbox_height
    else:
        bbox_area = 0

    # 판단
    if bbox_area > 0.4 and face_conf > 0.7:
        return 'face_closeup'
    elif lower_body_conf > 0.5:
        return 'full_body'
    else:
        return 'upper_body'


# ============================================================
# 테스트
# ============================================================

if __name__ == "__main__":
    # 테스트 이미지
    test_img = PROJECT_ROOT / "data" / "test_images" / "test1.jpg"

    if not test_img.exists():
        print(f"❌ Test image not found: {test_img}")
        print("Please provide a test image.")
        sys.exit(1)

    print("\n" + "="*60)
    print("🏃 MoveNet Analyzer Test")
    print("="*60)

    try:
        # MoveNet 분석
        analyzer = MoveNetAnalyzer()
        result = analyzer.analyze(str(test_img))

        print(f"\n📋 Analysis Result:")
        print(f"  Image: {test_img.name}")
        print(f"  Confidence: {result['confidence']:.2%}")
        print(f"  BBox: {result['bbox']}")

        print(f"\n🦴 Keypoints (top 10):")
        for kp in result['keypoints'][:10]:
            print(f"  {kp['name']:<15} ({kp['x']:.3f}, {kp['y']:.3f}) conf={kp['confidence']:.2f}")

        print("\n" + "="*60)
        print("✅ Test Completed!")
        print("="*60)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
