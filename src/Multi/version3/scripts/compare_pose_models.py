# ============================================================
# 📊 YOLO11 vs MoveNet Performance Comparison
# Phase 2-5: 포즈 모델 성능 비교 테스트
# ============================================================

import os
import sys
import time
import numpy as np
from pathlib import Path
from typing import Dict, List
import json

# Project root 설정
VERSION3_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = VERSION3_DIR
while PROJECT_ROOT != PROJECT_ROOT.parent and not ((PROJECT_ROOT / "data").exists() and (PROJECT_ROOT / "src").exists()):
    PROJECT_ROOT = PROJECT_ROOT.parent

if str(VERSION3_DIR) not in sys.path:
    sys.path.append(str(VERSION3_DIR))

from analysis.pose_analyzer import PoseAnalyzer


# ============================================================
# 테스트 함수
# ============================================================

def benchmark_model(analyzer: PoseAnalyzer, image_paths: List[str], model_name: str) -> Dict:
    """
    단일 모델 벤치마크

    Args:
        analyzer: PoseAnalyzer 인스턴스
        image_paths: 테스트 이미지 경로 리스트
        model_name: 모델 이름 (YOLO11 또는 MoveNet)

    Returns:
        벤치마크 결과 딕셔너리
    """
    print(f"\n{'='*60}")
    print(f"🏃 Testing {model_name}")
    print(f"{'='*60}")

    results = []
    inference_times = []
    confidences = []
    scenarios = {
        'full_body': 0,
        'upper_body': 0,
        'face_closeup': 0,
        'hand_gesture': 0,
        'back_view': 0,
        'no_person': 0
    }

    for i, img_path in enumerate(image_paths):
        print(f"[{i+1}/{len(image_paths)}] Testing {Path(img_path).name}...", end=" ")

        try:
            # 추론 시간 측정
            start_time = time.time()
            result = analyzer.analyze(str(img_path))
            inference_time = (time.time() - start_time) * 1000  # ms

            inference_times.append(inference_time)

            # 시나리오 카운트
            scenario = result.get('scenario', 'no_person')
            scenarios[scenario] += 1

            # Confidence 수집
            if result['confidence'] > 0:
                confidences.append(result['confidence'])

            results.append({
                'image': Path(img_path).name,
                'scenario': scenario,
                'confidence': result['confidence'],
                'inference_time_ms': inference_time,
                'keypoints_detected': len([kp for kp in result.get('yolo_keypoints', []) if kp['confidence'] > 0.3])
            })

            print(f"✅ ({inference_time:.1f}ms, conf={result['confidence']:.2f})")

        except Exception as e:
            print(f"❌ Error: {e}")
            results.append({
                'image': Path(img_path).name,
                'error': str(e)
            })

    # 통계 계산
    avg_time = np.mean(inference_times) if inference_times else 0
    std_time = np.std(inference_times) if inference_times else 0
    avg_conf = np.mean(confidences) if confidences else 0
    std_conf = np.std(confidences) if confidences else 0

    return {
        'model': model_name,
        'total_images': len(image_paths),
        'successful_detections': len([r for r in results if 'error' not in r and r.get('confidence', 0) > 0.15]),
        'avg_inference_time_ms': avg_time,
        'std_inference_time_ms': std_time,
        'avg_confidence': avg_conf,
        'std_confidence': std_conf,
        'scenarios': scenarios,
        'detailed_results': results
    }


def compare_models(test_images_dir: str, num_samples: int = 30):
    """
    YOLO11 vs MoveNet 비교

    Args:
        test_images_dir: 테스트 이미지 디렉토리
        num_samples: 테스트할 이미지 수 (기본 30장)
    """
    # 테스트 이미지 목록
    test_dir = Path(test_images_dir)
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_images_dir}")

    image_files = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.jpeg")) + list(test_dir.glob("*.png"))

    if len(image_files) == 0:
        raise ValueError(f"No images found in {test_images_dir}")

    # 샘플링
    if len(image_files) > num_samples:
        import random
        random.seed(42)
        image_files = random.sample(image_files, num_samples)

    print(f"\n{'='*60}")
    print(f"📊 YOLO11 vs MoveNet Performance Comparison")
    print(f"{'='*60}")
    print(f"Test images: {len(image_files)}")
    print(f"Location: {test_images_dir}\n")

    # YOLO11 테스트
    print("\n🔷 Phase 1: YOLO11 Benchmark")
    yolo_analyzer = PoseAnalyzer(use_movenet=False)
    yolo_results = benchmark_model(yolo_analyzer, image_files, "YOLO11-pose")

    # MoveNet 테스트
    print("\n🔶 Phase 2: MoveNet Benchmark")
    movenet_analyzer = PoseAnalyzer(use_movenet=True)
    movenet_results = benchmark_model(movenet_analyzer, image_files, "MoveNet-Thunder")

    # 결과 비교
    print(f"\n{'='*60}")
    print(f"📊 Comparison Results")
    print(f"{'='*60}\n")

    # 표 형식 출력
    print(f"{'Metric':<30} {'YOLO11-pose':<20} {'MoveNet-Thunder':<20} {'Winner':<10}")
    print("-" * 80)

    # 1. Detection Rate
    yolo_rate = yolo_results['successful_detections'] / yolo_results['total_images']
    movenet_rate = movenet_results['successful_detections'] / movenet_results['total_images']
    winner = "MoveNet" if movenet_rate > yolo_rate else "YOLO11" if yolo_rate > movenet_rate else "Tie"
    print(f"{'Detection Rate':<30} {yolo_rate:>6.1%}              {movenet_rate:>6.1%}              {winner:<10}")

    # 2. Inference Speed
    yolo_fps = 1000 / yolo_results['avg_inference_time_ms'] if yolo_results['avg_inference_time_ms'] > 0 else 0
    movenet_fps = 1000 / movenet_results['avg_inference_time_ms'] if movenet_results['avg_inference_time_ms'] > 0 else 0
    winner = "YOLO11" if yolo_fps > movenet_fps else "MoveNet" if movenet_fps > yolo_fps else "Tie"
    print(f"{'Avg Inference Time (ms)':<30} {yolo_results['avg_inference_time_ms']:>6.1f}              {movenet_results['avg_inference_time_ms']:>6.1f}              {winner:<10}")
    print(f"{'FPS':<30} {yolo_fps:>6.1f}              {movenet_fps:>6.1f}              {winner:<10}")

    # 3. Confidence
    winner = "MoveNet" if movenet_results['avg_confidence'] > yolo_results['avg_confidence'] else "YOLO11" if yolo_results['avg_confidence'] > movenet_results['avg_confidence'] else "Tie"
    print(f"{'Avg Confidence':<30} {yolo_results['avg_confidence']:>6.2%}              {movenet_results['avg_confidence']:>6.2%}              {winner:<10}")

    # 시나리오별 비교
    print(f"\n{'='*60}")
    print(f"📋 Scenario Detection Breakdown")
    print(f"{'='*60}\n")

    print(f"{'Scenario':<20} {'YOLO11':<15} {'MoveNet':<15}")
    print("-" * 50)
    for scenario in ['full_body', 'upper_body', 'face_closeup', 'hand_gesture', 'back_view', 'no_person']:
        yolo_count = yolo_results['scenarios'][scenario]
        movenet_count = movenet_results['scenarios'][scenario]
        print(f"{scenario:<20} {yolo_count:<15} {movenet_count:<15}")

    # 결론
    print(f"\n{'='*60}")
    print(f"💡 Conclusions")
    print(f"{'='*60}\n")

    # 승자 결정
    yolo_score = 0
    movenet_score = 0

    if yolo_rate > movenet_rate:
        yolo_score += 1
    elif movenet_rate > yolo_rate:
        movenet_score += 1

    if yolo_fps > movenet_fps:
        yolo_score += 1
    elif movenet_fps > yolo_fps:
        movenet_score += 1

    if yolo_results['avg_confidence'] > movenet_results['avg_confidence']:
        yolo_score += 1
    elif movenet_results['avg_confidence'] > yolo_results['avg_confidence']:
        movenet_score += 1

    print(f"Overall Score: YOLO11 ({yolo_score}/3) vs MoveNet ({movenet_score}/3)")

    if movenet_score > yolo_score:
        print("\n✅ MoveNet이 전반적으로 우수합니다!")
        print("   권장: MoveNet으로 전환 (특히 포즈 정확도가 중요한 경우)")
    elif yolo_score > movenet_score:
        print("\n✅ YOLO11이 전반적으로 우수합니다!")
        print("   권장: YOLO11 유지 (특히 속도가 중요한 경우)")
    else:
        print("\n✅ 두 모델이 비슷한 성능을 보입니다!")
        print("   권장: 사용 목적에 따라 선택")
        print("   - 정확도 우선: MoveNet")
        print("   - 속도 우선: YOLO11")

    # 상세 권장사항
    if movenet_rate > yolo_rate * 1.1:
        print("\n📌 MoveNet의 검출률이 10% 이상 높습니다")
        print("   → 측면 포즈, 얼굴 가린 포즈 등 어려운 케이스에 강점")

    if yolo_fps > movenet_fps * 1.2:
        print("\n📌 YOLO11의 속도가 20% 이상 빠릅니다")
        print("   → 실시간 처리가 중요한 경우 YOLO11 선택")

    # 결과 저장
    output_file = VERSION3_DIR / "pose_model_comparison_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            'yolo11': yolo_results,
            'movenet': movenet_results,
            'summary': {
                'yolo11_score': yolo_score,
                'movenet_score': movenet_score,
                'yolo11_detection_rate': yolo_rate,
                'movenet_detection_rate': movenet_rate,
                'yolo11_fps': yolo_fps,
                'movenet_fps': movenet_fps,
                'yolo11_avg_confidence': yolo_results['avg_confidence'],
                'movenet_avg_confidence': movenet_results['avg_confidence']
            }
        }, f, indent=2)

    print(f"\n💾 Results saved to: {output_file}")

    return {
        'yolo11': yolo_results,
        'movenet': movenet_results
    }


# ============================================================
# 실행
# ============================================================

if __name__ == "__main__":
    # 테스트 이미지 디렉토리
    test_dir = PROJECT_ROOT / "data" / "test_images"

    if not test_dir.exists():
        # 대안: clustered_images 사용
        test_dir = PROJECT_ROOT / "data" / "clustered_images" / "cluster_0"

    if not test_dir.exists():
        print(f"❌ Test directory not found: {test_dir}")
        print("Please provide test images in data/test_images/")
        sys.exit(1)

    try:
        results = compare_models(str(test_dir), num_samples=30)
    except Exception as e:
        print(f"\n❌ Error running comparison: {e}")
        import traceback
        traceback.print_exc()
