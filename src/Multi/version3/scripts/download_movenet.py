# ============================================================
# 📥 MoveNet Thunder Model Downloader
# Phase 2-1: MoveNet 모델 다운로드 및 TFLite 변환
# ============================================================

import os
import sys
from pathlib import Path

print("="*60)
print("📥 MoveNet Thunder Downloader")
print("="*60)

# TensorFlow 설치 확인
try:
    import tensorflow as tf
    import tensorflow_hub as hub
    print(f"✅ TensorFlow version: {tf.__version__}")
except ImportError:
    print("\n❌ TensorFlow not installed!")
    print("\n설치 명령어:")
    print("  pip install tensorflow==2.15.0")
    print("  pip install tensorflow-hub")
    sys.exit(1)

# 프로젝트 경로 설정
VERSION3_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = VERSION3_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# MoveNet 모델 URL
MOVENET_THUNDER_URL = "https://tfhub.dev/google/movenet/singlepose/thunder/4"
MOVENET_LIGHTNING_URL = "https://tfhub.dev/google/movenet/singlepose/lightning/4"

print(f"\nModels will be saved to: {MODELS_DIR}\n")

def download_and_convert(model_url, model_name):
    """
    MoveNet 모델 다운로드 및 TFLite 변환

    Args:
        model_url: TensorFlow Hub URL
        model_name: 저장할 모델 이름 (예: "movenet_thunder")
    """
    print(f"\n{'='*60}")
    print(f"📦 Downloading {model_name}...")
    print(f"{'='*60}\n")

    try:
        # Step 1: TensorFlow Hub에서 모델 다운로드
        print(f"1/4 Loading model from TensorFlow Hub...")
        print(f"    URL: {model_url}")

        model = hub.load(model_url)
        movenet = model.signatures['serving_default']

        print(f"✅ Model loaded successfully")

        # Step 2: 모델 정보 확인
        print(f"\n2/4 Model Information:")
        print(f"    Input shape: {movenet.inputs[0].shape}")
        print(f"    Output shape: {movenet.outputs[0].shape}")

        # Step 3: SavedModel 형식으로 저장
        saved_model_dir = MODELS_DIR / f"{model_name}_saved_model"
        print(f"\n3/4 Saving as SavedModel format...")
        print(f"    Location: {saved_model_dir}")

        tf.saved_model.save(model, str(saved_model_dir))
        print(f"✅ SavedModel saved")

        # Step 4: TFLite 변환
        print(f"\n4/4 Converting to TFLite...")

        converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))

        # 최적화 옵션
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        # 변환 실행
        tflite_model = converter.convert()

        # 저장
        tflite_path = MODELS_DIR / f"{model_name}.tflite"
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)

        model_size_mb = len(tflite_model) / (1024 * 1024)
        print(f"✅ TFLite model saved")
        print(f"    Location: {tflite_path}")
        print(f"    Size: {model_size_mb:.1f} MB")

        return tflite_path

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_model(tflite_path):
    """
    TFLite 모델 테스트

    Args:
        tflite_path: TFLite 모델 경로
    """
    print(f"\n{'='*60}")
    print(f"🧪 Testing TFLite Model")
    print(f"{'='*60}\n")

    try:
        import numpy as np

        # Interpreter 로드
        interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
        interpreter.allocate_tensors()

        # Input/Output 정보
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        print(f"Input Details:")
        print(f"  Shape: {input_details[0]['shape']}")
        print(f"  Type: {input_details[0]['dtype']}")

        print(f"\nOutput Details:")
        print(f"  Shape: {output_details[0]['shape']}")
        print(f"  Type: {output_details[0]['dtype']}")

        # 더미 입력으로 테스트
        input_shape = input_details[0]['shape']
        input_data = np.random.rand(*input_shape).astype(np.float32)

        print(f"\nRunning inference with dummy input...")
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # 결과
        output_data = interpreter.get_tensor(output_details[0]['index'])
        print(f"✅ Inference successful!")
        print(f"   Output shape: {output_data.shape}")
        print(f"   First keypoint: {output_data[0, 0, 0]}")  # [y, x, confidence]

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def main():
    """메인 실행 함수"""

    print("\nWhich model do you want to download?")
    print("1. MoveNet Thunder (정확도 우선, 12MB, 30fps)")
    print("2. MoveNet Lightning (속도 우선, 3MB, 60fps)")
    print("3. Both")

    choice = input("\nEnter your choice (1/2/3) [default: 1]: ").strip()

    if not choice:
        choice = "1"

    models_to_download = []

    if choice == "1":
        models_to_download = [
            (MOVENET_THUNDER_URL, "movenet_thunder")
        ]
    elif choice == "2":
        models_to_download = [
            (MOVENET_LIGHTNING_URL, "movenet_lightning")
        ]
    elif choice == "3":
        models_to_download = [
            (MOVENET_THUNDER_URL, "movenet_thunder"),
            (MOVENET_LIGHTNING_URL, "movenet_lightning")
        ]
    else:
        print("Invalid choice!")
        return

    # 다운로드 및 변환
    for model_url, model_name in models_to_download:
        tflite_path = download_and_convert(model_url, model_name)

        if tflite_path:
            # 테스트
            test_model(tflite_path)

    print(f"\n{'='*60}")
    print(f"✅ All Done!")
    print(f"{'='*60}")
    print(f"\nModels saved in: {MODELS_DIR}")
    print(f"\nNext steps:")
    print(f"  1. Python에서 사용: movenet_analyzer.py 참조")
    print(f"  2. iOS에서 사용: TFLite 모델을 Xcode 프로젝트에 추가")


if __name__ == "__main__":
    main()
