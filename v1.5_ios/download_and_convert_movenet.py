#!/usr/bin/env python3
"""
MoveNet Lightning SavedModel 다운로드 및 CoreML 변환
"""
import tensorflow as tf
import tensorflow_hub as hub
import coremltools as ct
import numpy as np

print("📥 MoveNet Lightning SavedModel 다운로드 중...")

# TensorFlow Hub에서 SavedModel 다운로드
model_url = "https://tfhub.dev/google/movenet/singlepose/lightning/4"
model = hub.load(model_url)
movenet = model.signatures['serving_default']

print(f"✅ 모델 다운로드 완료")
print(f"   Input shape: {movenet.inputs[0].shape}")
print(f"   Output shape: {movenet.outputs[0].shape}")

# SavedModel로 저장
print("\n💾 SavedModel 형식으로 저장 중...")
tf.saved_model.save(model, "movenet_saved_model")
print("✅ SavedModel 저장 완료: movenet_saved_model/")

# CoreML로 변환
print("\n🔄 CoreML 변환 중...")
mlmodel = ct.convert(
    "movenet_saved_model",
    source="tensorflow",
    inputs=[ct.ImageType(name="input", shape=(1, 192, 192, 3), scale=1/255.0)],
    outputs=[ct.TensorType(name="output")],
    minimum_deployment_target=ct.target.iOS14,
)

# 메타데이터 추가
mlmodel.author = "Google"
mlmodel.license = "Apache 2.0"
mlmodel.short_description = "MoveNet Lightning - Single Pose Estimation (17 keypoints)"
mlmodel.version = "4"

# 입력 설명
mlmodel.input_description["input"] = "Input image (192x192 RGB)"

# 출력 설명
mlmodel.output_description["output"] = "17 keypoints with [y, x, confidence] for each (shape: 1x1x17x3)"

# 저장
mlmodel.save("MoveNetLightning.mlpackage")
print("\n✅ MoveNet Lightning CoreML 변환 완료!")
print(f"   저장 위치: MoveNetLightning.mlpackage")
print(f"   크기: ~3MB")
print(f"   입력: 192x192 RGB 이미지")
print(f"   출력: 17개 keypoints (y, x, confidence)")
