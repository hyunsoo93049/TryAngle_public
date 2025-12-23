#!/usr/bin/env python3
"""
MoveNet Lightning TFLite를 CoreML로 변환
"""
import coremltools as ct
import tensorflow as tf
import numpy as np

# TFLite 모델 로드
interpreter = tf.lite.Interpreter(model_path="movenet_lightning.tflite")
interpreter.allocate_tensors()

# 입력/출력 정보 확인
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input details:", input_details)
print("Output details:", output_details)

# TFLite → CoreML 변환
# MoveNet은 이미지 입력 (192x192) → 17개 keypoints 출력
mlmodel = ct.convert(
    "movenet_lightning.tflite",
    inputs=[ct.TensorType(name="input", shape=(1, 192, 192, 3))],
    outputs=[ct.TensorType(name="output")],
    minimum_deployment_target=ct.target.iOS14,
)

# 메타데이터 추가
mlmodel.author = "Google"
mlmodel.license = "Apache 2.0"
mlmodel.short_description = "MoveNet Lightning - Single Pose Estimation"
mlmodel.version = "4"

# 저장
mlmodel.save("MoveNetLightning.mlpackage")
print("\n✅ MoveNet Lightning CoreML 변환 완료: MoveNetLightning.mlpackage")
