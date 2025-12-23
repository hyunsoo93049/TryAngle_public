#인물 포즈 감지(yolov8 - pose)
from ultralytics import YOLO
import numpy as np

class PoseEstimator:
    def __init__(self, model_path="yolov8s-pose.pt"):
        self.model = YOLO(model_path)

    def get_keypoints(self, image):
        results = self.model(image, verbose=False)
        keypoints = []
        for result in results:
            if result.keypoints is not None:
                keypoints.append(result.keypoints.xy.cpu().numpy())
        return keypoints
