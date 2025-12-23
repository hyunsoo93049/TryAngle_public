from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
while PROJECT_ROOT != PROJECT_ROOT.parent and not ((PROJECT_ROOT / "data").exists() and (PROJECT_ROOT / "src").exists()):
    PROJECT_ROOT = PROJECT_ROOT.parent

# 모델 불러오기
model = YOLO("yolo11n-pose.pt")

# 이미지 로드
path = PROJECT_ROOT / "data" / "sample_images" / "jott3.jpeg"
results = model(str(path), device='cuda', conf=0.5, imgsz=960)[0]

# COCO keypoint 연결 관계
SKELETON_COCO = [
    (15, 13), (13, 11), (16, 14), (14, 12),
    (11, 12), (5, 11), (6, 12),
    (5, 7), (6, 8), (7, 9), (8, 10),
    (1, 2), (0, 1), (0, 2), (1, 3), (2, 4),
    (3, 5), (4, 6)
]

# 결과 이미지 복사
frame = results.orig_img.copy()
keypoints = results.keypoints.xy.cpu().numpy()

# 각 사람별 처리
for i, pts in enumerate(keypoints):
    # 관절 점
    for (x, y) in pts:
        cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 255), -1)
    # 관절 연결선
    for (p1, p2) in SKELETON_COCO:
        if p1 < len(pts) and p2 < len(pts):
            x1, y1 = pts[p1]
            x2, y2 = pts[p2]
            if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 2)

cv2.imshow("YOLOv11 Skeleton (COCO)", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
