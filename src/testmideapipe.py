import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
while PROJECT_ROOT != PROJECT_ROOT.parent and not ((PROJECT_ROOT / "data").exists() and (PROJECT_ROOT / "src").exists()):
    PROJECT_ROOT = PROJECT_ROOT.parent

# -----------------------------------------------------
# 1️⃣ MediaPipe pose 초기화
# -----------------------------------------------------
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=True,        # 한 장의 이미지일 때 True
    model_complexity=2,            # 0~2 (2가 가장 정확)
    enable_segmentation=True,      # 실루엣(배경 분리)도 함께 수행
    min_detection_confidence=0.5
)

# -----------------------------------------------------
# 2️⃣ 이미지 읽기
# -----------------------------------------------------
image_path = PROJECT_ROOT / "data" / "sample_images" / "jott1.jpeg"
image = cv2.imread(str(image_path))
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# -----------------------------------------------------
# 3️⃣ 포즈 감지 실행
# -----------------------------------------------------
results = pose.process(image_rgb)

# -----------------------------------------------------
# 4️⃣ 포즈 & 실루엣 시각화
# -----------------------------------------------------
annotated_image = image.copy()

# 🔸 배경 분리 (실루엣 표시)
if results.segmentation_mask is not None:
    mask = results.segmentation_mask
    condition = mask > 0.5
    bg_color = (0, 0, 0)  # 배경을 검정으로
    bg_image = np.zeros_like(image, dtype=np.uint8)
    bg_image[:] = bg_color
    annotated_image = np.where(condition[..., None], image, bg_image)

# 🔸 관절 점 + 스켈레톤 연결
if results.pose_landmarks:
    mp_drawing.draw_landmarks(
        annotated_image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0,255,255), thickness=2, circle_radius=2),
        mp_drawing.DrawingSpec(color=(255,255,0), thickness=2, circle_radius=2)
    )

# -----------------------------------------------------
# 5️⃣ 결과 출력
# -----------------------------------------------------
cv2.imshow("MediaPipe Pose", annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
