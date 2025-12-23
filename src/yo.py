from ultralytics import YOLO
from transformers import pipeline
from PIL import Image
import cv2
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
while PROJECT_ROOT != PROJECT_ROOT.parent and not ((PROJECT_ROOT / "data").exists() and (PROJECT_ROOT / "src").exists()):
    PROJECT_ROOT = PROJECT_ROOT.parent

print("=== YOLO Seg + Pose + Depth 통합 ===\n")

# 모델 초기화
print("모델 로딩 중...")
yolo_seg = YOLO('yolov8s-seg.pt')
yolo_pose = YOLO('yolov8s-pose.pt')

# ✅ Depth Estimation 모델 추가
depth_estimator = pipeline(
    task="depth-estimation",
    model="depth-anything/Depth-Anything-V2-Small-hf"
)
print("✅ 모델 로딩 완료!\n")

# 이미지 로드
image_path = PROJECT_ROOT / "data" / "sample_images" / "p1.jpg"
image = cv2.imread(str(image_path))
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
h, w = image.shape[:2]

# 1. Segmentation
print("=== YOLO Segmentation ===")
seg_results = yolo_seg(image)
annotated_image = seg_results[0].plot()

# 2. Pose
print("\n=== YOLO Pose ===")
pose_results = yolo_pose(image)
annotated_image = pose_results[0].plot(img=annotated_image)

# 3. ✅ Depth Estimation
print("\n=== Depth Estimation ===")
image_pil = Image.fromarray(image_rgb)
depth_result = depth_estimator(image_pil)
depth_map = np.array(depth_result["depth"])
print(f"✅ Depth 맵 생성 완료! Shape: {depth_map.shape}")

# 4. ✅ 주인공(가장 가까운 사람) 찾기
print("\n=== 주인공 식별 ===")
persons = []

for i, box in enumerate(pose_results[0].boxes):
    if pose_results[0].names[int(box.cls[0])] == "person":
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        
        # Depth 값 추출 (높을수록 가까움)
        person_depth = depth_map[center_y, center_x]
        
        persons.append({
            'bbox': (x1, y1, x2, y2),
            'center': (center_x, center_y),
            'depth': person_depth
        })

if persons:
    # 가장 높은 depth 값 = 가장 가까운 사람
    main_subject = max(persons, key=lambda p: p['depth'])
    print(f"✅ 주인공 검출!")
    print(f"   위치: {main_subject['center']}")
    print(f"   Depth: {main_subject['depth']:.2f}")
    
    # 주인공에 특별 표시
    cv2.circle(annotated_image, main_subject['center'], 15, (0, 255, 0), -1)
    cv2.putText(annotated_image, "MAIN", 
                (main_subject['center'][0]-20, main_subject['center'][1]-20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
else:
    print("❌ 사람 검출 안 됨")

# Depth 맵 시각화 (옵션)
depth_colored = cv2.applyColorMap(
    cv2.convertScaleAbs(depth_map, alpha=255.0/depth_map.max()),
    cv2.COLORMAP_MAGMA
)

# 결과 표시
cv2.imshow("통합 결과", annotated_image)
cv2.imshow("Depth Map", depth_colored)
print("\n이미지 창에서 아무 키나 누르면 종료")
cv2.waitKey(0)
cv2.destroyAllWindows()

print("\n✅ 통합 테스트 완료!")
