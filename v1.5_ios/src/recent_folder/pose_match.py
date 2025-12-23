# reference_pose_matcher.py
# ✅ 단일 레퍼런스 이미지 기반 포즈 오버레이 + 실시간 웹캠 피드백

import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
while PROJECT_ROOT != PROJECT_ROOT.parent and not ((PROJECT_ROOT / "data").exists() and (PROJECT_ROOT / "src").exists()):
    PROJECT_ROOT = PROJECT_ROOT.parent

# 🔸 MediaPipe pose 초기화 (정지 이미지용 + 실시간용 각각 설정)
mp_pose = mp.solutions.pose
pose_estimator = mp_pose.Pose(static_image_mode=True, model_complexity=2)  # 레퍼런스 이미지 처리용
predictor = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)  # 실시간 웹캠 처리용
mp_drawing = mp.solutions.drawing_utils

# 🔸 레퍼런스 이미지 로딩 및 포즈 추출
REFERENCE_PATH = PROJECT_ROOT / "data" / "sample_images" / "test1.jpg"
ref_img = cv2.imread(str(REFERENCE_PATH))  # 이미지 읽기
ref_rgb = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)  # BGR -> RGB 변환
ref_result = pose_estimator.process(ref_rgb)  # 포즈 추론 수행

# 🔸 포즈 검출 실패 시 종료
if not ref_result.pose_landmarks:
    raise ValueError("레퍼런스 이미지에서 포즈를 감지하지 못했습니다.")

# 🔸 포즈 keypoint 좌표 추출 (정규화 좌표 → 실제 픽셀 좌표)
ref_landmarks = ref_result.pose_landmarks.landmark
ref_kps = [(int(l.x * ref_img.shape[1]), int(l.y * ref_img.shape[0])) for l in ref_landmarks]

# 🔸 웹캠 실행 시작
cap = cv2.VideoCapture(0)
print("웹캠 실행 중... 'q'를 눌러 종료")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 실시간 프레임 RGB 변환
    result = predictor.process(frame_rgb)  # 실시간 포즈 추정

    if result.pose_landmarks:
        # 🔸 현재 프레임의 포즈 keypoint 추출
        live_landmarks = result.pose_landmarks.landmark
        live_kps = [(int(l.x * frame.shape[1]), int(l.y * frame.shape[0])) for l in live_landmarks]

        # 🔸 실시간 스켈레톤 시각화
        mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # 🔸 레퍼런스와 현재 프레임의 keypoint 거리 비교 (평균 거리 → 유사도 점수 계산)
        distances = []
        for a, b in zip(ref_kps, live_kps):
            if a != (0, 0) and b != (0, 0):
                d = np.linalg.norm(np.array(a) - np.array(b))
                distances.append(d)
        score = max(0, 100 - int(np.mean(distances))) if distances else 0  # 점수는 0~100 범위로 표시

        # 🔸 점수 텍스트 출력
        cv2.putText(frame, f"Pose Match: {score}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    # 🔸 레퍼런스 스켈레톤을 반투명하게 오버레이 (파란색 선)
    overlay = frame.copy()
    for idx1, idx2 in mp_pose.POSE_CONNECTIONS:
        if ref_kps[idx1] != (0, 0) and ref_kps[idx2] != (0, 0):
            cv2.line(overlay, ref_kps[idx1], ref_kps[idx2], (255, 255, 0), 2)
    frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)  # 오버레이 적용 (0.4: 레퍼런스, 0.6: 실시간)

    # 🔸 화면 출력 및 종료 조건 확인
    cv2.imshow("Reference Pose Match", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 🔸 종료 처리
cap.release()
cv2.destroyAllWindows()
