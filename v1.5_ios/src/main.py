# main.py
# -----------------------------------------------------------------------------
# 캡처 루프:
#  - 웹캠에서 프레임 읽기 → 포즈 검출 → (1명 기준) 구도 분석 → 감성 유사도(옵션)
#  - 점수/피드백을 오버레이하여 화면에 표시
#  - 'q' 키로 종료
# -----------------------------------------------------------------------------

from typing import Optional
import cv2
import numpy as np

from pose_module import PoseEstimator
from composition_module import analyze_composition
from feedback_module import generate_feedback

# (선택) 감성 모듈을 사용하려면 주석 해제
USE_EMOTION = False
REFERENCE_PATH = "reference.jpg"  # 감성 비교용 참조 이미지 경로

if USE_EMOTION:
    from emotion_module import EmotionAnalyzer


def draw_guides(frame: np.ndarray) -> None:
    """화면 가이드(삼분할선) 오버레이."""
    h, w = frame.shape[:2]
    # 세로선(삼분할)
    cv2.line(frame, (w // 3, 0), (w // 3, h), (80, 80, 80), 1)
    cv2.line(frame, (2 * w // 3, 0), (2 * w // 3, h), (80, 80, 80), 1)
    # 가로선(삼분할)
    cv2.line(frame, (0, h // 3), (w, h // 3), (80, 80, 80), 1)
    cv2.line(frame, (0, 2 * h // 3), (w, 2 * h // 3), (80, 80, 80), 1)


def main(camera_index: int = 0):
    # 1) 모듈 초기화
    pose_estimator = PoseEstimator(model_path="yolov8s-pose.pt", conf_thres=0.25)
    emotion_analyzer = None
    if USE_EMOTION:
        emotion_analyzer = EmotionAnalyzer()

    # 2) 웹캠 열기
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # 가이드선 먼저 그리기
        draw_guides(frame)

        # 3) 포즈 검출
        detections = pose_estimator.infer(frame)  # bbox, score 포함
        pose_conf_avg: Optional[float] = None
        comp_score = None
        emo_score = None

        if len(detections) > 0:
            # 일단 1명만 분석(가장 자신있는 검출을 선택)
            det = max(detections, key=lambda d: d["score"] if d["score"] is not None else 0.0)
            kpts = det["keypoints"]
            bbox = det["bbox"]
            pose_conf_avg = det["score"]

            # 4) 구도 분석
            comp = analyze_composition(frame, kpts, bbox)
            comp_score = comp["score"]

            # 시각화: 인물 중심/바운딩박스
            cx, cy = map(int, comp["center"])
            cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)

            if bbox is not None:
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)

            # 5) 감성 유사도(옵션)
            if USE_EMOTION and emotion_analyzer is not None:
                emo_score = emotion_analyzer.compare_to_reference(REFERENCE_PATH, frame)

            # 6) 피드백 생성
            msgs = generate_feedback(pose_conf_avg, comp_score, emo_score)

            # 7) 텍스트 오버레이
            y = 28
            cv2.putText(frame, f"Composition: {comp_score:.1f}", (12, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (10, 230, 10), 2); y += 28
            if pose_conf_avg is not None:
                cv2.putText(frame, f"Pose conf: {pose_conf_avg:.2f}", (12, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 10), 2); y += 24
            if emo_score is not None:
                cv2.putText(frame, f"Emotion: {emo_score:.1f}%", (12, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 200, 230), 2); y += 24

            for m in msgs[:3]:  # 과도한 텍스트 방지(최대 3줄 표시)
                cv2.putText(frame, m, (12, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y += 22
        else:
            # 사람 미검출 시 기본 안내
            cv2.putText(frame, "사람을 찾지 못했어요. 프레임 안으로 들어오세요.",
                        (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 255), 2)

        # 8) 화면 출력
        cv2.imshow("TryAngle - Real-time Feedback", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 9) 정리
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
