# -*- coding: utf-8 -*-
"""
TryAngle 통합 비교 (CLIP + YOLO Pose + Tri-Path DINO + MiDaS + ColorTone)
안정화 + OpenCV 시각화 버전 v2.6
- 원본 비율 100% 유지
- 별도 창으로 표시
- 포즈 유사도 추가
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from composition_module import analyze_composition, calculate_pose_similarity
from feedback_module import generate_feedback
from emotion_module import EmotionAnalyzer
from dino_module import dino_similarity
from midas_module import camera_height_diff
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
while PROJECT_ROOT != PROJECT_ROOT.parent and not ((PROJECT_ROOT / "data").exists() and (PROJECT_ROOT / "src").exists()):
    PROJECT_ROOT = PROJECT_ROOT.parent


# ---------------------------------------------------------
# 안전 리사이즈 (긴 변 기준 1200px 이하로 제한)
# ---------------------------------------------------------
MAX_SIZE = 1200  # 화면에 맞게 조정
def safe_resize(img, max_size=MAX_SIZE):
    """이미지 크기가 너무 크면 비율 유지하며 축소"""
    h, w = img.shape[:2]
    scale = max_size / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


# ---------------------------------------------------------
# 색감/조명 유사도 (HSV 히스토그램 상관)
# ---------------------------------------------------------
def color_tone_similarity(img1_bgr, img2_bgr):
    hsv1 = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2HSV)
    hist1 = cv2.calcHist([hsv1],[0,1,2],None,[24,8,8],[0,180,0,256,0,256])
    hist2 = cv2.calcHist([hsv2],[0,1,2],None,[24,8,8],[0,180,0,256,0,256])
    cv2.normalize(hist1, hist1)
    cv2.normalize(hist2, hist2)
    sim = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return float((sim + 1.0) * 0.5)


# ---------------------------------------------------------
# YOLO Pose: 키포인트 & bbox 추출
# ---------------------------------------------------------
def extract_pose_info(image_bgr, yolo_model):
    """
    YOLO Pose 결과에서 인물 bbox, keypoints 추출 및 composition 분석 수행
    """
    res = yolo_model(image_bgr, verbose=False)
    if not res or res[0].keypoints is None:
        return None, None, None

    r = res[0]
    boxes = r.boxes.xyxy
    kpts = r.keypoints.xy

    if boxes is None or len(boxes) == 0:
        return None, None, None

    bbox = boxes[0].detach().cpu().numpy()
    x1, y1, x2, y2 = bbox

    # 상단 margin 추가 (머리 짤림 방지)
    margin_y = int((y2 - y1) * 0.15)
    y1 = max(0, y1 - margin_y)
    y2 = y2 + margin_y
    if y2 < y1:
        y1, y2 = y2, y1
    bbox = np.array([x1, y1, x2, y2])

    kpts = kpts[0].detach().cpu().numpy() if kpts is not None and len(kpts) > 0 else None

    comp = analyze_composition(image_bgr, kpts, bbox)
    return kpts, comp, bbox


# ---------------------------------------------------------
# 메인 함수
# ---------------------------------------------------------
def main(ref_path, tgt_path, yolo_weights="yolov8s-pose.pt"):
    ref_img = safe_resize(cv2.imread(ref_path))
    tgt_img = safe_resize(cv2.imread(tgt_path))
    if ref_img is None or tgt_img is None:
        raise FileNotFoundError("이미지 경로를 확인하세요.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("🔧 모델 로드 중...")
    yolo_pose = YOLO(yolo_weights)
    emo = EmotionAnalyzer(device=device)

    # 1️⃣ YOLO Pose
    ref_kp, ref_comp, ref_bbox = extract_pose_info(ref_img, yolo_pose)
    tgt_kp, tgt_comp, tgt_bbox = extract_pose_info(tgt_img, yolo_pose)

    # 포즈 유사도 계산
    pose_similarity = calculate_pose_similarity(ref_kp, tgt_kp)

    # 2️⃣ CLIP 감성
    clip_score = emo.compare_to_reference(ref_path, tgt_img)

    # 3️⃣ Tri-Path DINO
    try:
        dino_sim = dino_similarity(ref_path, tgt_path, ref_bbox, tgt_bbox, alpha=0.5, beta=0.3, gamma=0.2)
    except Exception as e:
        print(f"[경고] DINO 유사도 계산 실패: {e}")
        dino_sim = None

    # 4️⃣ MiDaS
    try:
        height_diff = camera_height_diff(ref_path, tgt_path)
    except Exception as e:
        print(f"[경고] MiDaS 시점 계산 실패: {e}")
        height_diff = None

    # 5️⃣ 색감
    color_sim = color_tone_similarity(ref_img, tgt_img)

    # -----------------------------------------------------
    # 결과 출력
    # -----------------------------------------------------
    print("\n===== ANALYSIS =====")
    if ref_comp: print(f"📷 [레퍼런스 구도]: {ref_comp['score']:.2f}")
    if tgt_comp: print(f"📸 [내 사진 구도]: {tgt_comp['score']:.2f}")
    print(f"🎨 [색감 유사도]: {color_sim*100:.2f}%")
    print(f"💫 [감성(CLIP) 유사도]: {clip_score:.2f}%")
    print(f"🕺 [포즈 유사도]: {pose_similarity['score']:.2f} ({pose_similarity['details']})")
    if dino_sim is not None:
        print(f"🧩 [DINO 구도 유사도 (Tri-Path)]: {dino_sim:.3f}")
    if height_diff is not None:
        trend = "하이앵글↑" if height_diff > 0 else ("로우앵글↓" if height_diff < 0 else "유사")
        print(f"📐 [MiDaS 시점 차이]: {height_diff:.3f}  → {trend}")

    # -----------------------------------------------------
    # 피드백 생성
    # -----------------------------------------------------
    reasons = []
    if tgt_comp:
        if not tgt_comp["on_rule_of_thirds"]:
            reasons.append("인물이 삼분할선에서 벗어나 있습니다.")
        if tgt_comp["size_ratio"] > 0.45:
            reasons.append("인물이 화면을 과도하게 차지하고 있습니다.")
        if tgt_comp["headroom_ratio"] < 0.08:
            reasons.append("머리 위 여백이 부족합니다.")
        if tgt_comp["headroom_ratio"] > 0.18:
            reasons.append("머리 위 여백이 넓어 인물이 작게 느껴집니다.")

    summary = "감성·프레임·시점·포즈를 종합 비교했습니다."
    extras = {
        "size_ratio": tgt_comp.get("size_ratio") if tgt_comp else None,
        "headroom_ratio": tgt_comp.get("headroom_ratio") if tgt_comp else None,
        "dino_sim": dino_sim,
        "height_diff": height_diff,
        "color_sim": color_sim,
        "pose_similarity": pose_similarity,
    }
    comp_score = tgt_comp["score"] if tgt_comp else None

    feedback = generate_feedback(
        pose_conf=None,
        composition_score=comp_score,
        emotion_score=clip_score,
        reasons=reasons,
        summary=summary,
        extras=extras
    )

    print("\n💬 [AI 피드백]")
    for line in feedback:
        print(line)

    # -----------------------------------------------------
    # 시각화 (OpenCV - 원본 비율 100% 유지)
    # -----------------------------------------------------
    
    # 타겟 이미지에 정보 오버레이
    overlay = tgt_img.copy()
    y = 30
    line_height = 35
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    
    # 텍스트 배경 추가 함수
    def add_text_with_bg(img, text, pos, color, bg_color=(0, 0, 0)):
        (w, h), _ = cv2.getTextSize(text, font, font_scale, thickness)
        x, y_pos = pos
        cv2.rectangle(img, (x, y_pos - h - 5), (x + w + 5, y_pos + 5), bg_color, -1)
        cv2.putText(img, text, (x, y_pos), font, font_scale, color, thickness)
    
    # 각 점수 표시
    if comp_score:
        add_text_with_bg(overlay, f"Composition: {comp_score:.1f}", (10, y), (0, 255, 0))
        y += line_height
    
    # 포즈 유사도 (색상 코드)
    if pose_similarity:
        pose_score = pose_similarity.get("score", 0.0)
        if pose_score >= 70:
            pose_color = (0, 255, 0)  # 초록
        elif pose_score >= 50:
            pose_color = (0, 165, 255)  # 주황
        else:
            pose_color = (0, 0, 255)  # 빨강
        add_text_with_bg(overlay, f"Pose: {pose_score:.1f}", (10, y), pose_color)
        y += line_height
    
    add_text_with_bg(overlay, f"Emotion(CLIP): {clip_score:.1f}%", (10, y), (255, 255, 0))
    y += line_height
    
    add_text_with_bg(overlay, f"ColorTone: {color_sim*100:.1f}%", (10, y), (0, 200, 255))
    y += line_height
    
    if dino_sim is not None:
        add_text_with_bg(overlay, f"DINO TriPath: {dino_sim:.3f}", (10, y), (255, 128, 0))
        y += line_height
    
    if height_diff is not None:
        angle_text = "High" if height_diff > 0.12 else ("Low" if height_diff < -0.12 else "Eye")
        add_text_with_bg(overlay, f"MiDaS: {height_diff:.3f} ({angle_text})", (10, y), (200, 200, 255))
        y += line_height
    
    # YOLO bbox 표시
    if tgt_bbox is not None:
        x1, y1, x2, y2 = map(int, tgt_bbox)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(overlay, "Person", (x1, y1 - 10), font, 0.6, (0, 255, 0), 2)
    
    # 별도 창으로 표시 (WINDOW_AUTOSIZE = 비율 유지)
    cv2.namedWindow("Reference Image", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("Reference Image", ref_img)
    
    cv2.namedWindow("Your Photo (Analyzed)", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("Your Photo (Analyzed)", overlay)
    
    # 창 위치 조정 (왼쪽/오른쪽)
    cv2.moveWindow("Reference Image", 50, 50)
    ref_w = ref_img.shape[1]
    cv2.moveWindow("Your Photo (Analyzed)", 50 + ref_w + 30, 50)
    
    print("\n✅ 이미지가 표시되었습니다. 아무 키나 눌러 종료하세요...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ---------------------------------------------------------
# 실행
# ---------------------------------------------------------
if __name__ == "__main__":
    default_ref = PROJECT_ROOT / "data" / "sample_images" / "1.jpg"
    default_tgt = PROJECT_ROOT / "data" / "sample_images" / "2.jpg"
    
    if not default_ref.exists() or not default_tgt.exists():
        print("⚠️ 기본 이미지 경로 확인 필요.")
        print(f"   레퍼런스: {default_ref}")
        print(f"   타겟: {default_tgt}")
    else:
        main(str(default_ref), str(default_tgt))
        
