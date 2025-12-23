# -*- coding: utf-8 -*-
"""
TryAngle 고도화 버전 (CLIP + YOLO Pose + Camera Angle + ColorTone)
---------------------------------------------------------------
1️⃣ CLIP : 장면 감성(배경, 톤, 분위기)
2️⃣ YOLO Pose : 인물의 구도·프레이밍
3️⃣ ColorTone : 색감 및 조명 유사도
4️⃣ Camera Angle : 시점(높이/거리) 차이 분석
"""

import cv2, torch, clip, numpy as np
from ultralytics import YOLO
from PIL import Image
from composition_module import analyze_composition
from feedback_module import generate_feedback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
while PROJECT_ROOT != PROJECT_ROOT.parent and not ((PROJECT_ROOT / "data").exists() and (PROJECT_ROOT / "src").exists()):
    PROJECT_ROOT = PROJECT_ROOT.parent

# ---------------------------------------------------------
# 모델 초기화
# ---------------------------------------------------------
print("모델 로드 중...")
device = "cuda" if torch.cuda.is_available() else "cpu"
pose_model = YOLO("yolov8s-pose.pt")
clip_model, preprocess = clip.load("ViT-B/32", device=device)
print("✅ 모델 로딩 완료")

# ---------------------------------------------------------
# 경로 설정
# ---------------------------------------------------------
ref_path = PROJECT_ROOT / "data" / "sample_images" / "cafe1.jpg"
tgt_path = PROJECT_ROOT / "data" / "sample_images" / "cafe5.jpg"

ref_img = cv2.imread(str(ref_path))
tgt_img = cv2.imread(str(tgt_path))
if ref_img is None or tgt_img is None:
    raise FileNotFoundError("이미지 경로 확인 필요")

# ---------------------------------------------------------
# 색감/조명 유사도 계산 (HSV 히스토그램)
# ---------------------------------------------------------
def color_tone_similarity(img1, img2):
    hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
    hist1 = cv2.calcHist([hsv1],[0,1,2],None,[24,8,8],[0,180,0,256,0,256])
    hist2 = cv2.calcHist([hsv2],[0,1,2],None,[24,8,8],[0,180,0,256,0,256])
    cv2.normalize(hist1,hist1); cv2.normalize(hist2,hist2)
    sim = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return float((sim + 1) / 2)  # 0~1

# ---------------------------------------------------------
# CLIP 감성(장면) 유사도 계산
# ---------------------------------------------------------
def clip_similarity(path1, path2):
    img1 = preprocess(Image.open(path1)).unsqueeze(0).to(device)
    img2 = preprocess(Image.open(path2)).unsqueeze(0).to(device)
    with torch.no_grad():
        f1 = clip_model.encode_image(img1); f2 = clip_model.encode_image(img2)
        f1 /= f1.norm(dim=-1, keepdim=True)
        f2 /= f2.norm(dim=-1, keepdim=True)
    sim = torch.cosine_similarity(f1, f2).item()
    # CLIP은 0.25~0.35가 '유사'한 범위이므로 정규화
    normed = np.clip((sim - 0.2) / (0.35 - 0.2), 0, 1)
    return float(normed)

# ---------------------------------------------------------
# Pose 분석 및 구도 정보
# ---------------------------------------------------------
def get_pose_info(image):
    res = pose_model(image)
    if res[0].keypoints is None:
        return None, None
    kpts = res[0].keypoints.xy[0].cpu().numpy()
    bbox = res[0].boxes.xyxy[0].cpu().numpy()
    comp = analyze_composition(image, kpts, bbox)
    return kpts, comp

# ---------------------------------------------------------
# 카메라 앵글 비교 (시점 높낮이)
# ---------------------------------------------------------
def camera_angle_difference(ref_kp, tgt_kp):
    # 코와 눈 위치를 기준으로 카메라 시점 추정
    if ref_kp is None or tgt_kp is None:
        return None
    # y 좌표(화면에서 아래로 증가)
    ref_eye = np.mean(ref_kp[[1,2,3,4]], axis=0)[1]  # 대략 눈라인
    tgt_eye = np.mean(tgt_kp[[1,2,3,4]], axis=0)[1]
    diff = (tgt_eye - ref_eye) / 480.0  # 480 기준 정규화
    return float(diff)

# ---------------------------------------------------------
# 실행
# ---------------------------------------------------------
ref_kp, ref_comp = get_pose_info(ref_img)
tgt_kp, tgt_comp = get_pose_info(tgt_img)

color_sim = color_tone_similarity(ref_img, tgt_img)
clip_sim = clip_similarity(ref_path, tgt_path)
angle_diff = camera_angle_difference(ref_kp, tgt_kp)

emotion_score = round(clip_sim * 100, 2)
color_score = round(color_sim * 100, 2)

print(f"📷 [레퍼런스 구도]: {ref_comp['score']:.2f}")
print(f"📸 [내 사진 구도]: {tgt_comp['score']:.2f}")
print(f"🎨 [색감 유사도]: {color_score}%")
print(f"💫 [감성(CLIP) 유사도]: {emotion_score}%")

# ---------------------------------------------------------
# 피드백 생성
# ---------------------------------------------------------
# 구체적 이유(Composition에서 넘어옴)
reasons = []
if not tgt_comp["on_rule_of_thirds"]:
    reasons.append("인물이 삼분할선에서 벗어나 있습니다.")
if tgt_comp["size_ratio"] > 0.45:
    reasons.append("인물이 화면을 과도하게 차지하고 있습니다. 약간 멀리서 촬영해보세요.")
if tgt_comp["headroom_ratio"] and tgt_comp["headroom_ratio"] < 0.05:
    reasons.append("머리 위 여백이 너무 적어요. 카메라를 약간 위로 올려보세요.")

# 카메라 앵글 차이 기반 피드백
if angle_diff is not None:
    if angle_diff > 0.1:
        reasons.append("카메라가 너무 낮아요. 약간 높여보세요.")
    elif angle_diff < -0.1:
        reasons.append("카메라가 너무 높아요. 인물 눈높이에 맞춰보세요.")

summary = "전반적으로 감성은 비슷하지만 구도와 시점 보정이 필요해요."

feedback = generate_feedback(
    pose_conf=None,
    composition_score=tgt_comp["score"],
    emotion_score=emotion_score,
    reasons=reasons,
    summary=summary
)

# ---------------------------------------------------------
# 결과 출력
# ---------------------------------------------------------
print("\n💬 [AI 피드백]")
for line in feedback:
    print(line)

# ---------------------------------------------------------
# 시각화
# ---------------------------------------------------------
cv2.putText(tgt_img, f"Composition: {tgt_comp['score']:.1f}", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
cv2.putText(tgt_img, f"Emotion: {emotion_score:.1f}%", (20,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
cv2.putText(tgt_img, f"ColorTone: {color_score:.1f}%", (20,120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,200,255), 2)
cv2.imshow("Reference", ref_img)
cv2.imshow("Your Photo (Analyzed)", tgt_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
