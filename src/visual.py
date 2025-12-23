# -*- coding: utf-8 -*-
"""
Camera Pose Assistant v2
- 수평선 기반 카메라 각도/높이 추정
- FaceMesh 눈라인 기반 하이/로우 앵글 판별
- 한글 폰트 + 안정화(민감도 완화/프레임 OK 표시)
"""

import cv2, math, numpy as np, mediapipe as mp
from PIL import ImageFont, ImageDraw, Image
from pathlib import Path

# ---------------------------
# 0. 설정
# ---------------------------

def _find_font_path() -> str:
    candidates = [
        Path(__file__).resolve().parent / "NanumGothic.ttf",
        Path("/System/Library/Fonts/Supplemental/NanumGothic.ttc"),
        Path("/Library/Fonts/NanumGothic.ttf"),
        Path("/System/Library/Fonts/Supplemental/AppleGothic.ttf"),
        Path("/Library/Fonts/AppleGothic.ttf"),
        Path("C:/Windows/Fonts/NanumGothic.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError("No suitable font found. Update FONT_PATH in visual.py.")


FONT_PATH = _find_font_path()
FONT_MAIN = ImageFont.truetype(FONT_PATH, 22)
FONT_SMALL = ImageFont.truetype(FONT_PATH, 18)

HORIZ_MAX_DEG = 12.0     # 수평선으로 간주할 최대 기울기
ROLL_WARN_DEG = 3.0      # 수평 기울기 경고 임계
EYE_HORIZ_THR = 0.10     # 허용 오차(10%) — 민감도 완화

# ---------------------------
# 1. 유틸
# ---------------------------
def put_kor(img, text, org, font=FONT_MAIN, color=(255,255,255)):
    im = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(im)
    draw.text(org, text, font=font, fill=color)
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)

def line_angle_deg(x1, y1, x2, y2):
    return math.degrees(math.atan2(y2-y1, x2-x1))

# ---------------------------
# 2. Mediapipe 초기화
# ---------------------------
mp_face = mp.solutions.face_mesh
face = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1,
                        refine_landmarks=True, min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)
L_EYE, R_EYE = 33, 263

# ---------------------------
# 3. 메인 루프
# ---------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("웹캠을 열 수 없습니다.")

print("실행 중... 'q'로 종료")

while True:
    ok, frame = cap.read()
    if not ok: break
    h, w = frame.shape[:2]
    vis = frame.copy()

    # ---------------- 3.1 배경 수평선 추정 ----------------
    small = cv2.resize(frame, (640, int(640*h/w)))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(gray, 80, 160)
    edges = cv2.dilate(edges, np.ones((3,3), np.uint8), 1)

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=90,
                            minLineLength=80, maxLineGap=15)

    horiz_lines = []
    if lines is not None:
        for l in lines[:,0,:]:
            x1,y1,x2,y2 = l.tolist()
            ang = line_angle_deg(x1,y1,x2,y2)
            if abs(ang) <= HORIZ_MAX_DEG:
                length = math.hypot(x2-x1, y2-y1)
                sx, sy = w/float(small.shape[1]), h/float(small.shape[0])
                X1,Y1,X2,Y2 = int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy)
                horiz_lines.append((X1,Y1,X2,Y2,ang,length))

    avg_roll, horiz_y = None, None
    if horiz_lines:
        weights = np.array([L for *_, L in horiz_lines], float)
        angles  = np.array([ang for *_, ang, L in horiz_lines], float)
        ys_mid  = np.array([(Y1+Y2)/2.0 for _,Y1,_,Y2,_,_ in horiz_lines], float)
        avg_roll = float(np.average(angles, weights=weights))
        horiz_y  = float(np.average(ys_mid, weights=weights))
        cv2.line(vis, (0,int(horiz_y)), (w,int(horiz_y)), (0,180,255), 3)

    # ---------------- 3.2 얼굴(눈라인) ----------------
    eye_line = None
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    f = face.process(rgb)
    if f.multi_face_landmarks:
        lm = f.multi_face_landmarks[0].landmark
        def to_xy(idx): return int(lm[idx].x * w), int(lm[idx].y * h)
        le, re = to_xy(L_EYE), to_xy(R_EYE)
        eye_y = (le[1]+re[1])/2.0
        eye_line = eye_y / float(h)
        cv2.circle(vis, le, 3, (255,180,0), -1)
        cv2.circle(vis, re, 3, (255,180,0), -1)
        cv2.line(vis, (0,int(eye_y)), (w,int(eye_y)), (120,220,120), 1)

    # ---------------- 3.3 피드백 ----------------
    messages = []
    diff = None

    # (a) 수평 기울기
    if avg_roll is not None:
        roll_abs = abs(avg_roll)
        if roll_abs > ROLL_WARN_DEG:
            dir_txt = "왼쪽이 낮음" if avg_roll < 0 else "오른쪽이 낮음"
            messages.append(f"수평 {roll_abs:.1f}° 기울어짐 ({dir_txt}) → 수평 맞추기")
        else:
            messages.append("수평 OK")
    else:
        messages.append("수평선 미검출")

    # (b) 카메라 높이(로우/하이 앵글)
    if eye_line is not None and horiz_y is not None:
        horiz_ratio = horiz_y / float(h)
        diff = horiz_ratio - eye_line
        # horizon이 눈보다 위면 → 로우앵글 / 아래면 → 하이앵글
        if diff > EYE_HORIZ_THR:
            messages.append("카메라가 낮음(로우앵글) → 카메라 올리기")
        elif diff < -EYE_HORIZ_THR:
            messages.append("카메라가 높음(하이앵글) → 카메라 낮추기")
        else:
            messages.append("카메라 높이 OK (눈높이 근처)")
        messages.append(f"[눈라인 {eye_line:.2f} / 수평선 {horiz_ratio:.2f}]")
    elif eye_line is None and horiz_y is not None:
        messages.append("얼굴 미검출 → 수평선 기준만 표시")
    elif eye_line is not None and horiz_y is None:
        messages.append("수평선 미검출 → 눈라인 기준만 표시")

    # ---------------- 3.4 안정 구도 표시 ----------------
    if diff is not None and abs(diff) <= EYE_HORIZ_THR and abs(avg_roll or 0) <= ROLL_WARN_DEG:
        cv2.rectangle(vis, (0,0), (w-1,h-1), (0,255,0), 6)
        vis = put_kor(vis, "✅ 구도 적정", (w//2-80, 40), FONT_MAIN, (0,255,0))

    # ---------------- 3.5 오버레이 ----------------
    cv2.rectangle(vis, (10,10), (480,90), (30,30,30), -1)
    if avg_roll is not None:
        color = (50,220,50) if abs(avg_roll)<=ROLL_WARN_DEG else (0,70,255)
        vis = put_kor(vis, f"수평각: {avg_roll:+.1f}°", (20,20), FONT_MAIN, color)
    if eye_line is not None:
        vis = put_kor(vis, f"눈라인: {eye_line:.2f}", (20,52), FONT_MAIN, (200,220,255))
    if horiz_y is not None:
        vis = put_kor(vis, f"수평선Y: {int(horiz_y)}px", (250,52), FONT_MAIN, (0,180,255))

    # 하단 피드백
    cv2.rectangle(vis, (10,h-90),(w-10,h-10),(25,25,25),-1)
    y = h-80
    for msg in messages[:3]:
        vis = put_kor(vis, msg, (20,y), FONT_SMALL, (255,255,255))
        y += 26

    cv2.imshow("Camera Pose Assistant v2", vis)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release(); cv2.destroyAllWindows()
