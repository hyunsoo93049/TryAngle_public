# ============================================================
# 🔥 TryAngle Feature Extractor v2
# CLIP + OpenCLIP + DINO + MiDaS(확장) + Color(확장) + YOLOv11-Pose + MediaPipe Face
# ============================================================

import os
import cv2
import numpy as np
import torch
from PIL import Image
import sys
from pathlib import Path

# Model cache
VERSION3_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = VERSION3_DIR
while PROJECT_ROOT != PROJECT_ROOT.parent and not ((PROJECT_ROOT / "data").exists() and (PROJECT_ROOT / "src").exists()):
    PROJECT_ROOT = PROJECT_ROOT.parent

if str(VERSION3_DIR) not in sys.path:
    sys.path.append(str(VERSION3_DIR))
from utils.model_cache import model_cache

# 기존 모델
import clip
import open_clip
from timm import create_model
from timm.data import resolve_model_data_config
from timm.data.transforms_factory import create_transform
from transformers import DPTImageProcessor, DPTForDepthEstimation

# 통계
from scipy.stats import skew, kurtosis
from skimage.feature import local_binary_pattern

# 🆕 YOLOv11-Pose
from ultralytics import YOLO

# 🆕 MediaPipe Face
import mediapipe as mp

# ------------------------------------------------------------
# Device
# ------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------------------------
# Model Loader (Singleton with cache)
# ------------------------------------------------------------
def load_models():
    """전체 모델 로드 (v2) - 싱글톤 캐싱"""

    def _load_all_models():
        print("🔧 Loading AI backbone models...")

        # ---- CLIP ----
        clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

        # ---- OpenCLIP ----
        openclip_model, _, openclip_preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained='laion2b_s34b_b79k', device=device
        )
        openclip_tokenizer = open_clip.get_tokenizer('ViT-B-32')

        # ---- DINOv2 ----
        dino = create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True).eval().to(device)
        dino_cfg = resolve_model_data_config(dino)
        dino_tf = create_transform(**dino_cfg)

        # ---- MiDaS (Depth) ----
        midas_processor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")
        midas_model = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to(device).eval()

        # ---- 🆕 YOLOv11-Pose ----
        yolo_pose = YOLO("yolo11s-pose.pt")  # YOLOv11s-Pose 모델

        # ---- 🆕 MediaPipe Face Mesh ----
        mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )

        _models = {
            "clip_model": clip_model,
            "clip_preprocess": clip_preprocess,
            "openclip_model": openclip_model,
            "openclip_preprocess": openclip_preprocess,
            "openclip_tokenizer": openclip_tokenizer,
            "dino_model": dino,
            "dino_tf": dino_tf,
            "midas_processor": midas_processor,
            "midas_model": midas_model,
            "yolo_pose": yolo_pose,
            "mp_face_mesh": mp_face_mesh,
        }

        print("✅ Backbone models loaded")
        return _models

    # 싱글톤 캐시에서 가져오기
    return model_cache.get_or_load("feature_extractor_models", _load_all_models)


# ============================================================
# 🆕 MiDaS 확장 (20D)
# ============================================================
def extract_midas_extended(depth_map):
    """
    MiDaS depth map에서 확장 특징 추출 (20D)
    
    depth_map: (H, W) numpy array
    """
    h, w = depth_map.shape
    
    # 기본 통계 (4D)
    depth_mean = float(np.mean(depth_map))
    depth_std = float(np.std(depth_map))
    depth_min = float(np.min(depth_map))
    depth_max = float(np.max(depth_map))
    
    # Depth 분포 (3D)
    p30 = np.percentile(depth_map, 30)
    p70 = np.percentile(depth_map, 70)
    
    foreground_ratio = float(np.sum(depth_map < p30) / depth_map.size)
    midground_ratio = float(np.sum((depth_map >= p30) & (depth_map <= p70)) / depth_map.size)
    background_ratio = float(np.sum(depth_map > p70) / depth_map.size)
    
    # Depth Gradient (2D) - 카메라 앵글 추정!
    depth_top = np.mean(depth_map[:h//3, :])
    depth_bottom = np.mean(depth_map[2*h//3:, :])
    depth_left = np.mean(depth_map[:, :w//3])
    depth_right = np.mean(depth_map[:, 2*w//3:])
    
    gradient_y = float((depth_top - depth_bottom) / (depth_mean + 1e-8))  # 양수 = 위쪽이 멀리 (high angle)
    gradient_x = float((depth_left - depth_right) / (depth_mean + 1e-8))
    
    # Spatial Depth (9D) - 3×3 그리드
    spatial_depth = []
    for i in range(3):
        for j in range(3):
            y_start = i * h // 3
            y_end = (i + 1) * h // 3
            x_start = j * w // 3
            x_end = (j + 1) * w // 3
            
            grid_mean = float(np.mean(depth_map[y_start:y_end, x_start:x_end]))
            spatial_depth.append(grid_mean)
    
    # Center Depth (1D)
    center_depth = float(np.mean(depth_map[h//3:2*h//3, w//3:2*w//3]))
    
    # Focus Sharpness (1D) - depth 분산
    focus_sharpness = float(np.std(depth_map) / (depth_mean + 1e-8))
    
    return np.array([
        depth_mean, depth_std, depth_min, depth_max,
        foreground_ratio, midground_ratio, background_ratio,
        gradient_y, gradient_x,
        *spatial_depth,
        center_depth,
        focus_sharpness
    ], dtype=np.float32)


# ============================================================
# 🆕 Color 확장 (150D)
# ============================================================
def extract_color_extended(img_bgr):
    """
    확장된 색감 특징 추출 (150D)
    """
    img = cv2.resize(img_bgr, (256, 256))
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    feats = []

    # ---- 기존 (119D) ----
    
    # HSV histogram (96D)
    for channel in cv2.split(img_hsv):
        hist = cv2.calcHist([channel], [0], None, [32], [0, 256]).flatten()
        hist = hist / (hist.sum() + 1e-6)
        feats.extend(hist)

    # LAB stats (12D)
    for channel in cv2.split(img_lab):
        flat = channel.flatten()
        feats.extend([flat.mean(), flat.std(), skew(flat), kurtosis(flat)])

    # LBP texture (10D)
    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10))
    lbp_hist = lbp_hist / (lbp_hist.sum() + 1e-6)
    feats.extend(lbp_hist)

    # Edge density (1D)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = edges.sum() / edges.size
    feats.append(edge_density)

    # ---- 🆕 추가 (31D) ----
    
    # Hue Distribution (7D) - 쿨톤/웜톤 판단!
    h_channel = img_hsv[:, :, 0]
    hue_ranges = {
        'red': ((0, 15), (165, 180)),    # 빨강
        'orange': (15, 30),              # 주황
        'yellow': (30, 60),              # 노랑
        'green': (60, 90),               # 초록
        'cyan': (90, 120),               # 청록
        'blue': (120, 150),              # 파랑
        'purple': (150, 165),            # 보라
    }
    
    for key, ranges in hue_ranges.items():
        if isinstance(ranges, tuple) and len(ranges) == 2 and isinstance(ranges[0], tuple):
            # Red (wrap around)
            mask1 = (h_channel >= ranges[0][0]) & (h_channel < ranges[0][1])
            mask2 = (h_channel >= ranges[1][0]) & (h_channel < ranges[1][1])
            ratio = float((np.sum(mask1) + np.sum(mask2)) / h_channel.size)
        else:
            mask = (h_channel >= ranges[0]) & (h_channel < ranges[1])
            ratio = float(np.sum(mask) / h_channel.size)
        feats.append(ratio)
    
    # Brightness Histogram (16D)
    brightness_hist, _ = np.histogram(gray.ravel(), bins=16, range=(0, 256))
    brightness_hist = brightness_hist / (brightness_hist.sum() + 1e-6)
    feats.extend(brightness_hist)
    
    # Contrast (1D)
    contrast = float(np.std(gray) / 128.0)
    feats.append(contrast)
    
    # Sharpness (1D) - Laplacian variance
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = float(laplacian.var())
    feats.append(sharpness)
    
    # Dominant Colors (6D) - K-Means로 주요 색 2개 추출
    pixels = img.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, _, centers = cv2.kmeans(pixels, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    dominant_colors = centers.flatten() / 255.0  # Normalize to 0-1
    feats.extend(dominant_colors)

    return np.array(feats, dtype=np.float32)


# ============================================================
# 🆕 YOLOv11-Pose 특징 (15D)
# ============================================================
def extract_yolo_pose_features(image_path, yolo_model):
    """
    YOLOv11-Pose로 포즈 특징 추출 (15D)
    
    Returns:
        numpy array (15,)
    """
    results = yolo_model.predict(image_path, verbose=False)
    
    if len(results) == 0 or results[0].keypoints is None or len(results[0].keypoints) == 0:
        # 사람 검출 실패
        return np.zeros(15, dtype=np.float32)
    
    # 첫 번째 사람의 keypoints (17개)
    kpts = results[0].keypoints.xy[0].cpu().numpy()  # (17, 2)
    confs = results[0].keypoints.conf[0].cpu().numpy()  # (17,)
    
    # Keypoints index (COCO format)
    # 0: nose, 1-2: eyes, 3-4: ears
    # 5-6: shoulders, 7-8: elbows, 9-10: wrists
    # 11-12: hips, 13-14: knees, 15-16: ankles
    
    # Pose Type (4D) - close_up/upper/half/full (onehot)
    visible_shoulders = (confs[5] > 0.3) and (confs[6] > 0.3)
    visible_hips = (confs[11] > 0.3) and (confs[12] > 0.3)
    visible_knees = (confs[13] > 0.3) and (confs[14] > 0.3)
    
    if visible_knees:
        pose_type = [0, 0, 0, 1]  # full_body
    elif visible_hips:
        pose_type = [0, 0, 1, 0]  # half_body
    elif visible_shoulders:
        pose_type = [0, 1, 0, 0]  # upper_body
    else:
        pose_type = [1, 0, 0, 0]  # close_up
    
    # Arm Angles (2D) - 팔꿈치 각도
    left_arm_angle = 0.0
    right_arm_angle = 0.0
    
    if confs[5] > 0.3 and confs[7] > 0.3 and confs[9] > 0.3:
        # Left arm
        shoulder = kpts[5]
        elbow = kpts[7]
        wrist = kpts[9]
        
        v1 = shoulder - elbow
        v2 = wrist - elbow
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        left_arm_angle = float(np.arccos(np.clip(cos_angle, -1, 1)) * 180 / np.pi)
    
    if confs[6] > 0.3 and confs[8] > 0.3 and confs[10] > 0.3:
        # Right arm
        shoulder = kpts[6]
        elbow = kpts[8]
        wrist = kpts[10]
        
        v1 = shoulder - elbow
        v2 = wrist - elbow
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        right_arm_angle = float(np.arccos(np.clip(cos_angle, -1, 1)) * 180 / np.pi)
    
    # Body Orientation (3D) - front/side/back (onehot)
    # 간단한 추정: 양쪽 어깨가 다 보이면 front, 한쪽만 보이면 side
    if visible_shoulders:
        shoulder_dist = np.linalg.norm(kpts[5] - kpts[6])
        # 임계값 기반 (이미지 크기 대비)
        if shoulder_dist > 50:  # 임의 임계값
            body_orientation = [1, 0, 0]  # front
        else:
            body_orientation = [0, 1, 0]  # side
    else:
        body_orientation = [0, 0, 1]  # back or unknown
    
    # Shoulder Tilt (1D)
    shoulder_tilt = 0.0
    if confs[5] > 0.3 and confs[6] > 0.3:
        dy = kpts[5][1] - kpts[6][1]
        dx = kpts[5][0] - kpts[6][0]
        shoulder_tilt = float(np.arctan2(dy, dx) * 180 / np.pi)
    
    # Visible Joints (5D) - binary
    visible_joints = [
        float(visible_shoulders),
        float((confs[7] > 0.3) and (confs[8] > 0.3)),  # elbows
        float((confs[9] > 0.3) and (confs[10] > 0.3)), # wrists
        float(visible_hips),
        float(visible_knees)
    ]
    
    return np.array([
        *pose_type,
        left_arm_angle / 180.0,  # Normalize to 0-1
        right_arm_angle / 180.0,
        *body_orientation,
        shoulder_tilt / 180.0,   # Normalize to -0.5~0.5
        *visible_joints
    ], dtype=np.float32)


# ============================================================
# 🆕 MediaPipe Face 특징 (7D)
# ============================================================
def extract_face_features(image_path, face_mesh):
    """
    MediaPipe Face Mesh로 얼굴 각도 추출 (7D)
    
    Returns:
        numpy array (7,)
    """
    img = cv2.imread(image_path)
    if img is None:
        return np.zeros(7, dtype=np.float32)
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape
    
    results = face_mesh.process(img_rgb)
    
    if not results.multi_face_landmarks:
        # 얼굴 검출 실패
        return np.zeros(7, dtype=np.float32)
    
    landmarks = results.multi_face_landmarks[0].landmark
    
    # 주요 랜드마크 추출
    # 코 끝: 1
    # 턱: 152
    # 왼쪽 눈: 33
    # 오른쪽 눈: 263
    # 왼쪽 입꼬리: 61
    # 오른쪽 입꼬리: 291
    
    nose_tip = np.array([landmarks[1].x, landmarks[1].y, landmarks[1].z])
    chin = np.array([landmarks[152].x, landmarks[152].y, landmarks[152].z])
    left_eye = np.array([landmarks[33].x, landmarks[33].y, landmarks[33].z])
    right_eye = np.array([landmarks[263].x, landmarks[263].y, landmarks[263].z])
    
    # Pitch (위/아래) - nose와 chin의 y 차이
    pitch = float((nose_tip[1] - chin[1]) * 90)  # -45 ~ +45 범위로 근사
    pitch = np.clip(pitch, -45, 45)
    
    # Yaw (좌/우) - 양쪽 눈의 x 차이
    eye_center = (left_eye + right_eye) / 2
    yaw = float((nose_tip[0] - eye_center[0]) * 90)
    yaw = np.clip(yaw, -45, 45)
    
    # Roll (기울기) - 양쪽 눈의 각도
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    roll = float(np.arctan2(dy, dx) * 180 / np.pi)
    roll = np.clip(roll, -45, 45)
    
    # Face Size (화면 대비 비율)
    face_x_coords = [lm.x for lm in landmarks]
    face_y_coords = [lm.y for lm in landmarks]
    
    face_width = max(face_x_coords) - min(face_x_coords)
    face_height = max(face_y_coords) - min(face_y_coords)
    face_size = float((face_width * face_height))
    
    # Face Position (중심 위치)
    face_center_x = float((max(face_x_coords) + min(face_x_coords)) / 2)
    face_center_y = float((max(face_y_coords) + min(face_y_coords)) / 2)
    
    # Confidence (간단히 1.0으로)
    confidence = 1.0
    
    return np.array([
        pitch / 45.0,         # Normalize to -1~1
        yaw / 45.0,
        roll / 45.0,
        face_size,
        face_center_x,
        face_center_y,
        confidence
    ], dtype=np.float32)


# ============================================================
# 🎯 Main Feature Extractor v2
# ============================================================
def extract_features_v2(image_path):
    """
    이미지 1장에서 모든 특징 추출 (v2)
    
    Returns:
        dict with keys:
        - clip: (512,)
        - openclip: (512,)
        - dino: (384,)
        - midas: (20,)
        - color: (150,)
        - yolo_pose: (15,)
        - face: (7,)
    """
    models = load_models()

    try:
        img_pil = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"[❌] 이미지 로드 실패: {image_path} : {e}")
        return None

    img_bgr = cv2.imread(image_path)

    # --------------------------------------------------------
    # 1) CLIP
    # --------------------------------------------------------
    clip_in = models["clip_preprocess"](img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        clip_feat = models["clip_model"].encode_image(clip_in)
        clip_feat = (clip_feat / clip_feat.norm(dim=-1, keepdim=True)).cpu().numpy().astype(np.float32).flatten()

    # --------------------------------------------------------
    # 2) OpenCLIP
    # --------------------------------------------------------
    openclip_in = models["openclip_preprocess"](img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        openclip_feat = models["openclip_model"].encode_image(openclip_in)
        openclip_feat = (openclip_feat / openclip_feat.norm(dim=-1, keepdim=True)).cpu().numpy().astype(np.float32).flatten()

    # --------------------------------------------------------
    # 3) DINOv2
    # --------------------------------------------------------
    dino_in = models["dino_tf"](img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        feats = models["dino_model"].forward_features(dino_in)
        
        if isinstance(feats, dict):
            if "x_norm_clstoken" in feats:
                dino_token = feats["x_norm_clstoken"]  # (1, 384)
            elif "pool" in feats:
                dino_token = feats["pool"]  # (1, 384)
            else:
                raise ValueError("❌ DINO dict에서 토큰을 찾을 수 없음.")
        elif isinstance(feats, torch.Tensor):
            # Tensor 형태인 경우
            if feats.ndim == 3:
                # (1, num_patches, 384) 형태
                # CLS token은 첫 번째 토큰
                dino_token = feats[:, 0, :]  # (1, 384)
            elif feats.ndim == 2:
                # (1, 384) 형태 - 이미 CLS token
                dino_token = feats
            else:
                raise ValueError(f"❌ DINO tensor 차원 오류: {feats.shape}")
            
            # 최종 shape 확인
            if dino_token.shape[-1] != 384:
                raise ValueError(f"❌ DINO token shape mismatch: {dino_token.shape}, expected (1, 384)")
        else:
            raise ValueError("❌ DINO 출력 형식이 완전히 다릅니다.")

        # Normalize
        dino_token = dino_token / (dino_token.norm(dim=-1, keepdim=True) + 1e-8)
        dino_feat = dino_token.squeeze().cpu().numpy().astype(np.float32)  # (384,)

    # --------------------------------------------------------
    # 4) MiDaS Depth (확장)
    # --------------------------------------------------------
    inputs = models["midas_processor"](images=img_pil, return_tensors="pt").to(device)
    with torch.no_grad():
        depth = models["midas_model"](inputs["pixel_values"]).predicted_depth
    
    depth_map = depth.squeeze().cpu().numpy()
    midas_feat = extract_midas_extended(depth_map)

    # --------------------------------------------------------
    # 5) Color + Texture (확장)
    # --------------------------------------------------------
    color_feat = extract_color_extended(img_bgr)

    # --------------------------------------------------------
    # 6) 🆕 YOLOv11-Pose
    # --------------------------------------------------------
    yolo_pose_feat = extract_yolo_pose_features(image_path, models["yolo_pose"])

    # --------------------------------------------------------
    # 7) 🆕 MediaPipe Face
    # --------------------------------------------------------
    face_feat = extract_face_features(image_path, models["mp_face_mesh"])

    return {
        "clip": clip_feat,
        "openclip": openclip_feat,
        "dino": dino_feat,
        "midas": midas_feat,
        "color": color_feat,
        "yolo_pose": yolo_pose_feat,
        "face": face_feat,
    }


# ============================================================
# ✅ Backward-compatible wrapper
# ============================================================
def extract_features_full_v2(image_path: str):
    """
    버전2 wrapper
    """
    return extract_features_v2(image_path)


# ============================================================
# 테스트
# ============================================================
if __name__ == "__main__":
    test_img = PROJECT_ROOT / "data" / "test_images" / "test1.jpg"
    
    print(f"\n🔧 Testing Feature Extractor v2")
    print(f"Image: {test_img}")
    
    result = extract_features_v2(str(test_img))
    
    if result:
        print("\n✅ Feature Extraction Success!")
        print(f"CLIP:         {result['clip'].shape}")
        print(f"OpenCLIP:     {result['openclip'].shape}")
        print(f"DINO:         {result['dino'].shape}")
        print(f"MiDaS:        {result['midas'].shape}")
        print(f"Color:        {result['color'].shape}")
        print(f"YOLO-Pose:    {result['yolo_pose'].shape}")
        print(f"Face:         {result['face'].shape}")
        
        total_dim = (
            result['clip'].shape[0] +
            result['openclip'].shape[0] +
            result['dino'].shape[0] +
            result['midas'].shape[0] +
            result['color'].shape[0] +
            result['yolo_pose'].shape[0] +
            result['face'].shape[0]
        )
        print(f"\n📊 Total Dimensions: {total_dim}D")
    else:
        print("❌ Feature extraction failed!")
