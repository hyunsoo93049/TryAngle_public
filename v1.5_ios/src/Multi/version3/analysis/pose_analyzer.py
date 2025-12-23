# ============================================================
# 🤸 TryAngle - Pose Analyzer
# Phase 2-3: YOLO11 / MoveNet + MediaPipe 하이브리드 포즈 분석
# ============================================================

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
import os
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

# YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("⚠️ ultralytics not installed. YOLO pose detection disabled.")
    YOLO_AVAILABLE = False

# Phase 2-3: MoveNet 추가
try:
    from analysis.movenet_analyzer import MoveNetAnalyzer
    MOVENET_AVAILABLE = True
except ImportError:
    print("⚠️ MoveNet not available. Install TensorFlow: pip install tensorflow==2.15.0")
    MOVENET_AVAILABLE = False

# MediaPipe
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    print("⚠️ mediapipe not installed. MediaPipe detection disabled.")
    MEDIAPIPE_AVAILABLE = False


class PoseAnalyzer:
    """
    Phase 2-3: YOLO11 / MoveNet + MediaPipe 하이브리드 포즈 분석기

    포즈 모델 선택:
    - YOLO11: 범용 객체 검출 기반 (정확도 62.5%, 속도 45fps)
    - MoveNet: 포즈 전문 모델 (정확도 77.6%, 속도 30fps) ⭐ 추천

    시나리오별 최적 모델 선택:
    - 전신/뒷모습/옆모습/멀리: 포즈 모델만
    - 얼굴 클로즈업: 포즈 모델 + MediaPipe Face
    - 손 제스처: 포즈 모델 + MediaPipe Hands
    - 디테일 필요: 포즈 모델 + MediaPipe Pose
    """

    # 17개 키포인트 (COCO format, YOLO & MoveNet 공통)
    KEYPOINTS = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]

    def __init__(self, use_movenet: bool = False, yolo_model_path: str = None, movenet_model_path: str = None):
        """
        Phase 2-3: MoveNet 지원 추가

        Args:
            use_movenet: True면 MoveNet, False면 YOLO11 (기본값)
            yolo_model_path: YOLO 모델 경로 (use_movenet=False일 때 사용)
            movenet_model_path: MoveNet 모델 경로 (use_movenet=True일 때 사용)
        """
        self.use_movenet = use_movenet
        self.pose_model = None

        if use_movenet:
            # MoveNet 사용
            if not MOVENET_AVAILABLE:
                raise ImportError(
                    "MoveNet not available. Install TensorFlow: pip install tensorflow==2.15.0\n"
                    "And download model: python scripts/download_movenet.py"
                )

            print("  🏃 Using MoveNet Thunder (정확도 우선)")
            self.pose_model = MoveNetAnalyzer(model_path=movenet_model_path)
            self.model_type = 'movenet'

        else:
            # YOLO11 사용 (기존)
            if not YOLO_AVAILABLE:
                raise ImportError("ultralytics package required. Install: pip install ultralytics")

            if yolo_model_path is None:
                yolo_model_path = VERSION3_DIR / "yolo11s-pose.pt"

            if not os.path.exists(yolo_model_path):
                raise FileNotFoundError(f"YOLO model not found: {yolo_model_path}")

            # 싱글톤 패턴으로 YOLO 모델 로드
            def load_yolo():
                print(f"  🔧 Loading YOLO11-pose from {os.path.basename(yolo_model_path)}...")
                return YOLO(yolo_model_path)

            self.yolo = model_cache.get_or_load("yolo_pose", load_yolo)
            self.pose_model = self.yolo
            self.model_type = 'yolo'

        # MediaPipe 초기화 (lazy loading)
        self.mp_pose = None
        self.mp_face = None
        self.mp_hands = None

        if MEDIAPIPE_AVAILABLE:
            self.mp = mp
            print("  ✅ MediaPipe available")
        else:
            print("  ⚠️ MediaPipe not available - YOLO only mode")

    def _init_mediapipe_pose(self):
        """MediaPipe Pose 초기화 (필요시)"""
        if self.mp_pose is None and MEDIAPIPE_AVAILABLE:
            self.mp_pose = self.mp.solutions.pose.Pose(
                static_image_mode=True,
                model_complexity=2,
                min_detection_confidence=0.5
            )

    def _init_mediapipe_face(self):
        """MediaPipe Face Mesh 초기화 (필요시)"""
        if self.mp_face is None and MEDIAPIPE_AVAILABLE:
            self.mp_face = self.mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                min_detection_confidence=0.5
            )

    def _init_mediapipe_hands(self):
        """MediaPipe Hands 초기화 (필요시)"""
        if self.mp_hands is None and MEDIAPIPE_AVAILABLE:
            self.mp_hands = self.mp.solutions.hands.Hands(
                static_image_mode=True,
                max_num_hands=2,
                min_detection_confidence=0.5
            )

    def analyze(self, image_path: str) -> Dict:
        """
        Phase 2-3: 이미지에서 포즈 추출 (MoveNet / YOLO11 선택 가능)

        Returns:
            {
                'scenario': 'full_body' | 'face_closeup' | 'hand_gesture' | 'back_view',
                'yolo_keypoints': [...],  # 호환성 위해 이름 유지 (MoveNet도 동일 포맷)
                'mediapipe_pose': [...] (optional),
                'mediapipe_face': [...] (optional),
                'mediapipe_hands': [...] (optional),
                'merged_keypoints': {...},
                'confidence': float,
                'bbox': [x1, y1, x2, y2],
                'model_type': 'yolo' | 'movenet'
            }
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # 이미지 로드
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        # Phase 2-3: MoveNet vs YOLO11 선택
        if self.use_movenet:
            # Step 1: MoveNet 실행
            pose_result = self._run_movenet(image_path)
        else:
            # Step 1: YOLO 실행 (기존)
            pose_result = self._run_yolo(img_rgb, h, w)

        # Phase 1-1: Threshold 최적화 (0.3 → 0.15)
        # 측면 포즈, 얼굴 가린 포즈 등에서 검출률 향상
        if pose_result is None or pose_result['confidence'] < 0.15:
            return {
                'scenario': 'no_person',
                'yolo_keypoints': None,
                'merged_keypoints': None,
                'confidence': 0.0,
                'bbox': None
            }

        # Step 2: 시나리오 판단
        scenario = self._detect_scenario(pose_result, h, w)

        # Step 3: 시나리오별 MediaPipe 추가 실행
        result = {
            'scenario': scenario,
            'yolo_keypoints': pose_result['keypoints'],  # 호환성 위해 이름 유지
            'yolo_confidence': pose_result['confidence'],
            'bbox': pose_result['bbox'],
            'model_type': self.model_type  # Phase 2-3: 사용된 모델 타입
        }

        if scenario == 'face_closeup' and MEDIAPIPE_AVAILABLE:
            self._init_mediapipe_face()
            mp_face_result = self._run_mediapipe_face(img_rgb)
            result['mediapipe_face'] = mp_face_result

        elif scenario == 'hand_gesture' and MEDIAPIPE_AVAILABLE:
            self._init_mediapipe_hands()
            mp_hands_result = self._run_mediapipe_hands(img_rgb)
            result['mediapipe_hands'] = mp_hands_result

        elif scenario in ['full_body', 'upper_body'] and MEDIAPIPE_AVAILABLE:
            self._init_mediapipe_pose()
            mp_pose_result = self._run_mediapipe_pose(img_rgb)
            result['mediapipe_pose'] = mp_pose_result

        # Step 4: 키포인트 병합
        result['merged_keypoints'] = self._merge_keypoints(result)
        result['confidence'] = pose_result['confidence']

        return result

    def _run_yolo(self, img_rgb: np.ndarray, h: int, w: int) -> Optional[Dict]:
        """YOLO 포즈 검출"""
        results = self.yolo(img_rgb, verbose=False)

        if len(results) == 0 or len(results[0].keypoints) == 0:
            return None

        # 첫 번째 사람만 (가장 confidence 높은)
        result = results[0]

        if result.keypoints is None or len(result.keypoints.data) == 0:
            return None

        keypoints_data = result.keypoints.data[0]  # [17, 3] (x, y, conf)
        boxes = result.boxes.data[0]  # [x1, y1, x2, y2, conf, class]

        # 정규화된 좌표로 변환
        keypoints = []
        for i, kp_name in enumerate(self.YOLO_KEYPOINTS):
            x, y, conf = keypoints_data[i]
            keypoints.append({
                'name': kp_name,
                'x': float(x) / w,  # 정규화 (0~1)
                'y': float(y) / h,
                'confidence': float(conf)
            })

        return {
            'keypoints': keypoints,
            'confidence': float(boxes[4]),
            'bbox': [float(boxes[0])/w, float(boxes[1])/h,
                     float(boxes[2])/w, float(boxes[3])/h]
        }

    def _run_movenet(self, image_path: str) -> Optional[Dict]:
        """
        MoveNet 포즈 검출

        Args:
            image_path: 이미지 파일 경로

        Returns:
            YOLO와 동일한 포맷의 결과
            {
                'keypoints': List[Dict],
                'confidence': float,
                'bbox': List[float]
            }
        """
        try:
            result = self.pose_model.analyze(image_path)

            # MoveNet 결과는 이미 YOLO와 동일한 포맷
            # (movenet_analyzer.py에서 호환 포맷으로 반환)
            return {
                'keypoints': result['keypoints'],
                'confidence': result['confidence'],
                'bbox': result['bbox']
            }

        except Exception as e:
            print(f"⚠️ MoveNet analysis failed: {e}")
            return None

    def _detect_scenario(self, pose_result: Dict, h: int, w: int) -> str:
        """
        시나리오 자동 판단

        Returns:
            'full_body' | 'upper_body' | 'face_closeup' | 'hand_gesture' | 'back_view'
        """
        bbox = pose_result['bbox']
        keypoints = pose_result['keypoints']

        # bbox 크기
        bbox_width = bbox[2] - bbox[0]
        bbox_height = bbox[3] - bbox[1]
        bbox_area = bbox_width * bbox_height

        # 키포인트 신뢰도
        kp_dict = {kp['name']: kp for kp in keypoints}

        # 얼굴 키포인트 신뢰도
        face_conf = np.mean([
            kp_dict['nose']['confidence'],
            kp_dict['left_eye']['confidence'],
            kp_dict['right_eye']['confidence']
        ])

        # 손 키포인트 신뢰도
        hand_conf = np.mean([
            kp_dict['left_wrist']['confidence'],
            kp_dict['right_wrist']['confidence']
        ])

        # 하체 키포인트 신뢰도
        lower_body_conf = np.mean([
            kp_dict['left_knee']['confidence'],
            kp_dict['right_knee']['confidence'],
            kp_dict['left_ankle']['confidence'],
            kp_dict['right_ankle']['confidence']
        ])

        # 판단 로직
        if bbox_area > 0.4 and face_conf > 0.7:
            return 'face_closeup'
        elif hand_conf > 0.7 and bbox_height < 0.6:
            return 'hand_gesture'
        elif lower_body_conf > 0.5:
            return 'full_body'
        elif face_conf < 0.3:
            return 'back_view'
        else:
            return 'upper_body'

    def _run_mediapipe_pose(self, img_rgb: np.ndarray) -> Optional[Dict]:
        """MediaPipe Pose 실행 (33 keypoints)"""
        if self.mp_pose is None:
            return None

        results = self.mp_pose.process(img_rgb)

        if results.pose_landmarks is None:
            return None

        keypoints = []
        for i, lm in enumerate(results.pose_landmarks.landmark):
            keypoints.append({
                'id': i,
                'x': lm.x,
                'y': lm.y,
                'z': lm.z,
                'visibility': lm.visibility
            })

        return {
            'keypoints': keypoints,
            'count': len(keypoints)
        }

    def _run_mediapipe_face(self, img_rgb: np.ndarray) -> Optional[Dict]:
        """MediaPipe Face Mesh 실행 (468 keypoints)"""
        if self.mp_face is None:
            return None

        results = self.mp_face.process(img_rgb)

        if results.multi_face_landmarks is None or len(results.multi_face_landmarks) == 0:
            return None

        face_landmarks = results.multi_face_landmarks[0]
        keypoints = []
        for i, lm in enumerate(face_landmarks.landmark):
            keypoints.append({
                'id': i,
                'x': lm.x,
                'y': lm.y,
                'z': lm.z
            })

        # 주요 포인트만 추출 (눈, 코, 입)
        key_indices = {
            'nose_tip': 1,
            'left_eye': 33,
            'right_eye': 263,
            'left_mouth': 61,
            'right_mouth': 291,
            'chin': 152
        }

        key_points = {}
        for name, idx in key_indices.items():
            key_points[name] = {
                'x': keypoints[idx]['x'],
                'y': keypoints[idx]['y'],
                'z': keypoints[idx]['z']
            }

        return {
            'keypoints': keypoints,
            'key_points': key_points,
            'count': len(keypoints)
        }

    def _run_mediapipe_hands(self, img_rgb: np.ndarray) -> Optional[Dict]:
        """MediaPipe Hands 실행 (21 keypoints per hand)"""
        if self.mp_hands is None:
            return None

        results = self.mp_hands.process(img_rgb)

        if results.multi_hand_landmarks is None:
            return None

        hands = []
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            keypoints = []
            for i, lm in enumerate(hand_landmarks.landmark):
                keypoints.append({
                    'id': i,
                    'x': lm.x,
                    'y': lm.y,
                    'z': lm.z
                })

            handedness = results.multi_handedness[hand_idx].classification[0].label

            hands.append({
                'handedness': handedness,  # 'Left' or 'Right'
                'keypoints': keypoints,
                'count': len(keypoints)
            })

        return {
            'hands': hands,
            'hand_count': len(hands)
        }

    def _merge_keypoints(self, result: Dict) -> Dict:
        """
        YOLO + MediaPipe 키포인트 병합

        Returns:
            {
                'base': {...},  # YOLO 17개 (항상)
                'face': {...},  # MediaPipe Face (optional)
                'hands': {...}, # MediaPipe Hands (optional)
                'pose_33': {...} # MediaPipe Pose 33개 (optional)
            }
        """
        merged = {
            'base': {}  # YOLO keypoints
        }

        # YOLO keypoints (base)
        for kp in result['yolo_keypoints']:
            merged['base'][kp['name']] = {
                'x': kp['x'],
                'y': kp['y'],
                'confidence': kp['confidence']
            }

        # MediaPipe Face
        if 'mediapipe_face' in result and result['mediapipe_face'] is not None:
            merged['face'] = result['mediapipe_face']['key_points']

        # MediaPipe Hands
        if 'mediapipe_hands' in result and result['mediapipe_hands'] is not None:
            merged['hands'] = result['mediapipe_hands']['hands']

        # MediaPipe Pose
        if 'mediapipe_pose' in result and result['mediapipe_pose'] is not None:
            merged['pose_33'] = {}
            for kp in result['mediapipe_pose']['keypoints']:
                merged['pose_33'][f'point_{kp["id"]}'] = {
                    'x': kp['x'],
                    'y': kp['y'],
                    'z': kp['z'],
                    'visibility': kp['visibility']
                }

        return merged


# ============================================================
# 포즈 비교 함수
# ============================================================

def compare_poses(ref_pose: Dict, user_pose: Dict) -> Dict:
    """
    레퍼런스 vs 사용자 포즈 비교

    Returns:
        {
            'similarity': float (0~1),
            'angle_differences': {...},
            'position_differences': {...},
            'feedback': [...]
        }
    """
    if ref_pose is None or user_pose is None:
        return {
            'similarity': 0.0,
            'feedback': ['포즈를 감지할 수 없습니다']
        }

    # 시나리오 체크
    if ref_pose['scenario'] != user_pose['scenario']:
        return {
            'similarity': 0.0,
            'feedback': [f"⚠️ 포즈 타입이 다릅니다 (레퍼런스: {ref_pose['scenario']}, 현재: {user_pose['scenario']})"]
        }

    ref_kp = ref_pose['merged_keypoints']['base']
    user_kp = user_pose['merged_keypoints']['base']

    # 각도 비교
    angle_diffs = _compare_angles(ref_kp, user_kp)

    # 위치 비교
    position_diffs = _compare_positions(ref_kp, user_kp)

    # 유사도 계산
    similarity = _calculate_similarity(angle_diffs, position_diffs)

    # 피드백 생성
    feedback = _generate_pose_feedback(angle_diffs, position_diffs, ref_kp, user_kp)

    return {
        'similarity': similarity,
        'angle_differences': angle_diffs,
        'position_differences': position_diffs,
        'feedback': feedback
    }


def _compare_angles(ref_kp: Dict, user_kp: Dict, conf_threshold: float = 0.25) -> Dict:
    """
    주요 관절 각도 비교

    Phase 1-1: conf_threshold 최적화 (0.5 → 0.25)
    낮은 confidence 키포인트도 활용하여 포즈 비교 정확도 향상
    """
    angles = {}

    # 팔꿈치 각도 (왼쪽)
    if all(k in ref_kp and ref_kp[k]['confidence'] > conf_threshold for k in ['left_shoulder', 'left_elbow', 'left_wrist']):
        ref_angle = _calculate_angle(
            ref_kp['left_shoulder'], ref_kp['left_elbow'], ref_kp['left_wrist']
        )
        user_angle = _calculate_angle(
            user_kp['left_shoulder'], user_kp['left_elbow'], user_kp['left_wrist']
        )
        angles['left_elbow'] = user_angle - ref_angle

    # 팔꿈치 각도 (오른쪽)
    if all(k in ref_kp and ref_kp[k]['confidence'] > conf_threshold for k in ['right_shoulder', 'right_elbow', 'right_wrist']):
        ref_angle = _calculate_angle(
            ref_kp['right_shoulder'], ref_kp['right_elbow'], ref_kp['right_wrist']
        )
        user_angle = _calculate_angle(
            user_kp['right_shoulder'], user_kp['right_elbow'], user_kp['right_wrist']
        )
        angles['right_elbow'] = user_angle - ref_angle

    # 어깨 각도 (팔 들어올림 정도)
    if all(k in ref_kp and ref_kp[k]['confidence'] > conf_threshold for k in ['left_shoulder', 'left_elbow', 'left_hip']):
        ref_angle = _calculate_angle(
            ref_kp['left_hip'], ref_kp['left_shoulder'], ref_kp['left_elbow']
        )
        user_angle = _calculate_angle(
            user_kp['left_hip'], user_kp['left_shoulder'], user_kp['left_elbow']
        )
        angles['left_shoulder'] = user_angle - ref_angle

    if all(k in ref_kp and ref_kp[k]['confidence'] > conf_threshold for k in ['right_shoulder', 'right_elbow', 'right_hip']):
        ref_angle = _calculate_angle(
            ref_kp['right_hip'], ref_kp['right_shoulder'], ref_kp['right_elbow']
        )
        user_angle = _calculate_angle(
            user_kp['right_hip'], user_kp['right_shoulder'], user_kp['right_elbow']
        )
        angles['right_shoulder'] = user_angle - ref_angle

    # 얼굴 각도 (고개 좌우)
    if all(k in ref_kp and ref_kp[k]['confidence'] > 0.5 for k in ['nose', 'left_eye', 'right_eye']):
        # 코와 양 눈으로 얼굴 각도 계산
        ref_angle = _calculate_angle(
            ref_kp['left_eye'], ref_kp['nose'], ref_kp['right_eye']
        )
        user_angle = _calculate_angle(
            user_kp['left_eye'], user_kp['nose'], user_kp['right_eye']
        )
        angles['face_angle'] = user_angle - ref_angle

    return angles


def _compare_positions(ref_kp: Dict, user_kp: Dict, conf_threshold: float = 0.2) -> Dict:
    """
    주요 키포인트 상대 위치 비교

    Phase 1-1: conf_threshold 최적화 (0.3 → 0.2)
    """
    positions = {}

    # 손목 높이 비교
    if 'left_wrist' in ref_kp and ref_kp['left_wrist']['confidence'] > conf_threshold:
        positions['left_wrist_y'] = user_kp['left_wrist']['y'] - ref_kp['left_wrist']['y']

    if 'right_wrist' in ref_kp and ref_kp['right_wrist']['confidence'] > conf_threshold:
        positions['right_wrist_y'] = user_kp['right_wrist']['y'] - ref_kp['right_wrist']['y']

    # 고개 기울기 (귀)
    if all(k in ref_kp and ref_kp[k]['confidence'] > 0.4 for k in ['left_ear', 'right_ear']):
        ref_head_tilt = (ref_kp['left_ear']['y'] - ref_kp['right_ear']['y'])
        user_head_tilt = (user_kp['left_ear']['y'] - user_kp['right_ear']['y'])
        positions['head_tilt'] = user_head_tilt - ref_head_tilt

    # 코 위치 (얼굴 상하 위치)
    if 'nose' in ref_kp and ref_kp['nose']['confidence'] > 0.7:
        positions['nose_y'] = user_kp['nose']['y'] - ref_kp['nose']['y']

    # 어깨 너비 비교
    if all(k in ref_kp and ref_kp[k]['confidence'] > 0.5 for k in ['left_shoulder', 'right_shoulder']):
        ref_shoulder_width = abs(ref_kp['left_shoulder']['x'] - ref_kp['right_shoulder']['x'])
        user_shoulder_width = abs(user_kp['left_shoulder']['x'] - user_kp['right_shoulder']['x'])
        positions['shoulder_width'] = user_shoulder_width - ref_shoulder_width

    return positions


def _calculate_angle(p1: Dict, p2: Dict, p3: Dict) -> float:
    """3점으로 각도 계산 (p2가 꼭짓점)"""
    v1 = np.array([p1['x'] - p2['x'], p1['y'] - p2['y']])
    v2 = np.array([p3['x'] - p2['x'], p3['y'] - p2['y']])

    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

    return float(angle)


def _calculate_similarity(angle_diffs: Dict, position_diffs: Dict) -> float:
    """전체 유사도 계산 (0~1)"""
    if not angle_diffs and not position_diffs:
        return 0.0

    # 각도 차이 점수
    angle_scores = []
    for diff in angle_diffs.values():
        score = max(0, 1 - abs(diff) / 90.0)  # 90도 이상 차이면 0점
        angle_scores.append(score)

    # 위치 차이 점수
    position_scores = []
    for diff in position_diffs.values():
        score = max(0, 1 - abs(diff) / 0.5)  # 50% 이상 차이면 0점
        position_scores.append(score)

    all_scores = angle_scores + position_scores

    if not all_scores:
        return 0.0

    return float(np.mean(all_scores))


def _generate_pose_feedback(angle_diffs: Dict, position_diffs: Dict,
                           ref_kp: Dict, user_kp: Dict) -> List[str]:
    """
    Phase 1-3: 구체적인 포즈 피드백 생성 (현재 각도 + 목표 각도 표시)

    기존: "왼팔 팔꿈치를 25도 더 펴세요"
    개선: "왼팔 팔꿈치를 25도 더 펴세요 (현재 90°, 목표 115°)"
    """
    feedback = []

    # 각도 피드백 (임계값 높여서 안정화)
    if 'left_elbow' in angle_diffs and abs(angle_diffs['left_elbow']) > 25:  # 15 -> 25
        # 현재 각도 계산
        current_angle = _calculate_angle(
            user_kp['left_shoulder'], user_kp['left_elbow'], user_kp['left_wrist']
        )
        target_angle = _calculate_angle(
            ref_kp['left_shoulder'], ref_kp['left_elbow'], ref_kp['left_wrist']
        )

        if angle_diffs['left_elbow'] > 0:
            feedback.append(
                f"왼팔 팔꿈치를 {abs(angle_diffs['left_elbow']):.0f}° 더 펴세요 "
                f"(현재 {current_angle:.0f}°, 목표 {target_angle:.0f}°)"
            )
        else:
            feedback.append(
                f"왼팔 팔꿈치를 {abs(angle_diffs['left_elbow']):.0f}° 더 구부리세요 "
                f"(현재 {current_angle:.0f}°, 목표 {target_angle:.0f}°)"
            )

    if 'right_elbow' in angle_diffs and abs(angle_diffs['right_elbow']) > 25:  # 15 -> 25
        current_angle = _calculate_angle(
            user_kp['right_shoulder'], user_kp['right_elbow'], user_kp['right_wrist']
        )
        target_angle = _calculate_angle(
            ref_kp['right_shoulder'], ref_kp['right_elbow'], ref_kp['right_wrist']
        )

        if angle_diffs['right_elbow'] > 0:
            feedback.append(
                f"오른팔 팔꿈치를 {abs(angle_diffs['right_elbow']):.0f}° 더 펴세요 "
                f"(현재 {current_angle:.0f}°, 목표 {target_angle:.0f}°)"
            )
        else:
            feedback.append(
                f"오른팔 팔꿈치를 {abs(angle_diffs['right_elbow']):.0f}° 더 구부리세요 "
                f"(현재 {current_angle:.0f}°, 목표 {target_angle:.0f}°)"
            )

    if 'left_shoulder' in angle_diffs and abs(angle_diffs['left_shoulder']) > 30:  # 20 -> 30
        current_angle = _calculate_angle(
            user_kp['left_hip'], user_kp['left_shoulder'], user_kp['left_elbow']
        )
        target_angle = _calculate_angle(
            ref_kp['left_hip'], ref_kp['left_shoulder'], ref_kp['left_elbow']
        )

        if angle_diffs['left_shoulder'] > 0:
            feedback.append(
                f"왼팔을 {abs(angle_diffs['left_shoulder']):.0f}° 더 올리세요 "
                f"(현재 {current_angle:.0f}°, 목표 {target_angle:.0f}°)"
            )
        else:
            feedback.append(
                f"왼팔을 {abs(angle_diffs['left_shoulder']):.0f}° 더 내리세요 "
                f"(현재 {current_angle:.0f}°, 목표 {target_angle:.0f}°)"
            )

    if 'right_shoulder' in angle_diffs and abs(angle_diffs['right_shoulder']) > 30:  # 20 -> 30
        current_angle = _calculate_angle(
            user_kp['right_hip'], user_kp['right_shoulder'], user_kp['right_elbow']
        )
        target_angle = _calculate_angle(
            ref_kp['right_hip'], ref_kp['right_shoulder'], ref_kp['right_elbow']
        )

        if angle_diffs['right_shoulder'] > 0:
            feedback.append(
                f"오른팔을 {abs(angle_diffs['right_shoulder']):.0f}° 더 올리세요 "
                f"(현재 {current_angle:.0f}°, 목표 {target_angle:.0f}°)"
            )
        else:
            feedback.append(
                f"오른팔을 {abs(angle_diffs['right_shoulder']):.0f}° 더 내리세요 "
                f"(현재 {current_angle:.0f}°, 목표 {target_angle:.0f}°)"
            )

    # 얼굴 각도
    if 'face_angle' in angle_diffs and abs(angle_diffs['face_angle']) > 5:
        if angle_diffs['face_angle'] > 0:
            feedback.append(f"얼굴을 {abs(angle_diffs['face_angle']):.0f}도 더 왼쪽으로 돌리세요")
        else:
            feedback.append(f"얼굴을 {abs(angle_diffs['face_angle']):.0f}도 더 오른쪽으로 돌리세요")

    # 위치 피드백
    if 'left_wrist_y' in position_diffs and abs(position_diffs['left_wrist_y']) > 0.1:
        if position_diffs['left_wrist_y'] > 0:
            feedback.append(f"왼손을 화면 기준 위쪽으로 이동하세요")
        else:
            feedback.append(f"왼손을 화면 기준 아래쪽으로 이동하세요")

    if 'right_wrist_y' in position_diffs and abs(position_diffs['right_wrist_y']) > 0.1:
        if position_diffs['right_wrist_y'] > 0:
            feedback.append(f"오른손을 화면 기준 위쪽으로 이동하세요")
        else:
            feedback.append(f"오른손을 화면 기준 아래쪽으로 이동하세요")

    if 'head_tilt' in position_diffs and abs(position_diffs['head_tilt']) > 0.05:
        if position_diffs['head_tilt'] > 0:
            feedback.append("고개를 왼쪽으로 기울이세요")
        else:
            feedback.append("고개를 오른쪽으로 기울이세요")

    if 'nose_y' in position_diffs and abs(position_diffs['nose_y']) > 0.08:
        if position_diffs['nose_y'] > 0:
            feedback.append("고개를 위로 들어 올리세요")
        else:
            feedback.append("고개를 아래로 숙이세요")

    if 'shoulder_width' in position_diffs and abs(position_diffs['shoulder_width']) > 0.1:
        if position_diffs['shoulder_width'] > 0:
            feedback.append("어깨를 좁혀주세요 (정면을 더 향하세요)")
        else:
            feedback.append("어깨를 펴주세요 (측면을 더 향하세요)")

    if not feedback:
        feedback.append("✅ 포즈가 적절합니다")

    return feedback


# ============================================================
# 테스트
# ============================================================
if __name__ == "__main__":
    test_img = PROJECT_ROOT / "data" / "test_images" / "test1.jpg"
    
    try:
        analyzer = PoseAnalyzer()
        result = analyzer.analyze(str(test_img))

        print("\n" + "="*60)
        print("🤸 POSE ANALYSIS RESULT")
        print("="*60)

        print(f"\n📋 Scenario: {result['scenario']}")
        print(f"🎯 Confidence: {result['confidence']:.2f}")
        print(f"📦 BBox: {result['bbox']}")

        print(f"\n🦴 YOLO Keypoints (17):")
        for kp in result['yolo_keypoints'][:5]:  # 상위 5개만
            print(f"  - {kp['name']}: ({kp['x']:.3f}, {kp['y']:.3f}) conf={kp['confidence']:.2f}")
        print("  ...")

        if 'mediapipe_face' in result:
            print(f"\n😊 MediaPipe Face: {result['mediapipe_face']['count']} keypoints")

        if 'mediapipe_hands' in result:
            print(f"\n🤚 MediaPipe Hands: {result['mediapipe_hands']['hand_count']} hands")

        if 'mediapipe_pose' in result:
            print(f"\n🧍 MediaPipe Pose: {result['mediapipe_pose']['count']} keypoints")

        print("\n" + "="*60)

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
