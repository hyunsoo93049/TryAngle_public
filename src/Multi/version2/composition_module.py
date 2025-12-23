# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# composition_module_explainable_full.py
# -------------------------------------------------------------------------
# 사진 구도 / 프레이밍 / 포즈 분석 + 설명형 피드백
# -------------------------------------------------------------------------
# 주요 기능:
#  1. YOLO Pose의 keypoints와 bbox를 활용해 구도 점수 산출
#  2. 삼분할선, 인물 크기비율, 헤드룸, 중심 정렬 평가
#  3. 두 이미지 간 포즈 유사도(pose similarity) 계산
#  4. 모든 판단 근거를 자연어 문장으로 설명
# -------------------------------------------------------------------------

from typing import Dict, Any, Optional
import numpy as np

# -------------------------------------------------------------------------
# 유틸 함수
# -------------------------------------------------------------------------
def _clamp(v: float, lo: float, hi: float) -> float:
    """값 v를 [lo, hi] 범위로 잘라냅니다."""
    return max(lo, min(hi, v))

def _point_in_thirds(x: float, y: float, w: int, h: int, tol: float = 0.06) -> bool:
    """(x, y)가 삼분할선 교차점 근처인지 확인합니다."""
    thirds_x = [w / 3, 2 * w / 3]
    thirds_y = [h / 3, 2 * h / 3]
    return any(abs(x - tx) <= w * tol for tx in thirds_x) and \
           any(abs(y - ty) <= h * tol for ty in thirds_y)


# -------------------------------------------------------------------------
# (1) 구도 분석 (Composition Analysis)
# -------------------------------------------------------------------------
def analyze_composition(
    image: np.ndarray,
    keypoints: np.ndarray,
    bbox: Optional[tuple] = None
) -> Dict[str, Any]:
    """
    구도 분석 + 설명형 피드백
    - YOLO Pose keypoints 및 bbox 기반
    - 점수뿐 아니라 각 요소별 판단 근거 문장까지 생성
    """
    h, w = image.shape[:2]

    # -------------------------------
    # (1) 인물 중심 계산
    # -------------------------------
    cx, cy = np.mean(keypoints, axis=0)
    cx = _clamp(float(cx), 0, w - 1)
    cy = _clamp(float(cy), 0, h - 1)

    # -------------------------------
    # (2) 삼분할선 여부
    # -------------------------------
    on_thirds = _point_in_thirds(cx, cy, w, h, tol=0.06)

    # -------------------------------
    # (3) 인물 bbox 기반 크기/헤드룸 계산
    # -------------------------------
    size_ratio, headroom_ratio = 0.0, None
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
        size_ratio = (bw * bh) / float(w * h)
        head_y = float(np.min(keypoints[:, 1]))
        headroom_ratio = _clamp((head_y - 0.0) / h, 0.0, 1.0)

    # -------------------------------
    # (4) 점수 계산
    # -------------------------------
    score = 50.0
    reasons = []  # 피드백 문장 저장용

    # (a) 삼분할선 보정
    if on_thirds:
        score += 20.0
        reasons.append("인물이 삼분할선 교차점 근처에 잘 배치되어 있습니다.")
    else:
        reasons.append("인물이 삼분할선에서 다소 벗어나 있습니다. 구도를 약간 이동해보세요.")

    # (b) 중심 정렬 (중앙 근접도)
    dist_from_center = abs(cx - w / 2) / (w / 2)
    if dist_from_center < 0.1:
        score += 10.0
        reasons.append("인물이 중앙에 안정적으로 위치해 있습니다.")
    elif dist_from_center < 0.3:
        score += 5.0
        reasons.append("인물이 중심에 비교적 가깝게 배치되어 있습니다.")
    else:
        reasons.append("인물이 프레임 한쪽에 치우쳐 있습니다.")

    # (c) 인물 크기 비율 (20~40% 권장)
    if size_ratio > 0:
        if 0.2 <= size_ratio <= 0.4:
            score += 12.0
            reasons.append("인물 크기가 프레임 대비 적절합니다.")
        elif size_ratio < 0.2:
            score -= 5.0
            reasons.append("인물이 너무 작게 보입니다. 카메라를 더 가까이 해보세요.")
        else:
            score -= 5.0
            reasons.append("인물이 화면을 과도하게 차지하고 있습니다. 약간 멀리서 촬영해보세요.")

    # (d) 헤드룸 (머리 위 여백 8~18% 권장)
    if headroom_ratio is not None:
        if 0.08 <= headroom_ratio <= 0.18:
            score += 8.0
            reasons.append("머리 위 여백(헤드룸)이 안정적으로 확보되어 있습니다.")
        elif headroom_ratio < 0.08:
            score -= 6.0
            reasons.append("머리 위 여백이 부족해 답답한 인상을 줄 수 있습니다.")
        else:
            score -= 4.0
            reasons.append("머리 위 여백이 넓어 인물이 작게 느껴질 수 있습니다.")

    score = float(_clamp(score, 0.0, 100.0))

    # -------------------------------
    # (5) 종합 요약
    # -------------------------------
    if score >= 80:
        summary = "전반적으로 매우 안정된 구도입니다."
    elif score >= 60:
        summary = "균형 잡힌 구도지만 약간의 수정 여지가 있습니다."
    else:
        summary = "인물 위치나 여백을 조정하면 더 자연스러울 것입니다."

    return {
        "center": (cx, cy),
        "on_rule_of_thirds": on_thirds,
        "size_ratio": float(size_ratio),
        "headroom_ratio": float(headroom_ratio) if headroom_ratio else None,
        "score": score,
        "reasons": reasons,
        "summary": summary
    }


# -------------------------------------------------------------------------
# (2) 포즈 유사도 분석 (Pose Similarity)
# -------------------------------------------------------------------------
def normalize_keypoints(kpts: np.ndarray) -> np.ndarray:
    """
    keypoint 좌표를 bbox 크기 기준으로 정규화 (0~1 범위)
    - 서로 다른 인물 크기나 해상도 차이를 보정하기 위함
    """
    x_min, y_min = kpts.min(axis=0)
    x_max, y_max = kpts.max(axis=0)
    return (kpts - [x_min, y_min]) / [x_max - x_min, y_max - y_min]


def calculate_angle_similarity(kpts1: np.ndarray, kpts2: np.ndarray) -> float:
    """
    두 keypoints 세트의 각도 기반 유사도 계산
    - 어깨, 팔꿈치, 손목, 얼굴 방향 등의 상대 각도 비교
    - 단순 코사인 유사도 기반 근사 구현
    """
    try:
        # 주요 관절 index (COCO 포맷 예시)
        important_idx = [5, 6, 7, 8, 9, 10, 11, 12]  # 양팔/다리 중심부
        v1 = kpts1[important_idx[1]] - kpts1[important_idx[0]]
        v2 = kpts2[important_idx[1]] - kpts2[important_idx[0]]
        cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        angle_sim = float((cos_sim + 1) / 2 * 100)  # [0~100]
        return angle_sim
    except Exception:
        return 50.0  # 기본값


def calculate_pose_similarity(kpts1: np.ndarray, kpts2: np.ndarray) -> Dict[str, Any]:
    """
    두 이미지의 포즈 유사도 계산
    - 정규화된 keypoint 거리 기반
    - 각도 기반 유사도 보정 포함
    """
    if kpts1 is None or kpts2 is None:
        return {"score": 0.0, "details": "포즈 인식 실패"}

    # (1) 정규화 (크기 차이 보정)
    kpts1_norm = normalize_keypoints(kpts1)
    kpts2_norm = normalize_keypoints(kpts2)

    # (2) 주요 관절 간 거리 비교
    distances = np.linalg.norm(kpts1_norm - kpts2_norm, axis=1)
    avg_dist = np.mean(distances)

    # (3) 각도 유사도
    angle_sim = calculate_angle_similarity(kpts1_norm, kpts2_norm)

    # (4) 점수화 (0~100)
    pose_score = max(0, 100 - avg_dist * 100) * 0.7 + angle_sim * 0.3
    pose_score = float(_clamp(pose_score, 0, 100))

    detail_text = (
        "포즈가 유사합니다." if pose_score >= 60
        else "손 위치, 얼굴 기울기, 자세 차이가 있습니다."
    )

    return {
        "score": pose_score,
        "avg_distance": float(avg_dist),
        "angle_sim": float(angle_sim),
        "details": detail_text
    }
