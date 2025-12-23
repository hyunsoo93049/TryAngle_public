# -*- coding: utf-8 -*-
# feedback_module.py
# -----------------------------------------------------------------------------
# 우선순위 기반 피드백 생성기 (v3.0)
# - 가장 중요한 차이점 1~3개만 강조
# - 중복 제거 및 명확한 행동 가이드
# - 포즈 > 시점 > 구도 > 감성 순서로 우선순위 적용
# -----------------------------------------------------------------------------

from typing import List, Optional, Dict, Any, Tuple
import math

# 권장 타깃
T_SIZE = 0.28
T_HEAD = 0.12
SIZE_OK_RANGE = (0.20, 0.40)
HEAD_OK_RANGE = (0.08, 0.18)

# 우선순위 레벨 정의
PRIORITY_CRITICAL = 1   # 포즈 불일치 (가장 중요)
PRIORITY_HIGH = 2       # 시점/여백 (즉시 수정 필요)
PRIORITY_MEDIUM = 3     # 프레이밍/구도
PRIORITY_LOW = 4        # 감성/색감 (미세 조정)
PRIORITY_INFO = 5       # 정보성 메시지


class FeedbackItem:
    """개별 피드백 항목"""
    def __init__(self, priority: int, category: str, message: str, detail: str = ""):
        self.priority = priority
        self.category = category  # "포즈", "구도", "감성" 등
        self.message = message
        self.detail = detail
    
    def __lt__(self, other):
        return self.priority < other.priority


def _calculate_camera_adjustment(height_diff: float) -> Tuple[str, int]:
    """MiDaS 시점 차이를 cm 단위 조정값으로 변환"""
    # height_diff: -1 (로우앵글) ~ +1 (하이앵글)
    cm = int(abs(height_diff) * 30)  # 최대 30cm
    cm = max(5, min(cm, 25))  # 5~25cm 범위로 제한
    
    if height_diff < -0.12:
        return "올리세요", cm
    elif height_diff > 0.12:
        return "내리세요", cm
    else:
        return "유사", 0


def _calculate_zoom_suggestion(size_ratio: float) -> Optional[str]:
    """인물 크기 비율 기반 줌 제안"""
    if size_ratio < SIZE_OK_RANGE[0]:
        zoom_factor = math.sqrt(T_SIZE / max(1e-6, size_ratio))
        zoom_factor = min(max(1.10, zoom_factor), 1.25)
        if zoom_factor >= 1.05:
            zoom_percent = int((zoom_factor - 1.0) * 100)
            return f"인물이 작아요. 약 {zoom_percent}% 더 당기거나 15~20cm 가까이 가세요."
    
    elif size_ratio > SIZE_OK_RANGE[1]:
        return "인물이 화면을 너무 많이 차지합니다. 10~15cm 뒤로 물러나세요."
    
    return None


def _calculate_headroom_suggestion(headroom_ratio: float) -> Optional[str]:
    """머리 위 여백 기반 수직 조정 제안"""
    if headroom_ratio > HEAD_OK_RANGE[1]:
        excess = int((headroom_ratio - T_HEAD) * 100)
        if excess > 3:
            return f"상단 여백이 {excess}% 많습니다. 카메라를 약간 올리거나 프레임을 위로 이동하세요."
    
    elif headroom_ratio < HEAD_OK_RANGE[0]:
        lack = int((T_HEAD - headroom_ratio) * 100)
        if lack > 3:
            return f"머리 위 여백이 {lack}% 부족합니다. 카메라를 약간 내리세요."
    
    return None


def generate_feedback(
    pose_conf: Optional[float] = None,
    composition_score: Optional[float] = None,
    emotion_score: Optional[float] = None,
    reasons: Optional[List[str]] = None,
    summary: Optional[str] = None,
    extras: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """
    우선순위 기반 피드백 생성
    - 최대 3개의 핵심 피드백만 표시
    - 중복 제거 및 명확한 행동 지침
    """
    
    feedback_items: List[FeedbackItem] = []
    
    # ===== 입력 정리 =====
    ext = extras or {}
    size_ratio = ext.get("size_ratio")
    headroom_ratio = ext.get("headroom_ratio")
    dino_sim = ext.get("dino_sim")
    height_diff = ext.get("height_diff")
    color_sim = ext.get("color_sim")
    emotion_factors = ext.get("emotion_factors", {})
    pose_similarity = ext.get("pose_similarity")  # 새로 추가될 포즈 유사도
    
    comp = composition_score if composition_score is not None else 0.0
    emo = emotion_score if emotion_score is not None else 0.0
    
    # ===== 1. 포즈 유사도 (최우선) =====
    if pose_similarity is not None:
        pose_score = pose_similarity.get("score", 0.0)
        
        if pose_score < 50:
            feedback_items.append(FeedbackItem(
                priority=PRIORITY_CRITICAL,
                category="포즈",
                message="포즈가 레퍼런스와 많이 다릅니다.",
                detail=f"손 위치, 얼굴 각도, 몸 기울기를 먼저 맞춰보세요. (유사도: {pose_score:.0f}%)"
            ))
        elif pose_score < 70:
            feedback_items.append(FeedbackItem(
                priority=PRIORITY_HIGH,
                category="포즈",
                message="포즈를 조금 더 조정해보세요.",
                detail=f"손 위치나 고개 각도를 미세 조정하면 더 비슷해질 거예요. (유사도: {pose_score:.0f}%)"
            ))
        else:
            feedback_items.append(FeedbackItem(
                priority=PRIORITY_INFO,
                category="포즈",
                message=f"포즈가 잘 맞습니다! (유사도: {pose_score:.0f}%)",
                detail=""
            ))
    
    # ===== 2. 시점 (카메라 높이) =====
    if height_diff is not None:
        direction, cm = _calculate_camera_adjustment(height_diff)
        
        if cm > 0:
            feedback_items.append(FeedbackItem(
                priority=PRIORITY_HIGH,
                category="시점",
                message=f"카메라를 {cm}cm {direction}",
                detail=f"({'로우앵글' if direction == '올리세요' else '하이앵글'} 경향)"
            ))
    
    # ===== 3. 구도 - 인물 크기 =====
    if size_ratio is not None:
        zoom_msg = _calculate_zoom_suggestion(size_ratio)
        if zoom_msg:
            feedback_items.append(FeedbackItem(
                priority=PRIORITY_MEDIUM,
                category="구도",
                message=zoom_msg,
                detail=""
            ))
    
    # ===== 4. 구도 - 머리 여백 =====
    # (줌과 여백은 중복 가능성이 있으므로 하나만 선택)
    if headroom_ratio is not None and size_ratio is not None:
        # 줌 문제가 없을 때만 여백 체크
        if SIZE_OK_RANGE[0] <= size_ratio <= SIZE_OK_RANGE[1]:
            headroom_msg = _calculate_headroom_suggestion(headroom_ratio)
            if headroom_msg:
                feedback_items.append(FeedbackItem(
                    priority=PRIORITY_MEDIUM,
                    category="구도",
                    message=headroom_msg,
                    detail=""
                ))
    
    # ===== 5. 프레이밍 (DINO) =====
    if dino_sim is not None:
        if dino_sim < 0.40:
            feedback_items.append(FeedbackItem(
                priority=PRIORITY_MEDIUM,
                category="구도",
                message="인물 위치가 레퍼런스와 많이 다릅니다.",
                detail="화면 내 인물 배치를 삼분할선 기준으로 조정해보세요."
            ))
        elif dino_sim < 0.65:
            feedback_items.append(FeedbackItem(
                priority=PRIORITY_LOW,
                category="구도",
                message="프레이밍이 거의 비슷합니다.",
                detail="소폭 위치 조정으로 더 가까워질 수 있어요."
            ))
    
    # ===== 6. 감성 - 색온도 =====
    if emotion_factors:
        color_temp = emotion_factors.get("color_temperature")
        if color_temp is not None and color_temp < 0.65:
            ref_warm = emotion_factors.get("ref_warm", True)
            feedback_items.append(FeedbackItem(
                priority=PRIORITY_LOW,
                category="감성",
                message=f"색온도가 다릅니다 (레퍼런스: {'따뜻' if ref_warm else '차가움'})",
                detail="화이트밸런스를 조정하거나 후보정으로 색감을 맞춰보세요."
            ))
    
    # ===== 7. 감성 - 조명 방향 =====
    if emotion_factors:
        lighting = emotion_factors.get("lighting_direction")
        if lighting is not None and lighting < 0.55:
            feedback_items.append(FeedbackItem(
                priority=PRIORITY_LOW,
                category="감성",
                message="조명 방향이 다릅니다.",
                detail="창가 또는 조명 위치를 조정해보세요."
            ))
    
    # ===== 8. 감성 - 전체 유사도 =====
    if emotion_score is not None and emotion_score < 60:
        if not emotion_factors:  # 세부 분석이 없을 때만 일반 메시지
            feedback_items.append(FeedbackItem(
                priority=PRIORITY_LOW,
                category="감성",
                message="분위기가 레퍼런스와 조금 다릅니다.",
                detail="조명 톤이나 색감을 조정해보세요."
            ))
    
    # ===== 9. 긍정 피드백 (감성 예외 처리) =====
    if comp < 60 and emo >= 75:
        feedback_items.append(FeedbackItem(
            priority=PRIORITY_INFO,
            category="종합",
            message="감성적으로는 매력적입니다!",
            detail="전통적 구도와 다르지만 의도된 느낌이라면 지금이 좋아요."
        ))
    elif comp >= 75 and emo >= 70:
        feedback_items.append(FeedbackItem(
            priority=PRIORITY_INFO,
            category="종합",
            message="구도와 감성 모두 훌륭합니다!",
            detail="레퍼런스와 매우 유사한 사진입니다."
        ))
    
    # ===== 우선순위 정렬 및 필터링 =====
    feedback_items.sort()
    
    # 최대 3개의 실질적 피드백 (INFO 제외)
    action_items = [f for f in feedback_items if f.priority < PRIORITY_INFO][:3]
    info_items = [f for f in feedback_items if f.priority == PRIORITY_INFO][:1]
    
    final_items = action_items + info_items
    
    # ===== 출력 포맷 =====
    if not final_items:
        return ["✅ 완벽합니다! 레퍼런스와 매우 유사한 구도입니다."]
    
    messages: List[str] = []
    
    # 카테고리별 그룹화
    categories = {}
    for item in final_items:
        if item.category not in categories:
            categories[item.category] = []
        categories[item.category].append(item)
    
    # 카테고리별 출력
    category_icons = {
        "포즈": "🕺",
        "시점": "📐",
        "구도": "📸",
        "감성": "💫",
        "종합": "✨"
    }
    
    for category, items in categories.items():
        icon = category_icons.get(category, "📌")
        messages.append(f"\n{icon} [{category}]")
        
        for item in items:
            if item.detail:
                messages.append(f" - {item.message}")
                messages.append(f"   → {item.detail}")
            else:
                messages.append(f" - {item.message}")
    
    # 요약 추가
    if summary:
        messages.append(f"\n🧾 [분석 요약]")
        messages.append(f" - {summary}")
    
    return messages


# ===== 하위 호환성 유지 =====
def _matrix_tone(composition_score: Optional[float], emotion_score: Optional[float]) -> str:
    """구도 × 감성 매트릭스 톤 (기존 호환)"""
    comp = composition_score if composition_score is not None else 0.0
    emo = emotion_score if emotion_score is not None else 0.0
    
    if comp < 60 and emo >= 75:
        return "감성적으로는 매력적입니다."
    if comp >= 80 and emo < 55:
        return "구도는 안정적이지만 감정 표현이 약해 보여요."
    if comp < 60 and emo < 55:
        return "구도와 감성 모두 개선 여지가 있어요."
    return "전반적으로 균형이 나쁘지 않아요."