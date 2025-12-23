import Foundation
import CoreGraphics

// MARK: - 피드백 단계 (Phase 3)

/// 6단계 피드백 시스템
enum FeedbackStage: Int, CaseIterable {
    case aspectRatio = 0    // 0단계: 비율 (4:3 vs 16:9)
    case shotType = 1       // 1단계: 샷 타입 (전신 vs 상반신 vs 얼굴)
    case coverage = 2       // 2단계: 점유율 (tight vs loose, 줌/거리)
    case position = 3       // 3단계: 인물 위치 (좌/중앙/우)
    case framing = 4        // 4단계: 프레이밍 디테일 (헤드룸, 리드룸)
    case pose = 5           // 5단계: 포즈 (팔 각도, 손가락)
    case complete = 99      // 완벽!

    var displayName: String {
        switch self {
        case .aspectRatio: return "비율"
        case .shotType: return "샷 타입"
        case .coverage: return "점유율"
        case .position: return "인물 위치"
        case .framing: return "프레이밍"
        case .pose: return "포즈"
        case .complete: return "완벽"
        }
    }

    var description: String {
        switch self {
        case .aspectRatio: return "카메라 비율을 맞추세요"
        case .shotType: return "전신/상반신/얼굴 구도를 맞추세요"
        case .coverage: return "프레임 내 점유율을 조정하세요"
        case .position: return "인물의 좌우/상하 위치를 맞추세요"
        case .framing: return "머리 위 공간, 시선 방향 여백을 조정하세요"
        case .pose: return "신체 포즈를 맞추세요"
        case .complete: return "완벽합니다!"
        }
    }
}

// MARK: - 단계별 피드백 생성기 (Phase 3)

/// 6단계 우선순위 기반 피드백 생성기
class StagedFeedbackGenerator {

    // MARK: - 임계값 정의

    // 샷 타입
    private let shotTypeDiffThreshold_Major: Int = 2    // 2단계 이상 차이
    private let shotTypeDiffThreshold_Minor: Int = 1    // 1단계 차이

    // 점유율
    private let coverageDiffThreshold: CGFloat = 0.15   // 15% 차이

    // 위치
    private let positionDiffThreshold_Major: CGFloat = 0.15  // 15% 차이
    private let positionDiffThreshold_Minor: CGFloat = 0.05  // 5% 차이

    // 프레이밍
    private let headroomDiffThreshold: CGFloat = 0.05   // 5% 차이
    private let leadRoomDiffThreshold: CGFloat = 0.10   // 10% 차이

    // 포즈
    private let angleDiffThreshold: Float = 15.0        // 15도

    // MARK: - 단계 결정 로직

    /// 현재 프레임의 피드백 단계 결정
    /// - Parameters:
    ///   - referenceFraming: 레퍼런스 프레이밍 분석 결과
    ///   - currentFraming: 현재 프레이밍 분석 결과
    ///   - referenceAspectRatio: 레퍼런스 비율
    ///   - currentAspectRatio: 현재 비율
    ///   - poseComparison: 포즈 비교 결과 (옵션)
    /// - Returns: 피드백 단계
    func determineFeedbackStage(
        referenceFraming: PhotographyFramingResult?,
        currentFraming: PhotographyFramingResult?,
        referenceAspectRatio: CameraAspectRatio,
        currentAspectRatio: CameraAspectRatio,
        poseComparison: PoseComparisonResult?
    ) -> FeedbackStage {

        // 0단계: 비율 체크
        if referenceAspectRatio != currentAspectRatio {
            return .aspectRatio
        }

        guard let refFraming = referenceFraming,
              let curFraming = currentFraming else {
            // 프레이밍 정보가 없으면 포즈만 비교
            return .pose
        }

        // 1단계: 샷 타입 차이
        let shotTypeDiff = shotTypeDistance(from: refFraming.shotType, to: curFraming.shotType)

        if shotTypeDiff >= shotTypeDiffThreshold_Major {
            return .shotType
        }

        // 2단계: 점유율 (샷 타입이 비슷할 때만)
        if shotTypeDiff <= shotTypeDiffThreshold_Minor {
            let coverageDiff = abs(refFraming.bodyCoverage - curFraming.bodyCoverage)
            if coverageDiff > coverageDiffThreshold {
                return .coverage
            }
        }

        // 3단계: 인물 위치 (샷 타입이 정확히 맞을 때만)
        if shotTypeDiff == 0 {
            // 얼굴(코) 위치로 비교
            let positionDiff = calculatePositionDifference(
                reference: refFraming,
                current: curFraming
            )

            if positionDiff > positionDiffThreshold_Major {
                return .position
            }

            // 4단계: 프레이밍 디테일 (위치가 대략 맞을 때만)
            if positionDiff <= positionDiffThreshold_Minor {
                let headroomDiff = abs(refFraming.headroom - curFraming.headroom)

                // leadRoom은 옵셔널이므로 안전하게 처리
                var leadRoomDiffExceedsThreshold = false
                if let refLead = refFraming.leadRoom, let curLead = curFraming.leadRoom {
                    leadRoomDiffExceedsThreshold = abs(refLead - curLead) > leadRoomDiffThreshold
                }

                if headroomDiff > headroomDiffThreshold || leadRoomDiffExceedsThreshold {
                    return .framing
                }

                // 5단계: 포즈 (프레이밍이 모두 맞을 때)
                if let poseResult = poseComparison {
                    // 각도 차이가 임계값을 넘으면 포즈 단계
                    let hasAngleDifference = poseResult.angleDifferences.values.contains { $0 > angleDiffThreshold }
                    if hasAngleDifference {
                        return .pose
                    }
                }

                // 모두 완벽!
                return .complete
            }
        }

        // 기본값: 포즈
        return .pose
    }

    // MARK: - 단계별 피드백 생성

    /// 단계에 맞는 피드백 메시지 생성
    /// - Parameters:
    ///   - stage: 피드백 단계
    ///   - referenceFraming: 레퍼런스 프레이밍
    ///   - currentFraming: 현재 프레이밍
    ///   - referenceAspectRatio: 레퍼런스 비율
    ///   - currentAspectRatio: 현재 비율
    ///   - poseComparison: 포즈 비교 결과
    ///   - croppedGroups: 잘린 그룹들
    ///   - isFrontCamera: 전면 카메라 여부
    /// - Returns: 피드백 아이템 배열
    func generateStagedFeedback(
        stage: FeedbackStage,
        referenceFraming: PhotographyFramingResult?,
        currentFraming: PhotographyFramingResult?,
        referenceAspectRatio: CameraAspectRatio,
        currentAspectRatio: CameraAspectRatio,
        poseComparison: PoseComparisonResult?,
        croppedGroups: [KeypointGroup],
        isFrontCamera: Bool
    ) -> [FeedbackItem] {

        switch stage {
        case .aspectRatio:
            return generateAspectRatioFeedback(
                reference: referenceAspectRatio,
                current: currentAspectRatio
            )

        case .shotType:
            return generateShotTypeFeedback(
                reference: referenceFraming,
                current: currentFraming,
                isFrontCamera: isFrontCamera
            )

        case .coverage:
            return generateCoverageFeedback(
                reference: referenceFraming,
                current: currentFraming,
                isFrontCamera: isFrontCamera
            )

        case .position:
            return generatePositionFeedback(
                reference: referenceFraming,
                current: currentFraming,
                isFrontCamera: isFrontCamera
            )

        case .framing:
            return generateFramingDetailFeedback(
                reference: referenceFraming,
                current: currentFraming,
                isFrontCamera: isFrontCamera
            )

        case .pose:
            var feedbacks: [FeedbackItem] = []

            // 잘림 피드백 우선
            if !croppedGroups.isEmpty {
                if let croppingFeedback = generateCroppingFeedback(croppedGroups: croppedGroups, isFrontCamera: isFrontCamera) {
                    feedbacks.append(croppingFeedback)
                }
            }

            // 포즈 피드백
            if let poseFeedback = generatePoseFeedback(poseComparison: poseComparison) {
                feedbacks.append(contentsOf: poseFeedback)
            }

            return feedbacks

        case .complete:
            return []  // 완벽한 상태, 피드백 없음
        }
    }

    // MARK: - Helper Functions

    /// 샷 타입 간 거리 계산 (0~7)
    private func shotTypeDistance(from ref: ShotType, to current: ShotType) -> Int {
        let levels: [ShotType] = [
            .extremeCloseUp,    // 0
            .closeUp,           // 1
            .mediumCloseUp,     // 2
            .mediumShot,        // 3
            .americanShot,      // 4
            .mediumFullShot,    // 5
            .fullShot,          // 6
            .longShot           // 7
        ]

        guard let refIndex = levels.firstIndex(of: ref),
              let curIndex = levels.firstIndex(of: current) else {
            return 0
        }

        return abs(refIndex - curIndex)
    }

    /// 인물 위치 차이 계산 (정규화된 거리)
    private func calculatePositionDifference(
        reference: PhotographyFramingResult,
        current: PhotographyFramingResult
    ) -> CGFloat {
        // 코(nose) 위치로 비교
        let refNose = reference.nosePosition
        let curNose = current.nosePosition

        let dx = refNose.x - curNose.x
        let dy = refNose.y - curNose.y

        return sqrt(dx * dx + dy * dy)
    }

    // MARK: - 각 단계별 피드백 생성 함수들

    private func generateAspectRatioFeedback(
        reference: CameraAspectRatio,
        current: CameraAspectRatio
    ) -> [FeedbackItem] {
        return [FeedbackItem(
            priority: -1,
            icon: "📐",
            message: "카메라 비율을 \(reference.displayName)로 변경하세요",
            category: "aspect_ratio",
            currentValue: nil,
            targetValue: nil,
            tolerance: nil,
            unit: nil
        )]
    }

    private func generateShotTypeFeedback(
        reference: PhotographyFramingResult?,
        current: PhotographyFramingResult?,
        isFrontCamera: Bool
    ) -> [FeedbackItem] {
        guard let ref = reference, let cur = current else { return [] }

        let message: String

        // 전면 카메라 (셀카): 사람이 움직임
        // 후면 카메라: 카메라가 움직임
        if ref.shotType == .fullShot && cur.shotType == .mediumShot {
            message = isFrontCamera ? "뒤로 물러나세요 (전신이 보이게)" : "카메라를 뒤로 멀리하세요 (전신이 보이게)"
        } else if ref.shotType == .fullShot && cur.shotType == .closeUp {
            message = isFrontCamera ? "뒤로 물러나세요 (전신이 보이게)" : "카메라를 뒤로 멀리하세요 (전신이 보이게)"
        } else if ref.shotType == .mediumShot && cur.shotType == .closeUp {
            message = isFrontCamera ? "뒤로 물러나세요 (상반신이 보이게)" : "카메라를 뒤로 멀리하세요 (상반신이 보이게)"
        } else if ref.shotType == .mediumShot && cur.shotType == .fullShot {
            message = isFrontCamera ? "가까이 다가오세요 (상반신만)" : "카메라를 가까이 당기세요 (상반신만)"
        } else if ref.shotType == .closeUp && cur.shotType == .fullShot {
            message = isFrontCamera ? "가까이 다가오세요 (얼굴 중심)" : "카메라를 가까이 당기세요 (얼굴 중심)"
        } else {
            message = isFrontCamera ? "거리를 조정하세요 (\(ref.shotType.userFriendlyDescription))" : "카메라 거리를 조정하세요 (\(ref.shotType.userFriendlyDescription))"
        }

        return [FeedbackItem(
            priority: 0,
            icon: "📸",
            message: message,
            category: "shot_type",
            currentValue: nil,
            targetValue: nil,
            tolerance: nil,
            unit: nil
        )]
    }

    private func generateCoverageFeedback(
        reference: PhotographyFramingResult?,
        current: PhotographyFramingResult?,
        isFrontCamera: Bool
    ) -> [FeedbackItem] {
        guard let ref = reference, let cur = current else { return [] }

        let coverageDiff = cur.bodyCoverage - ref.bodyCoverage

        let message: String
        if coverageDiff > 0 {
            // 현재가 더 꽉 참 → 전면: 사람 뒤로, 후면: 카메라 뒤로
            message = isFrontCamera ? "뒤로 물러나세요" : "카메라를 뒤로 멀리하세요"
        } else {
            // 현재가 더 여유 있음 → 전면: 사람 앞으로, 후면: 카메라 앞으로
            message = isFrontCamera ? "가까이 다가오세요" : "카메라를 가까이 당기세요"
        }

        return [FeedbackItem(
            priority: 0,
            icon: "🔍",
            message: message,
            category: "coverage",
            currentValue: Double(cur.bodyCoverage * 100),
            targetValue: Double(ref.bodyCoverage * 100),
            tolerance: 5.0,
            unit: "%"
        )]
    }

    private func generatePositionFeedback(
        reference: PhotographyFramingResult?,
        current: PhotographyFramingResult?,
        isFrontCamera: Bool
    ) -> [FeedbackItem] {
        guard let ref = reference, let cur = current else { return [] }

        var feedbacks: [FeedbackItem] = []

        // 좌우 위치 - 사람이 이동
        let xDiff = cur.nosePosition.x - ref.nosePosition.x
        if abs(xDiff) > positionDiffThreshold_Minor {
            var message: String
            if isFrontCamera {
                // 전면 카메라는 좌우 반전
                message = xDiff > 0 ? "왼쪽으로 이동하세요" : "오른쪽으로 이동하세요"
            } else {
                message = xDiff > 0 ? "오른쪽으로 이동하세요" : "왼쪽으로 이동하세요"
            }

            feedbacks.append(FeedbackItem(
                priority: 1,
                icon: "↔️",
                message: message,
                category: "position_x",
                currentValue: Double(cur.nosePosition.x * 100),
                targetValue: Double(ref.nosePosition.x * 100),
                tolerance: 5.0,
                unit: "%"
            ))
        }

        // 상하 위치 - 카메라를 조작
        // yDiff > 0: 현재 인물이 아래쪽에 있음 → 카메라를 아래로 내려서 인물을 위로
        // yDiff < 0: 현재 인물이 위쪽에 있음 → 카메라를 위로 올려서 인물을 아래로
        let yDiff = cur.nosePosition.y - ref.nosePosition.y
        if abs(yDiff) > positionDiffThreshold_Minor {
            let message = yDiff > 0 ? "카메라를 아래로 내려주세요" : "카메라를 위로 올려주세요"

            feedbacks.append(FeedbackItem(
                priority: 2,
                icon: "↕️",
                message: message,
                category: "position_y",
                currentValue: Double(cur.nosePosition.y * 100),
                targetValue: Double(ref.nosePosition.y * 100),
                tolerance: 5.0,
                unit: "%"
            ))
        }

        return feedbacks
    }

    private func generateFramingDetailFeedback(
        reference: PhotographyFramingResult?,
        current: PhotographyFramingResult?,
        isFrontCamera: Bool
    ) -> [FeedbackItem] {
        guard let ref = reference, let cur = current else { return [] }

        var feedbacks: [FeedbackItem] = []

        // 헤드룸 - 카메라 수직 조작
        // headroomDiff > 0: 머리 위 공간이 많음 (머리가 아래쪽) → 카메라를 아래로 내려서 머리를 위로
        // headroomDiff < 0: 머리 위 공간이 적음 (머리가 위쪽) → 카메라를 위로 올려서 머리를 아래로
        let headroomDiff = cur.headroom - ref.headroom
        if abs(headroomDiff) > headroomDiffThreshold {
            let message = headroomDiff > 0 ? "카메라를 아래로 내려주세요" : "카메라를 위로 올려주세요"

            feedbacks.append(FeedbackItem(
                priority: 1,
                icon: "⬆️",
                message: message,
                category: "headroom",
                currentValue: Double(cur.headroom * 100),
                targetValue: Double(ref.headroom * 100),
                tolerance: 5.0,
                unit: "%"
            ))
        }

        // 리드룸 (시선 방향 여백) - 사람이 좌우 이동
        if let curLeadRoom = cur.leadRoom, let refLeadRoom = ref.leadRoom {
            let leadRoomDiff = curLeadRoom - refLeadRoom
            if abs(leadRoomDiff) > leadRoomDiffThreshold {
                // 시선 방향을 기준으로 이동 방향 결정
                var message: String

                if leadRoomDiff > 0 {
                    // 시선 방향 여백이 많음 → 시선 반대 방향으로 이동
                    if cur.gazeDirection == .left {
                        message = isFrontCamera ? "왼쪽으로 이동하세요" : "오른쪽으로 이동하세요"
                    } else if cur.gazeDirection == .right {
                        message = isFrontCamera ? "오른쪽으로 이동하세요" : "왼쪽으로 이동하세요"
                    } else {
                        message = "시선 방향 여백을 줄이세요"
                    }
                } else {
                    // 시선 방향 여백이 부족 → 시선 방향으로 이동
                    if cur.gazeDirection == .left {
                        message = isFrontCamera ? "오른쪽으로 이동하세요" : "왼쪽으로 이동하세요"
                    } else if cur.gazeDirection == .right {
                        message = isFrontCamera ? "왼쪽으로 이동하세요" : "오른쪽으로 이동하세요"
                    } else {
                        message = "시선 방향에 여백을 더 주세요"
                    }
                }

                feedbacks.append(FeedbackItem(
                    priority: 2,
                    icon: "👁️",
                    message: message,
                    category: "leadroom",
                    currentValue: Double(curLeadRoom * 100),
                    targetValue: Double(refLeadRoom * 100),
                    tolerance: 10.0,
                    unit: "%"
                ))
            }
        }

        return feedbacks
    }

    private func generateCroppingFeedback(croppedGroups: [KeypointGroup], isFrontCamera: Bool) -> FeedbackItem? {
        guard !croppedGroups.isEmpty else { return nil }

        // 우선순위: legs > feet > arms > hands > head
        let message: String
        if croppedGroups.contains(.legs) {
            message = isFrontCamera ? "다리가 잘렸어요. 뒤로 물러나세요" : "다리가 잘렸어요. 카메라를 뒤로 멀리하세요"
        } else if croppedGroups.contains(.feet) {
            message = isFrontCamera ? "발이 잘렸어요. 뒤로 물러나세요" : "발이 잘렸어요. 카메라를 뒤로 멀리하세요"
        } else if croppedGroups.contains(.arms) {
            message = isFrontCamera ? "팔이 잘렸어요. 뒤로 물러나세요" : "팔이 잘렸어요. 카메라를 뒤로 멀리하세요"
        } else if croppedGroups.contains(.leftHand) || croppedGroups.contains(.rightHand) {
            message = isFrontCamera ? "손이 잘렸어요. 뒤로 물러나세요" : "손이 잘렸어요. 카메라를 뒤로 멀리하세요"
        } else if croppedGroups.contains(.head) {
            message = "머리가 잘렸어요. 카메라를 아래로 내려주세요"
        } else {
            message = "\(croppedGroups.first?.displayName ?? "신체 일부")가 잘렸어요. 카메라를 조정하세요"
        }

        return FeedbackItem(
            priority: 0,
            icon: "✂️",
            message: message,
            category: "pose_cropped",
            currentValue: nil,
            targetValue: nil,
            tolerance: nil,
            unit: nil
        )
    }

    private func generatePoseFeedback(poseComparison: PoseComparisonResult?) -> [FeedbackItem]? {
        guard let pose = poseComparison else { return nil }

        var feedbacks: [FeedbackItem] = []

        // 각 부위별 각도 차이 피드백 (🆕 구체적인 방향 메시지 사용)
        for (part, diff) in pose.angleDifferences {
            if diff > angleDiffThreshold {
                // 🆕 angleDirections에서 구체적인 메시지 가져오기
                let message = pose.angleDirections[part] ?? {
                    // fallback: 기존 메시지
                    switch part {
                    case "left_arm":
                        return "왼팔 각도를 조정하세요"
                    case "right_arm":
                        return "오른팔 각도를 조정하세요"
                    case "left_leg":
                        return "왼다리 각도를 조정하세요"
                    case "right_leg":
                        return "오른다리 각도를 조정하세요"
                    case "left_hand":
                        return "왼손 위치를 조정하세요"
                    case "right_hand":
                        return "오른손 위치를 조정하세요"
                    case "shoulder_tilt":
                        return "몸 기울기를 조정하세요"
                    case "face":
                        return "고개 방향을 조정하세요"
                    default:
                        return "\(part)를 조정하세요"
                    }
                }()

                // 아이콘 선택
                let icon: String
                switch part {
                case "shoulder_tilt":
                    icon = "↔️"  // 몸통 기울기
                case "face":
                    icon = "👤"  // 얼굴 방향
                default:
                    icon = "🤸"  // 포즈
                }

                feedbacks.append(FeedbackItem(
                    priority: feedbacks.count + 1,
                    icon: icon,
                    message: message,
                    category: "pose_\(part)",
                    currentValue: nil,
                    targetValue: nil,
                    tolerance: Double(angleDiffThreshold),
                    unit: "도"
                ))
            }
        }

        // 최대 3개까지만 표시
        return Array(feedbacks.prefix(3))
    }
}
