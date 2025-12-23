import Foundation
import CoreGraphics

// MARK: - 피드백 생성기
class FeedbackGenerator {

    // 헬퍼 컴포넌트
    private let cameraAngleDetector = CameraAngleDetector()
    private let compositionAnalyzer = CompositionAnalyzer()
    private let gazeTracker = GazeTracker()
    private let depthEstimator = DepthEstimator()
    private let poseComparator = AdaptivePoseComparator()

    /// Gap 목록을 FeedbackItem으로 변환
    /// - Parameters:
    ///   - gaps: Gap 목록
    ///   - reference: 레퍼런스 분석
    ///   - current: 현재 분석
    ///   - isFrontCamera: 전면 카메라 여부 (좌우반전 적용)
    /// - Returns: 피드백 아이템 목록
    func generateFeedback(
        from gaps: [Gap],
        reference: FrameAnalysis,
        current: (
            face: FaceAnalysisResult?,
            pose: PoseAnalysisResult?,
            bodyRect: CGRect?,
            brightness: Float,
            tilt: Float,
            cameraAngle: CameraAngle,
            compositionType: CompositionType?,
            gaze: GazeResult?,
            depth: DepthResult?
        ),
        isFrontCamera: Bool = false
    ) -> [FeedbackItem] {

        var feedbacks: [FeedbackItem] = []

        for gap in gaps {
            if let feedback = convertGapToFeedback(
                gap: gap,
                reference: reference,
                current: current,
                isFrontCamera: isFrontCamera
            ) {
                feedbacks.append(feedback)
            }
        }

        // 포즈 피드백 추가 (별도 처리)
        if let refPose = reference.poseKeypoints,
           refPose.count >= 17 {

            // 현재 포즈가 없으면 포즈 피드백 생성 안 함
            guard let curPose = current.pose?.keypoints,
                  curPose.count >= 17 else {
                print("⚠️ 현재 프레임에 포즈 없음 - 포즈 피드백 생성 안 함")
                return feedbacks
            }

            // 레퍼런스 포즈의 신뢰도 체크 (너무 낮으면 비교 불가)
            let refVisibleCount = refPose.filter { $0.confidence >= 0.5 }.count
            print("🔍 레퍼런스 포즈 - 전체: \(refPose.count)개, 신뢰도 0.5 이상: \(refVisibleCount)개")

            if refVisibleCount < 5 {
                print("⚠️ 레퍼런스 포즈의 신뢰도가 너무 낮음 (\(refVisibleCount)개) - 포즈 비교 건너뜀")
                return feedbacks
            }

            // 레퍼런스와 현재 프레임 모두 분석
            let referencePoseComparison = poseComparator.comparePoses(
                referenceKeypoints: refPose,
                currentKeypoints: refPose  // 레퍼런스 자체 분석
            )

            let currentPoseComparison = poseComparator.comparePoses(
                referenceKeypoints: refPose,
                currentKeypoints: curPose
            )

            print("🔍 레퍼런스 포즈 비교 결과 - 비교 가능 키포인트: \(referencePoseComparison.comparableKeypoints.count)개")
            print("🔍 현재 포즈 비교 결과 - 비교 가능 키포인트: \(currentPoseComparison.comparableKeypoints.count)개")

            // 레퍼런스 결과도 함께 전달
            let poseFeedbacks = poseComparator.generateFeedback(
                from: currentPoseComparison,
                referenceResult: referencePoseComparison
            )

            print("🔍 생성된 포즈 피드백: \(poseFeedbacks.count)개")

            for (message, category) in poseFeedbacks {
                feedbacks.append(FeedbackItem(
                    priority: 1,  // 🔥 포즈가 최우선!
                    icon: "💪",
                    message: message,
                    category: category,
                    currentValue: nil,
                    targetValue: nil,
                    tolerance: nil,
                    unit: nil
                ))
            }
        } else {
            print("⚠️ 레퍼런스에 포즈 키포인트가 없음 - 포즈 비교 건너뜀")
        }

        return feedbacks
    }

    // MARK: - Private Methods

    /// Gap을 FeedbackItem으로 변환
    private func convertGapToFeedback(
        gap: Gap,
        reference: FrameAnalysis,
        current: (
            face: FaceAnalysisResult?,
            pose: PoseAnalysisResult?,
            bodyRect: CGRect?,
            brightness: Float,
            tilt: Float,
            cameraAngle: CameraAngle,
            compositionType: CompositionType?,
            gaze: GazeResult?,
            depth: DepthResult?
        ),
        isFrontCamera: Bool
    ) -> FeedbackItem? {

        switch gap.type {
        case .distance:
            return generateDistanceFeedback(gap: gap, reference: reference, current: current)

        case .positionX:
            return generatePositionXFeedback(gap: gap, isFrontCamera: isFrontCamera)

        case .positionY:
            return generatePositionYFeedback(gap: gap)

        case .tilt:
            return generateTiltFeedback(gap: gap, current: current)

        case .faceYaw:
            return generateFaceYawFeedback(gap: gap, isFrontCamera: isFrontCamera)

        case .cameraAngle:
            return generateCameraAngleFeedback(gap: gap)

        case .gaze:
            return generateGazeFeedback(gap: gap)

        case .composition:
            return generateCompositionFeedback(gap: gap, reference: reference, current: current)

        case .aspectRatio:
            return generateAspectRatioFeedback(gap: gap)

        case .excessivePadding:
            return generatePaddingFeedback(gap: gap)

        default:
            return nil
        }
    }

    /// 거리 피드백 생성
    private func generateDistanceFeedback(
        gap: Gap,
        reference: FrameAnalysis,
        current: (
            face: FaceAnalysisResult?,
            pose: PoseAnalysisResult?,
            bodyRect: CGRect?,
            brightness: Float,
            tilt: Float,
            cameraAngle: CameraAngle,
            compositionType: CompositionType?,
            gaze: GazeResult?,
            depth: DepthResult?
        )
    ) -> FeedbackItem? {

        guard let refDepth = reference.depth, let curDepth = current.depth else {
            return nil
        }

        if let (message, shouldUseZoom) = depthEstimator.generateDistanceFeedback(
            reference: refDepth,
            current: curDepth
        ) {
            return FeedbackItem(
                priority: gap.priority,
                icon: shouldUseZoom ? "🔍" : "🚶",
                message: message,
                category: "distance",
                currentValue: gap.current,
                targetValue: gap.target,
                tolerance: gap.tolerance,
                unit: "m"
            )
        }

        return nil
    }

    /// X 위치 피드백
    private func generatePositionXFeedback(gap: Gap, isFrontCamera: Bool) -> FeedbackItem {
        guard let current = gap.current, let target = gap.target else {
            return FeedbackItem(
                priority: gap.priority,
                icon: "↔️",
                message: "좌우 위치를 맞춰주세요",
                category: "position_x",
                currentValue: nil,
                targetValue: nil,
                tolerance: nil,
                unit: nil
            )
        }

        // Vision 좌표계: X=0(왼쪽), X=1(오른쪽) (전면/후면 동일)
        // current > target: 인물이 오른쪽에 있음 → 왼쪽으로 이동
        // current < target: 인물이 왼쪽에 있음 → 오른쪽으로 이동
        //
        // ⚠️ 중요: Vision 좌표는 카메라 센서 기준이므로 전면/후면 동일!
        let diff = abs(current - target)
        let direction = current > target ? "왼쪽" : "오른쪽"

        // 차이에 따라 구체적인 안내 (화면 비율 기반)
        let message: String

        if diff > 0.3 {
            message = "\(direction)으로 많이 이동하세요"
        } else if diff > 0.15 {
            message = "\(direction)으로 조금 이동하세요"
        } else {
            message = "\(direction)으로 살짝 이동하세요"
        }

        return FeedbackItem(
            priority: gap.priority,
            icon: "↔️",
            message: message,
            category: "position_x",
            currentValue: current,
            targetValue: target,
            tolerance: gap.tolerance,
            unit: "%"
        )
    }

    /// Y 위치 피드백
    private func generatePositionYFeedback(gap: Gap) -> FeedbackItem {
        guard let current = gap.current, let target = gap.target else {
            return FeedbackItem(
                priority: gap.priority,
                icon: "↕️",
                message: "상하 위치를 맞춰주세요",
                category: "position_y",
                currentValue: nil,
                targetValue: nil,
                tolerance: nil,
                unit: nil
            )
        }

        // 🔄 카메라 움직임 기준으로 피드백
        // Vision 좌표계: Y=0(아래), Y=1(위)
        // current > target: 인물이 화면 위쪽에 있음
        //   → 인물을 아래로 내리려면 카메라를 위로 올려야 함
        // current < target: 인물이 화면 아래쪽에 있음
        //   → 인물을 위로 올리려면 카메라를 아래로 내려야 함
        let diff = abs(current - target)
        let cameraDirection = current > target ? "위로" : "아래로"

        // 차이에 따라 구체적인 거리 제시
        // 화면 기준 비율로 설명 (더 직관적)
        let percentage = Int(diff * 100)
        let message: String

        if percentage > 30 {
            message = "카메라를 \(cameraDirection) 많이 이동하세요"
        } else if percentage > 15 {
            message = "카메라를 \(cameraDirection) 조금 이동하세요"
        } else {
            message = "카메라를 \(cameraDirection) 살짝 이동하세요"
        }

        return FeedbackItem(
            priority: gap.priority,
            icon: "📷",
            message: message,
            category: "position_y",
            currentValue: current,
            targetValue: target,
            tolerance: gap.tolerance,
            unit: "%"
        )
    }

    /// 기울기 피드백
    private func generateTiltFeedback(
        gap: Gap,
        current: (
            face: FaceAnalysisResult?,
            pose: PoseAnalysisResult?,
            bodyRect: CGRect?,
            brightness: Float,
            tilt: Float,
            cameraAngle: CameraAngle,
            compositionType: CompositionType?,
            gaze: GazeResult?,
            depth: DepthResult?
        )
    ) -> FeedbackItem {
        guard let current = gap.current, let target = gap.target else {
            return FeedbackItem(
                priority: gap.priority,
                icon: "📐",
                message: "기울기 조정",
                category: "tilt",
                currentValue: nil,
                targetValue: nil,
                tolerance: nil,
                unit: nil
            )
        }

        // 기울기 방향 수정 ✅
        // atan2로 계산된 각도: 양수=반시계(왼쪽), 음수=시계(오른쪽)
        // current > target: 더 반시계 방향 → 시계 방향(오른쪽)으로 기울여야 함
        // current < target: 더 시계 방향 → 반시계 방향(왼쪽)으로 기울여야 함
        let diff = abs(current - target)
        let direction = current > target ? "오른쪽" : "왼쪽"
        let angle = Int(diff)

        let message: String
        if angle > 5 {
            message = "카메라를 \(direction)으로 \(angle)도 크게 기울이세요"
        } else if angle > 2 {
            message = "카메라를 \(direction)으로 \(angle)도 기울이세요"
        } else {
            message = "카메라를 \(direction)으로 약간만 기울이세요 (\(angle)도)"
        }

        return FeedbackItem(
            priority: gap.priority,
            icon: "📐",
            message: message,
            category: "tilt",
            currentValue: current,
            targetValue: target,
            tolerance: gap.tolerance,
            unit: "도"
        )
    }

    /// 얼굴 각도 피드백
    private func generateFaceYawFeedback(gap: Gap, isFrontCamera: Bool) -> FeedbackItem {
        guard let current = gap.current, let target = gap.target else {
            return FeedbackItem(
                priority: gap.priority,
                icon: "👤",
                message: "얼굴 각도 조정",
                category: "face_yaw",
                currentValue: nil,
                targetValue: nil,
                tolerance: nil,
                unit: nil
            )
        }

        // 🔥 Vision Yaw 좌표계 (전면/후면 동일)
        // Yaw > 0: 얼굴이 왼쪽을 향함 (실제 물리적 방향)
        // Yaw < 0: 얼굴이 오른쪽을 향함 (실제 물리적 방향)
        //
        // current > target: 현재 더 왼쪽 향함 → 오른쪽으로 돌려야 함
        // current < target: 현재 더 오른쪽 향함 → 왼쪽으로 돌려야 함
        //
        // ⚠️ 중요: 전면/후면 관계없이 동일한 로직!
        // Vision 값은 항상 실제 물리적 방향 기준
        let direction = current > target ? "오른쪽" : "왼쪽"

        // 🐛 디버그 로그
        print("🔍 Yaw 피드백 - current: \(current)도, target: \(target)도, 방향: \(direction), 전면카메라: \(isFrontCamera)")
        return FeedbackItem(
            priority: gap.priority,
            icon: "👤",
            message: "고개를 \(direction)으로 돌려주세요",
            category: "face_yaw",
            currentValue: current,
            targetValue: target,
            tolerance: gap.tolerance,
            unit: "도"
        )
    }

    /// 카메라 앵글 피드백
    private func generateCameraAngleFeedback(gap: Gap) -> FeedbackItem? {
        guard let metadata = gap.metadata,
              let refAngle = metadata["reference_angle"] as? CameraAngle,
              let curAngle = metadata["current_angle"] as? CameraAngle else {
            return nil
        }

        if let message = cameraAngleDetector.generateAngleFeedback(
            reference: refAngle,
            current: curAngle
        ) {
            return FeedbackItem(
                priority: gap.priority,
                icon: "📷",
                message: message,
                category: "camera_angle",
                currentValue: nil,
                targetValue: nil,
                tolerance: nil,
                unit: nil
            )
        }

        return nil
    }

    /// 시선 피드백
    private func generateGazeFeedback(gap: Gap) -> FeedbackItem? {
        guard let metadata = gap.metadata,
              let refGaze = metadata["reference_gaze"] as? GazeDirection,
              let _ = metadata["current_gaze"] as? GazeDirection else {
            return nil
        }

        let message: String
        switch refGaze {
        case .lookingAtCamera:
            message = "카메라를 바라봐주세요"
        case .lookingLeft:
            // 🔥 주의: "왼쪽을 본다" = 카메라 관점에서 왼쪽
            // 후면 카메라: 그대로 왼쪽
            message = "시선을 왼쪽으로"
        case .lookingRight:
            // 후면 카메라: 그대로 오른쪽
            message = "시선을 오른쪽으로"
        case .lookingUp:
            message = "시선을 위로"
        case .lookingDown:
            message = "시선을 아래로"
        default:
            message = "시선 방향 조정"
        }

        return FeedbackItem(
            priority: gap.priority,
            icon: "👀",
            message: message,
            category: "gaze",
            currentValue: nil,
            targetValue: nil,
            tolerance: nil,
            unit: nil
        )
    }

    /// 구도 피드백
    private func generateCompositionFeedback(
        gap: Gap,
        reference: FrameAnalysis,
        current: (
            face: FaceAnalysisResult?,
            pose: PoseAnalysisResult?,
            bodyRect: CGRect?,
            brightness: Float,
            tilt: Float,
            cameraAngle: CameraAngle,
            compositionType: CompositionType?,
            gaze: GazeResult?,
            depth: DepthResult?
        )
    ) -> FeedbackItem? {

        guard let refComp = reference.compositionType,
              let refFace = reference.faceRect,
              let curFace = current.face?.faceRect else {
            return nil
        }

        let refPosition = CGPoint(x: refFace.midX, y: refFace.midY)
        let curPosition = CGPoint(x: curFace.midX, y: curFace.midY)

        let (message, _, _) = compositionAnalyzer.generateCompositionFeedback(
            referenceType: refComp,
            referencePosition: refPosition,
            currentPosition: curPosition
        )

        return FeedbackItem(
            priority: gap.priority,
            icon: "🎨",
            message: message,
            category: "composition",
            currentValue: nil,
            targetValue: nil,
            tolerance: nil,
            unit: nil
        )
    }

    /// 화면 비율 피드백 생성
    private func generateAspectRatioFeedback(gap: Gap) -> FeedbackItem? {
        guard let metadata = gap.metadata,
              let refRatio = metadata["reference_ratio"] as? CameraAspectRatio,
              let curRatio = metadata["current_ratio"] as? CameraAspectRatio else {
            return nil
        }

        let message = "화면 비율을 \(refRatio.displayName)로 변경하세요 (현재: \(curRatio.displayName))"

        return FeedbackItem(
            priority: gap.priority,
            icon: "📐",
            message: message,
            category: "aspect_ratio",
            currentValue: nil,
            targetValue: nil,
            tolerance: nil,
            unit: nil
        )
    }

    /// 여백 피드백 생성
    private func generatePaddingFeedback(gap: Gap) -> FeedbackItem? {
        guard let metadata = gap.metadata,
              let left = metadata["left"] as? CGFloat,
              let right = metadata["right"] as? CGFloat,
              let refLeft = metadata["ref_left"] as? CGFloat,
              let refRight = metadata["ref_right"] as? CGFloat else {
            return nil
        }

        // 🔥 좌우 여백만 비교 (상하는 Y 위치로 해결)
        let diffs = [
            ("좌측", left - refLeft, left, refLeft),
            ("우측", right - refRight, right, refRight)
        ]

        // 가장 차이가 큰 방향 찾기
        let maxDiff = diffs.max(by: { abs($0.1) < abs($1.1) })!

        // 🔥 줌/거리 조정으로 해결하도록 유도
        let message: String
        if abs(maxDiff.1) > 0.1 {
            if maxDiff.1 > 0 {
                // 현재 여백이 더 많음 → 줌인 필요
                message = "\(maxDiff.0) 여백이 많습니다. 줌인하거나 가까이 가세요"
            } else {
                // 현재 여백이 더 적음 → 줌아웃 필요
                message = "\(maxDiff.0) 여백이 부족합니다. 줌아웃하거나 멀어지세요"
            }
        } else {
            message = "여백 조정 필요"
        }

        return FeedbackItem(
            priority: gap.priority,
            icon: "📐",
            message: message,
            category: "padding",
            currentValue: gap.current,
            targetValue: gap.target,
            tolerance: gap.tolerance,
            unit: "%"
        )
    }
}
