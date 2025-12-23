import Foundation
import CoreGraphics

// MARK: - Gap 타입
enum GapType: String {
    case distance          // 거리 (앞/뒤 이동)
    case positionX         // X 위치 (좌/우 이동)
    case positionY         // Y 위치 (상/하 이동)
    case tilt              // 기울기
    case faceYaw           // 얼굴 좌우 회전
    case facePitch         // 얼굴 상하 각도
    case cameraAngle       // 카메라 앵글
    case gaze              // 시선
    case composition       // 구도
    case leftArm           // 왼팔 포즈
    case rightArm          // 오른팔 포즈
    case leftLeg           // 왼다리 포즈
    case rightLeg          // 오른다리 포즈
    case missingParts      // 안 보이는 부위
    case aspectRatio       // 🆕 화면 비율
    case excessivePadding  // 🆕 과도한 여백
}

// MARK: - Gap (차이)
struct Gap {
    let type: GapType              // Gap 타입
    let current: Double?           // 현재 값
    let target: Double?            // 목표 값
    let difference: Double         // 차이 (절대값)
    let tolerance: Double          // 허용 오차
    let priority: Int              // 우선순위 (1=높음)
    let metadata: [String: Any]?   // 추가 정보

    /// Gap이 허용 범위 내인지
    var isWithinTolerance: Bool {
        return difference <= tolerance
    }
}

// MARK: - Gap 분석기
class GapAnalyzer {

    /// 프레임 분석 결과 비교
    /// - Parameters:
    ///   - reference: 레퍼런스 분석
    ///   - current: 현재 분석
    /// - Returns: 감지된 Gap 목록
    func analyzeGaps(
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
            depth: DepthResult?,
            aspectRatio: CameraAspectRatio,
            padding: ImagePadding?
        )
    ) -> [Gap] {

        var gaps: [Gap] = []

        // 🔄 우선순위 재정의: 포즈(1) > 위치(2) > 프레이밍(3) > 앵글(4) > 구도(5) > 시선(6)

        // 1. 위치 Gap (X) - 우선순위 2 (인물 위치)
        if let refFace = reference.faceRect, let curFace = current.face?.faceRect {
            let xDiff = abs(curFace.midX - refFace.midX)
            if xDiff > 0.08 {  // 🔄 8% 이상 차이 (더 관대하게)
                gaps.append(Gap(
                    type: .positionX,
                    current: Double(curFace.midX * 100),
                    target: Double(refFace.midX * 100),
                    difference: Double(xDiff * 100),
                    tolerance: 8.0,
                    priority: 2,  // 위치
                    metadata: nil
                ))
            }
        }

        // 2. 위치 Gap (Y) - 우선순위 2 (인물 위치)
        if let refFace = reference.faceRect, let curFace = current.face?.faceRect {
            let yDiff = abs(curFace.midY - refFace.midY)
            if yDiff > 0.08 {  // 🔄 8% 이상 차이 (더 관대하게)
                gaps.append(Gap(
                    type: .positionY,
                    current: Double(curFace.midY * 100),
                    target: Double(refFace.midY * 100),
                    difference: Double(yDiff * 100),
                    tolerance: 8.0,
                    priority: 2,  // 위치
                    metadata: nil
                ))
            }
        }

        // 3. 거리 Gap - 우선순위 3 (프레이밍)
        if let refDepth = reference.depth, let curDepth = current.depth {
            if let refDist = refDepth.distance, let curDist = curDepth.distance {
                let diff = abs(curDist - refDist)
                if diff > 0.3 {  // 🔄 30cm 이상 차이 (더 관대하게)
                    gaps.append(Gap(
                        type: .distance,
                        current: Double(curDist),
                        target: Double(refDist),
                        difference: Double(diff),
                        tolerance: 0.3,
                        priority: 3,  // 프레이밍 (거리/줌)
                        metadata: ["depth_method": curDepth.method]
                    ))
                }
            }
        }

        // 4. 카메라 앵글 Gap - 우선순위 4 (카메라 앵글)
        if reference.cameraAngle != current.cameraAngle {
            gaps.append(Gap(
                type: .cameraAngle,
                current: nil,
                target: nil,
                difference: 1.0,  // 불일치
                tolerance: 0.0,
                priority: 4,  // 카메라 앵글
                metadata: [
                    "reference_angle": reference.cameraAngle,
                    "current_angle": current.cameraAngle
                ]
            ))
        }

        // 5. 기울기 Gap - 우선순위 4 (카메라 앵글)
        let tiltDiff = abs(current.tilt - reference.tiltAngle)
        if tiltDiff > 5 {  // 🔄 5도 이상 (더 관대하게)
            gaps.append(Gap(
                type: .tilt,
                current: Double(current.tilt),
                target: Double(reference.tiltAngle),
                difference: Double(tiltDiff),
                tolerance: 5.0,
                priority: 4,  // 카메라 앵글 (기울기)
                metadata: nil
            ))
        }

        // 6. 구도 Gap - 우선순위 5 (구도)
        if let refComp = reference.compositionType, let curComp = current.compositionType {
            if refComp != curComp {
                gaps.append(Gap(
                    type: .composition,
                    current: nil,
                    target: nil,
                    difference: 1.0,
                    tolerance: 0.0,
                    priority: 5,  // 구도
                    metadata: [
                        "reference_composition": refComp,
                        "current_composition": curComp
                    ]
                ))
            }
        }

        // 7. 시선 Gap - 우선순위 6 (시선)
        if let refGaze = reference.gaze, let curGaze = current.gaze {
            if refGaze.direction != curGaze.direction {
                gaps.append(Gap(
                    type: .gaze,
                    current: nil,
                    target: nil,
                    difference: 1.0,
                    tolerance: 0.0,
                    priority: 6,  // 시선
                    metadata: [
                        "reference_gaze": refGaze.direction,
                        "current_gaze": curGaze.direction
                    ]
                ))
            }
        }

        // 8. 얼굴 각도 Gap (Yaw) - 우선순위 6 (시선)
        if let refYaw = reference.faceYaw, let curYaw = current.face?.yaw {
            let yawDiff = abs((curYaw - refYaw) * 180 / .pi)
            if yawDiff > 15 {  // 🔄 15도 이상 (더 관대하게)
                gaps.append(Gap(
                    type: .faceYaw,
                    current: Double(curYaw * 180 / .pi),
                    target: Double(refYaw * 180 / .pi),
                    difference: Double(yawDiff),
                    tolerance: 15.0,
                    priority: 6,  // 시선 (얼굴 방향)
                    metadata: nil
                ))
            }
        }

        // 9. 포즈 Gap (AdaptivePoseComparator 사용)
        // 이 부분은 FeedbackGenerator에서 처리

        // 10. 🆕 화면 비율 Gap
        if reference.aspectRatio != current.aspectRatio {
            gaps.append(Gap(
                type: .aspectRatio,
                current: nil,
                target: nil,
                difference: 1.0,
                tolerance: 0.0,
                priority: 3,  // 프레이밍 카테고리
                metadata: [
                    "reference_ratio": reference.aspectRatio,
                    "current_ratio": current.aspectRatio
                ]
            ))
        }

        // 11. 🆕 여백 Gap (레퍼런스와 비교)
        // 🔥 X 위치 Gap이 있으면 좌우 여백 Gap 무시 (위치 조정이 여백도 해결함)
        let hasPositionXGap = gaps.contains { $0.type == .positionX }

        if let refPadding = reference.imagePadding, let curPadding = current.padding {
            // 좌우 여백 차이만 계산 (상하는 Y 위치로 해결)
            let leftDiff = abs(refPadding.left - curPadding.left)
            let rightDiff = abs(refPadding.right - curPadding.right)
            let horizontalDiff = max(leftDiff, rightDiff)

            var shouldCreateGap = false
            var maxDiff: CGFloat = 0

            // 좌우 여백만 체크 (X 위치 Gap 없으면)
            if !hasPositionXGap && horizontalDiff > 0.1 {
                shouldCreateGap = true
                maxDiff = horizontalDiff
            }

            // 상하 여백은 무시 (Y 위치로 해결)
            // 거리/줌으로만 해결하도록 유도

            // 10% 이상 차이나고, 위치 Gap과 충돌하지 않으면 생성
            if shouldCreateGap {
                gaps.append(Gap(
                    type: .excessivePadding,
                    current: Double(curPadding.total * 100),
                    target: Double(refPadding.total * 100),
                    difference: Double(maxDiff * 100),
                    tolerance: 10.0,  // 10% 이하는 허용
                    priority: 3,  // 프레이밍 카테고리
                    metadata: [
                        "top": curPadding.top,
                        "bottom": curPadding.bottom,
                        "left": curPadding.left,
                        "right": curPadding.right,
                        "ref_top": refPadding.top,
                        "ref_bottom": refPadding.bottom,
                        "ref_left": refPadding.left,
                        "ref_right": refPadding.right
                    ]
                ))
            }
        }

        return gaps
    }

    /// 완성도 점수 계산
    /// - Parameter gaps: Gap 목록
    /// - Returns: 완성도 점수 (0~1)
    func calculateCompletionScore(gaps: [Gap]) -> Double {
        if gaps.isEmpty {
            return 1.0  // 완벽
        }

        var totalScore: Double = 0.0
        var count = 0

        for gap in gaps {
            // tolerance 내에 있으면 1.0점, 아니면 차이에 비례하여 감점
            let itemScore: Double
            if gap.isWithinTolerance {
                itemScore = 1.0
            } else {
                // tolerance를 넘어선 차이만 감점 대상
                let excessDiff = gap.difference - gap.tolerance

                if let _ = gap.current, let target = gap.target {
                    let maxDiff = max(abs(target) + gap.tolerance + 50, 100.0)
                    itemScore = max(0.0, 1.0 - (excessDiff / maxDiff))
                } else {
                    // current/target 없는 경우 (카테고리 불일치 등)
                    itemScore = 0.0
                }
            }

            totalScore += itemScore
            count += 1
        }

        if count == 0 {
            return gaps.isEmpty ? 1.0 : 0.0
        }

        // 평균을 0~1 범위로 확실히 제한
        return min(1.0, max(0.0, totalScore / Double(count)))
    }

    /// Gap 우선순위 정렬
    /// - Parameter gaps: Gap 목록
    /// - Returns: 우선순위순 정렬된 Gap
    func sortByPriority(gaps: [Gap]) -> [Gap] {
        return gaps.sorted { $0.priority < $1.priority }
    }
}
