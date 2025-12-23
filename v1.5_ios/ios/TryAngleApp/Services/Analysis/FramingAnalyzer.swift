import Foundation
import CoreGraphics
import Vision

// MARK: - 프레이밍 분석 결과
struct FramingAnalysis {
    let subjectSize: CGFloat          // 화면에서 인물이 차지하는 비율 (0.0~1.0)
    let headToTopDistance: CGFloat    // 머리에서 화면 상단까지 거리
    let bottomMargin: CGFloat          // 하단 여백
    let horizontalMargin: CGFloat      // 좌우 평균 여백
    let bodyVisibility: BodyVisibility // 보이는 신체 부위
    let suggestedAction: FramingAction // 제안 액션
}

// MARK: - 보이는 신체 부위
enum BodyVisibility {
    case fullBody       // 전신 (머리부터 발까지)
    case threeQuarter   // 3/4 (머리부터 무릎)
    case halfBody       // 반신 (머리부터 허리)
    case upperBody      // 상반신 (머리부터 가슴)
    case headAndShoulder // 헤드샷 (머리와 어깨)
    case closeUp        // 클로즈업 (얼굴만)

    var description: String {
        switch self {
        case .fullBody: return "전신"
        case .threeQuarter: return "무릎샷"
        case .halfBody: return "허리샷"
        case .upperBody: return "상반신"
        case .headAndShoulder: return "헤드샷"
        case .closeUp: return "클로즈업"
        }
    }
}

// MARK: - 프레이밍 액션
enum FramingAction {
    case zoomIn(amount: CGFloat)    // 줌인 (배율)
    case zoomOut(amount: CGFloat)   // 줌아웃 (배율)
    case moveCloser                 // 카메라 가까이
    case moveAway                   // 카메라 멀리
    case adjustRatio(CameraAspectRatio) // 비율 변경
    case perfect                    // 완벽

    var description: String {
        switch self {
        case .zoomIn(let amount):
            return "줌인 (\(String(format: "%.1fx", amount)))"
        case .zoomOut(let amount):
            return "줌아웃 (\(String(format: "%.1fx", amount)))"
        case .moveCloser:
            return "카메라를 가까이 이동"
        case .moveAway:
            return "카메라를 멀리 이동"
        case .adjustRatio(let ratio):
            return "\(ratio.displayName) 비율로 변경"
        case .perfect:
            return "완벽한 프레이밍"
        }
    }
}

// MARK: - 프레이밍 분석기
class FramingAnalyzer {

    // MARK: - 레퍼런스와 현재 프레임 비교 분석
    func analyzeFraming(
        reference: FrameAnalysis,
        current: FrameAnalysis,
        currentAspectRatio: CameraAspectRatio
    ) -> (analysis: FramingAnalysis, feedback: String?) {

        // 1. 레퍼런스 프레이밍 분석
        let refFraming = analyzeFrame(reference)

        // 2. 현재 프레이밍 분석
        let curFraming = analyzeFrame(current)

        // 3. 비교 및 액션 결정
        let suggestedAction = determineSuggestedAction(
            reference: refFraming,
            current: curFraming,
            currentRatio: currentAspectRatio,
            referenceRatio: reference.aspectRatio
        )

        // 4. 피드백 메시지 생성
        let feedback = generateFeedback(
            reference: refFraming,
            current: curFraming,
            action: suggestedAction
        )

        return (curFraming, feedback)
    }

    // MARK: - 개별 프레임 분석
    private func analyzeFrame(_ frame: FrameAnalysis) -> FramingAnalysis {
        var subjectSize: CGFloat = 0
        var headToTop: CGFloat = 0
        var bottomMargin: CGFloat = 0
        var horizontalMargin: CGFloat = 0
        var visibility = BodyVisibility.fullBody

        // 얼굴 기반 분석
        if let faceRect = frame.faceRect {
            // 얼굴 크기로 인물 크기 추정
            subjectSize = faceRect.width * faceRect.height * 10 // 얼굴은 전체 인물의 약 1/10
            headToTop = faceRect.minY

            // 얼굴 크기로 보이는 부위 추정
            let faceSize = faceRect.height
            if faceSize > 0.3 {
                visibility = .closeUp
            } else if faceSize > 0.2 {
                visibility = .headAndShoulder
            } else if faceSize > 0.15 {
                visibility = .upperBody
            } else if faceSize > 0.1 {
                visibility = .halfBody
            } else if faceSize > 0.07 {
                visibility = .threeQuarter
            } else {
                visibility = .fullBody
            }

            horizontalMargin = (faceRect.minX + (1.0 - faceRect.maxX)) / 2
        }

        // 포즈 키포인트 기반 보완
        if let keypoints = frame.poseKeypoints, keypoints.count >= 17 {
            // 어떤 키포인트가 보이는지 확인
            let hasAnkles = keypoints[15].confidence > 0.5 || keypoints[16].confidence > 0.5
            let hasKnees = keypoints[13].confidence > 0.5 || keypoints[14].confidence > 0.5
            let hasHips = keypoints[11].confidence > 0.5 || keypoints[12].confidence > 0.5
            let hasShoulders = keypoints[5].confidence > 0.5 || keypoints[6].confidence > 0.5

            // 보이는 부위에 따라 visibility 재설정
            if hasAnkles {
                visibility = .fullBody
            } else if hasKnees {
                visibility = .threeQuarter
            } else if hasHips {
                visibility = .halfBody
            } else if hasShoulders {
                visibility = .upperBody
            }

            // 실제 인물 영역 계산
            let visibleKeypoints = keypoints.filter { $0.confidence > 0.5 }
            if !visibleKeypoints.isEmpty {
                let minY = visibleKeypoints.map { $0.point.y }.min() ?? 0
                let maxY = visibleKeypoints.map { $0.point.y }.max() ?? 1
                let minX = visibleKeypoints.map { $0.point.x }.min() ?? 0
                let maxX = visibleKeypoints.map { $0.point.x }.max() ?? 1

                headToTop = minY
                bottomMargin = 1.0 - maxY
                horizontalMargin = (minX + (1.0 - maxX)) / 2
                subjectSize = (maxX - minX) * (maxY - minY)
            }
        }

        // 전신 영역으로 보완
        if let bodyRect = frame.bodyRect {
            bottomMargin = 1.0 - bodyRect.maxY
            if subjectSize == 0 {
                subjectSize = bodyRect.width * bodyRect.height
            }
        }

        return FramingAnalysis(
            subjectSize: subjectSize,
            headToTopDistance: headToTop,
            bottomMargin: bottomMargin,
            horizontalMargin: horizontalMargin,
            bodyVisibility: visibility,
            suggestedAction: .perfect
        )
    }

    // MARK: - 제안 액션 결정
    private func determineSuggestedAction(
        reference: FramingAnalysis,
        current: FramingAnalysis,
        currentRatio: CameraAspectRatio,
        referenceRatio: CameraAspectRatio
    ) -> FramingAction {

        // 1. 비율이 다르면 먼저 비율 변경 제안
        if currentRatio != referenceRatio {
            return .adjustRatio(referenceRatio)
        }

        // 2. 인물 크기 비교
        let sizeDiff = reference.subjectSize - current.subjectSize
        let sizeRatio = reference.subjectSize / max(0.01, current.subjectSize)

        // 크기 차이가 20% 이상이면 조정 필요
        if abs(sizeDiff) > 0.1 {
            if sizeDiff > 0 {
                // 레퍼런스가 더 크면 -> 줌인 또는 가까이
                if sizeRatio > 1.5 {
                    return .moveCloser
                } else {
                    return .zoomIn(amount: sizeRatio)
                }
            } else {
                // 레퍼런스가 더 작으면 -> 줌아웃 또는 멀리
                if sizeRatio < 0.7 {
                    return .moveAway
                } else {
                    return .zoomOut(amount: 1.0 / sizeRatio)
                }
            }
        }

        // 3. 여백 비교 (상단 여백이 너무 다르면)
        let headMarginDiff = abs(reference.headToTopDistance - current.headToTopDistance)
        if headMarginDiff > 0.1 {
            // 위치 조정이 필요하지만 여기서는 크기 조정으로 대체
            if current.headToTopDistance > reference.headToTopDistance {
                return .zoomIn(amount: 1.1)
            }
        }

        return .perfect
    }

    // MARK: - 피드백 메시지 생성
    private func generateFeedback(
        reference: FramingAnalysis,
        current: FramingAnalysis,
        action: FramingAction
    ) -> String? {

        switch action {
        case .zoomIn(let amount):
            if amount > 1.3 {
                return "📐 화면을 확대해주세요 (인물이 너무 작아요)"
            } else {
                return "📐 조금 더 확대해주세요"
            }

        case .zoomOut(let amount):
            if amount > 1.3 {
                return "📐 화면을 축소해주세요 (인물이 너무 커요)"
            } else {
                return "📐 조금 축소해주세요"
            }

        case .moveCloser:
            return "📐 카메라를 인물에게 가까이 이동하세요"

        case .moveAway:
            return "📐 카메라를 뒤로 이동하세요"

        case .adjustRatio(let ratio):
            return "📐 비율을 \(ratio.displayName)로 변경하세요"

        case .perfect:
            // 보이는 부위가 다른 경우
            if reference.bodyVisibility != current.bodyVisibility {
                return bodyVisibilityFeedback(
                    reference: reference.bodyVisibility,
                    current: current.bodyVisibility
                )
            }
            return nil
        }
    }

    // MARK: - 보이는 부위 피드백
    private func bodyVisibilityFeedback(
        reference: BodyVisibility,
        current: BodyVisibility
    ) -> String {

        let refLevel = visibilityLevel(reference)
        let curLevel = visibilityLevel(current)

        if curLevel < refLevel {
            // 더 많이 보여야 함 (줌아웃)
            return "📐 \(reference.description) 구도로 맞춰주세요 (현재: \(current.description))"
        } else if curLevel > refLevel {
            // 덜 보여야 함 (줌인)
            return "📐 \(reference.description) 구도로 맞춰주세요 (현재: \(current.description))"
        }

        return "📐 구도가 완벽합니다"
    }

    private func visibilityLevel(_ visibility: BodyVisibility) -> Int {
        switch visibility {
        case .fullBody: return 0
        case .threeQuarter: return 1
        case .halfBody: return 2
        case .upperBody: return 3
        case .headAndShoulder: return 4
        case .closeUp: return 5
        }
    }
}