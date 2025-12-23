import Foundation
import Vision
import CoreGraphics

// MARK: - 시선 방향
enum GazeDirection {
    case lookingAtCamera      // 카메라 응시
    case lookingLeft          // 왼쪽
    case lookingRight         // 오른쪽
    case lookingUp            // 위
    case lookingDown          // 아래
    case lookingLeftUp        // 왼쪽 위
    case lookingLeftDown      // 왼쪽 아래
    case lookingRightUp       // 오른쪽 위
    case lookingRightDown     // 오른쪽 아래
    case unknown              // 알 수 없음

    var description: String {
        switch self {
        case .lookingAtCamera:
            return "카메라 응시"
        case .lookingLeft:
            return "왼쪽 응시"
        case .lookingRight:
            return "오른쪽 응시"
        case .lookingUp:
            return "위쪽 응시"
        case .lookingDown:
            return "아래쪽 응시"
        case .lookingLeftUp:
            return "왼쪽 위 응시"
        case .lookingLeftDown:
            return "왼쪽 아래 응시"
        case .lookingRightUp:
            return "오른쪽 위 응시"
        case .lookingRightDown:
            return "오른쪽 아래 응시"
        case .unknown:
            return "알 수 없음"
        }
    }
}

// MARK: - 시선 추적 결과
struct GazeResult {
    let direction: GazeDirection        // 시선 방향
    let horizontalAngle: Float          // 수평 각도 (-1.0 ~ 1.0)
    let verticalAngle: Float            // 수직 각도 (-1.0 ~ 1.0)
    let confidence: Float               // 신뢰도 (0 ~ 1)
}

// MARK: - 시선 추적기
class GazeTracker {

    private let horizontalThreshold: Float = 0.15  // 좌우 판단 임계값
    private let verticalThreshold: Float = 0.15    // 상하 판단 임계값
    private let cameraLookThreshold: Float = 0.1   // 카메라 응시 판단 임계값

    /// Vision Framework 얼굴 관찰 결과로부터 시선 추적
    /// - Parameter faceObservation: VNFaceObservation
    /// - Returns: 시선 추적 결과
    func trackGaze(from faceObservation: VNFaceObservation) -> GazeResult? {
        guard let landmarks = faceObservation.landmarks else {
            return nil
        }

        // 방법 1: 눈 랜드마크 기반 시선 추정
        if let gazeFromEyes = estimateGazeFromEyes(landmarks: landmarks) {
            return gazeFromEyes
        }

        // 방법 2: 얼굴 각도(yaw/pitch) 기반 fallback
        return estimateGazeFromFaceAngles(
            yaw: faceObservation.yaw?.floatValue,
            pitch: faceObservation.pitch?.floatValue
        )
    }

    /// 시선 비교 (레퍼런스 vs 현재)
    /// - Parameters:
    ///   - reference: 레퍼런스 시선
    ///   - current: 현재 시선
    /// - Returns: 일치 여부
    func compareGaze(reference: GazeResult, current: GazeResult) -> Bool {
        return reference.direction == current.direction
    }

    /// 시선 차이 피드백 생성
    /// - Parameters:
    ///   - reference: 레퍼런스 시선
    ///   - current: 현재 시선
    /// - Returns: 피드백 메시지
    func generateGazeFeedback(reference: GazeResult, current: GazeResult) -> String? {
        if reference.direction == current.direction {
            return nil  // 일치
        }

        switch reference.direction {
        case .lookingAtCamera:
            return "카메라를 바라봐주세요"
        case .lookingLeft:
            return "시선을 왼쪽으로"
        case .lookingRight:
            return "시선을 오른쪽으로"
        case .lookingUp:
            return "시선을 위로"
        case .lookingDown:
            return "시선을 아래로"
        case .lookingLeftUp:
            return "시선을 왼쪽 위로"
        case .lookingLeftDown:
            return "시선을 왼쪽 아래로"
        case .lookingRightUp:
            return "시선을 오른쪽 위로"
        case .lookingRightDown:
            return "시선을 오른쪽 아래로"
        case .unknown:
            return nil
        }
    }

    // MARK: - Private Methods

    /// 눈 랜드마크 기반 시선 추정
    private func estimateGazeFromEyes(landmarks: VNFaceLandmarks2D) -> GazeResult? {
        guard let leftEye = landmarks.leftEye,
              let rightEye = landmarks.rightEye,
              let leftPupil = landmarks.leftPupil,
              let rightPupil = landmarks.rightPupil else {
            return nil
        }

        // 왼쪽 눈 중심과 동공 위치
        let leftEyeCenter = centroid(leftEye.normalizedPoints)
        let leftPupilCenter = centroid(leftPupil.normalizedPoints)

        // 오른쪽 눈 중심과 동공 위치
        let rightEyeCenter = centroid(rightEye.normalizedPoints)
        let rightPupilCenter = centroid(rightPupil.normalizedPoints)

        // 동공의 상대적 위치 계산 (눈 중심 기준)
        let leftPupilOffset = CGPoint(
            x: leftPupilCenter.x - leftEyeCenter.x,
            y: leftPupilCenter.y - leftEyeCenter.y
        )
        let rightPupilOffset = CGPoint(
            x: rightPupilCenter.x - rightEyeCenter.x,
            y: rightPupilCenter.y - rightEyeCenter.y
        )

        // 평균 offset
        let avgHorizontalOffset = Float((leftPupilOffset.x + rightPupilOffset.x) / 2.0)
        let avgVerticalOffset = Float((leftPupilOffset.y + rightPupilOffset.y) / 2.0)

        // Vision 좌표계: Y가 위로 갈수록 증가
        // offset이 음수 = 왼쪽/아래, 양수 = 오른쪽/위

        // 정규화된 각도 (-1.0 ~ 1.0)
        let horizontalAngle = avgHorizontalOffset * 10.0  // 스케일링
        let verticalAngle = avgVerticalOffset * 10.0

        // 시선 방향 분류
        let direction = classifyGazeDirection(
            horizontal: horizontalAngle,
            vertical: verticalAngle
        )

        // 신뢰도 계산 (간단히 거리 기반)
        let distance = sqrt(horizontalAngle * horizontalAngle + verticalAngle * verticalAngle)
        let confidence = max(0.0, min(1.0, 1.0 - distance / 2.0))

        return GazeResult(
            direction: direction,
            horizontalAngle: horizontalAngle,
            verticalAngle: verticalAngle,
            confidence: confidence
        )
    }

    /// 얼굴 각도 기반 시선 추정 (fallback)
    private func estimateGazeFromFaceAngles(yaw: Float?, pitch: Float?) -> GazeResult? {
        guard let yaw = yaw, let pitch = pitch else {
            return nil
        }

        // 라디안 → 정규화 (-1.0 ~ 1.0)
        let horizontalAngle = yaw * 2.0  // yaw ±0.5 rad 정도
        let verticalAngle = pitch * 2.0  // pitch ±0.5 rad 정도

        let direction = classifyGazeDirection(
            horizontal: horizontalAngle,
            vertical: verticalAngle
        )

        return GazeResult(
            direction: direction,
            horizontalAngle: horizontalAngle,
            verticalAngle: verticalAngle,
            confidence: 0.7  // 얼굴 각도 기반이므로 신뢰도 약간 낮음
        )
    }

    /// 시선 방향 분류
    private func classifyGazeDirection(horizontal: Float, vertical: Float) -> GazeDirection {
        // 카메라 응시 (중앙)
        if abs(horizontal) < cameraLookThreshold && abs(vertical) < cameraLookThreshold {
            return .lookingAtCamera
        }

        // 8방향 분류
        let isLeft = horizontal < -horizontalThreshold
        let isRight = horizontal > horizontalThreshold
        let isUp = vertical > verticalThreshold
        let isDown = vertical < -verticalThreshold

        if isLeft && isUp {
            return .lookingLeftUp
        } else if isLeft && isDown {
            return .lookingLeftDown
        } else if isRight && isUp {
            return .lookingRightUp
        } else if isRight && isDown {
            return .lookingRightDown
        } else if isLeft {
            return .lookingLeft
        } else if isRight {
            return .lookingRight
        } else if isUp {
            return .lookingUp
        } else if isDown {
            return .lookingDown
        } else {
            return .lookingAtCamera
        }
    }

    /// 포인트들의 중심점 계산
    private func centroid(_ points: [CGPoint]) -> CGPoint {
        guard !points.isEmpty else { return .zero }
        let sumX = points.reduce(0.0) { $0 + $1.x }
        let sumY = points.reduce(0.0) { $0 + $1.y }
        return CGPoint(x: sumX / CGFloat(points.count), y: sumY / CGFloat(points.count))
    }
}
