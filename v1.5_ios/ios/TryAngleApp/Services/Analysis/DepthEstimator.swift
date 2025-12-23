import Foundation
import ARKit
import AVFoundation
import CoreImage

// MARK: - 거리 측정 방법
enum DepthMethod {
    case lidar              // LiDAR 센서 (iPhone 12 Pro 이상)
    case arkit              // ARKit depth estimation
    case faceSize           // 얼굴 크기 기반 추정 (fallback)
    case unavailable        // 사용 불가
}

// MARK: - 거리 측정 결과
struct DepthResult {
    let distance: Float?              // 측정된 거리 (미터)
    let method: DepthMethod           // 사용된 측정 방법
    let confidence: Float             // 신뢰도 (0 ~ 1)
    let isZoomDetected: Bool          // 줌 사용 감지 여부
    let zoomFactor: Float?            // 줌 배율
}

// MARK: - 깊이 추정기
class DepthEstimator {

    // 평균 얼굴 폭 (미터) - 통계적 평균
    private let averageFaceWidth: Float = 0.14  // 14cm

    // LiDAR 사용 가능 여부
    private var isLiDARAvailable: Bool {
        if #available(iOS 14.0, *) {
            return ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth)
        }
        return false
    }

    /// 얼굴까지의 거리 추정
    /// - Parameters:
    ///   - faceRect: 얼굴 영역 (정규화 좌표)
    ///   - imageWidth: 이미지 너비 (픽셀)
    ///   - zoomFactor: 현재 줌 배율
    ///   - depthData: ARKit/LiDAR depth data (optional)
    /// - Returns: 거리 측정 결과
    func estimateDistance(
        faceRect: CGRect,
        imageWidth: Int,
        zoomFactor: Float = 1.0,
        depthData: AVDepthData? = nil
    ) -> DepthResult {

        // 방법 1: LiDAR/ARKit depth data 사용 (가장 정확)
        if let depthData = depthData {
            return estimateFromDepthData(
                depthData: depthData,
                faceRect: faceRect,
                zoomFactor: zoomFactor
            )
        }

        // 방법 2: 얼굴 크기 기반 추정 (fallback)
        return estimateFromFaceSize(
            faceRect: faceRect,
            imageWidth: imageWidth,
            zoomFactor: zoomFactor
        )
    }

    /// 거리 변화 감지 (줌 vs 실제 이동)
    /// - Parameters:
    ///   - referenceResult: 레퍼런스 거리 결과
    ///   - currentResult: 현재 거리 결과
    /// - Returns: (isZoomChange: Bool, isDistanceChange: Bool)
    func detectDistanceChange(
        reference: DepthResult,
        current: DepthResult
    ) -> (isZoomChange: Bool, isDistanceChange: Bool) {

        // 줌 배율 변화 감지
        var isZoomChange = false
        if let refZoom = reference.zoomFactor,
           let curZoom = current.zoomFactor {
            isZoomChange = abs(refZoom - curZoom) > 0.1  // 10% 이상 차이
        }

        // 실제 거리 변화 감지
        var isDistanceChange = false
        if let refDist = reference.distance,
           let curDist = current.distance {
            let distDiff = abs(refDist - curDist)
            isDistanceChange = distDiff > 0.2  // 20cm 이상 차이
        }

        return (isZoomChange, isDistanceChange)
    }

    /// 거리 조정 피드백 생성
    /// - Parameters:
    ///   - reference: 레퍼런스 거리
    ///   - current: 현재 거리
    /// - Returns: 피드백 메시지
    func generateDistanceFeedback(
        reference: DepthResult,
        current: DepthResult
    ) -> (message: String, shouldUseZoom: Bool)? {

        guard let refDist = reference.distance,
              let curDist = current.distance else {
            return nil
        }

        let distDiff = curDist - refDist  // 양수 = 현재가 더 멀리

        // 거리 차이가 20cm 미만이면 피드백 없음
        if abs(distDiff) < 0.2 {
            return nil
        }

        // 줌 사용 여부 판단
        let (isZoomChange, _) = detectDistanceChange(
            reference: reference,
            current: current
        )

        // Case 1: 줌으로 조정하는 중
        if isZoomChange {
            if distDiff > 0 {
                return ("줌 인하여 가까이", true)
            } else {
                return ("줌 아웃하여 멀리", true)
            }
        }

        // Case 2: 실제 이동 필요
        // 거리 범위 제한 (0.5m ~ 5m) - 비현실적인 값 방지
        let clampedCurDist = max(0.5, min(5.0, curDist))
        let clampedDiff = max(0.2, min(3.0, abs(distDiff)))  // 최대 3m 차이로 제한

        // 거리에 따른 구체적인 안내
        if clampedDiff < 0.5 {
            // 작은 거리 차이 - cm 단위로 표시
            let cm = Int(clampedDiff * 100)
            if distDiff > 0 {
                return ("약 \(cm)cm 앞으로 이동하세요", false)
            } else {
                return ("약 \(cm)cm 뒤로 이동하세요", false)
            }
        } else {
            // 큰 거리 차이 - 걸음 수로 표시
            let steps = max(1, Int(round(clampedDiff / 0.7)))  // 0.7m per step
            if distDiff > 0 {
                return ("\(steps)걸음 앞으로 이동하세요", false)
            } else {
                return ("\(steps)걸음 뒤로 이동하세요", false)
            }
        }
    }

    // MARK: - Private Methods

    /// LiDAR/ARKit depth data로부터 거리 추정
    private func estimateFromDepthData(
        depthData: AVDepthData,
        faceRect: CGRect,
        zoomFactor: Float
    ) -> DepthResult {

        // Depth map 추출
        let depthMap = depthData.depthDataMap

        // 얼굴 영역의 중심점
        let centerX = Int((faceRect.midX) * CGFloat(CVPixelBufferGetWidth(depthMap)))
        let centerY = Int((faceRect.midY) * CGFloat(CVPixelBufferGetHeight(depthMap)))

        // Depth 값 샘플링 (얼굴 중심 주변)
        var depthValues: [Float] = []
        let sampleRadius = 5

        CVPixelBufferLockBaseAddress(depthMap, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(depthMap, .readOnly) }

        let baseAddress = CVPixelBufferGetBaseAddress(depthMap)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(depthMap)
        let width = CVPixelBufferGetWidth(depthMap)
        let height = CVPixelBufferGetHeight(depthMap)

        for dy in -sampleRadius...sampleRadius {
            for dx in -sampleRadius...sampleRadius {
                let x = centerX + dx
                let y = centerY + dy

                if x >= 0 && x < width && y >= 0 && y < height {
                    let offset = y * bytesPerRow + x * MemoryLayout<Float32>.size
                    if let baseAddress = baseAddress {
                        let depthValue = baseAddress.load(fromByteOffset: offset, as: Float32.self)
                        if depthValue > 0 && depthValue < 10.0 {  // 유효 범위
                            depthValues.append(depthValue)
                        }
                    }
                }
            }
        }

        // 중앙값 계산 (노이즈 제거)
        if !depthValues.isEmpty {
            depthValues.sort()
            let medianDistance = depthValues[depthValues.count / 2]

            // 줌 감지 (depth data가 있으면 정확히 판단 가능)
            let isZoomDetected = zoomFactor > 1.05

            return DepthResult(
                distance: medianDistance,
                method: isLiDARAvailable ? .lidar : .arkit,
                confidence: 0.95,
                isZoomDetected: isZoomDetected,
                zoomFactor: zoomFactor
            )
        }

        // Depth data 실패 시 fallback
        return DepthResult(
            distance: nil,
            method: .unavailable,
            confidence: 0.0,
            isZoomDetected: false,
            zoomFactor: nil
        )
    }

    /// 얼굴 크기 기반 거리 추정 (fallback)
    private func estimateFromFaceSize(
        faceRect: CGRect,
        imageWidth: Int,
        zoomFactor: Float
    ) -> DepthResult {

        // 얼굴 너비 (픽셀)
        let faceWidthPixels = faceRect.width * CGFloat(imageWidth)

        // 카메라 FOV 추정 (iPhone 평균: 약 60도)
        let fovDegrees: Float = 60.0
        let fovRadians = fovDegrees * .pi / 180.0

        // 거리 계산 (핀홀 카메라 모델)
        // distance = (realWidth * imageWidth) / (2 * faceWidthPixels * tan(fov/2))
        let distance = (averageFaceWidth * Float(imageWidth)) /
                       (2.0 * Float(faceWidthPixels) * tan(fovRadians / 2.0))

        // 줌 보정 (줌하면 거리가 더 멀게 추정되므로 보정)
        let correctedDistance = distance / zoomFactor

        // 줌 사용 감지 (크기는 같은데 줌이 1이 아니면 줌 사용)
        let isZoomDetected = zoomFactor > 1.05

        return DepthResult(
            distance: correctedDistance,
            method: .faceSize,
            confidence: 0.6,  // 추정이므로 신뢰도 낮음
            isZoomDetected: isZoomDetected,
            zoomFactor: zoomFactor
        )
    }

    /// LiDAR 사용 가능 여부 확인
    func checkLiDARAvailability() -> Bool {
        return isLiDARAvailable
    }

    /// ARKit depth data 가용성 확인
    func checkARKitDepthAvailability() -> Bool {
        if #available(iOS 13.0, *) {
            return ARWorldTrackingConfiguration.supportsFrameSemantics(.personSegmentationWithDepth)
        }
        return false
    }
}
