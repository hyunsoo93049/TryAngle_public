import Foundation
import Vision
import UIKit
import CoreImage
import Combine

// MARK: - 실시간 분석을 위한 데이터 구조
struct FrameAnalysis {
    let faceRect: CGRect?          // 얼굴 위치 (정규화된 좌표)
    let bodyRect: CGRect?          // 전신 추정 영역
    let brightness: Float          // 평균 밝기
    let tiltAngle: Float           // 기울기 각도
}

// MARK: - 실시간 피드백 생성기
class RealtimeAnalyzer: ObservableObject {
    @Published var instantFeedback: [FeedbackItem] = []
    @Published var isPerfect: Bool = false  // 완벽한 상태 감지
    @Published var perfectScore: Double = 0.0  // 완성도 점수 (0~1)

    private var referenceAnalysis: FrameAnalysis?
    private var lastAnalysisTime = Date()
    private let analysisInterval: TimeInterval = 0.1  // 100ms마다 분석

    // 히스테리시스를 위한 상태 추적
    private var feedbackHistory: [String: Int] = [:]  // 카테고리별 연속 감지 횟수
    private let historyThreshold = 3  // 3번 연속 감지되어야 표시
    private var perfectFrameCount = 0  // 완벽한 프레임 연속 횟수
    private let perfectThreshold = 10  // 10프레임(약 1초) 연속 완벽해야 감지

    // Vision 요청 캐싱
    private lazy var faceDetectionRequest: VNDetectFaceRectanglesRequest = {
        let request = VNDetectFaceRectanglesRequest()
        request.revision = VNDetectFaceRectanglesRequestRevision3
        return request
    }()

    // MARK: - 레퍼런스 이미지 분석
    func analyzeReference(_ image: UIImage) {
        guard let cgImage = image.cgImage else { return }

        // Vision 요청 실행
        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        try? handler.perform([faceDetectionRequest])

        // 얼굴 영역 추출
        let faceRect = faceDetectionRequest.results?.first?.boundingBox

        // 밝기 계산
        let brightness = calculateBrightness(cgImage)

        // 기울기 계산 (간단한 엣지 검출 기반)
        let tiltAngle = calculateTilt(cgImage)

        // 전신 영역 추정 (얼굴 기준)
        let bodyRect = estimateBodyRect(from: faceRect)

        referenceAnalysis = FrameAnalysis(
            faceRect: faceRect,
            bodyRect: bodyRect,
            brightness: brightness,
            tiltAngle: tiltAngle
        )

        print("📸 레퍼런스 분석 완료:")
        print("   - 얼굴: \(faceRect != nil ? "감지됨" : "없음")")
        print("   - 밝기: \(brightness)")
        print("   - 기울기: \(tiltAngle)도")
    }

    // MARK: - 실시간 프레임 분석
    func analyzeFrame(_ image: UIImage) {
        // 너무 자주 분석하지 않도록 제한
        guard Date().timeIntervalSince(lastAnalysisTime) >= analysisInterval else { return }
        guard let reference = referenceAnalysis else { return }
        guard let cgImage = image.cgImage else { return }

        lastAnalysisTime = Date()

        // 빠른 Vision 분석
        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        try? handler.perform([faceDetectionRequest])

        let currentFaceRect = faceDetectionRequest.results?.first?.boundingBox
        let currentBodyRect = estimateBodyRect(from: currentFaceRect)
        let currentTilt = calculateTilt(cgImage)

        var feedback: [FeedbackItem] = []

        // 1순위: 프레이밍 (줌 레벨) 피드백
        if let refBody = reference.bodyRect, let curBody = currentBodyRect {
            let refSize = refBody.width * refBody.height
            let curSize = curBody.width * curBody.height
            let sizeRatio = curSize / refSize

            let zoomDiff = (1.0 - sizeRatio) * 100

            if abs(zoomDiff) > 10 {  // 10% 이상 차이날 때만
                let direction = zoomDiff > 0 ? "줌 아웃" : "줌 인"
                feedback.append(FeedbackItem(
                    priority: 1,
                    icon: "🔍",
                    message: direction,
                    category: "zoom",
                    currentValue: Double(curSize * 100),
                    targetValue: Double(refSize * 100),
                    tolerance: 10.0,
                    unit: "%"
                ))
            }
        }

        // 2순위: 구도 (위치) 피드백
        if let refFace = reference.faceRect, let curFace = currentFaceRect {
            let xDiff = (curFace.midX - refFace.midX) * 100
            let yDiff = (curFace.midY - refFace.midY) * 100

            if abs(xDiff) > 5 {  // 5% 이상 차이
                let direction = xDiff > 0 ? "왼쪽으로" : "오른쪽으로"
                feedback.append(FeedbackItem(
                    priority: 2,
                    icon: "↔️",
                    message: "\(direction) 이동",
                    category: "position_x",
                    currentValue: Double(curFace.midX * 100),
                    targetValue: Double(refFace.midX * 100),
                    tolerance: 5.0,
                    unit: "%"
                ))
            }

            if abs(yDiff) > 5 {
                let direction = yDiff > 0 ? "아래로" : "위로"
                feedback.append(FeedbackItem(
                    priority: 2,
                    icon: "↕️",
                    message: "\(direction) 이동",
                    category: "position_y",
                    currentValue: Double(curFace.midY * 100),
                    targetValue: Double(refFace.midY * 100),
                    tolerance: 5.0,
                    unit: "%"
                ))
            }
        }

        // 3순위: 기울기 피드백
        let tiltDiff = currentTilt - reference.tiltAngle
        if abs(tiltDiff) > 3 {
            let direction = tiltDiff > 0 ? "왼쪽" : "오른쪽"
            feedback.append(FeedbackItem(
                priority: 3,
                icon: "📐",
                message: "\(direction)으로 회전",
                category: "tilt",
                currentValue: Double(currentTilt),
                targetValue: Double(reference.tiltAngle),
                tolerance: 3.0,
                unit: "도"
            ))
        }

        // 히스테리시스 적용: 연속으로 감지된 피드백만 표시
        var stableFeedback: [FeedbackItem] = []
        var currentCategories = Set<String>()

        for fb in feedback {
            currentCategories.insert(fb.category)
            feedbackHistory[fb.category, default: 0] += 1

            // 히스테리시스 임계값 넘으면 표시
            if feedbackHistory[fb.category]! >= historyThreshold {
                stableFeedback.append(fb)
            }
        }

        // 사라진 카테고리는 히스토리 초기화
        for (category, _) in feedbackHistory {
            if !currentCategories.contains(category) {
                feedbackHistory[category] = 0
            }
        }

        // 완벽한 상태 감지
        let score = calculatePerfectScore(feedback: feedback)
        let isCurrentlyPerfect = stableFeedback.isEmpty && score > 0.95

        if isCurrentlyPerfect {
            perfectFrameCount += 1
        } else {
            perfectFrameCount = 0
        }

        // 즉시 피드백 업데이트
        DispatchQueue.main.async {
            self.instantFeedback = stableFeedback
            self.perfectScore = score
            self.isPerfect = self.perfectFrameCount >= self.perfectThreshold
        }
    }

    // MARK: - Helper Functions

    private func calculatePerfectScore(feedback: [FeedbackItem]) -> Double {
        // 피드백이 없으면 완벽
        if feedback.isEmpty {
            return 1.0
        }

        // 각 피드백의 완성도 계산
        var totalScore = 0.0
        var count = 0

        for fb in feedback {
            if let current = fb.currentValue,
               let target = fb.targetValue {
                let diff = abs(current - target)
                let maxDiff = max(abs(target) + 50, 100.0)  // 최대 차이
                let itemScore = max(0.0, 1.0 - (diff / maxDiff))
                totalScore += itemScore
                count += 1
            }
        }

        if count == 0 {
            return 0.0
        }

        // 평균 점수
        return totalScore / Double(count)
    }

    private func calculateBrightness(_ cgImage: CGImage) -> Float {
        // 간단한 밝기 계산 (샘플링)
        let width = min(cgImage.width, 100)  // 샘플링으로 속도 향상
        let height = min(cgImage.height, 100)

        guard let context = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { return 0.5 }

        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        guard let data = context.data else { return 0.5 }

        let buffer = data.bindMemory(to: UInt8.self, capacity: width * height * 4)
        var totalBrightness: Float = 0

        for i in stride(from: 0, to: width * height * 4, by: 4) {
            let r = Float(buffer[i]) / 255.0
            let g = Float(buffer[i + 1]) / 255.0
            let b = Float(buffer[i + 2]) / 255.0
            totalBrightness += (r + g + b) / 3.0
        }

        return totalBrightness / Float(width * height)
    }

    private func calculateTilt(_ cgImage: CGImage) -> Float {
        // 간단한 기울기 추정 (엣지 검출 기반)
        // 실제로는 더 복잡한 알고리즘 필요하지만 속도를 위해 간단하게
        return 0.0  // TODO: 구현 필요
    }

    private func estimateBodyRect(from faceRect: CGRect?) -> CGRect? {
        // 얼굴 위치로부터 전신 영역 추정
        guard let face = faceRect else { return nil }

        // 일반적으로 얼굴이 전신의 1/7 정도
        let bodyWidth = face.width * 3
        let bodyHeight = face.height * 7
        let bodyX = face.midX - bodyWidth / 2
        let bodyY = face.minY  // 얼굴 아래로 확장

        return CGRect(x: bodyX, y: bodyY, width: bodyWidth, height: bodyHeight)
    }
}