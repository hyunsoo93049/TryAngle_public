import Foundation
import Vision
import UIKit
import CoreGraphics

// MARK: - 얼굴 분석 결과
struct FaceAnalysisResult {
    let faceRect: CGRect                        // 얼굴 영역
    let landmarks: VNFaceLandmarks2D?           // 랜드마크
    let yaw: Float?                             // 좌우 회전
    let pitch: Float?                           // 상하 각도
    let roll: Float?                            // 기울기
    let observation: VNFaceObservation          // 원본 관찰 결과
}

// MARK: - 포즈 분석 결과
struct PoseAnalysisResult {
    let keypoints: [(point: CGPoint, confidence: Float)]  // 17개 키포인트
    let observation: VNHumanBodyPoseObservation?          // 원본 관찰 결과 (Vision일 때만)

    // Vision용 initializer
    init(observation: VNHumanBodyPoseObservation, keypoints: [(point: CGPoint, confidence: Float)]) {
        self.observation = observation
        self.keypoints = keypoints
    }

    // YOLO/MoveNet용 initializer (observation 없음)
    init(keypoints: [(point: CGPoint, confidence: Float)]) {
        self.observation = nil
        self.keypoints = keypoints
    }
}

// MARK: - Vision 분석기
class VisionAnalyzer {

    // Vision 요청 캐싱 (재사용)
    private lazy var faceDetectionRequest: VNDetectFaceLandmarksRequest = {
        let request = VNDetectFaceLandmarksRequest()
        request.revision = VNDetectFaceLandmarksRequestRevision3
        return request
    }()

    private lazy var poseDetectionRequest: VNDetectHumanBodyPoseRequest = {
        let request = VNDetectHumanBodyPoseRequest()
        return request
    }()

    /// 얼굴 감지 및 분석
    /// - Parameter image: 입력 이미지
    /// - Returns: 얼굴 분석 결과
    func detectFace(from image: UIImage) -> FaceAnalysisResult? {
        guard let cgImage = image.cgImage else { return nil }

        // 이미지 orientation을 Vision에 전달하여 좌표계 보정
        let handler = VNImageRequestHandler(
            cgImage: cgImage,
            orientation: image.cgImageOrientation,
            options: [:]
        )
        try? handler.perform([faceDetectionRequest])

        guard let observation = faceDetectionRequest.results?.first else {
            return nil
        }

        return FaceAnalysisResult(
            faceRect: observation.boundingBox,
            landmarks: observation.landmarks,
            yaw: observation.yaw?.floatValue,
            pitch: observation.pitch?.floatValue,
            roll: observation.roll?.floatValue,
            observation: observation
        )
    }

    /// 포즈 감지 및 분석
    /// - Parameter image: 입력 이미지
    /// - Returns: 포즈 분석 결과
    func detectPose(from image: UIImage) -> PoseAnalysisResult? {
        guard let cgImage = image.cgImage else { return nil }

        // 이미지 orientation을 Vision에 전달하여 좌표계 보정
        let handler = VNImageRequestHandler(
            cgImage: cgImage,
            orientation: image.cgImageOrientation,
            options: [:]
        )
        try? handler.perform([poseDetectionRequest])

        guard let observation = poseDetectionRequest.results?.first else {
            return nil
        }

        let keypoints = extractKeypoints(from: observation)

        return PoseAnalysisResult(
            observation: observation,
            keypoints: keypoints
        )
    }

    /// 얼굴 + 포즈 동시 분석 (최적화)
    /// - Parameter image: 입력 이미지
    /// - Returns: (얼굴 결과, 포즈 결과)
    func analyzeFaceAndPose(from image: UIImage) -> (face: FaceAnalysisResult?, pose: PoseAnalysisResult?) {
        guard let cgImage = image.cgImage else {
            return (nil, nil)
        }

        // 두 요청을 동시에 수행 (효율적)
        // 이미지 orientation을 Vision에 전달하여 좌표계 보정
        let handler = VNImageRequestHandler(
            cgImage: cgImage,
            orientation: image.cgImageOrientation,
            options: [:]
        )
        try? handler.perform([faceDetectionRequest, poseDetectionRequest])

        // 얼굴 결과
        var faceResult: FaceAnalysisResult? = nil
        if let faceObservation = faceDetectionRequest.results?.first {
            faceResult = FaceAnalysisResult(
                faceRect: faceObservation.boundingBox,
                landmarks: faceObservation.landmarks,
                yaw: faceObservation.yaw?.floatValue,
                pitch: faceObservation.pitch?.floatValue,
                roll: faceObservation.roll?.floatValue,
                observation: faceObservation
            )
        }

        // 포즈 결과
        var poseResult: PoseAnalysisResult? = nil
        if let poseObservation = poseDetectionRequest.results?.first {
            let keypoints = extractKeypoints(from: poseObservation)
            poseResult = PoseAnalysisResult(
                observation: poseObservation,
                keypoints: keypoints
            )
        }

        return (faceResult, poseResult)
    }

    /// 전신 영역 추정 (얼굴 기준)
    /// - Parameter faceRect: 얼굴 영역
    /// - Returns: 추정된 전신 영역
    func estimateBodyRect(from faceRect: CGRect?) -> CGRect? {
        guard let face = faceRect else { return nil }

        // 일반적으로 얼굴이 전신의 1/7 정도
        let bodyWidth = face.width * 3
        let bodyHeight = face.height * 7
        let bodyX = face.midX - bodyWidth / 2
        let bodyY = face.minY  // 얼굴 아래로 확장

        return CGRect(x: bodyX, y: bodyY, width: bodyWidth, height: bodyHeight)
    }

    /// 밝기 계산 (샘플링)
    /// - Parameter cgImage: 입력 이미지
    /// - Returns: 평균 밝기 (0~1)
    func calculateBrightness(from cgImage: CGImage) -> Float {
        // 샘플링으로 속도 향상
        let width = min(cgImage.width, 100)
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

    // MARK: - Private Helpers

    /// 키포인트 추출 (17개)
    private func extractKeypoints(from observation: VNHumanBodyPoseObservation) -> [(point: CGPoint, confidence: Float)] {
        var keypoints: [(point: CGPoint, confidence: Float)] = []

        let jointNames: [VNHumanBodyPoseObservation.JointName] = [
            .nose,           // 0: 코
            .leftEye,        // 1: 왼쪽 눈
            .rightEye,       // 2: 오른쪽 눈
            .leftEar,        // 3: 왼쪽 귀
            .rightEar,       // 4: 오른쪽 귀
            .leftShoulder,   // 5: 왼쪽 어깨
            .rightShoulder,  // 6: 오른쪽 어깨
            .leftElbow,      // 7: 왼쪽 팔꿈치
            .rightElbow,     // 8: 오른쪽 팔꿈치
            .leftWrist,      // 9: 왼쪽 손목
            .rightWrist,     // 10: 오른쪽 손목
            .leftHip,        // 11: 왼쪽 골반
            .rightHip,       // 12: 오른쪽 골반
            .leftKnee,       // 13: 왼쪽 무릎
            .rightKnee,      // 14: 오른쪽 무릎
            .leftAnkle,      // 15: 왼쪽 발목
            .rightAnkle      // 16: 오른쪽 발목
        ]

        for jointName in jointNames {
            if let point = try? observation.recognizedPoint(jointName) {
                keypoints.append((point: point.location, confidence: point.confidence))
            } else {
                keypoints.append((point: .zero, confidence: 0.0))
            }
        }

        return keypoints
    }
}
