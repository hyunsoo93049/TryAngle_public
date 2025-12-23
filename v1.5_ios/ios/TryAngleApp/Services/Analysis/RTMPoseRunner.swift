import Foundation
import UIKit
import CoreGraphics
import Accelerate

// MARK: - RTMPose 결과 구조체
struct RTMPoseResult {
    let keypoints: [(point: CGPoint, confidence: Float)]  // 133개 키포인트
    let boundingBox: CGRect?  // 인물 검출 박스
}

// MARK: - RTMPose Runner (ONNX Runtime Objective-C API)
class RTMPoseRunner {

    private var detectorSession: ORTSession?
    private var poseSession: ORTSession?
    private var env: ORTEnv?

    // 모델 경로
    private let detectorModelPath: String
    private let poseModelPath: String

    // 모델 입력 크기
    private let detectorInputSize = CGSize(width: 640, height: 640)
    private let poseInputSize = CGSize(width: 192, height: 256)

    init?() {
        print("🚀 RTMPoseRunner init() 시작")

        // ONNX format 모델 사용 (전체 ONNX Runtime 사용)
        guard let detectorURL = Bundle.main.url(forResource: "yolox_int8", withExtension: "onnx") else {
            print("❌ yolox_int8.onnx 파일을 찾을 수 없습니다")
            print("   Bundle path: \(Bundle.main.bundlePath)")
            print("   Bundle resources: \(Bundle.main.paths(forResourcesOfType: "onnx", inDirectory: nil))")
            return nil
        }

        guard let poseURL = Bundle.main.url(forResource: "rtmpose_int8", withExtension: "onnx") else {
            print("❌ rtmpose_int8.onnx 파일을 찾을 수 없습니다")
            print("   Bundle path: \(Bundle.main.bundlePath)")
            print("   Bundle resources: \(Bundle.main.paths(forResourcesOfType: "onnx", inDirectory: nil))")
            return nil
        }

        detectorModelPath = detectorURL.path
        poseModelPath = poseURL.path

        print("✅ ONNX 모델 경로 확인 (전체 Runtime 사용):")
        print("   Detector (YOLOX): \(detectorModelPath)")
        print("   Pose (RTMPose): \(poseModelPath)")

        setupONNXRuntime()
    }

    deinit {
        print("🗑️ RTMPoseRunner deinit")
    }

    // MARK: - ONNX Runtime 초기화
    private func setupONNXRuntime() {
        print("🔧 ONNX Runtime 초기화 시작...")

        do {
            // 1. Environment 생성
            env = try ORTEnv(loggingLevel: ORTLoggingLevel.warning)
            print("✅ Environment 생성 성공")

            // 2. YOLOX용 Session Options (CoreML GPU 가속)
            let detectorOptions = try ORTSessionOptions()

            // 🔥 YOLOX도 CoreML GPU 가속 활성화
            do {
                try detectorOptions.appendCoreMLExecutionProvider()
                print("✅ YOLOX: CoreML GPU 가속 활성화")
            } catch {
                print("⚠️ YOLOX CoreML 활성화 실패, CPU 폴백: \(error)")
            }

            try detectorOptions.setIntraOpNumThreads(6)  // 병렬 처리
            try detectorOptions.setGraphOptimizationLevel(.all)

            // 3. RTMPose용 Session Options (CoreML GPU 가속)
            let poseOptions = try ORTSessionOptions()

            // 🔥 CoreML Execution Provider 활성화 (GPU 가속)
            do {
                try poseOptions.appendCoreMLExecutionProvider()
                print("✅ RTMPose: CoreML GPU 가속 활성화")
            } catch {
                print("⚠️ RTMPose CoreML 활성화 실패, CPU 폴백: \(error)")
            }

            // 병렬 처리 설정 (최대 성능)
            try poseOptions.setIntraOpNumThreads(6)  // 🔥 스레드 6개로 증가
            try poseOptions.setGraphOptimizationLevel(.all)

            print("✅ 최대 성능 최적화 설정 완료 (YOLOX: CoreML GPU, RTMPose: CoreML GPU)")

            // 4. 세션 생성
            print("📦 Detector 모델 로딩 중... (\(detectorModelPath))")
            detectorSession = try ORTSession(env: env!, modelPath: detectorModelPath, sessionOptions: detectorOptions)
            print("✅ YOLOX Detector 로드 성공 (CoreML GPU)")

            print("📦 Pose 모델 로딩 중... (\(poseModelPath))")
            poseSession = try ORTSession(env: env!, modelPath: poseModelPath, sessionOptions: poseOptions)
            print("✅ RTMPose 로드 성공 (CoreML GPU)")

            print("🔧 ONNX Runtime 초기화 완료")

        } catch {
            print("❌ ONNX Runtime 초기화 실패: \(error)")
            env = nil
            detectorSession = nil
            poseSession = nil
        }
    }

    // MARK: - 포즈 추정
    func detectPose(from image: UIImage) -> RTMPoseResult? {
        guard let detectorSession = detectorSession,
              let poseSession = poseSession,
              let env = env else {
            print("❌ RTMPose 세션이 초기화되지 않음")
            return nil
        }

        // 1. YOLOX로 사람 검출
        let boundingBox: CGRect
        if let detectedBox = detectPerson(from: image, using: detectorSession, env: env) {
            print("✅ YOLOX: 사람 검출 성공 - \(detectedBox)")
            boundingBox = detectedBox
        } else {
            // YOLOX가 사람을 검출하지 못하면 전체 이미지 사용
            print("⚠️ YOLOX: 사람을 검출하지 못함 → 전체 이미지로 포즈 추정 시도")
            guard let cgImage = image.cgImage else { return nil }
            boundingBox = CGRect(x: 0, y: 0, width: cgImage.width, height: cgImage.height)
        }

        // 2. 검출된 영역으로 포즈 추정
        let keypoints = estimatePose(from: image, boundingBox: boundingBox, using: poseSession, env: env)

        if let keypoints = keypoints {
            print("✅ RTMPose: \(keypoints.count)개 키포인트 검출 성공")
        } else {
            print("❌ RTMPose: 포즈 추정 실패")
        }

        return keypoints.map { RTMPoseResult(keypoints: $0, boundingBox: boundingBox) }
    }

    // MARK: - YOLOX 사람 검출
    private func detectPerson(from image: UIImage, using session: ORTSession, env: ORTEnv) -> CGRect? {
        guard let cgImage = image.cgImage else { return nil }

        // 이미지를 640x640으로 리사이즈
        let inputSize = detectorInputSize
        guard let resizedImage = resizeImage(cgImage, targetSize: inputSize) else { return nil }

        // 이미지를 Float 배열로 변환 (RGB, 정규화)
        let pixelData = preprocessImage(resizedImage, size: inputSize)

        do {
            // 입력 텐서 생성 - [1, 3, 640, 640]
            let inputShape: [NSNumber] = [1, 3, NSNumber(value: Int(inputSize.height)), NSNumber(value: Int(inputSize.width))]
            let inputTensor = try ORTValue(
                tensorData: NSMutableData(data: pixelData),
                elementType: .float,
                shape: inputShape
            )

            // 추론 실행
            let outputs = try session.run(
                withInputs: ["input": inputTensor],
                outputNames: ["dets", "labels"],
                runOptions: nil
            )

            guard let detsTensor = outputs["dets"],
                  let labelsTensor = outputs["labels"] else {
                print("❌ YOLOX 출력을 찾을 수 없음")
                return nil
            }

            // 출력 파싱하여 바운딩 박스 추출
            return parseYOLOXOutput(detsTensor, labels: labelsTensor, imageSize: CGSize(width: cgImage.width, height: cgImage.height))

        } catch {
            print("❌ YOLOX 추론 오류: \(error)")
            return nil
        }
    }

    // MARK: - RTMPose 포즈 추정
    private func estimatePose(from image: UIImage, boundingBox: CGRect, using session: ORTSession, env: ORTEnv) -> [(point: CGPoint, confidence: Float)]? {
        guard let cgImage = image.cgImage else { return nil }

        // 바운딩 박스 영역 크롭
        guard let croppedImage = cropImage(cgImage, rect: boundingBox) else { return nil }

        // 192x256으로 리사이즈
        let inputSize = poseInputSize
        guard let resizedImage = resizeImage(croppedImage, targetSize: inputSize) else { return nil }

        // 이미지를 Float 배열로 변환
        let pixelData = preprocessImage(resizedImage, size: inputSize)

        do {
            // 입력 텐서 생성 - [1, 3, 256, 192]
            let inputShape: [NSNumber] = [1, 3, NSNumber(value: Int(inputSize.height)), NSNumber(value: Int(inputSize.width))]
            let inputTensor = try ORTValue(
                tensorData: NSMutableData(data: pixelData),
                elementType: .float,
                shape: inputShape
            )

            // 추론 실행
            let outputs = try session.run(
                withInputs: ["input": inputTensor],
                outputNames: ["simcc_x", "simcc_y"],
                runOptions: nil
            )

            guard let simccX = outputs["simcc_x"],
                  let simccY = outputs["simcc_y"] else {
                print("❌ RTMPose 출력(SimCC)을 찾을 수 없음")
                return nil
            }

            // SimCC 출력 파싱하여 키포인트 추출 (133개)
            return parseRTMPoseSimCCOutput(simccX: simccX, simccY: simccY, boundingBox: boundingBox)

        } catch {
            print("❌ RTMPose 추론 오류: \(error)")
            return nil
        }
    }

    // MARK: - 이미지 전처리 헬퍼 함수들
    private func resizeImage(_ cgImage: CGImage, targetSize: CGSize) -> CGImage? {
        let width = Int(targetSize.width)
        let height = Int(targetSize.height)

        guard let context = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
        ) else { return nil }

        context.interpolationQuality = .high
        context.draw(cgImage, in: CGRect(origin: .zero, size: targetSize))
        return context.makeImage()
    }

    private func cropImage(_ cgImage: CGImage, rect: CGRect) -> CGImage? {
        // 바운딩 박스를 충분히 확장 (손이 포함되도록 패딩 증가)
        // 🔥 손 인식 개선: 패딩을 0.2에서 0.4로 증가
        let padding: CGFloat = 0.4  // 40% 패딩으로 손까지 포함
        let expandedRect = CGRect(
            x: rect.minX - rect.width * padding,
            y: rect.minY - rect.height * padding,
            width: rect.width * (1 + 2 * padding),
            height: rect.height * (1 + 2 * padding)
        ).intersection(CGRect(x: 0, y: 0, width: cgImage.width, height: cgImage.height))

        return cgImage.cropping(to: expandedRect)
    }

    private func preprocessImage(_ cgImage: CGImage, size: CGSize) -> Data {
        let width = Int(size.width)
        let height = Int(size.height)
        let bytesPerPixel = 4
        let bytesPerRow = bytesPerPixel * width
        let bitsPerComponent = 8

        var rawData = [UInt8](repeating: 0, count: width * height * bytesPerPixel)

        guard let context = CGContext(
            data: &rawData,
            width: width,
            height: height,
            bitsPerComponent: bitsPerComponent,
            bytesPerRow: bytesPerRow,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
        ) else {
            return Data()
        }

        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))

        // RGBA를 RGB로 변환하고 정규화 (ImageNet 평균/표준편차)
        var floatData = [Float](repeating: 0, count: width * height * 3)
        let mean: [Float] = [0.485, 0.456, 0.406]
        let std: [Float] = [0.229, 0.224, 0.225]

        for y in 0..<height {
            for x in 0..<width {
                let pixelIndex = y * width + x
                let dataIndex = pixelIndex * 4

                // RGB 채널별로 정규화
                for c in 0..<3 {
                    let value = Float(rawData[dataIndex + c]) / 255.0
                    floatData[c * width * height + pixelIndex] = (value - mean[c]) / std[c]
                }
            }
        }

        return Data(bytes: &floatData, count: floatData.count * MemoryLayout<Float>.size)
    }

    // MARK: - 출력 파싱
    private func parseYOLOXOutput(_ dets: ORTValue, labels: ORTValue, imageSize: CGSize) -> CGRect? {
        // YOLOX 출력 형식:
        // dets: [1, num_boxes, 5] - (x1, y1, x2, y2, score)
        // labels: [1, num_boxes] - class_id

        guard let detsData = try? dets.tensorData() as NSData,
              let labelsData = try? labels.tensorData() as NSData else { return nil }
        guard let detsShape = try? dets.tensorTypeAndShapeInfo().shape else { return nil }

        let numBoxes = detsShape[1].intValue
        if numBoxes == 0 {
            print("⚠️ YOLOX: 검출된 박스 없음")
            return nil
        }

        var bestBox: CGRect?
        var bestScore: Float = 0.3  // 최소 임계값

        let detsPointer = detsData.bytes.bindMemory(to: Float.self, capacity: detsData.length / MemoryLayout<Float>.size)
        let labelsPointer = labelsData.bytes.bindMemory(to: Int64.self, capacity: labelsData.length / MemoryLayout<Int64>.size)

        // 640x640 좌표를 원본 이미지 좌표로 변환
        let scaleX = imageSize.width / detectorInputSize.width
        let scaleY = imageSize.height / detectorInputSize.height

        for i in 0..<numBoxes {
            let label = labelsPointer[i]
            // person class = 0
            guard label == 0 else { continue }

            let offset = i * 5
            let x1 = CGFloat(detsPointer[offset + 0]) * scaleX
            let y1 = CGFloat(detsPointer[offset + 1]) * scaleY
            let x2 = CGFloat(detsPointer[offset + 2]) * scaleX
            let y2 = CGFloat(detsPointer[offset + 3]) * scaleY
            let score = detsPointer[offset + 4]

            if score > bestScore {
                bestBox = CGRect(
                    x: x1,
                    y: y1,
                    width: x2 - x1,
                    height: y2 - y1
                )
                bestScore = score
            }
        }

        return bestBox
    }

    private func parseRTMPoseSimCCOutput(simccX: ORTValue, simccY: ORTValue, boundingBox: CGRect) -> [(point: CGPoint, confidence: Float)]? {
        // SimCC 출력 형식:
        // simcc_x: [1, num_keypoints, 384] - x 좌표 확률 분포
        // simcc_y: [1, num_keypoints, 512] - y 좌표 확률 분포

        guard let xData = try? simccX.tensorData() as NSData,
              let yData = try? simccY.tensorData() as NSData else { return nil }
        guard let xShape = try? simccX.tensorTypeAndShapeInfo().shape,
              let yShape = try? simccY.tensorTypeAndShapeInfo().shape else { return nil }

        let numKeypoints = xShape[1].intValue
        let xBins = xShape[2].intValue  // 384
        let yBins = yShape[2].intValue  // 512

        if numKeypoints != 133 {
            print("⚠️ 예상치 못한 키포인트 수: \(numKeypoints)")
            return nil
        }

        var keypoints: [(point: CGPoint, confidence: Float)] = []
        let xPointer = xData.bytes.bindMemory(to: Float.self, capacity: xData.length / MemoryLayout<Float>.size)
        let yPointer = yData.bytes.bindMemory(to: Float.self, capacity: yData.length / MemoryLayout<Float>.size)

        for i in 0..<numKeypoints {
            // x 좌표: argmax 찾기
            let xOffset = i * xBins
            var maxXIdx = 0
            var maxXVal: Float = -Float.infinity
            for j in 0..<xBins {
                let val = xPointer[xOffset + j]
                if val > maxXVal {
                    maxXVal = val
                    maxXIdx = j
                }
            }

            // y 좌표: argmax 찾기
            let yOffset = i * yBins
            var maxYIdx = 0
            var maxYVal: Float = -Float.infinity
            for j in 0..<yBins {
                let val = yPointer[yOffset + j]
                if val > maxYVal {
                    maxYVal = val
                    maxYIdx = j
                }
            }

            // SimCC 좌표를 픽셀 좌표로 변환
            // 384 bins -> 192 pixels, 512 bins -> 256 pixels (각각 2배 해상도)
            let xNorm = CGFloat(maxXIdx) / CGFloat(xBins) * poseInputSize.width
            let yNorm = CGFloat(maxYIdx) / CGFloat(yBins) * poseInputSize.height

            // 바운딩 박스 기준으로 변환
            let point = CGPoint(
                x: boundingBox.minX + (xNorm / poseInputSize.width) * boundingBox.width,
                y: boundingBox.minY + (yNorm / poseInputSize.height) * boundingBox.height
            )

            // 신뢰도: 두 확률의 평균
            let confidence = (maxXVal + maxYVal) / 2.0

            keypoints.append((point: point, confidence: confidence))

            // 🔍 손 키포인트 디버그 (91-132번)
            if i >= 91 && i <= 132 {
                if confidence < 0.3 {
                    let handName = i <= 111 ? "왼손" : "오른손"
                    let keypointIndex = i <= 111 ? i - 91 : i - 112
                    if keypointIndex % 5 == 0 {  // 5개마다 한 번만 로그
                        print("⚠️ \(handName) 키포인트 \(keypointIndex): 신뢰도 낮음 (\(String(format: "%.2f", confidence)))")
                    }
                }
            }
        }

        // 손 키포인트 요약 통계
        let leftHandConfidences = (91...111).compactMap { keypoints[$0].confidence }
        let rightHandConfidences = (112...132).compactMap { keypoints[$0].confidence }

        let leftHandAvg = leftHandConfidences.reduce(0, +) / Float(leftHandConfidences.count)
        let rightHandAvg = rightHandConfidences.reduce(0, +) / Float(rightHandConfidences.count)

        if leftHandAvg < 0.5 || rightHandAvg < 0.5 {
            print("📊 손 인식 평균 신뢰도 - 왼손: \(String(format: "%.2f", leftHandAvg)), 오른손: \(String(format: "%.2f", rightHandAvg))")
            if leftHandAvg < 0.3 || rightHandAvg < 0.3 {
                print("💡 손이 화면에서 잘렸거나 가려졌을 수 있습니다. 전체 신체가 프레임 안에 들어오도록 조정해보세요.")
            }
        }

        return keypoints
    }
}
