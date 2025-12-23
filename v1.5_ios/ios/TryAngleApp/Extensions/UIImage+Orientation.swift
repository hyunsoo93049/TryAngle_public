import UIKit
import ImageIO

extension UIImage {
    /// UIImage.Orientation을 CGImagePropertyOrientation으로 변환
    var cgImageOrientation: CGImagePropertyOrientation {
        switch imageOrientation {
        case .up: return .up
        case .down: return .down
        case .left: return .left
        case .right: return .right
        case .upMirrored: return .upMirrored
        case .downMirrored: return .downMirrored
        case .leftMirrored: return .leftMirrored
        case .rightMirrored: return .rightMirrored
        @unknown default: return .up
        }
    }

    /// 파일에 로그 기록 (디버깅용)
    private func logRotation(_ message: String) {
        if let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first {
            let logFile = documentsPath.appendingPathComponent("rotation_debug.txt")
            let timestamp = DateFormatter.localizedString(from: Date(), dateStyle: .none, timeStyle: .medium)
            let logMessage = "[\(timestamp)] \(message)\n"

            if let data = logMessage.data(using: .utf8) {
                if FileManager.default.fileExists(atPath: logFile.path) {
                    if let fileHandle = try? FileHandle(forWritingTo: logFile) {
                        fileHandle.seekToEndOfFile()
                        fileHandle.write(data)
                        fileHandle.closeFile()
                    }
                } else {
                    try? data.write(to: logFile)
                }
            }
        }
        print(message)
    }

    /// 이미지 방향을 올바르게 수정
    func fixedOrientation() -> UIImage {
        let orientationName: String
        switch imageOrientation {
        case .up: orientationName = "up(0)"
        case .down: orientationName = "down(1)"
        case .left: orientationName = "left(2)"
        case .right: orientationName = "right(3)"
        case .upMirrored: orientationName = "upMirrored(4)"
        case .downMirrored: orientationName = "downMirrored(5)"
        case .leftMirrored: orientationName = "leftMirrored(6)"
        case .rightMirrored: orientationName = "rightMirrored(7)"
        @unknown default: orientationName = "unknown"
        }

        logRotation("📐 fixedOrientation() 호출 - orientation: \(orientationName)")
        logRotation("📐 원본 크기(size): \(size.width) x \(size.height)")

        if let cgImage = cgImage {
            logRotation("📐 실제 픽셀(cgImage): \(cgImage.width) x \(cgImage.height)")
        }

        // 이미 올바른 방향이면 그대로 반환
        if imageOrientation == .up {
            logRotation("📐 이미 .up 방향이므로 그대로 반환")
            return self
        }

        // UIGraphicsImageRenderer를 사용하여 올바른 방향으로 다시 그리기
        guard let cgImage = cgImage else { return self }

        // .right orientation의 경우: 실제 픽셀은 가로지만, 메타데이터상 세로로 표시됨
        // 따라서 cgImage.width와 cgImage.height를 사용하여 실제 픽셀 기준으로 계산
        let actualWidth = CGFloat(cgImage.width)
        let actualHeight = CGFloat(cgImage.height)

        // 회전 후의 크기 계산
        var targetWidth = actualWidth
        var targetHeight = actualHeight

        switch imageOrientation {
        case .left, .leftMirrored, .right, .rightMirrored:
            // 90도 또는 270도 회전 시 너비와 높이가 바뀜
            targetWidth = actualHeight
            targetHeight = actualWidth
        default:
            break
        }

        // 렌더러로 새 이미지 생성
        let renderer = UIGraphicsImageRenderer(size: CGSize(width: targetWidth, height: targetHeight))

        let rotatedImage = renderer.image { context in
            let cgContext = context.cgContext

            // 변환 적용
            // Core Graphics: rotate(+각도) = 반시계방향, rotate(-각도) = 시계방향
            switch imageOrientation {
            case .down, .downMirrored:
                cgContext.translateBy(x: targetWidth, y: targetHeight)
                cgContext.rotate(by: .pi)

            case .left, .leftMirrored:
                // .left: 이미지를 반시계방향 90도 회전해서 봐야 함
                // 픽셀을 바로잡으려면 반시계방향 90도 회전
                cgContext.translateBy(x: 0, y: targetHeight)
                cgContext.rotate(by: -.pi / 2)

            case .right, .rightMirrored:
                // .right: 이미지를 시계방향 90도 회전해서 봐야 함
                // 픽셀을 바로잡으려면 반시계방향 90도 회전
                cgContext.translateBy(x: targetWidth, y: 0)
                cgContext.rotate(by: .pi / 2)

            default:
                break
            }

            // 미러링 처리
            switch imageOrientation {
            case .upMirrored, .downMirrored:
                cgContext.translateBy(x: actualWidth, y: 0)
                cgContext.scaleBy(x: -1, y: 1)

            case .leftMirrored, .rightMirrored:
                cgContext.translateBy(x: actualHeight, y: 0)
                cgContext.scaleBy(x: -1, y: 1)

            default:
                break
            }

            // 원본 이미지 그리기 (실제 픽셀 크기 사용)
            cgContext.draw(cgImage, in: CGRect(x: 0, y: 0, width: actualWidth, height: actualHeight))
        }

        logRotation("📐 회전 후 크기: \(rotatedImage.size.width) x \(rotatedImage.size.height)")
        logRotation("📐 회전 후 orientation: \(rotatedImage.imageOrientation.rawValue) (should be 0 = .up)")
        logRotation("========================================")

        return rotatedImage
    }
}
