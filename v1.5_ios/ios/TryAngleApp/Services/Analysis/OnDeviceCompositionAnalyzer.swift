import Foundation
import Vision
import CoreML
import UIKit

// MARK: - 온디바이스 구도 분석기
class OnDeviceCompositionAnalyzer {

    // MARK: - 구도 점수 계산
    struct CompositionScore {
        let overall: Float          // 전체 점수 (0-1)
        let ruleOfThirds: Float     // 3분할 규칙
        let balance: Float          // 균형감
        let leadingSpace: Float     // 시선 여백
        let goldenRatio: Float      // 황금비
        let symmetry: Float         // 대칭성

        // 추가 정보 (구체적 지시를 위해)
        let bodyPosition: CGPoint?      // 신체 중심 위치
        let imageSize: CGSize?          // 이미지 크기
        let faceDirection: Float?       // 얼굴 방향 (yaw)

        var feedback: String {
            if overall > 0.8 {
                return "✓ 구도 완벽!"
            } else if overall > 0.6 {
                return "구도 양호 - 미세 조정 필요"
            } else if overall > 0.4 {
                return "구도 개선 필요"
            } else {
                return "구도를 다시 잡으세요"
            }
        }

        var detailedFeedback: [String] {
            var feedbacks: [String] = []

            guard let bodyPos = bodyPosition, let imgSize = imageSize else {
                return feedbacks
            }

            // 1. 3분할 규칙 - 구체적 위치 지시
            if ruleOfThirds < 0.5 {
                let currentX = bodyPos.x / imgSize.width
                let thirdX1 = 1.0 / 3.0
                let thirdX2 = 2.0 / 3.0

                let distToLeft = abs(currentX - thirdX1)
                let distToRight = abs(currentX - thirdX2)

                if currentX < 0.33 {
                    feedbacks.append("→ 오른쪽으로 약간 이동 (3분할선 맞추기)")
                } else if currentX > 0.67 {
                    feedbacks.append("← 왼쪽으로 약간 이동 (3분할선 맞추기)")
                } else {
                    // 중앙에 있는 경우
                    if distToLeft < distToRight {
                        feedbacks.append("← 왼쪽으로 10cm 이동 (3분할 구도)")
                    } else {
                        feedbacks.append("→ 오른쪽으로 10cm 이동 (3분할 구도)")
                    }
                }
            }

            // 2. 균형감 - 구체적 방향 지시
            if balance < 0.5 {
                if let bodyX = bodyPosition?.x, let width = imageSize?.width {
                    let centerX = width / 2
                    let offset = bodyX - centerX

                    if offset > width * 0.1 {
                        feedbacks.append("← 왼쪽으로 이동 (좌우 균형 맞추기)")
                    } else if offset < -width * 0.1 {
                        feedbacks.append("→ 오른쪽으로 이동 (좌우 균형 맞추기)")
                    }
                }
            }

            // 3. 시선 여백 - 구체적 지시
            if leadingSpace < 0.5, let yaw = faceDirection, let bodyX = bodyPosition?.x, let width = imageSize?.width {
                let lookingRight = yaw > 0
                let bodyRatio = Float(bodyX / width)

                if lookingRight {
                    // 오른쪽을 보는데 오른쪽에 위치 → 왼쪽으로 이동해야 함
                    if bodyRatio > 0.5 {
                        feedbacks.append("← 왼쪽으로 20-30cm 이동 (시선 공간 확보)")
                    }
                } else {
                    // 왼쪽을 보는데 왼쪽에 위치 → 오른쪽으로 이동해야 함
                    if bodyRatio < 0.5 {
                        feedbacks.append("→ 오른쪽으로 20-30cm 이동 (시선 공간 확보)")
                    }
                }
            }

            // 4. 황금비
            if goldenRatio < 0.5 {
                feedbacks.append("↕ 카메라 높이 조정 (상하 비율 개선)")
            }

            // 5. 대칭성
            if symmetry < 0.3 {
                feedbacks.append("⚖️ 자세를 좌우 대칭으로 만드세요")
            } else if symmetry > 0.7 {
                feedbacks.append("💃 좌우 대칭을 깨고 자연스러운 포즈를 취하세요")
            }

            return feedbacks
        }
    }

    // MARK: - 3분할 규칙 체크
    func checkRuleOfThirds(keypoints: [PoseKeypoint], in imageSize: CGSize) -> Float {
        // 3분할 선 위치
        let thirdX1 = imageSize.width / 3
        let thirdX2 = imageSize.width * 2 / 3
        let thirdY1 = imageSize.height / 3
        let thirdY2 = imageSize.height * 2 / 3

        // 주요 키포인트 (얼굴, 가슴, 골반)
        let importantIndices = [0, 1, 2, 5, 6, 11, 12]  // 얼굴, 어깨, 골반
        var score: Float = 0
        var count = 0

        for index in importantIndices {
            guard index < keypoints.count else { continue }
            let point = keypoints[index]

            // 3분할 선과의 거리 계산
            let distX = min(
                abs(point.location.x - thirdX1),
                abs(point.location.x - thirdX2)
            )
            let distY = min(
                abs(point.location.y - thirdY1),
                abs(point.location.y - thirdY2)
            )

            // 거리가 가까울수록 높은 점수 (tolerance: 5%)
            let tolerance = imageSize.width * 0.05
            let xScore = Float(max(0, 1 - distX / tolerance))
            let yScore = Float(max(0, 1 - distY / tolerance))

            score += (xScore + yScore) / 2
            count += 1
        }

        return count > 0 ? score / Float(count) : 0
    }

    // MARK: - 균형감 체크
    func checkBalance(keypoints: [PoseKeypoint], in imageSize: CGSize) -> Float {
        let centerX = imageSize.width / 2

        // 좌우 키포인트 분리
        var leftWeight: Float = 0
        var rightWeight: Float = 0

        for keypoint in keypoints {
            let weight = keypoint.confidence
            if keypoint.location.x < centerX {
                leftWeight += weight * Float(centerX - keypoint.location.x)
            } else {
                rightWeight += weight * Float(keypoint.location.x - centerX)
            }
        }

        // 균형 비율 계산 (1에 가까울수록 균형적)
        let balance = min(leftWeight, rightWeight) / max(leftWeight, rightWeight, 0.001)
        return balance
    }

    // MARK: - 시선 여백 체크
    func checkLeadingSpace(faceYaw: Float?, bodyCenter: CGPoint, in imageSize: CGSize) -> Float {
        guard let yaw = faceYaw else { return 0.5 }

        // 얼굴이 향하는 방향
        let lookingRight = yaw > 0

        // 신체 중심 위치
        let bodyRatio = bodyCenter.x / imageSize.width

        // 시선 방향에 여백이 있는지 체크
        if lookingRight {
            // 오른쪽을 보는데 왼쪽에 위치 (좋음)
            return Float(max(0, 1 - bodyRatio * 2))
        } else {
            // 왼쪽을 보는데 오른쪽에 위치 (좋음)
            return Float(max(0, bodyRatio * 2 - 1))
        }
    }

    // MARK: - 황금비 체크
    func checkGoldenRatio(bodyRect: CGRect, in imageSize: CGSize) -> Float {
        let goldenRatio: Float = 1.618

        // 신체 비율 체크
        let bodyRatio = Float(bodyRect.height / bodyRect.width)
        let ratioScore = 1 - abs(bodyRatio - goldenRatio) / goldenRatio

        // 위치 비율 체크 (상단에서의 위치)
        let topRatio = Float(bodyRect.minY / imageSize.height)
        let positionScore = abs(topRatio - (1 / goldenRatio))

        return (ratioScore + (1 - positionScore)) / 2
    }

    // MARK: - 대칭성 체크
    func checkSymmetry(keypoints: [PoseKeypoint]) -> Float {
        // 좌우 대응 키포인트 쌍
        let pairs = [
            (5, 6),   // 어깨
            (7, 8),   // 팔꿈치
            (9, 10),  // 손목
            (11, 12), // 골반
            (13, 14), // 무릎
            (15, 16)  // 발목
        ]

        var symmetryScore: Float = 0
        var count = 0

        for (leftIdx, rightIdx) in pairs {
            guard leftIdx < keypoints.count, rightIdx < keypoints.count else { continue }

            let left = keypoints[leftIdx]
            let right = keypoints[rightIdx]

            // Y축 대칭성 (높이 차이)
            let yDiff = abs(left.location.y - right.location.y)
            let ySymmetry = Float(1 - min(yDiff / 100, 1))  // 100픽셀 이상 차이나면 0점

            symmetryScore += ySymmetry
            count += 1
        }

        return count > 0 ? symmetryScore / Float(count) : 0
    }

    // MARK: - 종합 분석
    func analyzeComposition(
        keypoints: [PoseKeypoint],
        faceResult: FaceAnalysisResult?,
        imageSize: CGSize
    ) -> CompositionScore {

        // 신체 중심점 계산
        let bodyCenter = calculateBodyCenter(from: keypoints)

        // 신체 영역 계산
        let bodyRect = calculateBodyRect(from: keypoints)

        // 각 요소 점수 계산
        let ruleOfThirds = checkRuleOfThirds(keypoints: keypoints, in: imageSize)
        let balance = checkBalance(keypoints: keypoints, in: imageSize)
        let leadingSpace = checkLeadingSpace(faceYaw: faceResult?.yaw, bodyCenter: bodyCenter, in: imageSize)
        let goldenRatio = checkGoldenRatio(bodyRect: bodyRect, in: imageSize)
        let symmetry = checkSymmetry(keypoints: keypoints)

        // 가중 평균 (구도에서 중요한 순서대로 가중치)
        let overall = (
            ruleOfThirds * 0.3 +    // 3분할이 가장 중요
            balance * 0.25 +         // 균형감
            leadingSpace * 0.2 +     // 시선 여백
            goldenRatio * 0.15 +     // 황금비
            symmetry * 0.1           // 대칭성 (약간만)
        )

        return CompositionScore(
            overall: overall,
            ruleOfThirds: ruleOfThirds,
            balance: balance,
            leadingSpace: leadingSpace,
            goldenRatio: goldenRatio,
            symmetry: symmetry,
            bodyPosition: bodyCenter,
            imageSize: imageSize,
            faceDirection: faceResult?.yaw
        )
    }

    // MARK: - Helper Methods
    private func calculateBodyCenter(from keypoints: [PoseKeypoint]) -> CGPoint {
        guard !keypoints.isEmpty else { return .zero }

        var sumX: CGFloat = 0
        var sumY: CGFloat = 0
        var count = 0

        for keypoint in keypoints {
            if keypoint.confidence > 0.3 {
                sumX += keypoint.location.x
                sumY += keypoint.location.y
                count += 1
            }
        }

        guard count > 0 else { return .zero }
        return CGPoint(x: sumX / CGFloat(count), y: sumY / CGFloat(count))
    }

    private func calculateBodyRect(from keypoints: [PoseKeypoint]) -> CGRect {
        guard !keypoints.isEmpty else { return .zero }

        let validPoints = keypoints.filter { $0.confidence > 0.3 }
        guard !validPoints.isEmpty else { return .zero }

        let minX = validPoints.map { $0.location.x }.min() ?? 0
        let maxX = validPoints.map { $0.location.x }.max() ?? 0
        let minY = validPoints.map { $0.location.y }.min() ?? 0
        let maxY = validPoints.map { $0.location.y }.max() ?? 0

        return CGRect(x: minX, y: minY, width: maxX - minX, height: maxY - minY)
    }
}

// MARK: - 앵글 분석기
extension OnDeviceCompositionAnalyzer {

    enum CameraAngle {
        case high    // 하이앵글
        case eye     // 아이레벨
        case low     // 로우앵글

        var description: String {
            switch self {
            case .high: return "하이 앵글"
            case .eye: return "아이 레벨"
            case .low: return "로우 앵글"
            }
        }

        // 레퍼런스 앵글과 비교하여 구체적 피드백 생성
        func feedbackComparedTo(reference: CameraAngle?) -> String {
            guard let ref = reference else {
                // 레퍼런스가 없으면 현재 상태만 알려줌
                switch self {
                case .high: return "📷 하이 앵글 (위에서 촬영 중)"
                case .eye: return "📷 아이 레벨 (눈높이 촬영 중)"
                case .low: return "📷 로우 앵글 (아래에서 촬영 중)"
                }
            }

            // 레퍼런스와 같으면 OK
            if self == ref {
                return "✓ 카메라 앵글 일치"
            }

            // 레퍼런스와 다르면 구체적 지시
            switch (self, ref) {
            case (.high, .eye):
                return "↓ 카메라를 15-20cm 낮추세요 (아이레벨로)"
            case (.high, .low):
                return "↓ 카메라를 30-40cm 낮추세요 (로우앵글로)"
            case (.eye, .high):
                return "↑ 카메라를 15-20cm 높이세요 (하이앵글로)"
            case (.eye, .low):
                return "↓ 카메라를 15-20cm 낮추세요 (로우앵글로)"
            case (.low, .high):
                return "↑ 카메라를 30-40cm 높이세요 (하이앵글로)"
            case (.low, .eye):
                return "↑ 카메라를 15-20cm 높이세요 (아이레벨로)"
            default:
                return "✓ 카메라 앵글 OK"
            }
        }
    }

    func detectCameraAngle(from keypoints: [PoseKeypoint]) -> CameraAngle {
        // 머리와 발 키포인트 찾기
        guard keypoints.count > 16 else { return .eye }

        let headY = (keypoints[0].location.y + keypoints[1].location.y + keypoints[2].location.y) / 3
        let footY = (keypoints[15].location.y + keypoints[16].location.y) / 2

        let bodyHeight = footY - headY
        let headRatio = headY / bodyHeight

        // 머리 위치 비율로 앵글 판단
        if headRatio < 0.15 {
            return .high  // 머리가 너무 위에 있음
        } else if headRatio > 0.25 {
            return .low   // 머리가 상대적으로 아래 있음
        } else {
            return .eye
        }
    }
}