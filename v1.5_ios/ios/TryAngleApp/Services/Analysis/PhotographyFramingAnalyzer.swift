import Foundation
import CoreGraphics

// MARK: - RTMPose 133 키포인트 인덱스 맵
/// 전문 사진 프레이밍 분석을 위한 키포인트 매핑
struct KeypointIndex {
    // Body (0-16): COCO 17 keypoints
    static let nose = 0
    static let leftEye = 1
    static let rightEye = 2
    static let leftEar = 3
    static let rightEar = 4
    static let leftShoulder = 5
    static let rightShoulder = 6
    static let leftElbow = 7
    static let rightElbow = 8
    static let leftWrist = 9
    static let rightWrist = 10
    static let leftHip = 11
    static let rightHip = 12
    static let leftKnee = 13
    static let rightKnee = 14
    static let leftAnkle = 15
    static let rightAnkle = 16

    // Face (17-84): 68 facial landmarks
    static let faceStart = 17
    static let faceEnd = 84
    static let faceJawStart = 17      // 턱 라인 (0-16)
    static let faceJawEnd = 33
    static let faceBrowStart = 34     // 눈썹 (17-26)
    static let faceBrowEnd = 43
    static let faceNoseStart = 44     // 코 (27-35)
    static let faceNoseEnd = 52
    static let faceEyeStart = 53      // 눈 (36-47)
    static let faceEyeEnd = 64
    static let faceMouthStart = 65    // 입 (48-67)
    static let faceMouthEnd = 84

    // Hands (85-126): 21 keypoints per hand
    static let leftHandStart = 85
    static let leftHandEnd = 105
    static let rightHandStart = 106
    static let rightHandEnd = 126

    // Feet (127-132): 3 keypoints per foot
    static let leftFootStart = 127
    static let leftFootEnd = 129
    static let rightFootStart = 130
    static let rightFootEnd = 132
}

// MARK: - 샷 타입 (사진학 기준)
enum ShotType: String, CaseIterable {
    case extremeCloseUp = "익스트림 클로즈업"  // 눈, 입 등 특정 부위만
    case closeUp = "클로즈업"                 // 얼굴 중심
    case mediumCloseUp = "미디엄 클로즈업"    // 어깨선 위
    case mediumShot = "미디엄샷"              // 허리 위 (바스트샷)
    case americanShot = "아메리칸샷"          // 무릎 위 (카우보이샷)
    case mediumFullShot = "미디엄 풀샷"       // 무릎~발목
    case fullShot = "풀샷"                   // 전신
    case longShot = "롱샷"                   // 전신 + 환경

    /// 사용자 친화적인 설명
    var userFriendlyDescription: String {
        switch self {
        case .extremeCloseUp: return "얼굴 일부만"
        case .closeUp: return "얼굴 중심"
        case .mediumCloseUp: return "어깨 위까지"
        case .mediumShot: return "허리 위까지"
        case .americanShot: return "무릎 위까지"
        case .mediumFullShot: return "무릎 아래까지"
        case .fullShot: return "전신"
        case .longShot: return "전신 + 배경"
        }
    }

    var headroomRange: ClosedRange<CGFloat> {
        switch self {
        case .extremeCloseUp: return 0.02...0.08
        case .closeUp: return 0.05...0.15
        case .mediumCloseUp: return 0.08...0.18
        case .mediumShot: return 0.10...0.20
        case .americanShot: return 0.08...0.15
        case .mediumFullShot: return 0.05...0.12
        case .fullShot: return 0.03...0.10
        case .longShot: return 0.02...0.08
        }
    }
}

// MARK: - 카메라 앵글 (사진학 기준)
enum PhotoCameraAngle: String {
    case highAngle = "하이앵글"      // 위에서 아래로 (연약함, 귀여움)
    case eyeLevel = "아이레벨"       // 눈높이 (평등, 자연스러움)
    case lowAngle = "로우앵글"       // 아래에서 위로 (강인함, 권위)
    case birdsEye = "버즈아이"       // 직하방 (객관적, 전체적)
    case dutchAngle = "더치앵글"     // 기울임 (불안정, 역동적)

    var description: String {
        switch self {
        case .highAngle: return "위에서 촬영 (부드러운 느낌)"
        case .eyeLevel: return "눈높이 촬영 (자연스러운 느낌)"
        case .lowAngle: return "아래에서 촬영 (강인한 느낌)"
        case .birdsEye: return "직하방 촬영"
        case .dutchAngle: return "기울임 촬영 (역동적)"
        }
    }
}

// MARK: - 프레이밍 분석 결과
struct PhotographyFramingResult {
    // 샷 타입
    let shotType: ShotType
    let shotTypeConfidence: Float

    // 헤드룸 (정규화된 값: 0.0~1.0)
    let headroom: CGFloat           // 머리 위 여백
    let optimalHeadroom: CGFloat    // 샷 타입에 맞는 최적 헤드룸
    let headroomStatus: SpaceStatus

    // 리드룸 (시선 방향 여백)
    let leadRoom: CGFloat?          // 시선 방향 여백
    let gazeDirection: FramingGazeDirection
    let leadRoomStatus: SpaceStatus?

    // 카메라 앵글
    let cameraAngle: PhotoCameraAngle
    let cameraAngleValue: CGFloat   // 실제 각도 (도)

    // 잘림 체크 (사진 규칙 위반)
    let croppingViolations: [CroppingViolation]

    // 🆕 신체 점유율 (0.0~1.0)
    let bodyCoverage: CGFloat       // 구조적 키포인트가 차지하는 프레임 비율

    // 🆕 코(nose) 위치 (정규화된 좌표: 0.0~1.0) - Phase 3
    let nosePosition: CGPoint       // 인물 위치 비교용

    // 전체 점수 (0.0~1.0)
    let overallScore: Float
}

// MARK: - 여백 상태
enum SpaceStatus: String {
    case tooMuch = "과다"
    case optimal = "적정"
    case tooLittle = "부족"
    case none = "없음"
}

// MARK: - 프레이밍용 시선 방향 (단순화)
/// 기존 GazeDirection과 별도로, 프레이밍 분석용 단순화된 시선 방향
enum FramingGazeDirection: String {
    case left = "왼쪽"
    case right = "오른쪽"
    case center = "정면"
    case up = "위"
    case down = "아래"

    /// 기존 GazeDirection에서 변환
    init(from gazeDirection: GazeDirection) {
        switch gazeDirection {
        case .lookingLeft, .lookingLeftUp, .lookingLeftDown:
            self = .left
        case .lookingRight, .lookingRightUp, .lookingRightDown:
            self = .right
        case .lookingUp:
            self = .up
        case .lookingDown:
            self = .down
        case .lookingAtCamera, .unknown:
            self = .center
        }
    }
}

// MARK: - 잘림 규칙 위반
struct CroppingViolation {
    let jointName: String       // 잘린 관절명
    let position: CGFloat       // 위치 (정규화)
    let severity: ViolationSeverity
}

enum ViolationSeverity: String {
    case critical = "심각"      // 관절에서 직접 잘림
    case warning = "경고"       // 관절 근처에서 잘림
    case minor = "경미"         // 약간 어색함
}

// MARK: - 전문 사진 프레이밍 분석기
class PhotographyFramingAnalyzer {

    // MARK: - 구조적 키포인트 (프레이밍 분석용)
    /// 손가락, 얼굴 랜드마크를 제외한 신체 구조 키포인트
    struct StructuralKeypoints {
        // 프레이밍 분석에 사용할 구조적 키포인트만 (0-16번)
        static let head = [0, 1, 2, 3, 4]        // 코, 눈, 귀
        static let shoulders = [5, 6]            // 어깨
        static let elbows = [7, 8]               // 팔꿈치
        static let wrists = [9, 10]              // 손목
        static let hips = [11, 12]               // 엉덩이
        static let knees = [13, 14]              // 무릎
        static let ankles = [15, 16]             // 발목

        // 전체 구조적 키포인트 (손가락, 얼굴 랜드마크 제외)
        static let all = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

        /// 샷 타입별 중요 키포인트 반환 (구조적 키포인트만, 0-16)
        /// 손(91-132), 얼굴(23-90)은 별도 처리
        static func importantKeypoints(for shotType: ShotType) -> [Int] {
            switch shotType {
            case .extremeCloseUp, .closeUp:
                return head + shoulders  // 머리 + 어깨 (구조적 키포인트만)

            case .mediumCloseUp, .mediumShot:
                return head + shoulders + elbows + wrists + hips  // 상반신 구조

            case .americanShot:
                return head + shoulders + elbows + wrists + hips + knees  // + 무릎

            case .fullShot, .mediumFullShot, .longShot:
                return all  // 전신 (0-16 전부)
            }
        }
    }

    // 신뢰도 임계값
    private let confidenceThreshold: Float = 0.3

    // MARK: - 메인 분석 함수
    /// RTMPose 133개 키포인트를 분석하여 전문적인 프레이밍 정보 반환
    /// - Parameters:
    ///   - keypoints: RTMPose 133개 키포인트 (정규화된 좌표: 0.0~1.0)
    ///   - imageSize: 이미지 크기 (픽셀)
    /// - Returns: 사진학 기반 프레이밍 분석 결과
    func analyze(
        keypoints: [(point: CGPoint, confidence: Float)],
        imageSize: CGSize
    ) -> PhotographyFramingResult? {

        guard keypoints.count >= 133 else {
            print("⚠️ 키포인트 수 부족: \(keypoints.count)/133")
            return nil
        }

        // 1. 샷 타입 결정
        let (shotType, shotConfidence) = determineShotType(keypoints: keypoints)

        // 2. 헤드룸 계산
        let (headroom, headroomStatus) = calculateHeadroom(
            keypoints: keypoints,
            shotType: shotType
        )
        let optimalHeadroom = (shotType.headroomRange.lowerBound + shotType.headroomRange.upperBound) / 2

        // 3. 시선 방향 및 리드룸 계산
        let gazeDirection = detectGazeDirection(keypoints: keypoints)
        let (leadRoom, leadRoomStatus) = calculateLeadRoom(
            keypoints: keypoints,
            gazeDirection: gazeDirection
        )

        // 4. 카메라 앵글 분석
        let (cameraAngle, angleValue) = analyzeCameraAngle(keypoints: keypoints)

        // 5. 잘림 규칙 체크
        let violations = checkCroppingViolations(keypoints: keypoints)

        // 🆕 6. 신체 점유율 계산
        let bodyCoverage = calculateBodyCoverage(
            keypoints: keypoints,
            shotType: shotType
        )

        // 7. 전체 점수 계산
        let overallScore = calculateOverallScore(
            headroomStatus: headroomStatus,
            leadRoomStatus: leadRoomStatus,
            violations: violations,
            shotConfidence: shotConfidence
        )

        // 🆕 코(nose) 위치 추출 (Phase 3)
        let nosePosition = keypoints[0].point  // index 0 = nose

        return PhotographyFramingResult(
            shotType: shotType,
            shotTypeConfidence: shotConfidence,
            headroom: headroom,
            optimalHeadroom: optimalHeadroom,
            headroomStatus: headroomStatus,
            leadRoom: leadRoom,
            gazeDirection: gazeDirection,
            leadRoomStatus: leadRoomStatus,
            cameraAngle: cameraAngle,
            cameraAngleValue: angleValue,
            croppingViolations: violations,
            bodyCoverage: bodyCoverage,
            nosePosition: nosePosition,
            overallScore: overallScore
        )
    }

    // MARK: - 샷 타입 결정
    private func determineShotType(
        keypoints: [(point: CGPoint, confidence: Float)]
    ) -> (ShotType, Float) {

        // 보이는 키포인트 체크
        let hasHead = keypoints[KeypointIndex.nose].confidence > confidenceThreshold
        let hasShoulders = (keypoints[KeypointIndex.leftShoulder].confidence > confidenceThreshold ||
                          keypoints[KeypointIndex.rightShoulder].confidence > confidenceThreshold)
        let hasHips = (keypoints[KeypointIndex.leftHip].confidence > confidenceThreshold ||
                      keypoints[KeypointIndex.rightHip].confidence > confidenceThreshold)
        let hasKnees = (keypoints[KeypointIndex.leftKnee].confidence > confidenceThreshold ||
                       keypoints[KeypointIndex.rightKnee].confidence > confidenceThreshold)
        let hasAnkles = (keypoints[KeypointIndex.leftAnkle].confidence > confidenceThreshold ||
                        keypoints[KeypointIndex.rightAnkle].confidence > confidenceThreshold)
        let hasFeet = (keypoints[KeypointIndex.leftFootStart].confidence > confidenceThreshold ||
                      keypoints[KeypointIndex.rightFootStart].confidence > confidenceThreshold)

        // 얼굴 크기로 보완
        let faceSize = calculateFaceSize(keypoints: keypoints)

        // 샷 타입 결정 로직
        var shotType: ShotType
        var confidence: Float = 0.8

        if !hasHead {
            // 머리가 안 보임 - 익스트림 클로즈업이거나 잘못된 프레이밍
            shotType = .extremeCloseUp
            confidence = 0.5
        } else if !hasShoulders {
            // 어깨가 안 보임 - 클로즈업
            shotType = faceSize > 0.35 ? .extremeCloseUp : .closeUp
            confidence = 0.85
        } else if !hasHips {
            // 어깨는 보이지만 허리가 안 보임
            shotType = faceSize > 0.25 ? .closeUp : .mediumCloseUp
            confidence = 0.85
        } else if !hasKnees {
            // 허리까지 보임 - 미디엄샷
            shotType = .mediumShot
            confidence = 0.9
        } else if !hasAnkles {
            // 무릎까지 보임 - 아메리칸샷
            shotType = .americanShot
            confidence = 0.9
        } else if !hasFeet {
            // 발목까지 보임 - 미디엄 풀샷
            shotType = .mediumFullShot
            confidence = 0.85
        } else {
            // 전신 보임
            let bodyHeight = calculateBodyHeight(keypoints: keypoints)
            if bodyHeight < 0.6 {
                shotType = .longShot
                confidence = 0.8
            } else {
                shotType = .fullShot
                confidence = 0.9
            }
        }

        return (shotType, confidence)
    }

    // MARK: - 헤드룸 계산
    private func calculateHeadroom(
        keypoints: [(point: CGPoint, confidence: Float)],
        shotType: ShotType
    ) -> (CGFloat, SpaceStatus) {

        // 머리 최상단 찾기 (얼굴 키포인트 중 가장 위)
        var topY: CGFloat = 1.0

        // 얼굴 키포인트에서 최상단 찾기
        for i in KeypointIndex.faceStart...KeypointIndex.faceEnd {
            if keypoints[i].confidence > confidenceThreshold {
                topY = min(topY, keypoints[i].point.y)
            }
        }

        // 코/눈으로 보정
        if keypoints[KeypointIndex.nose].confidence > confidenceThreshold {
            let noseY = keypoints[KeypointIndex.nose].point.y
            // 머리 높이 추정 (코에서 얼굴 높이의 1.5배 위)
            let faceHeight = calculateFaceSize(keypoints: keypoints)
            let estimatedTop = noseY - faceHeight * 0.7
            topY = min(topY, max(0, estimatedTop))
        }

        // 헤드룸 = 머리 최상단 Y 좌표 (0이 상단)
        let headroom = topY

        // 샷 타입에 따른 상태 판정
        let optimalRange = shotType.headroomRange
        let status: SpaceStatus

        if headroom < optimalRange.lowerBound * 0.5 {
            status = .tooLittle  // 너무 적음 (머리가 잘릴 위험)
        } else if headroom < optimalRange.lowerBound {
            status = .tooLittle  // 약간 부족
        } else if headroom > optimalRange.upperBound * 1.5 {
            status = .tooMuch    // 너무 많음
        } else if headroom > optimalRange.upperBound {
            status = .tooMuch    // 약간 많음
        } else {
            status = .optimal    // 적정
        }

        return (headroom, status)
    }

    // MARK: - 시선 방향 감지
    private func detectGazeDirection(
        keypoints: [(point: CGPoint, confidence: Float)]
    ) -> FramingGazeDirection {

        // 얼굴 방향으로 시선 추정 (더 정확한 gaze tracking은 별도 모델 필요)
        let leftEye = keypoints[KeypointIndex.leftEye]
        let rightEye = keypoints[KeypointIndex.rightEye]
        let nose = keypoints[KeypointIndex.nose]
        let leftEar = keypoints[KeypointIndex.leftEar]
        let rightEar = keypoints[KeypointIndex.rightEar]

        guard leftEye.confidence > confidenceThreshold,
              rightEye.confidence > confidenceThreshold,
              nose.confidence > confidenceThreshold else {
            return .center
        }

        // 눈 중심과 코의 상대 위치로 시선 방향 추정
        let eyeCenter = CGPoint(
            x: (leftEye.point.x + rightEye.point.x) / 2,
            y: (leftEye.point.y + rightEye.point.y) / 2
        )

        // 코가 눈 중심에서 얼마나 벗어났는지 (좌우)
        let horizontalOffset = nose.point.x - eyeCenter.x
        let eyeDistance = abs(rightEye.point.x - leftEye.point.x)
        let normalizedOffset = horizontalOffset / max(eyeDistance, 0.01)

        // 귀 가시성으로 보완 (한쪽 귀만 보이면 그 반대 방향 보고 있음)
        let leftEarVisible = leftEar.confidence > confidenceThreshold
        let rightEarVisible = rightEar.confidence > confidenceThreshold

        if !leftEarVisible && rightEarVisible {
            return .left  // 왼쪽 귀가 안 보임 = 왼쪽 보고 있음
        } else if leftEarVisible && !rightEarVisible {
            return .right // 오른쪽 귀가 안 보임 = 오른쪽 보고 있음
        }

        // 코 위치 기반 판정
        if normalizedOffset < -0.15 {
            return .left
        } else if normalizedOffset > 0.15 {
            return .right
        } else {
            return .center
        }
    }

    // MARK: - 리드룸 계산
    private func calculateLeadRoom(
        keypoints: [(point: CGPoint, confidence: Float)],
        gazeDirection: FramingGazeDirection
    ) -> (CGFloat?, SpaceStatus?) {

        guard gazeDirection == .left || gazeDirection == .right else {
            // 정면이면 리드룸 계산 불필요
            return (nil, nil)
        }

        // 얼굴 중심 위치 찾기
        let nose = keypoints[KeypointIndex.nose]
        guard nose.confidence > confidenceThreshold else {
            return (nil, nil)
        }

        let faceX = nose.point.x

        // 시선 방향의 여백 계산
        let leadRoom: CGFloat
        if gazeDirection == .left {
            leadRoom = faceX  // 왼쪽 보고 있으면 왼쪽 여백
        } else {
            leadRoom = 1.0 - faceX  // 오른쪽 보고 있으면 오른쪽 여백
        }

        // 권장 리드룸: 15~35%
        let status: SpaceStatus
        if leadRoom < 0.10 {
            status = .tooLittle  // 너무 적음
        } else if leadRoom < 0.15 {
            status = .tooLittle  // 약간 부족
        } else if leadRoom > 0.45 {
            status = .tooMuch    // 너무 많음
        } else if leadRoom > 0.35 {
            status = .tooMuch    // 약간 많음
        } else {
            status = .optimal    // 적정 (15~35%)
        }

        return (leadRoom, status)
    }

    // MARK: - 카메라 앵글 분석
    private func analyzeCameraAngle(
        keypoints: [(point: CGPoint, confidence: Float)]
    ) -> (PhotoCameraAngle, CGFloat) {

        // 어깨와 눈의 상대적 위치로 카메라 앵글 추정
        let leftShoulder = keypoints[KeypointIndex.leftShoulder]
        let rightShoulder = keypoints[KeypointIndex.rightShoulder]
        let leftEye = keypoints[KeypointIndex.leftEye]
        let rightEye = keypoints[KeypointIndex.rightEye]

        guard leftShoulder.confidence > confidenceThreshold || rightShoulder.confidence > confidenceThreshold,
              leftEye.confidence > confidenceThreshold || rightEye.confidence > confidenceThreshold else {
            return (.eyeLevel, 0)
        }

        // 어깨 중심
        let shoulderY: CGFloat
        if leftShoulder.confidence > confidenceThreshold && rightShoulder.confidence > confidenceThreshold {
            shoulderY = (leftShoulder.point.y + rightShoulder.point.y) / 2
        } else if leftShoulder.confidence > confidenceThreshold {
            shoulderY = leftShoulder.point.y
        } else {
            shoulderY = rightShoulder.point.y
        }

        // 눈 중심
        let eyeY: CGFloat
        if leftEye.confidence > confidenceThreshold && rightEye.confidence > confidenceThreshold {
            eyeY = (leftEye.point.y + rightEye.point.y) / 2
        } else if leftEye.confidence > confidenceThreshold {
            eyeY = leftEye.point.y
        } else {
            eyeY = rightEye.point.y
        }

        // 눈-어깨 거리 비율로 앵글 추정
        // 정상: 눈이 어깨보다 위에 있음 (Y 값이 작음)
        // 하이앵글: 눈-어깨 거리가 짧아짐 (위에서 내려다봄)
        // 로우앵글: 눈-어깨 거리가 길어짐 (아래에서 올려다봄)

        let eyeShoulderDist = shoulderY - eyeY  // 양수가 정상

        // 얼굴 크기로 정규화
        let faceSize = calculateFaceSize(keypoints: keypoints)
        let normalizedDist = eyeShoulderDist / max(faceSize, 0.1)

        // 각도 추정 (대략적)
        let estimatedAngle = (normalizedDist - 1.0) * 30  // 정규화된 값 기준

        let cameraAngle: PhotoCameraAngle
        if normalizedDist < 0.5 {
            cameraAngle = .birdsEye  // 직하방
        } else if normalizedDist < 0.8 {
            cameraAngle = .highAngle  // 하이앵글
        } else if normalizedDist > 1.5 {
            cameraAngle = .lowAngle   // 로우앵글
        } else {
            cameraAngle = .eyeLevel   // 아이레벨
        }

        // 기울기 체크 (더치앵글)
        let shoulderTilt = calculateShoulderTilt(keypoints: keypoints)
        if abs(shoulderTilt) > 10 {
            return (.dutchAngle, shoulderTilt)
        }

        return (cameraAngle, estimatedAngle)
    }

    // MARK: - 잘림 규칙 체크
    private func checkCroppingViolations(
        keypoints: [(point: CGPoint, confidence: Float)]
    ) -> [CroppingViolation] {

        var violations: [CroppingViolation] = []

        // 사진 규칙: 관절에서 자르지 말 것
        let jointChecks: [(index: Int, name: String)] = [
            (KeypointIndex.leftAnkle, "왼쪽 발목"),
            (KeypointIndex.rightAnkle, "오른쪽 발목"),
            (KeypointIndex.leftKnee, "왼쪽 무릎"),
            (KeypointIndex.rightKnee, "오른쪽 무릎"),
            (KeypointIndex.leftHip, "왼쪽 엉덩이"),
            (KeypointIndex.rightHip, "오른쪽 엉덩이"),
            (KeypointIndex.leftWrist, "왼쪽 손목"),
            (KeypointIndex.rightWrist, "오른쪽 손목"),
            (KeypointIndex.leftElbow, "왼쪽 팔꿈치"),
            (KeypointIndex.rightElbow, "오른쪽 팔꿈치"),
        ]

        for (index, name) in jointChecks {
            let joint = keypoints[index]

            // 관절이 프레임 가장자리에 있는지 체크
            if joint.confidence > confidenceThreshold {
                let x = joint.point.x
                let y = joint.point.y

                // 하단 경계 근처 (0.95~1.0)
                if y > 0.92 && y <= 1.0 {
                    let severity: ViolationSeverity = y > 0.98 ? .critical : .warning
                    violations.append(CroppingViolation(
                        jointName: name,
                        position: y,
                        severity: severity
                    ))
                }

                // 좌우 경계 근처
                if x < 0.05 || x > 0.95 {
                    let severity: ViolationSeverity = (x < 0.02 || x > 0.98) ? .critical : .warning
                    violations.append(CroppingViolation(
                        jointName: name,
                        position: x,
                        severity: severity
                    ))
                }
            }
        }

        return violations
    }

    // MARK: - 🆕 신체 점유율 계산
    /// 구조적 키포인트가 차지하는 프레임 비율 계산
    /// - Parameters:
    ///   - keypoints: RTMPose 133개 키포인트
    ///   - shotType: 샷 타입
    /// - Returns: 점유율 (0.0~1.0)
    private func calculateBodyCoverage(
        keypoints: [(point: CGPoint, confidence: Float)],
        shotType: ShotType
    ) -> CGFloat {

        // 1. 샷 타입별 중요 키포인트 선택
        let importantIndices = StructuralKeypoints.importantKeypoints(for: shotType)

        // 2. 신뢰도 0.3 이상인 점만 필터링
        let validPoints = importantIndices.compactMap { idx -> CGPoint? in
            guard idx < keypoints.count else { return nil }
            return keypoints[idx].confidence > 0.3 ? keypoints[idx].point : nil
        }

        guard validPoints.count >= 3 else {
            return 0.5  // 기본값
        }

        // 3. 바운딩 박스 계산
        let minX = validPoints.map { $0.x }.min() ?? 0
        let maxX = validPoints.map { $0.x }.max() ?? 1
        let minY = validPoints.map { $0.y }.min() ?? 0
        let maxY = validPoints.map { $0.y }.max() ?? 1

        // 4. 점유율 = 바운딩 박스 면적
        let width = maxX - minX
        let height = maxY - minY
        let coverage = width * height

        return coverage
    }

    // MARK: - 전체 점수 계산
    private func calculateOverallScore(
        headroomStatus: SpaceStatus,
        leadRoomStatus: SpaceStatus?,
        violations: [CroppingViolation],
        shotConfidence: Float
    ) -> Float {

        var score: Float = 1.0

        // 헤드룸 점수
        switch headroomStatus {
        case .optimal: break
        case .tooMuch: score -= 0.15
        case .tooLittle: score -= 0.25
        case .none: score -= 0.1
        }

        // 리드룸 점수
        if let leadStatus = leadRoomStatus {
            switch leadStatus {
            case .optimal: break
            case .tooMuch: score -= 0.1
            case .tooLittle: score -= 0.2
            case .none: break
            }
        }

        // 잘림 위반 감점
        for violation in violations {
            switch violation.severity {
            case .critical: score -= 0.2
            case .warning: score -= 0.1
            case .minor: score -= 0.05
            }
        }

        // 샷 타입 신뢰도 반영
        score *= shotConfidence

        return max(0, min(1, score))
    }

    // MARK: - 헬퍼 함수들

    /// 얼굴 크기 계산 (정규화)
    private func calculateFaceSize(
        keypoints: [(point: CGPoint, confidence: Float)]
    ) -> CGFloat {
        let leftEye = keypoints[KeypointIndex.leftEye]
        let rightEye = keypoints[KeypointIndex.rightEye]

        guard leftEye.confidence > confidenceThreshold,
              rightEye.confidence > confidenceThreshold else {
            return 0.1  // 기본값
        }

        // 눈 사이 거리로 얼굴 크기 추정 (얼굴 너비의 약 1/3)
        let eyeDistance = abs(rightEye.point.x - leftEye.point.x)
        return eyeDistance * 3.0
    }

    /// 신체 높이 계산 (정규화)
    private func calculateBodyHeight(
        keypoints: [(point: CGPoint, confidence: Float)]
    ) -> CGFloat {
        var topY: CGFloat = 1.0
        var bottomY: CGFloat = 0.0

        // 모든 키포인트에서 최상단/최하단 찾기
        for kp in keypoints.prefix(133) {
            if kp.confidence > confidenceThreshold {
                topY = min(topY, kp.point.y)
                bottomY = max(bottomY, kp.point.y)
            }
        }

        return bottomY - topY
    }

    /// 어깨 기울기 계산 (도)
    private func calculateShoulderTilt(
        keypoints: [(point: CGPoint, confidence: Float)]
    ) -> CGFloat {
        let leftShoulder = keypoints[KeypointIndex.leftShoulder]
        let rightShoulder = keypoints[KeypointIndex.rightShoulder]

        guard leftShoulder.confidence > confidenceThreshold,
              rightShoulder.confidence > confidenceThreshold else {
            return 0
        }

        let dx = rightShoulder.point.x - leftShoulder.point.x
        let dy = rightShoulder.point.y - leftShoulder.point.y

        // atan2로 각도 계산
        let angleRadians = atan2(dy, dx)
        let angleDegrees = angleRadians * 180 / .pi

        return angleDegrees
    }
}

// MARK: - 피드백 생성 확장
extension PhotographyFramingResult {

    /// 한국어 피드백 메시지 생성
    func generateFeedback() -> String? {
        var feedbacks: [String] = []

        // 1. 헤드룸 피드백
        switch headroomStatus {
        case .tooLittle:
            feedbacks.append("머리 위 여백이 부족해요. 카메라를 살짝 위로 올리거나 뒤로 물러서세요.")
        case .tooMuch:
            feedbacks.append("머리 위 공간이 너무 많아요. 인물을 프레임 위쪽으로 이동하거나 줌인하세요.")
        case .optimal, .none:
            break
        }

        // 2. 리드룸 피드백
        if let leadStatus = leadRoomStatus {
            switch leadStatus {
            case .tooLittle:
                feedbacks.append("시선 방향(\(gazeDirection.rawValue))에 여백을 더 주세요.")
            case .tooMuch:
                feedbacks.append("시선 방향에 여백이 너무 많아요. 인물을 \(gazeDirection == .left ? "오른쪽" : "왼쪽")으로 이동하세요.")
            case .optimal, .none:
                break
            }
        }

        // 3. 잘림 위반 피드백
        let criticalViolations = croppingViolations.filter { $0.severity == .critical }
        if !criticalViolations.isEmpty {
            let jointNames = criticalViolations.map { $0.jointName }.joined(separator: ", ")
            feedbacks.append("관절(\(jointNames))이 프레임 경계에서 잘려요. 조금 물러서세요.")
        }

        // 4. 최종 피드백
        if feedbacks.isEmpty {
            return nil  // 완벽
        }

        return feedbacks.first  // 가장 중요한 피드백만 반환
    }

    /// 레퍼런스와 비교하여 피드백 생성
    /// - Parameters:
    ///   - reference: 레퍼런스 프레이밍 분석 결과
    ///   - isFrontCamera: 전면 카메라 여부 (좌우 피드백 반전)
    /// - Returns: 피드백 메시지 (일치하면 nil)
    func generateFeedbackComparedTo(reference: PhotographyFramingResult, isFrontCamera: Bool = false) -> String? {
        var feedbacks: [String] = []

        // 1. 샷 타입 비교 (가장 중요)
        // 레퍼런스 샷 타입 기준으로 현재 프레임이 맞춰야 함
        // ShotType 순서: extremeCloseUp(0) → closeUp(1) → ... → fullShot(6) → longShot(7)
        // 인덱스가 작을수록 클로즈업 (좁음), 클수록 롱샷 (넓음)
        let currentLevel = ShotType.allCases.firstIndex(of: self.shotType) ?? 0
        let refLevel = ShotType.allCases.firstIndex(of: reference.shotType) ?? 0
        let levelDiff = currentLevel - refLevel  // 양수: 현재가 더 넓음, 음수: 현재가 더 좁음

        if abs(levelDiff) >= 2 {  // 2단계 이상 차이나야 피드백 (여유 제공)
            if levelDiff > 0 {
                // 현재가 더 넓음 (전신이 보이는데 레퍼런스는 상반신) → 가까이 와야 함
                feedbacks.append("더 가까이 오세요 (\(reference.shotType.userFriendlyDescription) 구도)")
            } else {
                // 현재가 더 좁음 (얼굴만 보이는데 레퍼런스는 전신) → 뒤로 가야 함
                feedbacks.append("조금 뒤로 가세요 (\(reference.shotType.userFriendlyDescription) 구도)")
            }
        }

        // 2. 헤드룸 비교 (레퍼런스 기준) - 여유있게
        let headroomDiff = self.headroom - reference.headroom
        if abs(headroomDiff) > 0.10 {  // 🔄 10% 이상 차이 (5% → 10%)
            if headroomDiff > 0 {
                feedbacks.append("머리 위 공간을 줄여주세요.")
            } else {
                feedbacks.append("머리 위 공간을 늘려주세요.")
            }
        }

        // 3. 시선/고개 방향 비교
        // 전면카메라는 거울모드이므로 좌우 피드백 반전 필요
        if self.gazeDirection != reference.gazeDirection &&
           self.gazeDirection != .center && reference.gazeDirection != .center {
            // 시선 방향이 다르면 피드백
            let targetDirection: String
            switch reference.gazeDirection {
            case .left:
                // 전면카메라: 사용자 입장에서 왼쪽은 화면상 오른쪽
                targetDirection = isFrontCamera ? "오른쪽" : "왼쪽"
            case .right:
                targetDirection = isFrontCamera ? "왼쪽" : "오른쪽"
            case .up:
                targetDirection = "위쪽"
            case .down:
                targetDirection = "아래쪽"
            case .center:
                targetDirection = "정면"
            }
            feedbacks.append("고개를 \(targetDirection)으로 돌려주세요")
        }

        // 리드룸 비교 (시선 방향이 같을 때만)
        if self.gazeDirection == reference.gazeDirection,
           let currentLead = self.leadRoom,
           let refLead = reference.leadRoom {
            let leadDiff = currentLead - refLead
            if abs(leadDiff) > 0.12 {  // 12% 이상 차이
                // 전면카메라는 좌우 반전
                let moveDirection: String
                if leadDiff > 0 {
                    // 시선 방향 여백이 많음 → 반대 방향으로 이동
                    if self.gazeDirection == .left {
                        moveDirection = isFrontCamera ? "왼쪽" : "오른쪽"
                    } else {
                        moveDirection = isFrontCamera ? "오른쪽" : "왼쪽"
                    }
                    feedbacks.append("\(moveDirection)으로 조금 이동하세요")
                } else {
                    // 시선 방향 여백이 부족 → 시선 방향으로 이동
                    if self.gazeDirection == .left {
                        moveDirection = isFrontCamera ? "오른쪽" : "왼쪽"
                    } else {
                        moveDirection = isFrontCamera ? "왼쪽" : "오른쪽"
                    }
                    feedbacks.append("\(moveDirection)으로 조금 이동하세요")
                }
            }
        }

        // 4. 카메라 앵글 비교 - 쉬운 표현
        if self.cameraAngle != reference.cameraAngle {
            switch reference.cameraAngle {
            case .highAngle:
                feedbacks.append("카메라를 위에서 아래로 향하게 하세요.")
            case .lowAngle:
                feedbacks.append("카메라를 아래에서 위로 향하게 하세요.")
            case .eyeLevel:
                feedbacks.append("카메라를 눈높이에 맞춰주세요.")
            case .birdsEye:
                feedbacks.append("바로 위에서 아래로 촬영하세요.")
            case .dutchAngle:
                feedbacks.append("카메라를 살짝 기울여주세요.")
            }
        }

        // 5. 잘림 규칙 위반 (항상 체크) - 쉬운 표현
        let criticalViolations = croppingViolations.filter { $0.severity == .critical }
        if !criticalViolations.isEmpty {
            feedbacks.append("신체 일부가 잘려요. 조금 뒤로 가세요.")
        }

        // 가장 중요한 피드백만 반환 (기준 충족하면 nil → 피드백 사라짐)
        return feedbacks.first
    }

    /// 상세 분석 정보 (디버그용)
    func debugDescription() -> String {
        """
        📷 프레이밍 분석 결과
        ━━━━━━━━━━━━━━━━━━━━━━
        샷 타입: \(shotType.rawValue) (\(String(format: "%.0f%%", shotTypeConfidence * 100)))
        헤드룸: \(String(format: "%.1f%%", headroom * 100)) [\(headroomStatus.rawValue)]
        최적 헤드룸: \(String(format: "%.1f%%", optimalHeadroom * 100))
        시선 방향: \(gazeDirection.rawValue)
        리드룸: \(leadRoom.map { String(format: "%.1f%%", $0 * 100) } ?? "N/A") [\(leadRoomStatus?.rawValue ?? "N/A")]
        카메라 앵글: \(cameraAngle.rawValue)
        잘림 위반: \(croppingViolations.count)건
        전체 점수: \(String(format: "%.0f", overallScore * 100))점
        """
    }
}
