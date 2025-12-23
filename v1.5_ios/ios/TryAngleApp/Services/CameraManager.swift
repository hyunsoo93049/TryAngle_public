import AVFoundation
import UIKit
import Combine
import Metal
import CoreImage

// MARK: - 렌즈 타입 정의
enum CameraLensType: String, CaseIterable {
    case ultraWide = "0.5"   // 초광각 (13mm, 0.5x)
    case wide = "1"          // 광각 (26mm, 1x) - 기본
    case telephoto = "3"     // 망원 (77mm, 3x)

    var displayName: String {
        return rawValue + "x"
    }

    var zoomFactor: CGFloat {
        switch self {
        case .ultraWide: return 0.5
        case .wide: return 1.0
        case .telephoto: return 3.0
        }
    }

    var deviceType: AVCaptureDevice.DeviceType {
        switch self {
        case .ultraWide: return .builtInUltraWideCamera
        case .wide: return .builtInWideAngleCamera
        case .telephoto: return .builtInTelephotoCamera
        }
    }
}

class CameraManager: NSObject, ObservableObject {
    // MARK: - Published Properties
    @Published var isAuthorized = false
    @Published var currentFrame: UIImage?
    @Published var isSessionRunning = false
    @Published var isFlashOn = false
    @Published var currentFPS: Double = 0.0
    @Published var currentZoom: CGFloat = 1.0
    @Published var aspectRatio: CameraAspectRatio = .ratio4_3  // 카메라 비율
    @Published var isFrontCamera: Bool = false  // 전면 카메라 여부
    @Published var currentLens: CameraLensType = .wide  // 🆕 현재 렌즈
    @Published var availableLenses: [CameraLensType] = [.wide]  // 🆕 사용 가능한 렌즈 목록

    // MARK: - Camera Properties
    private let session = AVCaptureSession()
    private var videoOutput = AVCaptureVideoDataOutput()
    private var currentCamera: AVCaptureDevice?
    private var currentInput: AVCaptureDeviceInput?

    // 🆕 Virtual Device 관련 (심리스 줌 전환)
    private var isUsingVirtualDevice = false  // 가상 디바이스 사용 여부
    private var minZoomFactor: CGFloat = 1.0  // 최소 줌 (초광각 시 0.5 등)
    private var maxZoomFactor: CGFloat = 10.0  // 최대 줌

    // 개별 렌즈 지원 (Virtual Device 미지원 시 폴백)
    private var availableCameras: [CameraLensType: AVCaptureDevice] = [:]

    // MARK: - Performance Optimization
    private let ciContext: CIContext = {
        // 🔥 Metal GPU 가속 사용
        if let metalDevice = MTLCreateSystemDefaultDevice() {
            return CIContext(mtlDevice: metalDevice, options: [
                .workingColorSpace: NSNull(),  // 컬러 변환 스킵
                .outputColorSpace: NSNull(),   // 출력 컬러 변환 스킵
                .cacheIntermediates: false     // 메모리 절약
            ])
        } else {
            // Metal 없으면 CPU 폴백
            return CIContext(options: [
                .useSoftwareRenderer: false,
                .workingColorSpace: NSNull(),
                .outputColorSpace: NSNull()
            ])
        }
    }()

    // 🔥 중복 버퍼 방지
    private var lastBufferTime: TimeInterval = 0

    // MARK: - Settings
    private var currentISO: Float?
    private var currentExposureCompensation: Float?

    // MARK: - FPS Tracking
    private var frameCount = 0
    private var lastFPSUpdate = Date()
    private var fpsFrameCount = 0

    // Preview layer (UIKit에서 사용)
    var previewLayer: AVCaptureVideoPreviewLayer {
        let layer = AVCaptureVideoPreviewLayer(session: session)
        layer.videoGravity = .resizeAspectFill  // 기본 카메라처럼 화면 전체 채우기
        return layer
    }

    // MARK: - Initialization
    override init() {
        super.init()
        checkAuthorization()
    }

    // MARK: - Authorization
    private func checkAuthorization() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            isAuthorized = true
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
                DispatchQueue.main.async {
                    self?.isAuthorized = granted
                }
            }
        default:
            isAuthorized = false
        }
    }

    // MARK: - Session Setup
    func setupSession() {
        guard isAuthorized else { return }

        session.beginConfiguration()

        // 🆕 Virtual Device 우선 탐색 (심리스 줌 전환 지원)
        let camera = findBestBackCamera()

        guard let camera = camera else {
            session.commitConfiguration()
            return
        }

        currentCamera = camera
        currentLens = .wide
        isFrontCamera = false

        do {
            let input = try AVCaptureDeviceInput(device: camera)
            if session.canAddInput(input) {
                session.addInput(input)
                currentInput = input
            }

            // 🆕 고해상도 포맷 설정 (뿌옇게 나오는 문제 해결)
            configureHighQualityFormat(for: camera)

            // 비디오 출력 설정
            let videoQueue = DispatchQueue(label: "videoQueue", qos: .userInteractive, attributes: [])
            videoOutput.setSampleBufferDelegate(self, queue: videoQueue)
            videoOutput.videoSettings = [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
            ]

            if session.canAddOutput(videoOutput) {
                session.addOutput(videoOutput)
            }

            // 비디오 방향 설정
            if let connection = videoOutput.connection(with: .video) {
                connection.videoOrientation = .portrait
                connection.isVideoMirrored = false
            }

        } catch {
            print("❌ Camera setup error: \(error)")
        }

        session.commitConfiguration()

        // Virtual Device 줌 범위 설정 (commitConfiguration 이후)
        setupZoomFactors(for: camera)

        // 개별 렌즈 정보도 탐색 (UI 표시용)
        discoverAvailableLenses()

        print("📷 Virtual Device 사용: \(isUsingVirtualDevice)")
        print("📷 줌 범위: \(minZoomFactor)x ~ \(maxZoomFactor)x")
        print("📷 사용 가능한 렌즈: \(availableLenses.map { $0.displayName })")
    }

    // 🆕 고화질 포맷 설정 (30fps 보장)
    private func configureHighQualityFormat(for device: AVCaptureDevice) {
        // 30fps를 지원하는 포맷만 필터링
        let targetFPS: Float64 = 30.0

        let formats = device.formats.filter { format in
            let dimensions = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
            let mediaType = CMFormatDescriptionGetMediaSubType(format.formatDescription)

            // 420v 또는 420f 포맷 (표준 비디오 포맷)
            let isVideoFormat = mediaType == kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange ||
                               mediaType == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange

            // 최소 1080p 이상
            guard isVideoFormat && dimensions.height >= 1080 else { return false }

            // 30fps 지원 여부 확인
            let supports30fps = format.videoSupportedFrameRateRanges.contains { range in
                range.minFrameRate <= targetFPS && range.maxFrameRate >= targetFPS
            }

            return supports30fps
        }

        // 해상도 기준으로 정렬 (높은 것 우선, 하지만 1080p 선호)
        // 4K는 처리 부하가 크므로 1080p가 최적
        let sortedFormats = formats.sorted { f1, f2 in
            let d1 = CMVideoFormatDescriptionGetDimensions(f1.formatDescription)
            let d2 = CMVideoFormatDescriptionGetDimensions(f2.formatDescription)

            // 1080p (1920x1080)를 우선 선택
            let is1080p_1 = d1.height == 1080 || d1.width == 1920
            let is1080p_2 = d2.height == 1080 || d2.width == 1920

            if is1080p_1 && !is1080p_2 { return true }
            if !is1080p_1 && is1080p_2 { return false }

            // 같은 등급이면 해상도 높은 것 선택
            return d1.width * d1.height > d2.width * d2.height
        }

        if let bestFormat = sortedFormats.first {
            do {
                try device.lockForConfiguration()
                device.activeFormat = bestFormat

                // 30fps 설정
                device.activeVideoMinFrameDuration = CMTime(value: 1, timescale: 30)
                device.activeVideoMaxFrameDuration = CMTime(value: 1, timescale: 30)

                device.unlockForConfiguration()

                let dimensions = CMVideoFormatDescriptionGetDimensions(bestFormat.formatDescription)
                let maxFPS = bestFormat.videoSupportedFrameRateRanges.map { $0.maxFrameRate }.max() ?? 0
                print("📷 포맷 설정: \(dimensions.width)x\(dimensions.height) @ 30fps (최대 \(Int(maxFPS))fps 지원)")
            } catch {
                print("❌ 포맷 설정 실패: \(error)")
                if session.canSetSessionPreset(.hd1920x1080) {
                    session.sessionPreset = .hd1920x1080
                }
            }
        } else {
            // 적합한 포맷이 없으면 1080p preset 사용
            if session.canSetSessionPreset(.hd1920x1080) {
                session.sessionPreset = .hd1920x1080
                print("📷 기본 preset 사용: 1920x1080")
            }
        }
    }

    // 🆕 최적의 후면 카메라 찾기 (Virtual Device 우선)
    private func findBestBackCamera() -> AVCaptureDevice? {
        // 1순위: Triple Camera (0.5x, 1x, 3x) - iPhone 11 Pro 이상
        if let tripleCamera = AVCaptureDevice.default(.builtInTripleCamera, for: .video, position: .back) {
            isUsingVirtualDevice = true
            print("📷 Triple Camera 사용 (심리스 줌 지원)")
            return tripleCamera
        }

        // 2순위: Dual Wide Camera (0.5x, 1x) - iPhone 11 이상
        if let dualWideCamera = AVCaptureDevice.default(.builtInDualWideCamera, for: .video, position: .back) {
            isUsingVirtualDevice = true
            print("📷 Dual Wide Camera 사용 (심리스 줌 지원)")
            return dualWideCamera
        }

        // 3순위: Dual Camera (1x, 2x) - iPhone 7 Plus ~ iPhone X
        if let dualCamera = AVCaptureDevice.default(.builtInDualCamera, for: .video, position: .back) {
            isUsingVirtualDevice = true
            print("📷 Dual Camera 사용 (심리스 줌 지원)")
            return dualCamera
        }

        // 4순위: Wide Angle Camera (1x만) - 모든 iPhone
        isUsingVirtualDevice = false
        print("📷 Wide Angle Camera 사용 (단일 렌즈)")
        return AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back)
    }

    // 🆕 Virtual Device 렌즈 전환 포인트 (0.5x→1x, 1x→2x 전환 줌 팩터)
    private var switchOverZoomFactors: [CGFloat] = []
    private var wideAngleZoomFactor: CGFloat = 2.0  // 광각(1x) 줌 팩터 (기본값)
    private var telephotoZoomFactor: CGFloat = 4.0  // 망원(2x) 줌 팩터 (기본값)

    // 🆕 줌 팩터 범위 설정
    private func setupZoomFactors(for device: AVCaptureDevice) {
        minZoomFactor = device.minAvailableVideoZoomFactor
        // 사용자 표시 15x = 실제 videoZoomFactor 30.0 (wideAngleZoomFactor=2.0 기준)
        maxZoomFactor = min(device.maxAvailableVideoZoomFactor, 30.0)

        // 🆕 Virtual Device의 렌즈 전환 포인트 가져오기
        // Triple Camera: [2.0, 4.0] → 1.0=초광각, 2.0=광각, 4.0=망원
        // Dual Wide: [2.0] → 1.0=초광각, 2.0=광각
        switchOverZoomFactors = device.virtualDeviceSwitchOverVideoZoomFactors.map { CGFloat(truncating: $0) }

        if let firstSwitchOver = switchOverZoomFactors.first {
            wideAngleZoomFactor = firstSwitchOver  // 광각 렌즈 시작점
        }
        if switchOverZoomFactors.count > 1 {
            telephotoZoomFactor = switchOverZoomFactors[1]  // 망원 렌즈 시작점
        }

        print("📷 렌즈 전환 포인트: \(switchOverZoomFactors)")
        print("📷 광각(1x) = videoZoomFactor \(wideAngleZoomFactor)")

        // 초기값 설정 (광각 렌즈 = 사용자에게 1x로 표시)
        currentZoom = wideAngleZoomFactor
        virtualZoom = 1.0  // 사용자에게 보이는 값
        currentLens = .wide

        // 🆕 광각(1x)으로 시작 (wideAngleZoomFactor 사용)
        do {
            try device.lockForConfiguration()
            device.cancelVideoZoomRamp()
            device.videoZoomFactor = wideAngleZoomFactor
            device.unlockForConfiguration()
            print("📷 초기 줌 광각(1x)으로 설정 완료: videoZoomFactor = \(wideAngleZoomFactor)")
        } catch {
            print("❌ 초기 줌 설정 실패: \(error)")
        }

        print("📷 디바이스 줌 범위: \(device.minAvailableVideoZoomFactor) ~ \(device.maxAvailableVideoZoomFactor)")
    }

    // 🆕 사용 가능한 렌즈 탐지 (UI 표시용)
    private func discoverAvailableLenses() {
        availableCameras.removeAll()
        var lenses: [CameraLensType] = []

        // Virtual Device 사용 시에도 개별 렌즈 정보 필요 (UI 버튼 표시)
        let deviceTypes: [AVCaptureDevice.DeviceType] = [
            .builtInUltraWideCamera,
            .builtInWideAngleCamera,
            .builtInTelephotoCamera
        ]

        let discoverySession = AVCaptureDevice.DiscoverySession(
            deviceTypes: deviceTypes,
            mediaType: .video,
            position: .back
        )

        for device in discoverySession.devices {
            switch device.deviceType {
            case .builtInUltraWideCamera:
                availableCameras[.ultraWide] = device
                lenses.append(.ultraWide)
            case .builtInWideAngleCamera:
                availableCameras[.wide] = device
                lenses.append(.wide)
            case .builtInTelephotoCamera:
                availableCameras[.telephoto] = device
                lenses.append(.telephoto)
            default:
                break
            }
        }

        availableLenses = lenses.sorted { $0.zoomFactor < $1.zoomFactor }
    }

    // MARK: - Session Control
    func startSession() {
        guard !isSessionRunning else { return }

        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            self?.session.startRunning()
            DispatchQueue.main.async {
                self?.isSessionRunning = true
            }
        }
    }

    func stopSession() {
        guard isSessionRunning else { return }

        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            self?.session.stopRunning()
            DispatchQueue.main.async {
                self?.isSessionRunning = false
            }
        }
    }

    // MARK: - Camera Settings
    func applyCameraSettings(_ settings: CameraSettings) {
        guard let device = currentCamera else { return }

        do {
            try device.lockForConfiguration()

            // ISO 설정
            if let iso = settings.iso {
                let isoValue = Float(iso)
                let clampedISO = min(max(isoValue, device.activeFormat.minISO), device.activeFormat.maxISO)
                device.setExposureModeCustom(duration: AVCaptureDevice.currentExposureDuration, iso: clampedISO)
                currentISO = clampedISO
            }

            // 노출 보정 (EV)
            if let ev = settings.evCompensation {
                let evValue = Float(ev)
                let clampedEV = min(max(evValue, device.minExposureTargetBias), device.maxExposureTargetBias)
                device.setExposureTargetBias(clampedEV)
                currentExposureCompensation = clampedEV
            }

            // 화이트밸런스 (Kelvin)
            if let kelvin = settings.wbKelvin {
                // AVFoundation은 Kelvin 직접 설정을 지원하지 않으므로
                // Temperature/Tint 기반으로 근사치 설정
                let temp = kelvinToTemperature(kelvin)
                let gains = AVCaptureDevice.WhiteBalanceGains(
                    redGain: temp.red,
                    greenGain: 1.0,
                    blueGain: temp.blue
                )
                device.setWhiteBalanceModeLocked(with: gains)
            }

            device.unlockForConfiguration()

        } catch {
            print("❌ Failed to apply camera settings: \(error)")
        }
    }

    // Kelvin을 RGB gain으로 근사 변환
    private func kelvinToTemperature(_ kelvin: Int) -> (red: Float, blue: Float) {
        switch kelvin {
        case ..<3500:
            return (2.2, 1.1)
        case 3500..<4500:
            return (1.8, 1.3)
        case 4500..<5500:
            return (1.5, 1.5)
        case 5500..<6500:
            return (1.3, 1.8)
        default:
            return (1.1, 2.2)
        }
    }

    // MARK: - Camera Switch
    func switchCamera() {
        guard let input = currentInput else { return }

        session.beginConfiguration()

        // 현재 입력 제거
        session.removeInput(input)

        // 반대 카메라 선택
        let newPosition: AVCaptureDevice.Position = (currentCamera?.position == .back) ? .front : .back

        guard let newCamera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: newPosition) else {
            session.commitConfiguration()
            return
        }

        do {
            let newInput = try AVCaptureDeviceInput(device: newCamera)
            if session.canAddInput(newInput) {
                session.addInput(newInput)
                currentInput = newInput
                currentCamera = newCamera
                isFrontCamera = (newPosition == .front)  // 카메라 위치 업데이트

                // 전면/후면 카메라에 따라 미러링 설정
                if let connection = videoOutput.connection(with: .video) {
                    // 전면 카메라: 미러링 활성화 (거울처럼)
                    // 후면 카메라: 미러링 비활성화
                    connection.isVideoMirrored = (newPosition == .front)

                    // 전면/후면 모두 portrait 방향 사용
                    connection.videoOrientation = .portrait
                }
            }
        } catch {
            print("❌ Failed to switch camera: \(error)")
        }

        session.commitConfiguration()
    }

    // MARK: - Flash Control
    func toggleFlash() {
        guard let device = currentCamera else { return }

        guard device.hasTorch && device.hasFlash else {
            print("⚠️ Flash not available on this camera")
            return
        }

        do {
            try device.lockForConfiguration()

            if isFlashOn {
                // Flash OFF
                if device.torchMode == .on {
                    device.torchMode = .off
                }
                isFlashOn = false
            } else {
                // Flash ON
                if device.isTorchModeSupported(.on) {
                    device.torchMode = .on
                }
                isFlashOn = true
            }

            device.unlockForConfiguration()
        } catch {
            print("❌ Failed to toggle flash: \(error)")
        }
    }

    // MARK: - Zoom Control (Virtual Device 심리스 줌)

    /// 가상 줌 팩터 (사용자에게 표시되는 값: 0.5x, 1x, 2x 등)
    @Published var virtualZoom: CGFloat = 1.0

    /// 사용자 표시 줌 → 실제 videoZoomFactor 변환
    private func displayZoomToDeviceZoom(_ displayZoom: CGFloat) -> CGFloat {
        // Virtual Device에서:
        // - 사용자 0.5x = videoZoomFactor 1.0 (초광각)
        // - 사용자 1.0x = videoZoomFactor 2.0 (광각) = wideAngleZoomFactor
        // - 사용자 2.0x = videoZoomFactor 4.0 (망원) = telephotoZoomFactor

        if !isUsingVirtualDevice {
            return displayZoom  // Virtual Device 아니면 그대로
        }

        // 0.5x 기준으로 스케일 계산 (0.5x = 1.0, 1.0x = 2.0, 2.0x = 4.0)
        return displayZoom * wideAngleZoomFactor
    }

    /// 실제 videoZoomFactor → 사용자 표시 줌 변환
    private func deviceZoomToDisplayZoom(_ deviceZoom: CGFloat) -> CGFloat {
        if !isUsingVirtualDevice {
            return deviceZoom
        }
        return deviceZoom / wideAngleZoomFactor
    }

    /// 줌 설정 (사용자 표시 줌 기준, 예: 0.5, 1.0, 2.0)
    func setZoom(_ displayFactor: CGFloat) {
        guard let device = currentCamera else { return }

        // 사용자 표시 줌 → 실제 디바이스 줌으로 변환
        let deviceFactor = displayZoomToDeviceZoom(displayFactor)
        let clampedFactor = max(minZoomFactor, min(deviceFactor, maxZoomFactor))

        do {
            try device.lockForConfiguration()

            // 핀치 줌: 빠른 반응
            device.ramp(toVideoZoomFactor: clampedFactor, withRate: 150.0)

            currentZoom = clampedFactor
            virtualZoom = deviceZoomToDisplayZoom(clampedFactor)

            updateCurrentLensDisplay(for: clampedFactor)

            device.unlockForConfiguration()
        } catch {
            print("❌ Failed to set zoom: \(error)")
        }
    }

    /// 핀치 줌 적용
    func applyPinchZoom(_ scale: CGFloat) {
        let newDisplayZoom = virtualZoom * scale
        setZoom(newDisplayZoom)
    }

    /// 특정 배율로 부드럽게 줌 (버튼 클릭 시, 사용자 표시 줌 기준)
    func setZoomAnimated(_ displayFactor: CGFloat) {
        guard let device = currentCamera else { return }

        let deviceFactor = displayZoomToDeviceZoom(displayFactor)
        let clampedFactor = max(minZoomFactor, min(deviceFactor, maxZoomFactor))

        do {
            try device.lockForConfiguration()

            // 버튼 클릭 시 더 부드러운 전환
            device.ramp(toVideoZoomFactor: clampedFactor, withRate: 30.0)

            currentZoom = clampedFactor
            virtualZoom = deviceZoomToDisplayZoom(clampedFactor)

            updateCurrentLensDisplay(for: clampedFactor)

            device.unlockForConfiguration()
        } catch {
            print("❌ Failed to set zoom: \(error)")
        }
    }

    /// 현재 줌에 따라 렌즈 표시 업데이트 (실제 videoZoomFactor 기준)
    private func updateCurrentLensDisplay(for deviceZoom: CGFloat) {
        // 실제 videoZoomFactor 기준으로 렌즈 결정
        if deviceZoom < wideAngleZoomFactor {
            currentLens = .ultraWide
        } else if deviceZoom >= telephotoZoomFactor && availableCameras[.telephoto] != nil {
            currentLens = .telephoto
        } else {
            currentLens = .wide
        }
    }

    // MARK: - Lens Control (버튼으로 렌즈 전환)

    /// 특정 렌즈로 전환 (사용자 표시 줌 기준: 0.5x, 1x, 2x)
    func switchLens(to lens: CameraLensType) {
        guard !isFrontCamera else {
            print("⚠️ 전면 카메라에서는 렌즈 전환 불가")
            return
        }

        // 사용자 표시 줌으로 전환 (0.5, 1.0, 2.0)
        // setZoomAnimated가 내부에서 실제 deviceZoom으로 변환
        setZoomAnimated(lens.zoomFactor)
    }

    /// 다음 렌즈로 순환 전환 (0.5x → 1x → 2x → 0.5x ...)
    func cycleToNextLens() {
        guard availableLenses.count > 1 else { return }

        if let currentIndex = availableLenses.firstIndex(of: currentLens) {
            let nextIndex = (currentIndex + 1) % availableLenses.count
            let nextLens = availableLenses[nextIndex]
            switchLens(to: nextLens)
        }
    }

    // MARK: - Aspect Ratio Control
    func setAspectRatio(_ ratio: CameraAspectRatio) {
        guard aspectRatio != ratio else { return }

        aspectRatio = ratio

        // 세션 재구성
        session.beginConfiguration()

        // 비율에 따라 적절한 preset 설정
        switch ratio {
        case .ratio16_9:
            if session.canSetSessionPreset(.hd1920x1080) {
                session.sessionPreset = .hd1920x1080
            }
        case .ratio4_3:
            if session.canSetSessionPreset(.photo) {
                session.sessionPreset = .photo
            }
        case .ratio1_1:
            // 1:1은 별도 preset이 없으므로 .photo 사용 후 크롭
            if session.canSetSessionPreset(.photo) {
                session.sessionPreset = .photo
            }
        }

        session.commitConfiguration()

        print("📷 Camera aspect ratio changed to: \(ratio.rawValue)")
    }
}

// MARK: - AVCaptureVideoDataOutputSampleBufferDelegate
extension CameraManager: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        let frameStart = CACurrentMediaTime()  // 🔍 프로파일링

        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }

        // 🔥 타임스탬프 체크로 중복 버퍼 방지
        let timestamp = CMSampleBufferGetPresentationTimeStamp(sampleBuffer).seconds
        if timestamp == lastBufferTime { return }
        lastBufferTime = timestamp

        // CVPixelBuffer → UIImage 변환 (최적화된 방식)
        let convertStart = CACurrentMediaTime()  // 🔍
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)

        // 재사용 가능한 ciContext 사용
        guard let cgImage = ciContext.createCGImage(ciImage, from: ciImage.extent) else { return }
        let convertEnd = CACurrentMediaTime()  // 🔍

        // 디바이스 방향에 따라 적절한 orientation 설정
        let deviceOrientation = UIDevice.current.orientation
        var imageOrientation: UIImage.Orientation = .right  // 기본값 (세로)

        // 후면 카메라 기준으로 orientation 매핑
        if currentCamera?.position == .back {
            switch deviceOrientation {
            case .portrait:
                imageOrientation = .up
            case .portraitUpsideDown:
                imageOrientation = .down
            case .landscapeLeft:
                imageOrientation = .right  // landscapeRight와 같은 값 사용
            case .landscapeRight:
                imageOrientation = .right
            default:
                imageOrientation = .up
            }
        } else {
            // 전면 카메라: 화면에 보이는 그대로 저장 (회전 없음)
            switch deviceOrientation {
            case .portrait:
                imageOrientation = .up  // 회전 없음
            case .portraitUpsideDown:
                imageOrientation = .down
            case .landscapeLeft:
                imageOrientation = .up
            case .landscapeRight:
                imageOrientation = .up
            default:
                imageOrientation = .up
            }
        }

        let image = UIImage(cgImage: cgImage, scale: 1.0, orientation: imageOrientation)

        let frameEnd = CACurrentMediaTime()  // 🔍

        // FPS 계산
        fpsFrameCount += 1
        let now = Date()
        let elapsed = now.timeIntervalSince(lastFPSUpdate)

        if elapsed >= 1.0 {
            let fps = Double(fpsFrameCount) / elapsed

            // 🔍 프로파일링 로그 (1초마다)
            let convertTime = (convertEnd - convertStart) * 1000
            let totalTime = (frameEnd - frameStart) * 1000
            print("📊 [CameraManager] 이미지변환: \(String(format: "%.1f", convertTime))ms, 총: \(String(format: "%.1f", totalTime))ms, FPS: \(String(format: "%.1f", fps))")

            DispatchQueue.main.async { [weak self] in
                self?.currentFPS = fps
            }
            fpsFrameCount = 0
            lastFPSUpdate = now
        }

        DispatchQueue.main.async { [weak self] in
            self?.currentFrame = image
        }
    }
}
