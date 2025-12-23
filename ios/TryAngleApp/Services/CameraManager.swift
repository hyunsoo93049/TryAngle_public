import AVFoundation
import UIKit
import Combine

class CameraManager: NSObject, ObservableObject {
    // MARK: - Published Properties
    @Published var isAuthorized = false
    @Published var currentFrame: UIImage?
    @Published var isSessionRunning = false
    @Published var isFlashOn = false
    @Published var currentFPS: Double = 0.0
    @Published var currentZoom: CGFloat = 1.0

    // MARK: - Camera Properties
    private let session = AVCaptureSession()
    private var videoOutput = AVCaptureVideoDataOutput()
    private var currentCamera: AVCaptureDevice?
    private var currentInput: AVCaptureDeviceInput?

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
        layer.videoGravity = .resizeAspectFill
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

        // 세션 preset 설정
        if session.canSetSessionPreset(.photo) {
            session.sessionPreset = .photo
        }

        // 카메라 디바이스 설정 (후면 카메라)
        guard let camera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else {
            session.commitConfiguration()
            return
        }

        currentCamera = camera

        do {
            // 입력 추가
            let input = try AVCaptureDeviceInput(device: camera)
            if session.canAddInput(input) {
                session.addInput(input)
                currentInput = input
            }

            // 비디오 출력 설정
            videoOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "videoQueue"))
            videoOutput.videoSettings = [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
            ]

            if session.canAddOutput(videoOutput) {
                session.addOutput(videoOutput)
            }

            // 비디오 방향 설정
            if let connection = videoOutput.connection(with: .video) {
                connection.videoOrientation = .portrait
            }

        } catch {
            print("❌ Camera setup error: \(error)")
        }

        session.commitConfiguration()
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

    // MARK: - Zoom Control
    func setZoom(_ factor: CGFloat) {
        guard let device = currentCamera else { return }

        do {
            try device.lockForConfiguration()

            // 줌 범위 제한 (1.0 ~ maxZoomFactor)
            let maxZoom = min(device.activeFormat.videoMaxZoomFactor, 10.0) // 최대 10배로 제한
            let zoom = max(1.0, min(factor, maxZoom))

            device.videoZoomFactor = zoom
            currentZoom = zoom

            device.unlockForConfiguration()
        } catch {
            print("❌ Failed to set zoom: \(error)")
        }
    }

    func applyPinchZoom(_ scale: CGFloat) {
        let newZoom = currentZoom * scale
        setZoom(newZoom)
    }
}

// MARK: - AVCaptureVideoDataOutputSampleBufferDelegate
extension CameraManager: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }

        // CVPixelBuffer → UIImage 변환
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let context = CIContext()

        guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else { return }

        let image = UIImage(cgImage: cgImage, scale: 1.0, orientation: .right)

        // FPS 계산
        fpsFrameCount += 1
        let now = Date()
        let elapsed = now.timeIntervalSince(lastFPSUpdate)

        if elapsed >= 1.0 {
            let fps = Double(fpsFrameCount) / elapsed
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
