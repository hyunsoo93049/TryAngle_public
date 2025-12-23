import SwiftUI
import AVFoundation

struct CameraView: UIViewRepresentable {
    let cameraManager: CameraManager

    func makeUIView(context: Context) -> UIView {
        let view = UIView(frame: .zero)
        view.backgroundColor = .black

        let previewLayer = cameraManager.previewLayer
        previewLayer.frame = view.bounds
        view.layer.addSublayer(previewLayer)

        // 핀치 제스처 추가
        let pinchGesture = UIPinchGestureRecognizer(target: context.coordinator, action: #selector(Coordinator.handlePinch(_:)))
        view.addGestureRecognizer(pinchGesture)

        return view
    }

    func updateUIView(_ uiView: UIView, context: Context) {
        if let previewLayer = uiView.layer.sublayers?.first as? AVCaptureVideoPreviewLayer {
            DispatchQueue.main.async {
                previewLayer.frame = uiView.bounds
            }
        }
    }

    func makeCoordinator() -> Coordinator {
        Coordinator(cameraManager: cameraManager)
    }

    class Coordinator: NSObject {
        let cameraManager: CameraManager
        private var lastScale: CGFloat = 1.0

        init(cameraManager: CameraManager) {
            self.cameraManager = cameraManager
        }

        @objc func handlePinch(_ gesture: UIPinchGestureRecognizer) {
            switch gesture.state {
            case .began:
                lastScale = gesture.scale
            case .changed:
                let deltaScale = gesture.scale / lastScale
                cameraManager.applyPinchZoom(deltaScale)
                lastScale = gesture.scale
            case .ended, .cancelled:
                lastScale = 1.0
            default:
                break
            }
        }
    }
}
