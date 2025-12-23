import SwiftUI

struct ContentView: View {
    // MARK: - State
    @StateObject private var cameraManager = CameraManager()
    @StateObject private var realtimeAnalyzer = RealtimeAnalyzer()  // 실시간 분석
    @State private var referenceImage: UIImage?
    @State private var feedbackItems: [FeedbackItem] = []
    @State private var serverFeedbackItems: [FeedbackItem] = []  // 서버 피드백 (포즈 등)
    @State private var processingTime: String = ""
    @State private var isAnalyzing = false
    @State private var errorMessage: String?
    @State private var analysisTimer: Timer?
    @State private var frameUpdateTimer: Timer?  // 실시간 프레임 분석용

    // 안드로이드 기능 추가
    @State private var showGrid = false
    @State private var showFPS = false
    @State private var zoomLevel: CGFloat = 1.0
    @State private var analysisEnabled = true  // 분석 모드 on/off
    @State private var autoCapture = true  // 자동 촬영 모드
    @State private var capturedImage: UIImage?  // 촬영된 이미지
    @State private var showCaptureFlash = false  // 촬영 플래시 효과

    // 통합 피드백 (실시간 + 서버)
    private var combinedFeedback: [FeedbackItem] {
        var combined: [FeedbackItem] = []

        // 1순위: 실시간 피드백 (프레이밍, 구도)
        combined.append(contentsOf: realtimeAnalyzer.instantFeedback)

        // 2순위: 서버 피드백 (포즈) - 실시간 피드백과 중복되지 않는 것만
        let realtimeCategories = Set(realtimeAnalyzer.instantFeedback.map { $0.category })
        let uniqueServerFeedback = serverFeedbackItems.filter {
            !realtimeCategories.contains($0.category) && $0.category == "pose"
        }
        combined.append(contentsOf: uniqueServerFeedback)

        // 우선순위로 정렬하고 상위 5개만 반환
        return Array(combined.sorted { $0.priority < $1.priority }.prefix(5))
    }

    // 피드백 업데이트
    private func updateCombinedFeedback() {
        // combinedFeedback은 computed property라서 자동 업데이트됨
        // 필요시 추가 로직
    }

    // 사진 촬영
    private func performCapture() {
        guard let currentFrame = cameraManager.currentFrame else { return }

        // 플래시 효과
        withAnimation(.easeInOut(duration: 0.2)) {
            showCaptureFlash = true
        }

        // 이미지 저장
        capturedImage = currentFrame

        // 사진 앨범에 저장
        UIImageWriteToSavedPhotosAlbum(currentFrame, nil, nil, nil)

        // 플래시 효과 제거
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.2) {
            withAnimation(.easeInOut(duration: 0.2)) {
                showCaptureFlash = false
            }

            // 성공 알림
            print("📸 사진 촬영 완료!")
        }

        // 5초 후 다시 촬영 가능
        DispatchQueue.main.asyncAfter(deadline: .now() + 5.0) {
            capturedImage = nil
        }
    }

    var body: some View {
        ZStack {
            // 1. 카메라 프리뷰
            if cameraManager.isAuthorized {
                CameraView(cameraManager: cameraManager)
                    .ignoresSafeArea()
                    .onAppear {
                        cameraManager.setupSession()
                        cameraManager.startSession()
                    }
                    .onDisappear {
                        cameraManager.stopSession()
                        stopAnalysis()
                    }
            } else {
                // 권한 없을 때
                VStack(spacing: 20) {
                    Image(systemName: "camera.fill")
                        .font(.system(size: 60))
                        .foregroundColor(.gray)

                    Text("카메라 권한이 필요합니다")
                        .font(.title3)
                        .foregroundColor(.white)

                    Text("설정 > TryAngle > 카메라 허용")
                        .font(.caption)
                        .foregroundColor(.gray)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .background(Color.black)
            }

            // 2. 그리드 오버레이
            if showGrid {
                GridOverlay()
                    .ignoresSafeArea()
            }

            // 3. 상단 툴바
            VStack {
                // 첫번째 행: 그리드, 플래시, 분석 모드
                HStack(spacing: 16) {
                    // 그리드 토글
                    Button(action: {
                        showGrid.toggle()
                    }) {
                        Image(systemName: showGrid ? "grid" : "grid.circle")
                            .font(.title3)
                            .foregroundColor(.white)
                            .frame(width: 44, height: 44)
                            .background(Color.black.opacity(0.5))
                            .clipShape(Circle())
                    }

                    // Flash 토글
                    Button(action: {
                        cameraManager.toggleFlash()
                    }) {
                        Image(systemName: cameraManager.isFlashOn ? "bolt.fill" : "bolt.slash.fill")
                            .font(.title3)
                            .foregroundColor(cameraManager.isFlashOn ? .yellow : .white)
                            .frame(width: 44, height: 44)
                            .background(Color.black.opacity(0.5))
                            .clipShape(Circle())
                    }

                    // 분석 모드 토글
                    Button(action: {
                        analysisEnabled.toggle()
                        if !analysisEnabled {
                            feedbackItems = []
                            processingTime = ""
                        }
                    }) {
                        Image(systemName: analysisEnabled ? "wand.and.stars" : "wand.and.stars.inverse")
                            .font(.title3)
                            .foregroundColor(analysisEnabled ? .cyan : .white)
                            .frame(width: 44, height: 44)
                            .background(Color.black.opacity(0.5))
                            .clipShape(Circle())
                    }

                    // 자동 촬영 토글
                    Button(action: {
                        autoCapture.toggle()
                    }) {
                        Image(systemName: autoCapture ? "camera.fill" : "camera")
                            .font(.title3)
                            .foregroundColor(autoCapture ? .yellow : .white)
                            .frame(width: 44, height: 44)
                            .background(Color.black.opacity(0.5))
                            .clipShape(Circle())
                    }

                    Spacer()
                }
                .padding(.horizontal, 16)
                .padding(.top, 60)

                // 두번째 행: FPS 토글 및 정보
                HStack(spacing: 16) {
                    // FPS 토글
                    Button(action: {
                        showFPS.toggle()
                    }) {
                        Image(systemName: showFPS ? "info.circle.fill" : "info.circle")
                            .font(.title3)
                            .foregroundColor(.white)
                            .frame(width: 44, height: 44)
                            .background(Color.black.opacity(0.5))
                            .clipShape(Circle())
                    }

                    // FPS 표시
                    if showFPS {
                        Text(String(format: "%.1f FPS", cameraManager.currentFPS))
                            .font(.caption)
                            .foregroundColor(.white)
                            .padding(.horizontal, 12)
                            .padding(.vertical, 6)
                            .background(Color.black.opacity(0.6))
                            .cornerRadius(8)
                    }

                    Spacer()
                }
                .padding(.horizontal, 16)
                .padding(.top, 8)

                Spacer()
            }

            // 4. 피드백 오버레이 (실시간 + 서버 피드백 통합)
            FeedbackOverlay(
                feedbackItems: combinedFeedback,
                processingTime: processingTime
            )
            .onChange(of: realtimeAnalyzer.instantFeedback) { newFeedback in
                updateCombinedFeedback()
            }
            .onChange(of: serverFeedbackItems) { _ in
                updateCombinedFeedback()
            }

            // 완벽한 상태 표시
            if realtimeAnalyzer.isPerfect {
                VStack {
                    Spacer()
                        .frame(height: 200)

                    HStack {
                        Spacer()
                        VStack(spacing: 16) {
                            // 완벽 표시
                            ZStack {
                                Circle()
                                    .fill(Color.green.opacity(0.9))
                                    .frame(width: 100, height: 100)
                                    .overlay(
                                        Circle()
                                            .stroke(Color.white, lineWidth: 4)
                                    )

                                VStack(spacing: 4) {
                                    Image(systemName: "checkmark")
                                        .font(.system(size: 40, weight: .bold))
                                        .foregroundColor(.white)
                                    Text("완벽!")
                                        .font(.headline)
                                        .foregroundColor(.white)
                                }
                            }
                            .scaleEffect(showCaptureFlash ? 1.2 : 1.0)
                            .animation(.easeInOut(duration: 0.5).repeatForever(autoreverses: true), value: showCaptureFlash)

                            // 자동 촬영 카운트다운
                            if autoCapture {
                                Text("자동 촬영!")
                                    .font(.caption)
                                    .foregroundColor(.white)
                                    .padding(.horizontal, 16)
                                    .padding(.vertical, 8)
                                    .background(Color.green.opacity(0.8))
                                    .cornerRadius(20)
                            }
                        }
                        Spacer()
                    }

                    Spacer()
                }
            }

            // 촬영 플래시 효과
            if showCaptureFlash {
                Color.white
                    .ignoresSafeArea()
                    .opacity(0.8)
                    .transition(.opacity)
            }

            // 완성도 점수 표시 (디버깅용)
            if showFPS {
                VStack {
                    HStack {
                        Spacer()
                        Text(String(format: "완성도: %.0f%%", realtimeAnalyzer.perfectScore * 100))
                            .font(.caption)
                            .foregroundColor(.white)
                            .padding(.horizontal, 12)
                            .padding(.vertical, 6)
                            .background(Color.blue.opacity(0.6))
                            .cornerRadius(8)
                            .padding(.trailing, 16)
                    }
                    .padding(.top, 200)
                    Spacer()
                }
            }

            // 5. 하단 컨트롤
            VStack {
                Spacer()

                HStack(alignment: .center, spacing: 20) {
                    // 레퍼런스 선택
                    ReferenceSelector(selectedImage: $referenceImage)
                        .onChange(of: referenceImage) { newImage in
                            if let image = newImage {
                                // 레퍼런스 분석
                                realtimeAnalyzer.analyzeReference(image)
                                startRealtimeAnalysis()  // 실시간 분석 시작
                                startAnalysis()          // 서버 분석도 병행 (포즈용)
                            } else {
                                stopRealtimeAnalysis()
                                stopAnalysis()
                            }
                        }

                    Spacer()

                    // 카메라 전환 버튼
                    Button(action: {
                        cameraManager.switchCamera()
                    }) {
                        Image(systemName: "camera.rotate")
                            .font(.title2)
                            .foregroundColor(.white)
                            .frame(width: 50, height: 50)
                            .background(Color.black.opacity(0.6))
                            .clipShape(Circle())
                    }
                }
                .padding(.horizontal, 20)
                .padding(.bottom, 40)
            }

            // 4. 에러 메시지
            if let error = errorMessage {
                VStack {
                    Text("⚠️ \(error)")
                        .font(.caption)
                        .foregroundColor(.white)
                        .padding()
                        .background(Color.red.opacity(0.8))
                        .cornerRadius(8)
                        .padding(.top, 100)

                    Spacer()
                }
            }

            // 5. 분석 중 인디케이터
            if isAnalyzing {
                VStack {
                    HStack {
                        ProgressView()
                            .progressViewStyle(CircularProgressViewStyle(tint: .white))
                        Text("분석 중...")
                            .font(.caption)
                            .foregroundColor(.white)
                    }
                    .padding()
                    .background(Color.black.opacity(0.7))
                    .cornerRadius(8)
                    .padding(.top, 100)

                    Spacer()
                }
            }
        }
        .onChange(of: realtimeAnalyzer.isPerfect) { isPerfect in
            if isPerfect && autoCapture && capturedImage == nil {
                performCapture()
            }
        }
    }

    // MARK: - Analysis Control

    /// 실시간 프레임 분석 시작 (클라이언트 사이드)
    private func startRealtimeAnalysis() {
        // 기존 타이머 중지
        stopRealtimeAnalysis()

        // 60fps로 프레임 분석 (16ms마다)
        frameUpdateTimer = Timer.scheduledTimer(withTimeInterval: 0.016, repeats: true) { _ in
            if let currentFrame = cameraManager.currentFrame {
                realtimeAnalyzer.analyzeFrame(currentFrame)
            }
        }
    }

    /// 실시간 분석 중지
    private func stopRealtimeAnalysis() {
        frameUpdateTimer?.invalidate()
        frameUpdateTimer = nil
        realtimeAnalyzer.instantFeedback = []
    }

    /// 서버 분석 시작 (포즈 등 복잡한 분석용)
    private func startAnalysis() {
        guard referenceImage != nil else { return }

        // 기존 타이머 중지
        stopAnalysis()

        // 2초마다 서버 분석 (포즈만)
        analysisTimer = Timer.scheduledTimer(withTimeInterval: 2.0, repeats: true) { _ in
            Task {
                await performAnalysis()
            }
        }
    }

    /// 서버 분석 중지
    private func stopAnalysis() {
        analysisTimer?.invalidate()
        analysisTimer = nil
        serverFeedbackItems = []
        processingTime = ""
    }

    /// 실제 분석 수행
    private func performAnalysis() async {
        // 분석 모드가 꺼져있으면 스킵
        guard analysisEnabled else {
            return
        }

        // 이미 분석 중이면 스킵 (중복 요청 방지)
        guard !isAnalyzing else {
            print("⏭️ 이전 분석이 진행 중이므로 스킵")
            return
        }

        guard let refImage = referenceImage,
              let currentFrame = cameraManager.currentFrame else {
            return
        }

        isAnalyzing = true
        errorMessage = nil

        do {
            let response = try await APIService.shared.analyzeFrame(
                referenceImage: refImage,
                currentFrame: currentFrame
            )

            // UI 업데이트 (메인 스레드)
            await MainActor.run {
                serverFeedbackItems = response.userFeedback  // 서버 피드백 별도 저장
                processingTime = response.processingTime

                // 카메라 설정 자동 적용 비활성화 (초록색 문제 때문에)
                // TODO: 설정을 수동으로 조정할 수 있는 UI 추가
                // if analysisEnabled {
                //     cameraManager.applyCameraSettings(response.cameraSettings)
                // }

                isAnalyzing = false
            }

        } catch {
            await MainActor.run {
                errorMessage = "서버 연결 실패: \(error.localizedDescription)"
                isAnalyzing = false
            }
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
