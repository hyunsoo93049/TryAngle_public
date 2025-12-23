import SwiftUI
import Photos

struct ContentView: View {
    // MARK: - State
    @Binding var referenceImage: UIImage?  // 레퍼런스 이미지 (MainTabView에서 전달)
    @StateObject private var cameraManager = CameraManager()
    @StateObject private var realtimeAnalyzer = RealtimeAnalyzer()  // 실시간 분석
    @StateObject private var thermalManager = ThermalStateManager()  // 🔥 발열/배터리 관리
    @State private var feedbackItems: [FeedbackItem] = []
    @State private var serverFeedbackItems: [FeedbackItem] = []  // 서버 피드백 (포즈 등)
    @State private var processingTime: String = ""
    @State private var isAnalyzing = false
    @State private var errorMessage: String?
    @State private var analysisTimer: Timer?
    @State private var frameUpdateTimer: Timer?  // 실시간 프레임 분석용

    // UI 상태
    @State private var showGrid = false
    @State private var showFPS = false
    @State private var autoCapture = true
    @State private var capturedImage: UIImage?
    @State private var showCaptureFlash = false
    @State private var selectedAspectRatio: CameraAspectRatio = .ratio4_3
    @State private var showSettings = false  // 설정 시트
    @State private var showCameraOptions = false  // 카메라 옵션 펼침/접기

    // AI 분석은 레퍼런스 선택 시 자동 활성화
    private var analysisEnabled: Bool {
        referenceImage != nil
    }

    // 통합 피드백 (실시간 + 서버)
    private var combinedFeedback: [FeedbackItem] {
        var combined: [FeedbackItem] = []

        // 1순위: 실시간 피드백 (프레이밍, 구도)
        combined.append(contentsOf: realtimeAnalyzer.instantFeedback)

        // 🔍 디버그
        if !realtimeAnalyzer.instantFeedback.isEmpty {
            print("🎯 ContentView combinedFeedback: \(realtimeAnalyzer.instantFeedback.count)개")
        }

        // 2순위: 서버 피드백 (포즈) - 실시간 피드백과 중복되지 않는 것만
        let realtimeCategories = Set(realtimeAnalyzer.instantFeedback.map { $0.category })
        let uniqueServerFeedback = serverFeedbackItems.filter {
            !realtimeCategories.contains($0.category) && $0.category == "pose"
        }
        combined.append(contentsOf: uniqueServerFeedback)

        // 우선순위로 정렬하고 상위 5개만 반환
        let result = Array(combined.sorted { $0.priority < $1.priority }.prefix(5))

        if !result.isEmpty {
            print("✅ 최종 combinedFeedback: \(result.count)개")
        }

        return result
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

        let fixedImage: UIImage

        // 픽셀 비율 확인하여 landscape/portrait 처리
        guard let cgImage = currentFrame.cgImage else { return }
        let actualWidth = cgImage.width
        let actualHeight = cgImage.height

        if actualWidth > actualHeight {
            // 픽셀이 가로 방향 (landscape)이면 회전하지 않고 orientation만 .up으로 설정
            // 이렇게 해야 사진 앱에서 가로 사진으로 올바르게 표시됨
            fixedImage = UIImage(cgImage: cgImage, scale: currentFrame.scale, orientation: .up)
        } else {
            // 픽셀이 세로 방향 (portrait)이면 fixedOrientation() 적용
            fixedImage = currentFrame.fixedOrientation()
        }

        // 회전된 이미지를 크롭
        let croppedImage = cropImage(fixedImage, to: selectedAspectRatio)

        // 이미지 저장
        capturedImage = croppedImage

        // 🔧 사진 앨범에 저장
        saveImageToPhotoLibrary(croppedImage)

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

    // 이미지를 선택한 비율로 크롭
    private func cropImage(_ image: UIImage, to aspectRatio: CameraAspectRatio) -> UIImage {
        guard let cgImage = image.cgImage else { return image }

        let imageWidth = CGFloat(cgImage.width)
        let imageHeight = CGFloat(cgImage.height)
        let targetRatio = aspectRatio.ratio  // 가로:세로 비율 (예: 4:3 = 1.333)

        var cropRect: CGRect

        // fixedOrientation() 후의 이미지는 세로 모드
        // 세로 모드에서의 가로:세로 비율 계산
        let currentRatio = imageWidth / imageHeight  // 예: 3024 / 4032 = 0.75
        let targetVerticalRatio = 1.0 / targetRatio   // 예: 3/4 = 0.75

        if currentRatio > targetVerticalRatio {
            // 이미지가 목표보다 더 가로로 넓으면 (또는 덜 세로로 길면), 좌우를 크롭
            let targetWidth = imageHeight * targetVerticalRatio
            let xOffset = (imageWidth - targetWidth) / 2
            cropRect = CGRect(x: xOffset, y: 0, width: targetWidth, height: imageHeight)
        } else {
            // 이미지가 목표보다 더 세로로 길면, 위아래를 크롭
            let targetHeight = imageWidth / targetVerticalRatio
            let yOffset = (imageHeight - targetHeight) / 2
            cropRect = CGRect(x: 0, y: yOffset, width: imageWidth, height: targetHeight)
        }

        guard let croppedCGImage = cgImage.cropping(to: cropRect) else { return image }

        return UIImage(cgImage: croppedCGImage, scale: image.scale, orientation: image.imageOrientation)
    }

    // 🔧 사진을 올바른 방향으로 저장
    private func saveImageToPhotoLibrary(_ image: UIImage) {
        PHPhotoLibrary.requestAuthorization { status in
            guard status == .authorized else {
                print("⚠️ 사진 라이브러리 권한 없음")
                return
            }

            PHPhotoLibrary.shared().performChanges {
                // 이미지는 이미 CameraManager에서 fixedOrientation 처리됨
                let request = PHAssetChangeRequest.creationRequestForAsset(from: image)
                _ = request.placeholderForCreatedAsset
            } completionHandler: { success, error in
                if success {
                    print("✅ 사진 저장 성공")
                } else if let error = error {
                    print("❌ 사진 저장 실패: \(error.localizedDescription)")
                }
            }
        }
    }

    var body: some View {
        GeometryReader { geometry in
            let safeAreaTop = geometry.safeAreaInsets.top
            let _ = geometry.safeAreaInsets.bottom  // 예약용
            let screenHeight = geometry.size.height
            let screenWidth = geometry.size.width

            // 카메라 뷰박스 높이 계산 (AspectRatioMaskView와 동일)
            let captureHeight: CGFloat = {
                switch selectedAspectRatio {
                case .ratio4_3:
                    return screenWidth * 4.0 / 3.0
                case .ratio1_1:
                    return screenWidth
                case .ratio16_9:
                    return screenWidth * 16.0 / 9.0
                }
            }()
            let maskHeight = max(0, (screenHeight - captureHeight) / 2)

            ZStack {
                // 1. 카메라 프리뷰 (비율에 따라 캡처 영역 표시)
                if cameraManager.isAuthorized {
                ZStack {
                    // 전체 화면 카메라 프리뷰
                    CameraView(cameraManager: cameraManager)
                        .ignoresSafeArea()

                    // 비율에 따른 마스크 오버레이 (캡처되지 않는 영역 어둡게)
                    AspectRatioMaskView(selectedRatio: selectedAspectRatio)
                        .ignoresSafeArea()
                        .allowsHitTesting(false)
                }
                .onAppear {
                    cameraManager.setupSession()
                    cameraManager.startSession()
                    setupBackgroundHandling()
                }
                .onDisappear {
                    cameraManager.stopSession()
                    stopAnalysis()
                    stopRealtimeAnalysis()
                    removeBackgroundHandling()
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

            // 3. 피드백 오버레이 (실시간 + 서버 피드백 통합)
            // 🔥 버튼보다 먼저 선언하여 z-index를 낮춤
            // 🔥 카메라 뷰박스 영역으로 제한
            VStack(spacing: 0) {
                // 상단 safe area + 마스크 영역
                Spacer()
                    .frame(height: maskHeight)

                // 피드백 표시 영역 (카메라 뷰박스 내부만)
                FeedbackOverlay(
                    feedbackItems: combinedFeedback,
                    categoryStatuses: realtimeAnalyzer.categoryStatuses,
                    completedFeedbacks: realtimeAnalyzer.completedFeedbacks,
                    processingTime: processingTime
                )
                .frame(height: captureHeight)
                .clipped()  // 뷰박스 밖으로 나가는 것 방지
                .onChange(of: realtimeAnalyzer.instantFeedback) { newFeedback in
                    updateCombinedFeedback()
                }
                .onChange(of: serverFeedbackItems) { _ in
                    updateCombinedFeedback()
                }

                // 하단 마스크 영역
                Spacer()
                    .frame(height: maskHeight)
            }
            .allowsHitTesting(false)  // 피드백 오버레이는 터치 이벤트 통과

            // 4. 접었다 펼칠 수 있는 상단바 (오른쪽 아래)
            // 🔥 버튼들이 FeedbackOverlay 위에 오도록 나중에 선언
            VStack {
                Spacer()  // ← ① Spacer를 위로 이동 (버튼을 아래로 보냄)

                HStack {
                    Spacer()  // ← 오른쪽 정렬 유지

                    if showCameraOptions {
                        // 펼쳐진 상태: 검은색 배경 + 플래시, 비율, 설정, 닫기
                        ZStack{
                            //검은색 반투명 배경
                            Rectangle()
                                .foregroundColor(.clear)
                                .background(.black.opacity(0.48))
                                .cornerRadius(20)

                            HStack(spacing: 12) {
                                // 플래시
                                Button(action: {
                                    cameraManager.toggleFlash()
                                }) {
                                    Image(systemName: cameraManager.isFlashOn ? "bolt.fill" : "bolt.slash.fill")
                                        .font(.system(size: 20))
                                        .foregroundColor(cameraManager.isFlashOn ? .yellow : .white)
                                        .frame(width: 40, height: 40)
                                        .background(.ultraThinMaterial)
                                        .clipShape(Circle())
                                }

                                // 비율 선택
                                Menu {
                                    ForEach(CameraAspectRatio.allCases, id: \.self) { ratio in
                                        Button(action: {
                                            selectedAspectRatio = ratio
                                        }) {
                                            HStack {
                                                Text(ratio.displayName)
                                                if ratio == selectedAspectRatio {
                                                    Image(systemName: "checkmark")
                                                }
                                            }
                                        }
                                    }
                                } label: {
                                    Text(selectedAspectRatio.displayName)
                                        .font(.system(size: 13, weight: .medium))
                                        .foregroundColor(.white)
                                        .padding(.horizontal, 12)
                                        .padding(.vertical, 8)
                                        .background(.ultraThinMaterial)
                                        .cornerRadius(20)
                                }

                                // 설정
                                Button(action: {
                                    showSettings = true
                                }) {
                                    Image(systemName: "gearshape.fill")
                                        .font(.system(size: 20))
                                        .foregroundColor(.white)
                                        .frame(width: 40, height: 40)
                                        .background(.ultraThinMaterial)
                                        .clipShape(Circle())
                                }

                                // 닫기 버튼
                                Button(action: {
                                    withAnimation(.easeInOut(duration: 0.2)) {
                                        showCameraOptions = false
                                    }
                                }) {
                                    Image(systemName: "xmark")
                                        .font(.system(size: 16))
                                        .foregroundColor(.white)
                                        .frame(width: 40, height: 40)
                                        .background(.ultraThinMaterial)
                                        .clipShape(Circle())
                                }
                            }
                            .padding(.horizontal, 12)
                            .padding(.vertical, 8)
                        }
                        .frame(height: 60)
                        .padding(.horizontal, 20)//좌우 패딩 동일하게
                        .transition(.move(edge: .trailing).combined(with: .opacity))
                    } else {
                        // 접힌 상태: 점4개 사각형 버튼
                        Button(action: {
                            withAnimation(.easeInOut(duration: 0.2)) {
                                showCameraOptions = true
                            }
                        }) {
                            Image(systemName: "square.grid.2x2")
                                .font(.system(size: 20))
                                .foregroundColor(.white)
                                .frame(width: 40, height: 40)
                                .background(.ultraThinMaterial)
                                .clipShape(Circle())
                        }
                        .transition(.move(edge: .trailing).combined(with: .opacity))
                    }
                }
                .padding(.horizontal, 20)
                .padding(.bottom,210)  // 카메라 뷰박스 하단 기준
            }

            // 레퍼런스 선택 안내 (탭바로 이동)
            if referenceImage == nil {
                VStack(spacing: 0) {
                    Spacer()
                        .frame(height: safeAreaTop + screenHeight * 0.25)

                    Text("📸 레퍼런스 이미지를 선택하세요")
                        .font(.title3)
                        .fontWeight(.semibold)
                        .foregroundColor(.white)
                        .padding(.horizontal, 24)
                        .padding(.vertical, 16)
                        .background(Color.blue.opacity(0.8))
                        .cornerRadius(16)
                        .shadow(radius: 10)

                    Text("하단 '레퍼런스' 탭에서\n따라 찍고 싶은 사진을 선택하세요")
                        .font(.caption)
                        .foregroundColor(.white.opacity(0.8))
                        .multilineTextAlignment(.center)
                        .padding(.top, 8)

                    Spacer()
                }
            }

            // 완벽한 상태 표시 (레퍼런스가 있을 때만)
            else if referenceImage != nil && realtimeAnalyzer.isPerfect {
                VStack(spacing: 0) {
                    Spacer()
                        .frame(height: safeAreaTop + screenHeight * 0.25)

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


            // 5. 레퍼런스 썸네일 (왼쪽 하단)
            if let refImage = referenceImage {
                VStack {
                    Spacer()
                    HStack {
                        Image(uiImage: refImage)
                            .resizable()
                            .scaledToFill()
                            .frame(width: 70, height: 70)
                            .clipShape(RoundedRectangle(cornerRadius: 12))
                            .overlay(
                                RoundedRectangle(cornerRadius: 12)
                                    .strokeBorder(Color.white, lineWidth: 2)
                            )
                            .shadow(color: .black.opacity(0.3), radius: 8, x: 0, y: 4)
                            .padding(.leading, 20)
                            .padding(.bottom, 200)  // 하단 버튼 위에 표시

                        Spacer()
                    }
                }
            }

            // 6. 하단 카메라 컨트롤 (고정 위치)
            VStack {
                Spacer()

                // 🆕 렌즈 선택 버튼 (0.5x, 1x, 2x) - 셔터 버튼 위
                if !cameraManager.isFrontCamera && cameraManager.availableLenses.count > 1 {
                    HStack(spacing: 8) {
                        ForEach(cameraManager.availableLenses, id: \.self) { lens in
                            Button(action: {
                                withAnimation(.easeInOut(duration: 0.2)) {
                                    cameraManager.switchLens(to: lens)
                                }
                            }) {
                                // 현재 렌즈면 실제 가상 줌 값 표시, 아니면 렌즈 기본값 표시
                                let displayZoom = cameraManager.currentLens == lens ?
                                    String(format: "%.1f", cameraManager.virtualZoom) :
                                    lens.rawValue
                                Text(displayZoom + "x")
                                    .font(.system(size: 13, weight: cameraManager.currentLens == lens ? .bold : .medium))
                                    .foregroundColor(cameraManager.currentLens == lens ? .yellow : .white)
                                    .frame(width: 44, height: 44)
                                    .background(
                                        Circle()
                                            .fill(cameraManager.currentLens == lens ?
                                                  Color.black.opacity(0.6) : Color.black.opacity(0.3))
                                    )
                            }
                        }
                    }
                    .padding(.horizontal, 16)
                    .padding(.vertical, 8)
                    .background(
                        Capsule()
                            .fill(Color.black.opacity(0.3))
                    )
                    .padding(.bottom, 16)
                }

                ZStack {
                    // 셔터 버튼 (정중앙)
                    Button(action: {
                        performCapture()
                    }) {
                        ZStack {
                            Circle()
                                .stroke(Color.white, lineWidth: 4)
                                .frame(width: 80, height: 80)

                            Circle()
                                .fill(capturedImage != nil ? Color.green : Color.white)
                                .frame(width: 68, height: 68)

                            if capturedImage != nil {
                                Image(systemName: "checkmark")
                                    .font(.system(size: 28, weight: .bold))
                                    .foregroundColor(.white)
                            }
                        }
                    }
                    .disabled(capturedImage != nil)
                    .opacity(capturedImage != nil ? 0.8 : 1.0)

                    // 카메라 전환 (오른쪽 고정)
                    HStack {
                        Spacer()

                        Button(action: {
                            cameraManager.switchCamera()
                        }) {
                            Image(systemName: "arrow.triangle.2.circlepath.camera")
                                .font(.system(size: 24))
                                .foregroundColor(.white)
                                .frame(width: 50, height: 50)
                                .background(.ultraThinMaterial)
                                .clipShape(Circle())
                        }
                        .padding(.trailing, 30)
                    }
                }
                .frame(maxWidth: .infinity)
                .padding(.bottom, 70)  // 탭바 위에 고정
            }

            // 에러 메시지
            if let error = errorMessage {
                VStack {
                    Text("⚠️ \(error)")
                        .font(.caption)
                        .foregroundColor(.white)
                        .padding()
                        .background(Color.red.opacity(0.8))
                        .cornerRadius(8)
                        .padding(.top, safeAreaTop + 80)

                    Spacer()
                }
            }

            // 분석 중 인디케이터
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
                    .padding(.top, safeAreaTop + 80)

                    Spacer()
                }
            }

            // 디버그 오버레이 (showFPS 활성화 시에만)
            if showFPS {
                VStack {
                    Spacer()
                    HStack {
                        Spacer()
                        VStack(alignment: .trailing, spacing: 4) {
                            // FPS
                            Text(String(format: "%.1f FPS", cameraManager.currentFPS))
                                .font(.system(size: 11, weight: .semibold))
                                .foregroundColor(.white)
                                .padding(.horizontal, 8)
                                .padding(.vertical, 4)
                                .background(.ultraThinMaterial)
                                .cornerRadius(6)

                            // 🔥 발열 상태
                            let thermalEmoji = thermalStateEmoji(thermalManager.currentThermalState)
                            let targetFPS = Int(1.0 / thermalManager.recommendedAnalysisInterval)
                            Text("\(thermalEmoji) 목표: \(targetFPS)fps")
                                .font(.system(size: 10, weight: .bold))
                                .foregroundColor(thermalColor(thermalManager.currentThermalState))
                                .padding(.horizontal, 8)
                                .padding(.vertical, 4)
                                .background(.ultraThinMaterial)
                                .cornerRadius(6)

                            // 🔋 배터리/저전력 모드
                            if thermalManager.isLowPowerMode {
                                Text("⚡️ 저전력")
                                    .font(.system(size: 10, weight: .bold))
                                    .foregroundColor(.yellow)
                                    .padding(.horizontal, 8)
                                    .padding(.vertical, 4)
                                    .background(.ultraThinMaterial)
                                    .cornerRadius(6)
                            }

                            // 레퍼런스 포즈 키포인트
                            if let refPose = realtimeAnalyzer.referenceAnalysis?.poseKeypoints {
                                let visibleCount = refPose.filter { $0.confidence >= 0.5 }.count
                                let color: Color = visibleCount >= 10 ? .green : (visibleCount >= 5 ? .yellow : .red)
                                Text("레퍼런스: \(visibleCount)/\(refPose.count)개")
                                    .font(.system(size: 10, weight: .bold))
                                    .foregroundColor(color)
                                    .padding(.horizontal, 8)
                                    .padding(.vertical, 4)
                                    .background(.ultraThinMaterial)
                                    .cornerRadius(6)
                            }

                            // 완성도
                            if referenceImage != nil {
                                let score = Int(realtimeAnalyzer.perfectScore * 100)
                                Text("완성도: \(score)%")
                                    .font(.system(size: 10, weight: .bold))
                                    .foregroundColor(.white)
                                    .padding(.horizontal, 8)
                                    .padding(.vertical, 4)
                                    .background(.ultraThinMaterial)
                                    .cornerRadius(6)
                            }
                        }
                        .padding(.trailing, 16)
                        .padding(.bottom, 200)  // 하단 버튼 위에 표시
                    }
                }
            }
            }
        }
        .onChange(of: realtimeAnalyzer.isPerfect) { isPerfect in
            if isPerfect && autoCapture && capturedImage == nil {
                performCapture()
            }
        }
        .onChange(of: selectedAspectRatio) { newRatio in
            cameraManager.setAspectRatio(newRatio)

            // 비율 변경시 즉시 프레임 재분석하여 피드백 갱신
            if let currentFrame = cameraManager.currentFrame {
                realtimeAnalyzer.analyzeFrame(
                    currentFrame,
                    isFrontCamera: cameraManager.isFrontCamera,
                    currentAspectRatio: cameraManager.aspectRatio
                )
            }
        }
        .onChange(of: referenceImage) { newImage in
            // 레퍼런스 이미지 변경 시 분석 시작
            if let image = newImage {
                print("🎯🎯🎯 레퍼런스 이미지 선택됨!")

                // 🔥 레퍼런스 이미지를 먼저 analyzeReference로 분석
                realtimeAnalyzer.analyzeReference(image)
                print("✅ 레퍼런스 이미지 분석 완료")

                // 레퍼런스 분석 결과 확인
                if realtimeAnalyzer.referenceAnalysis != nil {
                    print("✅ referenceAnalysis 설정됨")
                } else {
                    print("❌ referenceAnalysis가 nil!")
                }

                // 실시간 분석 자동 시작
                startRealtimeAnalysis()
                print("🎯 실시간 피드백 모드 시작!")
            } else {
                // 레퍼런스 이미지가 없으면 분석 중지
                print("⏹️ 레퍼런스 제거됨 - 분석 중지")
                stopRealtimeAnalysis()
            }
        }
        .sheet(isPresented: $showSettings) {
            SettingsSheet(
                showGrid: $showGrid,
                showFPS: $showFPS,
                autoCapture: $autoCapture
            )
        }
    }

    // MARK: - Analysis Control

    /// 실시간 프레임 분석 시작 (클라이언트 사이드) - 적응형 속도
    private func startRealtimeAnalysis() {
        // 기존 타이머 중지
        stopRealtimeAnalysis()

        // 🔥 적응형 분석: 발열/배터리 상태에 따라 자동 조절
        // - 정상: 60fps (0.016초)
        // - 약간 따뜻: 45fps (0.022초)
        // - 뜨거움: 30fps (0.033초)
        // - 매우 뜨거움: 15fps (0.066초)
        var frameCount = 0
        frameUpdateTimer = Timer.scheduledTimer(withTimeInterval: 0.001, repeats: true) { _ in
            // 발열 관리자가 분석을 허용하는지 체크
            if self.thermalManager.shouldPerformAnalysis(),
               let currentFrame = self.cameraManager.currentFrame {
                frameCount += 1
                if frameCount % 100 == 0 {
                    print("🎬 프레임 분석 중... (\(frameCount)번째)")
                }
                self.realtimeAnalyzer.analyzeFrame(
                    currentFrame,
                    isFrontCamera: self.cameraManager.isFrontCamera,
                    currentAspectRatio: self.cameraManager.aspectRatio
                )
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

    // MARK: - Helper Functions

    /// 발열 상태 이모지
    private func thermalStateEmoji(_ state: ProcessInfo.ThermalState) -> String {
        switch state {
        case .nominal: return "❄️"
        case .fair: return "☁️"
        case .serious: return "🔥"
        case .critical: return "🚨"
        @unknown default: return "❓"
        }
    }

    /// 발열 상태 색상
    private func thermalColor(_ state: ProcessInfo.ThermalState) -> Color {
        switch state {
        case .nominal: return .green
        case .fair: return .yellow
        case .serious: return .orange
        case .critical: return .red
        @unknown default: return .gray
        }
    }

    // MARK: - Background Handling (배터리 절약)

    /// 백그라운드/포어그라운드 처리 설정
    private func setupBackgroundHandling() {
        // 백그라운드 진입 시
        NotificationCenter.default.addObserver(
            forName: UIApplication.willResignActiveNotification,
            object: nil,
            queue: .main
        ) { _ in
            print("🌙 백그라운드 진입: 카메라 및 분석 중단 (배터리 절약)")
            self.cameraManager.stopSession()
            self.stopRealtimeAnalysis()
        }

        // 포어그라운드 진입 시
        NotificationCenter.default.addObserver(
            forName: UIApplication.didBecomeActiveNotification,
            object: nil,
            queue: .main
        ) { _ in
            print("☀️ 포어그라운드 진입: 카메라 및 분석 재개")
            self.cameraManager.startSession()
            if self.referenceImage != nil {
                self.startRealtimeAnalysis()
            }
        }
    }

    /// 백그라운드 처리 해제
    private func removeBackgroundHandling() {
        NotificationCenter.default.removeObserver(self, name: UIApplication.willResignActiveNotification, object: nil)
        NotificationCenter.default.removeObserver(self, name: UIApplication.didBecomeActiveNotification, object: nil)
    }

    /// 실제 분석 수행 (V1: 온디바이스만 사용, 서버 연결 비활성화)
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

        guard referenceImage != nil,
              cameraManager.currentFrame != nil else {
            return
        }

        isAnalyzing = true
        errorMessage = nil

        // V1: 온디바이스 분석만 사용 (서버 연결 안 함)
        // RealtimeAnalyzer가 모든 분석을 처리함 (YOLO + MoveNet + Vision)
        // serverFeedbackItems는 사용하지 않음 (combinedFeedback에서 realtimeAnalyzer.instantFeedback만 사용)

        await MainActor.run {
            serverFeedbackItems = []  // 서버 피드백 비우기
            processingTime = "On-Device"  // 온디바이스 표시
            isAnalyzing = false
        }
    }
}

// MARK: - Aspect Ratio Mask View

struct AspectRatioMaskView: View {
    let selectedRatio: CameraAspectRatio

    var body: some View {
        GeometryReader { geometry in
            let screenWidth = geometry.size.width
            let screenHeight = geometry.size.height

            // 실제 iPhone 카메라처럼: 4:3이 기본(전체 화면), 나머지는 위아래 크롭
            // iPhone 화면 비율은 대략 19.5:9 (2.16:1)

            let captureHeight: CGFloat = {
                switch selectedRatio {
                case .ratio4_3:
                    // 4:3 - 세로 모드에서 4:3 비율 (width x width*4/3)
                    // iPhone 카메라 센서의 기본 비율 (가로:세로 = 4:3)
                    // 세로 모드에서는 너비를 기준으로 높이 계산
                    return screenWidth * 4.0 / 3.0

                case .ratio1_1:
                    // 1:1 - 정사각형, 너비를 기준으로 높이 설정
                    return screenWidth

                case .ratio16_9:
                    // 16:9 - 와이드, 가장 좁은 높이
                    return screenWidth * 16.0 / 9.0
                }
            }()

            // 위아래 마스크 높이 계산
            let maskHeight = max(0, (screenHeight - captureHeight) / 2)

            ZStack {
                if maskHeight > 0 {
                    // 상단 마스크
                    VStack {
                        Rectangle()
                            .fill(Color.black.opacity(0.7))
                            .frame(height: maskHeight)
                        Spacer()
                    }

                    // 하단 마스크
                    VStack {
                        Spacer()
                        Rectangle()
                            .fill(Color.black.opacity(0.7))
                            .frame(height: maskHeight)
                    }
                }
            }
            .ignoresSafeArea()
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView(referenceImage: .constant(nil))
    }
}
