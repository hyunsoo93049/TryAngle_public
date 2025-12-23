import SwiftUI

// MARK: - 간소화된 피드백 오버레이
struct SimplifiedFeedbackOverlay: View {
    let primaryFeedback: String?  // 가장 중요한 피드백 1개
    let completionScore: Double   // 완성도 (0~100)
    let isCapturing: Bool        // 촬영 중 여부

    var body: some View {
        VStack(spacing: 0) {
            // 상단 피드백 영역
            if let feedback = primaryFeedback {
                HStack {
                    Text(feedback)
                        .font(.system(size: 16, weight: .semibold))
                        .foregroundColor(.white)
                        .padding(.horizontal, 16)
                        .padding(.vertical, 12)
                        .background(feedbackBackground)
                        .cornerRadius(20)
                        .shadow(radius: 8)
                }
                .padding(.top, 60)
                .padding(.horizontal, 20)
            }

            Spacer()

            // 하단 완성도 표시 (간단하게)
            if !isCapturing {
                HStack(spacing: 12) {
                    // 완성도 프로그레스
                    ProgressView(value: completionScore / 100)
                        .progressViewStyle(LinearProgressViewStyle(tint: progressColor))
                        .frame(width: 120, height: 4)

                    // 완성도 퍼센트
                    Text("\(Int(completionScore))%")
                        .font(.system(size: 14, weight: .medium))
                        .foregroundColor(.white)
                        .frame(width: 40)
                }
                .padding(.horizontal, 20)
                .padding(.vertical, 12)
                .background(Color.black.opacity(0.5))
                .cornerRadius(16)
                .padding(.bottom, 100)
            }
        }
    }

    // 피드백 배경색 (중요도에 따라)
    private var feedbackBackground: Color {
        if completionScore > 90 {
            return Color.green.opacity(0.9)
        } else if completionScore > 70 {
            return Color.blue.opacity(0.9)
        } else if completionScore > 50 {
            return Color.orange.opacity(0.9)
        } else {
            return Color.red.opacity(0.9)
        }
    }

    // 프로그레스 색상
    private var progressColor: Color {
        if completionScore > 90 {
            return .green
        } else if completionScore > 70 {
            return .blue
        } else if completionScore > 50 {
            return .orange
        } else {
            return .red
        }
    }
}

// MARK: - 미니멀 컨트롤 뷰
struct MinimalControlsView: View {
    @Binding var selectedAspectRatio: CameraAspectRatio
    @Binding var showGrid: Bool
    let isFlashOn: Bool
    let onFlashToggle: () -> Void
    let onCapture: () -> Void
    let onSwitchCamera: () -> Void

    var body: some View {
        VStack(spacing: 0) {
            // 상단 컨트롤 (최소화)
            HStack {
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
                    HStack(spacing: 4) {
                        Image(systemName: "aspectratio")
                            .font(.system(size: 16))
                        Text(selectedAspectRatio.displayName)
                            .font(.system(size: 14, weight: .medium))
                    }
                    .foregroundColor(.white)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                    .background(Color.black.opacity(0.4))
                    .cornerRadius(16)
                }

                Spacer()

                // 그리드 토글
                Button(action: {
                    showGrid.toggle()
                }) {
                    Image(systemName: showGrid ? "grid" : "grid.circle")
                        .font(.system(size: 20))
                        .foregroundColor(.white)
                        .frame(width: 36, height: 36)
                        .background(Color.black.opacity(0.4))
                        .clipShape(Circle())
                }

                // 플래시 토글
                Button(action: onFlashToggle) {
                    Image(systemName: isFlashOn ? "bolt.fill" : "bolt.slash.fill")
                        .font(.system(size: 20))
                        .foregroundColor(isFlashOn ? .yellow : .white)
                        .frame(width: 36, height: 36)
                        .background(Color.black.opacity(0.4))
                        .clipShape(Circle())
                }
            }
            .padding(.horizontal, 16)
            .padding(.top, 50)

            Spacer()

            // 하단 촬영 컨트롤
            HStack(alignment: .center, spacing: 60) {
                // 빈 공간 (균형)
                Spacer()
                    .frame(width: 44)

                // 촬영 버튼
                Button(action: onCapture) {
                    ZStack {
                        Circle()
                            .stroke(Color.white, lineWidth: 3)
                            .frame(width: 70, height: 70)

                        Circle()
                            .fill(Color.white)
                            .frame(width: 60, height: 60)
                    }
                }

                // 카메라 전환
                Button(action: onSwitchCamera) {
                    Image(systemName: "camera.rotate")
                        .font(.system(size: 22))
                        .foregroundColor(.white)
                        .frame(width: 44, height: 44)
                        .background(Color.black.opacity(0.4))
                        .clipShape(Circle())
                }
            }
            .padding(.bottom, 40)
        }
    }
}

// MARK: - 레퍼런스 썸네일 뷰
struct ReferenceThumbnailView: View {
    let image: UIImage?
    let isVisible: Bool

    var body: some View {
        if let image = image, isVisible {
            VStack {
                HStack {
                    Image(uiImage: image)
                        .resizable()
                        .aspectRatio(contentMode: .fill)
                        .frame(width: 60, height: 80)
                        .cornerRadius(8)
                        .overlay(
                            RoundedRectangle(cornerRadius: 8)
                                .stroke(Color.white, lineWidth: 2)
                        )
                        .shadow(radius: 4)

                    Spacer()
                }
                .padding(.leading, 16)
                .padding(.top, 100)

                Spacer()
            }
        }
    }
}

struct SimplifiedFeedbackOverlay_Previews: PreviewProvider {
    static var previews: some View {
        ZStack {
            Color.black
            SimplifiedFeedbackOverlay(
                primaryFeedback: "왼팔을 더 올려주세요",
                completionScore: 75,
                isCapturing: false
            )
        }
    }
}