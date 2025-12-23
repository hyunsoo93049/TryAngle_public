import SwiftUI

struct FeedbackOverlay: View {
    let feedbackItems: [FeedbackItem]
    let categoryStatuses: [CategoryStatus]  // 🆕 카테고리 상태
    let completedFeedbacks: [CompletedFeedback]  // 🆕 완료된 피드백
    let processingTime: String

    var body: some View {
        let _ = {
            if !feedbackItems.isEmpty {
                print("🖥️ FeedbackOverlay: \(feedbackItems.count)개 피드백 받음")
            }
        }()

        ZStack {
            // 🔄 왼쪽 상단: 카테고리 체크리스트 (프레이밍 박스 왼쪽 상단 모서리)
            if !categoryStatuses.isEmpty {
                VStack {
                    HStack {
                        CategoryChecklistView(categoryStatuses: categoryStatuses)
                            .padding(.leading, 12)
                            .padding(.top, 120)  // 상단에서 약간 아래로 (상태바/노치 피하기)
                        Spacer()
                    }
                    Spacer()
                }
            }

            // 기존 레이아웃: 상단 + 하단 피드백
            VStack(alignment: .leading, spacing: 12) {
                // 상단: 처리 시간
                HStack {
                    Spacer()
                    Text("⚡ \(processingTime)")
                        .font(.caption)
                        .foregroundColor(.white)
                        .padding(.horizontal, 12)
                        .padding(.vertical, 6)
                        .background(Color.black.opacity(0.6))
                        .cornerRadius(8)
                }
                .padding(.top, 60)
                .padding(.trailing, 16)

                Spacer()

                // 하단: 피드백 리스트 (우선순위 높은 것만 3개)
                VStack(alignment: .leading, spacing: 8) {
                    // 🆕 완료된 피드백들 (먼저 표시)
                    ForEach(completedFeedbacks) { completed in
                        CompletedFeedbackView(completed: completed)
                            .transition(.asymmetric(
                                insertion: .scale.combined(with: .opacity),
                                removal: .opacity
                            ))
                            .id(completed.id)  // 고유 ID로 애니메이션 추적
                    }

                    // 진행 중인 피드백들 (🔄 최대 2개로 제한)
                    if !feedbackItems.isEmpty {
                        ForEach(feedbackItems.prefix(2)) { item in
                            FeedbackItemView(item: item)
                                .transition(.move(edge: .bottom).combined(with: .opacity))
                                .id(item.id)
                        }
                    }
                }
                .animation(.spring(response: 0.4, dampingFraction: 0.8), value: completedFeedbacks.map { $0.id })
                .animation(.spring(response: 0.4, dampingFraction: 0.8), value: feedbackItems.map { $0.id })
                .padding(.horizontal, 16)
                .padding(.bottom, 120)
            }
        }
    }

    // 카테고리별 강조 색상
    private func categoryColor(_ category: String) -> Color {
        switch category {
        case "pose":
            return .purple
        case "distance":
            return .blue
        case "composition":
            return .orange
        default:
            return .gray
        }
    }
}

// MARK: - 개별 피드백 아이템 뷰 (실시간 진행도 표시)
struct FeedbackItemView: View {
    let item: FeedbackItem

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            // 상단: 아이콘 + 메시지
            HStack(spacing: 12) {
                Text(item.icon)
                    .font(.title2)

                Text(item.message)
                    .font(.body)
                    .foregroundColor(.white)
                    .lineLimit(2)

                Spacer()

                // 완료 체크 표시
                if item.isCompleted {
                    Image(systemName: "checkmark.circle.fill")
                        .font(.title2)
                        .foregroundColor(.green)
                }
            }

            // 하단: 실시간 진행도 표시
            if let current = item.currentValue,
               let target = item.targetValue,
               let unit = item.unit {

                HStack(spacing: 12) {
                    // 현재값 → 목표값
                    Text(String(format: "%.0f%@ → %.0f%@", current, unit, target, unit))
                        .font(.caption)
                        .foregroundColor(.white.opacity(0.8))
                        .monospacedDigit()

                    Spacer()

                    // 차이값 표시
                    let diff = abs(target - current)
                    Text(String(format: "차이: %.0f%@", diff, unit))
                        .font(.caption)
                        .foregroundColor(diff <= (item.tolerance ?? 3.0) ? .green : .orange)
                        .monospacedDigit()
                }

                // 프로그레스 바
                GeometryReader { geometry in
                    ZStack(alignment: .leading) {
                        // 배경
                        RoundedRectangle(cornerRadius: 4)
                            .fill(Color.white.opacity(0.2))
                            .frame(height: 8)

                        // 진행 바
                        RoundedRectangle(cornerRadius: 4)
                            .fill(progressColor)
                            .frame(width: geometry.size.width * progressWidth, height: 8)
                            .animation(.easeInOut(duration: 0.3), value: progressWidth)
                    }
                }
                .frame(height: 8)
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
        .background(
            Color.black.opacity(0.7)
                .overlay(
                    categoryColor(item.category)
                        .frame(width: 4),
                    alignment: .leading
                )
        )
        .cornerRadius(12)
    }

    // 진행도 바 너비 계산
    private var progressWidth: CGFloat {
        guard let current = item.currentValue,
              let target = item.targetValue else {
            return 0.0
        }

        let diff = abs(target - current)
        let tolerance = item.tolerance ?? 3.0

        // 차이가 허용 오차 이내면 100%
        if diff <= tolerance {
            return 1.0
        }

        // 차이가 클수록 진행도 낮음 (최대 50도 기준)
        let maxDiff = 50.0
        return max(0.0, min(1.0, 1.0 - (diff / maxDiff)))
    }

    // 진행도에 따른 색상
    private var progressColor: Color {
        if item.isCompleted {
            return .green
        } else if progressWidth > 0.7 {
            return .yellow
        } else if progressWidth > 0.4 {
            return .orange
        } else {
            return .red
        }
    }

    // 카테고리별 강조 색상
    private func categoryColor(_ category: String) -> Color {
        switch category {
        case "pose":
            return .purple
        case "distance":
            return .blue
        case "composition":
            return .orange
        default:
            return .gray
        }
    }
}

// MARK: - 완료된 피드백 뷰 (초록색 + 페이드아웃)
struct CompletedFeedbackView: View {
    let completed: CompletedFeedback

    var body: some View {
        HStack(spacing: 12) {
            // 체크 아이콘
            Image(systemName: "checkmark.circle.fill")
                .font(.title2)
                .foregroundColor(.white)

            Text(completed.item.icon)
                .font(.title2)

            VStack(alignment: .leading, spacing: 4) {
                Text(completed.item.message)
                    .font(.body)
                    .foregroundColor(.white)
                    .lineLimit(2)

                Text("완료!")
                    .font(.caption)
                    .fontWeight(.bold)
                    .foregroundColor(.white)
            }

            Spacer()
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
        .background(
            Color.green.opacity(0.9)
                .overlay(
                    Color.white.opacity(0.2)
                        .frame(width: 4),
                    alignment: .leading
                )
        )
        .cornerRadius(12)
        .shadow(color: .green.opacity(0.5), radius: 10, x: 0, y: 5)
        .opacity(completed.fadeProgress)
        .scaleEffect(completed.fadeProgress * 0.1 + 0.9)  // 살짝 작아지면서 사라짐
    }
}

// MARK: - 카테고리 체크리스트 뷰
struct CategoryChecklistView: View {
    let categoryStatuses: [CategoryStatus]

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            ForEach(categoryStatuses) { status in
                CategoryCheckItem(status: status)
            }
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 6)
        .background(
            RoundedRectangle(cornerRadius: 8)
                .fill(Color.black.opacity(0.6))
        )
    }
}

// MARK: - 개별 카테고리 체크 아이템
struct CategoryCheckItem: View {
    let status: CategoryStatus

    var body: some View {
        HStack(spacing: 6) {
            // 카테고리 이름
            Text(status.category.displayName)
                .font(.system(size: 13, weight: .medium))
                .foregroundColor(status.isSatisfied ? .white.opacity(0.7) : .white)

            // 체크 아이콘 (글자 바로 옆)
            Image(systemName: status.isSatisfied ? "checkmark.circle.fill" : "circle")
                .font(.system(size: 14))
                .foregroundColor(status.isSatisfied ? .green : .white.opacity(0.5))
                .animation(.easeInOut(duration: 0.3), value: status.isSatisfied)
        }
    }
}

struct FeedbackOverlay_Previews: PreviewProvider {
    static var previews: some View {
        FeedbackOverlay(
            feedbackItems: [
                FeedbackItem(
                    priority: 1,
                    icon: "💪",
                    message: "왼팔을 더 올려주세요",
                    category: "pose_left_arm",
                    currentValue: 45.0,
                    targetValue: 90.0,
                    tolerance: 10.0,
                    unit: "도"
                ),
                FeedbackItem(
                    priority: 2,
                    icon: "📍",
                    message: "왼쪽으로 서주세요",
                    category: "position_x",
                    currentValue: 55.0,
                    targetValue: 50.0,
                    tolerance: 5.0,
                    unit: "%"
                )
            ],
            categoryStatuses: [
                CategoryStatus(category: .pose, isSatisfied: false, activeFeedbacks: []),
                CategoryStatus(category: .position, isSatisfied: false, activeFeedbacks: []),
                CategoryStatus(category: .framing, isSatisfied: true, activeFeedbacks: []),
                CategoryStatus(category: .angle, isSatisfied: true, activeFeedbacks: []),
                CategoryStatus(category: .composition, isSatisfied: true, activeFeedbacks: []),
                CategoryStatus(category: .gaze, isSatisfied: true, activeFeedbacks: [])
            ],
            completedFeedbacks: [
                CompletedFeedback(
                    item: FeedbackItem(
                        priority: 3,
                        icon: "🔍",
                        message: "거리 조정 완료",
                        category: "distance",
                        currentValue: 1.5,
                        targetValue: 1.5,
                        tolerance: 0.2,
                        unit: "m"
                    ),
                    completedAt: Date().addingTimeInterval(-0.5)
                )
            ],
            processingTime: "0.8s"
        )
        .background(Color.black)
    }
}
