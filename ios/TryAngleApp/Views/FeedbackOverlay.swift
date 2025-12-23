import SwiftUI

struct FeedbackOverlay: View {
    let feedbackItems: [FeedbackItem]
    let processingTime: String

    var body: some View {
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

            // 하단: 피드백 리스트
            if !feedbackItems.isEmpty {
                VStack(alignment: .leading, spacing: 8) {
                    ForEach(feedbackItems.prefix(5)) { item in
                        FeedbackItemView(item: item)
                    }
                }
                .padding(.horizontal, 16)
                .padding(.bottom, 120)
            } else {
                // 피드백 없을 때
                Text("✅ 완벽합니다!")
                    .font(.headline)
                    .foregroundColor(.white)
                    .padding(.horizontal, 20)
                    .padding(.vertical, 12)
                    .background(Color.green.opacity(0.8))
                    .cornerRadius(12)
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

struct FeedbackOverlay_Previews: PreviewProvider {
    static var previews: some View {
        FeedbackOverlay(
            feedbackItems: [
                FeedbackItem(
                    priority: 1,
                    icon: "📐",
                    message: "왼쪽으로 기울이세요",
                    category: "composition",
                    currentValue: 10.0,
                    targetValue: 0.0,
                    tolerance: 3.0,
                    unit: "도"
                ),
                FeedbackItem(
                    priority: 2,
                    icon: "📏",
                    message: "뒤로 가세요",
                    category: "distance",
                    currentValue: 1.0,
                    targetValue: 3.0,
                    tolerance: 0.5,
                    unit: "걸음"
                )
            ],
            processingTime: "0.8s"
        )
        .background(Color.black)
    }
}
