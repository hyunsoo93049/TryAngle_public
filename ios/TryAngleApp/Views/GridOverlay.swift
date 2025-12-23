import SwiftUI

struct GridOverlay: View {
    var body: some View {
        GeometryReader { geometry in
            let width = geometry.size.width
            let height = geometry.size.height

            Path { path in
                // 세로선 2개 (3등분)
                let third = width / 3
                path.move(to: CGPoint(x: third, y: 0))
                path.addLine(to: CGPoint(x: third, y: height))

                path.move(to: CGPoint(x: third * 2, y: 0))
                path.addLine(to: CGPoint(x: third * 2, y: height))

                // 가로선 2개 (3등분)
                let thirdH = height / 3
                path.move(to: CGPoint(x: 0, y: thirdH))
                path.addLine(to: CGPoint(x: width, y: thirdH))

                path.move(to: CGPoint(x: 0, y: thirdH * 2))
                path.addLine(to: CGPoint(x: width, y: thirdH * 2))
            }
            .stroke(Color.white.opacity(0.3), lineWidth: 1)
        }
    }
}
