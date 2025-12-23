import SwiftUI

struct MainTabView: View {
    @State private var selectedTab = 1  // 0: 갤러리, 1: 카메라, 2: 레퍼런스
    @State private var selectedReferenceImage: UIImage?  // 선택된 레퍼런스 이미지

    var body: some View {
        ZStack(alignment: .bottom) {
            // 탭별 콘텐츠 - ContentView는 항상 유지 (재생성 방지)
            ZStack {
                // 갤러리
                if selectedTab == 0 {
                    GalleryView()
                }

                // 카메라 (항상 백그라운드에 유지)
                ContentView(referenceImage: $selectedReferenceImage)
                    .opacity(selectedTab == 1 ? 1 : 0)
                    .allowsHitTesting(selectedTab == 1)

                // 레퍼런스
                if selectedTab == 2 {
                    ReferenceGalleryViewSimple(
                        selectedTab: $selectedTab,
                        onSelectImage: { image in
                            print("🟢 [MainTabView] onSelectImage 콜백 호출됨!")
                            selectedReferenceImage = image
                            print("🟢 [MainTabView] selectedReferenceImage 설정 완료")
                            selectedTab = 1  // 카메라 탭으로 이동
                            print("🟢 [MainTabView] 카메라 탭(1)으로 이동")
                        }
                    )
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)

            // 커스텀 하단 탭바
            //레퍼런스 이미지 창에서는 하단 탭바 숨기기
            if selectedTab != 2 {
                 CustomTabBar(selectedTab: $selectedTab)
             }

        }
        .ignoresSafeArea()
    }
}

// MARK: - 커스텀 탭바
struct CustomTabBar: View {
    @Binding var selectedTab: Int

    var body: some View {
        ZStack {
            VStack(spacing: 0) {
                Spacer()

                // 경계선
                Rectangle()
                    .fill(Color.white.opacity(0.3))
                    .frame(height: 1)

                // 탭 레이블만 표시 (셔터/카메라 버튼은 ContentView에서 관리)
                HStack(spacing: 77) {
                    TabButton(title: "갤러리", isSelected: selectedTab == 0) {
                        selectedTab = 0
                    }

                    TabButton(title: "카메라", isSelected: selectedTab == 1) {
                        selectedTab = 1
                    }

                    TabButton(title: "레퍼런스", isSelected: selectedTab == 2) {
                        selectedTab = 2
                    }
                }
                .padding(.bottom, 27)
            }
        }
        .frame(height: 90)  // 탭 레이블만 표시하므로 높이 축소
    }
}

// MARK: - 탭 버튼
struct TabButton: View {
    let title: String
    let isSelected: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Text(title)
                .font(.system(size: 16, weight: .medium))
                .foregroundColor(isSelected ? .white : Color(hex: "#5c5c5c"))
        }
    }
}

