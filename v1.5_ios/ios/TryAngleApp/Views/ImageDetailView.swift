import SwiftUI

struct ImageDetailView: View {
    let image: UIImage
    let onSelectImage: () -> Void
    @Environment(\.dismiss) var dismiss

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            VStack {
                // 확대된 이미지
                Image(uiImage: image)
                    .resizable()
                    .scaledToFit()
                    .frame(maxWidth: .infinity, maxHeight: .infinity)

                Spacer()

                // 선택 버튼
                Button(action: {
                    print("🔵 [ImageDetailView] 이 사진 선택하기 버튼 클릭!")
                    onSelectImage()
                    print("🔵 [ImageDetailView] onSelectImage() 호출 완료")
                    dismiss()
                }) {
                    Text("이 사진 선택하기")
                        .font(.system(size: 18, weight: .semibold))
                        .foregroundColor(.white)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 16)
                        .background(Color.blue)
                        .cornerRadius(12)
                }
                .padding(.horizontal, 20)
                .padding(.bottom, 40)
            }

            // 닫기 버튼 (오른쪽 상단)
            VStack {
                HStack {
                    Spacer()
                    Button(action: {
                        dismiss()
                    }) {
                        Image(systemName: "xmark")
                            .font(.system(size: 20, weight: .semibold))
                            .foregroundColor(.white)
                            .frame(width: 40, height: 40)
                            .background(Color.black.opacity(0.5))
                            .clipShape(Circle())
                    }
                    .padding(.top, 50)
                    .padding(.trailing, 20)
                }
                Spacer()
            }
        }
    }
}
