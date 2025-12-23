import SwiftUI
import PhotosUI

struct ReferenceGalleryView: View {
    @Binding var selectedTab: Int  // 탭 전환을 위한 바인딩
    @State private var selectedCategoryIndex = 1  // Hot부터 시작 (index 1)
    @State private var searchText = ""
    @State private var mySavedPhotos: [SavedPhoto] = []  // My 테마 저장된 사진들
    @State private var showingImagePicker = false
    @State private var selectedPhotoItem: PhotosPickerItem?
    @Environment(\.presentationMode) var presentationMode

    // 저장된 사진 모델
    struct SavedPhoto: Identifiable, Codable {
        let id: UUID
        let fileName: String
        let dateAdded: Date

        init(fileName: String) {
            self.id = UUID()
            self.fileName = fileName
            self.dateAdded = Date()
        }
    }

    let categories = ["My", "Hot", "Cafe☕️", "Winter ☃️", "Street 🚶‍♂️", "랜드마크🗽"]

    private var selectedCategory: String {
        categories[selectedCategoryIndex]
    }

    // 카테고리별 실제 이미지 불러오기
    private func getImagesForCategory(_ index: Int) -> [String] {
        let categoryNames = ["My", "Hot", "Cafe", "Winter", "Street", "Landmark"]
        let folderName = categoryNames[index]

        // 하드코딩된 이미지 리스트
        switch folderName {
        case "Hot":
            return [
                "Hot/hot1.jpg", "Hot/hot2.jpg", "Hot/hot3.jpg", "Hot/hot4.jpg",
                "Hot/hot5.jpg", "Hot/hot6.jpg", "Hot/hot7.jpg", "Hot/hot8.jpg"
            ]
        case "Cafe":
            return [
                "Cafe/IMG_9593.JPG", "Cafe/IMG_9594.JPG", "Cafe/IMG_9595.JPG", "Cafe/IMG_9596.JPG",
                "Cafe/IMG_9597.JPG", "Cafe/IMG_9598.JPG", "Cafe/IMG_9599.JPG", "Cafe/IMG_9600.JPG"
            ]
        case "Winter":
            return [
                "Winter/winter1.jpg", "Winter/winter2.jpg", "Winter/winter3.jpg", "Winter/winter4.jpg",
                "Winter/winter5.jpg", "Winter/winter6.jpg", "Winter/winter7.jpg", "Winter/winter8.jpg"
            ]
        case "Street":
            return [
                "Street/IMG_9617.JPG", "Street/IMG_9618.JPG", "Street/IMG_9619.JPG", "Street/IMG_9620.JPG",
                "Street/IMG_9621.JPG", "Street/IMG_9622.JPG", "Street/IMG_9623.JPG", "Street/IMG_9624.JPG"
            ]
        case "Landmark":
            return [
                "Landmark/landmark1.jpg", "Landmark/landmark2.jpg", "Landmark/landmark3.jpg", "Landmark/landmark4.jpg",
                "Landmark/landmark5.jpg", "Landmark/landmark6.jpg", "Landmark/landmark7.jpg", "Landmark/landmark8.jpg",
                "Landmark/landmark9.jpg"
            ]
        case "My":
            return []  // My 탭은 별도로 처리
        default:
            return []
        }
    }

    var body: some View {
        GeometryReader { geometry in
            let safeAreaTop = geometry.safeAreaInsets.top

            ZStack(alignment: .top) {
                Color.white.ignoresSafeArea()

                VStack(spacing: 0) {
                    // 다이나믹 아일랜드까지 흰색 공간
                    Color.white
                        .frame(height: safeAreaTop)

                    // 상단 바 + 검색
                    VStack(spacing: 8) {
                        // 흰색 공백
                        Color.white
                            .frame(height: 40)

                        // 뒤로가기 버튼
                        HStack {
                            Button(action: {
                                selectedTab = 1  // 카메라 탭으로 이동
                            }) {
                                Image(systemName: "chevron.left")
                                    .font(.system(size: 18, weight: .medium))
                                    .foregroundColor(.black)
                                    .frame(width: 29, height: 29)
                                    .background(Color(hex: "#ececec"))
                                    .clipShape(Circle())
                            }
                            .padding(.leading, 10)

                            Spacer()
                        }

                    // 카테고리 탭 (자동 스크롤)
                    ScrollViewReader { scrollProxy in
                        ScrollView(.horizontal, showsIndicators: false) {
                            HStack(spacing: 3) {
                                ForEach(Array(categories.enumerated()), id: \.offset) { index, category in
                                    CategoryTab(
                                        title: category,
                                        isSelected: selectedCategoryIndex == index,
                                        action: {
                                            withAnimation {
                                                selectedCategoryIndex = index
                                            }
                                        }
                                    )
                                    .id(index)  // 각 카테고리에 ID 부여
                                }
                            }
                            .padding(.horizontal, 10)
                        }
                        .frame(height: 37)
                        .onChange(of: selectedCategoryIndex) { newIndex in
                            // 페이지 스와이프 시 카테고리도 자동 스크롤
                            withAnimation {
                                scrollProxy.scrollTo(newIndex, anchor: .center)
                            }
                        }
                    }

                    // 검색바
                    HStack(spacing: 10) {
                        Image(systemName: "magnifyingglass")
                            .font(.system(size: 14))
                            .foregroundColor(Color(hex: "#454545"))

                        Text("스타일 검색")
                            .font(.system(size: 16, weight: .medium))
                            .foregroundColor(Color(hex: "#454545"))

                        Spacer()
                    }
                    .padding(.horizontal, 10)
                    .padding(.vertical, 6)
                    .background(Color(hex: "#ececec"))
                    .cornerRadius(99)
                    .padding(.horizontal, 10)
                    }
                }
                .background(Color.white)

                // 스와이프 가능한 그리드 콘텐츠
                TabView(selection: $selectedCategoryIndex) {
                    ForEach(Array(categories.enumerated()), id: \.offset) { index, category in
                        Group {
                            if index == 0 {
                                // My 테마 - 항상 + 버튼 카드를 첫 번째로 표시
                                ScrollView {
                                    LazyVGrid(columns: [
                                        GridItem(.flexible(), spacing: 5),
                                        GridItem(.flexible(), spacing: 5)
                                    ], spacing: 15) {
                                        // + 버튼 카드 (항상 첫 번째)
                                        AddPhotoCard {
                                            showingImagePicker = true
                                        }

                                        // 저장된 사진들
                                        ForEach(mySavedPhotos) { photo in
                                            MyPhotoCard(
                                                photo: photo,
                                                onDelete: { deletePhoto(photo) }
                                            )
                                        }
                                    }
                                    .padding(.horizontal, 10)
                                    .padding(.top, 10)
                                    .padding(.bottom, 90)
                                }
                            } else {
                                // 다른 테마는 실제 이미지 표시
                                let images = getImagesForCategory(index)
                                ScrollView {
                                    LazyVGrid(columns: [
                                        GridItem(.flexible(), spacing: 5),
                                        GridItem(.flexible(), spacing: 5)
                                    ], spacing: 15) {
                                        ForEach(images, id: \.self) { imagePath in
                                            RealPhotoCard(imagePath: imagePath)
                                        }
                                    }
                                    .padding(.horizontal, 10)
                                    .padding(.top, 10)
                                    .padding(.bottom, 90) // 하단 탭바 공간 확보
                                }
                            }
                        }
                        .tag(index)
                    }
                }
                .tabViewStyle(.page(indexDisplayMode: .never))
                .padding(.top, safeAreaTop + 5)  // 포토카드랑 검색창 사이 여백
            }
        }
        .photosPicker(isPresented: $showingImagePicker, selection: $selectedPhotoItem, matching: .images)
        .onChange(of: selectedPhotoItem) { newItem in
            Task {
                if let newItem = newItem {
                    await loadAndSavePhoto(from: newItem)
                }
            }
        }
        .onAppear {
            loadSavedPhotos()
        }
    }

    // MARK: - 사진 저장/로드 함수

    /// 선택한 사진을 앱 저장소에 저장
    private func loadAndSavePhoto(from item: PhotosPickerItem) async {
        guard let data = try? await item.loadTransferable(type: Data.self),
              let image = UIImage(data: data) else {
            print("❌ 이미지 로드 실패")
            return
        }

        // 파일명 생성 (UUID 기반)
        let fileName = "my_\(UUID().uuidString).jpg"

        // Documents 디렉토리에 저장
        if let documentsURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first {
            let fileURL = documentsURL.appendingPathComponent(fileName)

            // JPEG로 압축하여 저장
            if let jpegData = image.jpegData(compressionQuality: 0.8) {
                do {
                    try jpegData.write(to: fileURL)
                    print("✅ 사진 저장 완료: \(fileName)")

                    // 목록에 추가
                    DispatchQueue.main.async {
                        let newPhoto = SavedPhoto(fileName: fileName)
                        mySavedPhotos.insert(newPhoto, at: 0)  // 맨 앞에 추가
                        saveSavedPhotosList()
                    }
                } catch {
                    print("❌ 사진 저장 실패: \(error)")
                }
            }
        }

        // 선택 초기화
        selectedPhotoItem = nil
    }

    /// 사진 삭제
    private func deletePhoto(_ photo: SavedPhoto) {
        // 파일 삭제
        if let documentsURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first {
            let fileURL = documentsURL.appendingPathComponent(photo.fileName)
            try? FileManager.default.removeItem(at: fileURL)
        }

        // 목록에서 제거
        mySavedPhotos.removeAll { $0.id == photo.id }
        saveSavedPhotosList()
    }

    /// 저장된 사진 목록 불러오기
    private func loadSavedPhotos() {
        if let data = UserDefaults.standard.data(forKey: "mySavedPhotos"),
           let photos = try? JSONDecoder().decode([SavedPhoto].self, from: data) {
            mySavedPhotos = photos
        }
    }

    /// 저장된 사진 목록 저장
    private func saveSavedPhotosList() {
        if let data = try? JSONEncoder().encode(mySavedPhotos) {
            UserDefaults.standard.set(data, forKey: "mySavedPhotos")
        }
    }
}

// MARK: - 카테고리 탭
struct CategoryTab: View {
    let title: String
    let isSelected: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            VStack(spacing: 0) {
                Text(title)
                    .font(.system(size: 15, weight: isSelected ? .bold : .medium))
                    .foregroundColor(Color(hex: "#555555"))
                    .padding(.horizontal, 10)
                    .padding(.top, 10)
                    .padding(.bottom, 2)

                // 밑줄
                Rectangle()
                    .fill(isSelected ? Color(hex: "#555555") : Color.clear)
                    .frame(height: 4)
            }
        }
    }
}

// MARK: - + 버튼 카드 (사진 추가)
struct AddPhotoCard: View {
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            VStack(spacing: 12) {
                // + 버튼 영역
                RoundedRectangle(cornerRadius: 8)
                    .fill(Color(hex: "#f5f5f5"))
                    .frame(width: 184, height: 184)
                    .overlay(
                        VStack(spacing: 8) {
                            Image(systemName: "plus")
                                .font(.system(size: 40, weight: .light))
                                .foregroundColor(Color(hex: "#888888"))

                            Text("직접 추가하기")
                                .font(.system(size: 14, weight: .medium))
                                .foregroundColor(Color(hex: "#888888"))
                        }
                    )
                    .overlay(
                        RoundedRectangle(cornerRadius: 8)
                            .stroke(style: StrokeStyle(lineWidth: 2, dash: [8, 4]))
                            .foregroundColor(Color(hex: "#cccccc"))
                    )

                // 하트 버튼 자리 (빈 공간으로 다른 카드와 높이 맞춤)
                Color.clear
                    .frame(height: 18)
                    .padding(.trailing, 8)
            }
        }
    }
}

// MARK: - My 사진 카드 (사용자가 추가한 사진)
struct MyPhotoCard: View {
    let photo: ReferenceGalleryView.SavedPhoto
    let onDelete: () -> Void
    @State private var uiImage: UIImage?
    @State private var showDeleteConfirm = false

    var body: some View {
        VStack(alignment: .trailing, spacing: 15) {
            // 이미지
            ZStack(alignment: .topTrailing) {
                if let uiImage = uiImage {
                    Image(uiImage: uiImage)
                        .resizable()
                        .scaledToFill()
                        .frame(width: 184, height: 184)
                        .clipShape(RoundedRectangle(cornerRadius: 8))
                } else {
                    // 로딩 중
                    RoundedRectangle(cornerRadius: 8)
                        .fill(Color.gray.opacity(0.3))
                        .frame(width: 184, height: 184)
                        .overlay(
                            ProgressView()
                        )
                }

                // 삭제 버튼
                Button(action: { showDeleteConfirm = true }) {
                    Image(systemName: "xmark.circle.fill")
                        .font(.system(size: 22))
                        .foregroundColor(.white)
                        .shadow(radius: 2)
                }
                .padding(8)
            }

            // 하트 버튼 (My 사진은 이미 저장된 것이므로 채워진 하트)
            Image(systemName: "heart.fill")
                .font(.system(size: 18))
                .foregroundColor(.red)
                .padding(.trailing, 8)
        }
        .onAppear {
            loadImage()
        }
        .alert("사진 삭제", isPresented: $showDeleteConfirm) {
            Button("취소", role: .cancel) { }
            Button("삭제", role: .destructive) {
                onDelete()
            }
        } message: {
            Text("이 사진을 삭제하시겠습니까?")
        }
    }

    private func loadImage() {
        if let documentsURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first {
            let fileURL = documentsURL.appendingPathComponent(photo.fileName)
            if let image = UIImage(contentsOfFile: fileURL.path) {
                self.uiImage = image
            }
        }
    }
}

// MARK: - 포토카드 (placeholder)
struct PhotoCard: View {
    let imageName: String
    @State private var isFavorite = false

    var body: some View {
        VStack(alignment: .trailing, spacing: 15) {
            // 이미지 (placeholder)
            RoundedRectangle(cornerRadius: 8)
                .fill(
                    LinearGradient(
                        colors: [Color(hex: "#e0e0e0"), Color(hex: "#f5f5f5")],
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    )
                )
                .frame(width: 184, height: 184)
                .overlay(
                    Image(systemName: "photo")
                        .font(.system(size: 40))
                        .foregroundColor(.gray.opacity(0.5))
                )

            // 하트 버튼
            Button(action: { isFavorite.toggle() }) {
                Image(systemName: isFavorite ? "heart.fill" : "heart")
                    .font(.system(size: 18))
                    .foregroundColor(isFavorite ? .red : .black)
            }
            .padding(.trailing, 8)
        }
    }
}

// MARK: - 실제 이미지 포토카드
struct RealPhotoCard: View {
    let imagePath: String
    @State private var isFavorite = false
    @State private var uiImage: UIImage?

    var body: some View {
        VStack(alignment: .trailing, spacing: 15) {
            // 실제 이미지
            if let uiImage = uiImage {
                Image(uiImage: uiImage)
                    .resizable()
                    .scaledToFill()
                    .frame(width: 184, height: 184)
                    .clipShape(RoundedRectangle(cornerRadius: 8))
            } else {
                // 로딩 중
                RoundedRectangle(cornerRadius: 8)
                    .fill(Color.gray.opacity(0.3))
                    .frame(width: 184, height: 184)
            }

            // 하트 버튼
            Button(action: { isFavorite.toggle() }) {
                Image(systemName: isFavorite ? "heart.fill" : "heart")
                    .font(.system(size: 18))
                    .foregroundColor(isFavorite ? .red : .black)
            }
            .padding(.trailing, 8)
        }
        .onAppear {
            loadImage()
        }
    }

    private func loadImage() {
        // imagePath 형식: "Cafe/IMG_9593.JPG" 또는 "Hot/hot1.jpg"
        let pathComponents = imagePath.split(separator: "/").map(String.init)

        if pathComponents.count == 2 {
            let _ = pathComponents[0]  // folder는 사용하지 않음
            let fileName = pathComponents[1]

            // 파일명에서 확장자 분리
            let nameWithoutExtension: String
            let fileExtension: String

            if let dotIndex = fileName.lastIndex(of: ".") {
                nameWithoutExtension = String(fileName[..<dotIndex])
                fileExtension = String(fileName[fileName.index(after: dotIndex)...])
            } else {
                nameWithoutExtension = fileName
                fileExtension = "jpg"
            }

            // 1. 먼저 정확한 이름으로 시도
            if let bundlePath = Bundle.main.path(forResource: fileName, ofType: nil) {
                if let image = UIImage(contentsOfFile: bundlePath) {
                    self.uiImage = image
                    print("✅ 이미지 로드 성공 (파일명): \(fileName)")
                    return
                }
            }

            // 2. 이름과 확장자를 분리해서 시도
            if let bundlePath = Bundle.main.path(forResource: nameWithoutExtension, ofType: fileExtension) {
                if let image = UIImage(contentsOfFile: bundlePath) {
                    self.uiImage = image
                    print("✅ 이미지 로드 성공 (분리): \(nameWithoutExtension).\(fileExtension)")
                    return
                }
            }

            // 3. 대소문자 변형 시도
            for ext in [fileExtension, fileExtension.lowercased(), fileExtension.uppercased(), "jpg", "JPG", "jpeg", "JPEG", "png", "PNG"] {
                for name in [nameWithoutExtension, nameWithoutExtension.lowercased(), nameWithoutExtension.uppercased()] {
                    if let bundlePath = Bundle.main.path(forResource: name, ofType: ext) {
                        if let image = UIImage(contentsOfFile: bundlePath) {
                            self.uiImage = image
                            print("✅ 이미지 로드 성공 (변형): \(name).\(ext)")
                            return
                        }
                    }
                }
            }

            // 4. UIImage(named:) 시도
            if let image = UIImage(named: fileName) {
                self.uiImage = image
                print("✅ 이미지 로드 성공 (named): \(fileName)")
                return
            }

            // 5. 확장자 없이 시도
            if let image = UIImage(named: nameWithoutExtension) {
                self.uiImage = image
                print("✅ 이미지 로드 성공 (named without extension): \(nameWithoutExtension)")
                return
            }
        }

        print("❌ 이미지 로드 실패: \(imagePath)")
    }
}

struct ReferenceGalleryView_Previews: PreviewProvider {
    static var previews: some View {
        ReferenceGalleryView(selectedTab: .constant(2))
    }
}
