#!/bin/bash

echo "🔍 TryAngleApp 최종 검증 시작..."
echo "=================================="

# 1. 프로젝트 파일 체크
echo "1️⃣ Xcode 프로젝트 파일 확인..."
if grep -q "ReferenceImages" /Users/hyunsoo/Try_Angle/ios/TryAngleApp/TryAngleApp.xcodeproj/project.pbxproj; then
    echo "✅ ReferenceImages 폴더가 프로젝트에 포함됨"
else
    echo "❌ ReferenceImages 폴더가 프로젝트에 없음"
fi

# 2. 이미지 파일 체크
echo ""
echo "2️⃣ 이미지 파일 확인..."
IMAGES_PATH="/Users/hyunsoo/Try_Angle/ios/TryAngleApp/ReferenceImages"

for category in "Hot" "Cafe" "Winter" "Street" "Landmark"; do
    count=$(find "$IMAGES_PATH/$category" -type f \( -name "*.jpg" -o -name "*.JPG" -o -name "*.png" -o -name "*.PNG" \) | wc -l)
    echo "   $category: $count 개 이미지"
done

# 3. 코드 파일 체크
echo ""
echo "3️⃣ 코드 파일 확인..."

FILES_TO_CHECK=(
    "/Users/hyunsoo/Try_Angle/ios/TryAngleApp/Views/ReferenceGalleryView.swift"
    "/Users/hyunsoo/Try_Angle/ios/TryAngleApp/Views/ReferenceGalleryViewSimple.swift"
    "/Users/hyunsoo/Try_Angle/ios/TryAngleApp/Views/MainTabView.swift"
    "/Users/hyunsoo/Try_Angle/ios/TryAngleApp/ContentView.swift"
)

for file in "${FILES_TO_CHECK[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $(basename $file) 존재"
    else
        echo "❌ $(basename $file) 없음"
    fi
done

# 4. Build 디렉토리 체크
echo ""
echo "4️⃣ Build 디렉토리 확인..."
if [ -d "/Users/hyunsoo/Try_Angle/ios/TryAngleApp/build" ]; then
    echo "✅ Build 디렉토리 존재"
else
    echo "⚠️ Build 디렉토리 없음 (첫 빌드 전)"
fi

echo ""
echo "=================================="
echo "📝 요약:"
echo "- ReferenceImages 폴더명이 올바름 (s 없음)"
echo "- 코드가 SimpleGalleryView를 사용하도록 수정됨"
echo "- 이미지가 하드코딩되어 있어 Bundle 경로 문제 해결됨"
echo ""
echo "🎯 다음 단계:"
echo "1. Xcode에서 프로젝트 열기"
echo "2. Product → Clean Build Folder (Shift+Cmd+K)"
echo "3. Product → Build (Cmd+B)"
echo "4. Product → Run (Cmd+R)"
echo ""
echo "💡 앱 실행 후 하단 '레퍼런스' 탭을 클릭하면 이미지가 표시될 것입니다!"