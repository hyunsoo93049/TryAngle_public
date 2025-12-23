#!/usr/bin/env python3
import os
import json
import shutil

BASE_PATH = "/Users/hyunsoo/Try_Angle/ios/TryAngleApp"
ASSETS_PATH = os.path.join(BASE_PATH, "Assets.xcassets")
REFERENCE_IMAGES_PATH = os.path.join(BASE_PATH, "ReferenceImages")

# Assets 폴더가 없으면 생성
os.makedirs(ASSETS_PATH, exist_ok=True)

# Contents.json for Assets.xcassets
assets_contents = {
    "info": {
        "author": "xcode",
        "version": 1
    }
}
with open(os.path.join(ASSETS_PATH, "Contents.json"), 'w') as f:
    json.dump(assets_contents, f, indent=2)

added_count = 0
category_list = ["Hot", "Cafe", "Winter", "Street", "Landmark"]

for category in category_list:
    category_path = os.path.join(REFERENCE_IMAGES_PATH, category)

    if not os.path.exists(category_path):
        print(f"⚠️ {category} 폴더가 없습니다")
        continue

    # 폴더의 모든 파일 가져오기
    files = os.listdir(category_path)

    for file_name in files:
        # 이미지 파일만 처리
        if not (file_name.lower().endswith('.jpg') or file_name.lower().endswith('.jpeg') or file_name.lower().endswith('.png')):
            continue

        # 파일명 앞뒤 공백 제거
        clean_file_name = file_name.strip()
        src = os.path.join(category_path, file_name)

        # 공백이 있으면 파일명 변경
        if file_name != clean_file_name:
            new_src = os.path.join(category_path, clean_file_name)
            os.rename(src, new_src)
            src = new_src
            print(f"🔧 파일명 수정: '{file_name}' → '{clean_file_name}'")

        # 이미지 이름 (확장자 제거)
        image_name = os.path.splitext(clean_file_name)[0]

        # imageset 폴더 생성
        imageset_path = os.path.join(ASSETS_PATH, f"{image_name}.imageset")
        os.makedirs(imageset_path, exist_ok=True)

        # 이미지 파일 복사
        dst = os.path.join(imageset_path, clean_file_name)

        if os.path.exists(src):
            shutil.copy2(src, dst)

            # Contents.json 생성
            contents = {
                "images": [
                    {
                        "filename": clean_file_name,
                        "idiom": "universal",
                        "scale": "1x"
                    },
                    {
                        "idiom": "universal",
                        "scale": "2x"
                    },
                    {
                        "idiom": "universal",
                        "scale": "3x"
                    }
                ],
                "info": {
                    "author": "xcode",
                    "version": 1
                }
            }

            with open(os.path.join(imageset_path, "Contents.json"), 'w') as f:
                json.dump(contents, f, indent=2)

            added_count += 1
            print(f"✅ {image_name} 추가됨 ({category})")

print(f"\n🎉 총 {added_count}개 이미지가 Assets.xcassets에 추가되었습니다!")
print(f"📁 위치: {ASSETS_PATH}")

# 각 카테고리별로 추가된 이미지 개수 확인
print("\n📊 카테고리별 이미지 개수:")
for category in category_list:
    category_path = os.path.join(REFERENCE_IMAGES_PATH, category)
    if os.path.exists(category_path):
        files = [f for f in os.listdir(category_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"  {category}: {len(files)}개")
