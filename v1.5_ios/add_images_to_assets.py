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

# 각 카테고리별 이미지 처리
categories = {
    "Hot": ["hot1.jpg", "hot2.jpg", "hot3.jpg", "hot4.jpg", "hot5.jpg", "hot6.jpg", "hot7.jpg", "hot8.jpg"],
    "Cafe": ["IMG_9593.JPG", "IMG_9594.JPG", "IMG_9595.JPG", "IMG_9596.JPG", "IMG_9597.JPG", "IMG_9598.JPG", "IMG_9599.JPG", "IMG_9600.JPG"],
    "Winter": ["winter1.jpg", "winter2.jpg", "winter3.jpg", "winter4.jpg", "winter5.jpg", "winter6.jpg", "winter7.jpg", "winter8.jpg"],
    "Street": ["IMG_9617.JPG", "IMG_9618.JPG", "IMG_9619.JPG", "IMG_9620.JPG", "IMG_9621.JPG", "IMG_9622.JPG", "IMG_9623.JPG", "IMG_9624.JPG"],
    "Landmark": ["landmark1.jpg", "landmark2.jpg", "landmark3.jpg", "landmark4.jpg", "landmark5.jpg", "landmark6.jpg", "landmark7.jpg", "landmark8.jpg", "landmark9.jpg"]
}

added_count = 0

for category, images in categories.items():
    for image_file in images:
        # 이미지 파일명에서 확장자 제거
        image_name = os.path.splitext(image_file)[0]

        # imageset 폴더 생성
        imageset_path = os.path.join(ASSETS_PATH, f"{image_name}.imageset")
        os.makedirs(imageset_path, exist_ok=True)

        # 이미지 파일 복사
        src = os.path.join(REFERENCE_IMAGES_PATH, category, image_file)
        dst = os.path.join(imageset_path, image_file)

        if os.path.exists(src):
            shutil.copy2(src, dst)

            # Contents.json 생성
            contents = {
                "images": [
                    {
                        "filename": image_file,
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
            print(f"✅ {image_name} 추가됨")
        else:
            print(f"❌ {src} 파일이 없습니다")

print(f"\n🎉 총 {added_count}개 이미지가 Assets.xcassets에 추가되었습니다!")
print(f"📁 위치: {ASSETS_PATH}")