#!/usr/bin/env python3
import os
import json
import shutil

BASE_PATH = "/Users/hyunsoo/Try_Angle/ios/TryAngleApp"
ASSETS_PATH = os.path.join(BASE_PATH, "Assets.xcassets")
REFERENCE_IMAGES_PATH = os.path.join(BASE_PATH, "ReferenceImages")

def create_image_set(asset_path, image_name, image_file):
    """이미지 세트 생성"""
    imageset_path = os.path.join(asset_path, f"{image_name}.imageset")
    os.makedirs(imageset_path, exist_ok=True)

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

    # 이미지 파일 복사
    src = os.path.join(REFERENCE_IMAGES_PATH, image_file)
    dst = os.path.join(imageset_path, image_file)
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"✅ 이미지 추가됨: {image_name}")

def setup_reference_images():
    """모든 레퍼런스 이미지를 Asset Catalog에 추가"""

    # ReferenceImages 폴더 생성
    ref_images_asset = os.path.join(ASSETS_PATH, "ReferenceImages")
    os.makedirs(ref_images_asset, exist_ok=True)

    # 각 카테고리별로 처리
    categories = {
        "Hot": ["hot1.jpg", "hot2.jpg", "hot3.jpg", "hot4.jpg", "hot5.jpg", "hot6.jpg", "hot7.jpg", "hot8.jpg"],
        "Cafe": ["IMG_9593.JPG", "IMG_9594.JPG", "IMG_9595.JPG", "IMG_9596.JPG", "IMG_9597.JPG", "IMG_9598.JPG", "IMG_9599.JPG", "IMG_9600.JPG"],
        "Winter": ["winter1.jpg", "winter2.jpg", "winter3.jpg", "winter4.jpg", "winter5.jpg", "winter6.jpg", "winter7.jpg", "winter8.jpg"],
        "Street": ["IMG_9617.JPG", "IMG_9618.JPG", "IMG_9619.JPG", "IMG_9620.JPG", "IMG_9621.JPG", "IMG_9622.JPG", "IMG_9623.JPG", "IMG_9624.JPG"],
        "Landmark": ["landmark1.jpg", "landmark2.jpg", "landmark3.jpg", "landmark4.jpg", "landmark5.jpg", "landmark6.jpg", "landmark7.jpg", "landmark8.jpg", "landmark9.jpg"]
    }

    for category, images in categories.items():
        category_path = os.path.join(ref_images_asset, category)
        os.makedirs(category_path, exist_ok=True)

        # Contents.json for folder
        folder_contents = {
            "info": {
                "author": "xcode",
                "version": 1
            },
            "properties": {
                "provides-namespace": True
            }
        }
        with open(os.path.join(category_path, "Contents.json"), 'w') as f:
            json.dump(folder_contents, f, indent=2)

        for image in images:
            image_path = os.path.join(REFERENCE_IMAGES_PATH, category, image)
            if os.path.exists(image_path):
                # 이미지 이름 정리 (확장자 제거, 특수문자 처리)
                clean_name = os.path.splitext(image)[0].replace("_", "")
                create_image_set(category_path, clean_name, os.path.join(category, image))

    print("🎉 Asset Catalog 생성 완료!")

if __name__ == "__main__":
    setup_reference_images()