#!/usr/bin/env python3
import os
import json
import shutil

ASSETS_PATH = "/Users/hyunsoo/Try_Angle/ios/TryAngleApp/Assets.xcassets"

# 1. AppIcon.appiconset 생성
appicon_path = os.path.join(ASSETS_PATH, "AppIcon.appiconset")
os.makedirs(appicon_path, exist_ok=True)

appicon_contents = {
    "images": [
        {"idiom": "universal", "platform": "ios", "size": "1024x1024"}
    ],
    "info": {
        "author": "xcode",
        "version": 1
    }
}

with open(os.path.join(appicon_path, "Contents.json"), 'w') as f:
    json.dump(appicon_contents, f, indent=2)

print("✅ AppIcon.appiconset 생성 완료")

# 2. AccentColor.colorset 생성
accentcolor_path = os.path.join(ASSETS_PATH, "AccentColor.colorset")
os.makedirs(accentcolor_path, exist_ok=True)

accentcolor_contents = {
    "colors": [
        {
            "idiom": "universal"
        }
    ],
    "info": {
        "author": "xcode",
        "version": 1
    }
}

with open(os.path.join(accentcolor_path, "Contents.json"), 'w') as f:
    json.dump(accentcolor_contents, f, indent=2)

print("✅ AccentColor.colorset 생성 완료")

# 3. JPG 확장자를 소문자로 변경
fixed_count = 0
for root, dirs, files in os.walk(ASSETS_PATH):
    for file in files:
        if file.endswith('.JPG'):
            old_path = os.path.join(root, file)
            new_file = file.replace('.JPG', '.jpg')
            new_path = os.path.join(root, new_file)

            # 파일명 변경
            os.rename(old_path, new_path)

            # Contents.json도 업데이트
            contents_path = os.path.join(root, 'Contents.json')
            if os.path.exists(contents_path):
                with open(contents_path, 'r') as f:
                    contents = json.load(f)

                # filename을 소문자 확장자로 변경
                for image in contents.get('images', []):
                    if 'filename' in image and image['filename'] == file:
                        image['filename'] = new_file

                with open(contents_path, 'w') as f:
                    json.dump(contents, f, indent=2)

            fixed_count += 1
            print(f"🔧 {file} → {new_file}")

print(f"\n✅ {fixed_count}개 파일 확장자 수정 완료")
print("🎉 모든 Assets 문제 수정 완료!")