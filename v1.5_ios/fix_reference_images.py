#!/usr/bin/env python3
import os
import re
import uuid
import plistlib

PROJECT_FILE = "/Users/hyunsoo/Try_Angle/ios/TryAngleApp/TryAngleApp.xcodeproj/project.pbxproj"
REFERENCE_IMAGES_PATH = "/Users/hyunsoo/Try_Angle/ios/TryAngleApp/ReferenceImages"

def generate_uuid():
    return uuid.uuid4().hex[:24].upper()

def fix_project_file():
    """Xcode 프로젝트 파일에서 ReferenceImages 폴더 참조 수정"""

    # 프로젝트 파일 읽기
    with open(PROJECT_FILE, 'r') as f:
        content = f.read()

    # 백업 생성
    with open(PROJECT_FILE + '.backup_fix', 'w') as f:
        f.write(content)

    # 모든 개별 이미지 파일 참조 찾기 및 제거
    image_patterns = [
        r'.*landmark\d+\.jpg.*\n',
        r'.*winter\d+\.jpg.*\n',
        r'.*hot\d+\.jpg.*\n',
        r'.*IMG_\d+\.JPG.*\n',
        r'.*ReferenceImages/.*\.jpg.*\n',
        r'.*ReferenceImages/.*\.JPG.*\n',
    ]

    for pattern in image_patterns:
        content = re.sub(pattern, '', content, flags=re.IGNORECASE)

    # ReferencesImages를 ReferenceImages로 변경
    content = content.replace('ReferencesImages', 'ReferenceImages')

    # ReferenceImages 폴더가 Copy Bundle Resources에 있는지 확인
    if 'ReferenceImages in Resources' not in content:
        # Build Files 섹션 찾기
        build_files_match = re.search(r'/\* Begin PBXBuildFile section \*/(.*?)/\* End PBXBuildFile section \*/', content, re.DOTALL)
        if build_files_match:
            new_uuid = generate_uuid()
            folder_uuid = generate_uuid()

            # PBXBuildFile 추가
            new_build_file = f"\t\t{new_uuid} /* ReferenceImages in Resources */ = {{isa = PBXBuildFile; fileRef = {folder_uuid} /* ReferenceImages */; }};\n"

            # Build Files 섹션에 추가
            build_files = build_files_match.group(1)
            build_files = build_files.rstrip() + "\n" + new_build_file
            content = content[:build_files_match.start(1)] + build_files + content[build_files_match.end(1):]

            # PBXFileReference 추가 (폴더 참조로)
            file_ref_section = re.search(r'/\* Begin PBXFileReference section \*/(.*?)/\* End PBXFileReference section \*/', content, re.DOTALL)
            if file_ref_section:
                new_file_ref = f"\t\t{folder_uuid} /* ReferenceImages */ = {{isa = PBXFileReference; lastKnownFileType = folder; path = ReferenceImages; sourceTree = \"<group>\"; }};\n"
                file_refs = file_ref_section.group(1)
                file_refs = file_refs.rstrip() + "\n" + new_file_ref
                content = content[:file_ref_section.start(1)] + file_refs + content[file_ref_section.end(1):]

            # Resources build phase에 추가
            resources_match = re.search(r'(/\* Resources \*/ = \{[^}]*files = \([^)]*)', content)
            if resources_match:
                resources_section = resources_match.group(1)
                if f"{new_uuid} /* ReferenceImages in Resources */" not in resources_section:
                    resources_section += f",\n\t\t\t\t{new_uuid} /* ReferenceImages in Resources */"
                    content = content[:resources_match.start(1)] + resources_section + content[resources_match.end(1):]

    # 파일 저장
    with open(PROJECT_FILE, 'w') as f:
        f.write(content)

    print("✅ Xcode 프로젝트 파일 수정 완료")

if __name__ == "__main__":
    fix_project_file()