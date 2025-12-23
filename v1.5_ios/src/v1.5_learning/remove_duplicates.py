#!/usr/bin/env python3
"""
중복 이미지 제거 스크립트
MD5 해시 기반으로 완전히 동일한 파일 제거
"""

import os
import hashlib
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import argparse


def get_file_hash(file_path: Path) -> str:
    """파일의 MD5 해시 계산"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def find_duplicates(root_dir: Path):
    """중복 파일 찾기"""
    print(f"\n[SCAN] {root_dir} 검색 중...")

    # 이미지 파일 찾기
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
    image_files = []

    for ext in image_extensions:
        image_files.extend(root_dir.rglob(f'*{ext}'))
        image_files.extend(root_dir.rglob(f'*{ext.upper()}'))

    print(f"[FOUND] 총 {len(image_files)}개 이미지 파일\n")

    # 해시 계산
    hash_to_files = defaultdict(list)

    for img_path in tqdm(image_files, desc="해시 계산"):
        try:
            file_hash = get_file_hash(img_path)
            hash_to_files[file_hash].append(img_path)
        except Exception as e:
            print(f"\n[ERROR] {img_path}: {e}")

    # 중복 찾기
    duplicates = {h: files for h, files in hash_to_files.items() if len(files) > 1}

    return duplicates, len(image_files)


def remove_duplicates(duplicates: dict, dry_run: bool = True):
    """중복 파일 제거 (첫 번째 제외)"""
    total_duplicates = sum(len(files) - 1 for files in duplicates.values())

    print(f"\n[DUPLICATES] 중복 그룹: {len(duplicates)}개")
    print(f"[DUPLICATES] 중복 파일: {total_duplicates}개\n")

    if total_duplicates == 0:
        print("[OK] 중복 파일 없음!")
        return

    # 상위 10개 중복 그룹 표시
    print("[TOP 10 중복 그룹]")
    print("-" * 60)

    sorted_dupes = sorted(duplicates.items(), key=lambda x: len(x[1]), reverse=True)
    for i, (hash_val, files) in enumerate(sorted_dupes[:10], 1):
        print(f"\n{i}. {len(files)}개 중복:")
        for f in files[:3]:  # 처음 3개만 표시
            print(f"   - {f.name}")
        if len(files) > 3:
            print(f"   ... 외 {len(files) - 3}개")

    print("\n" + "=" * 60)

    if dry_run:
        print("\n[DRY RUN] 실제 삭제 안 함")
        print(f"[INFO] 실제 삭제하려면 --remove 옵션 사용")
        return

    # 실제 삭제
    print("\n[DELETE] 중복 파일 삭제 중...")
    deleted_count = 0

    for hash_val, files in tqdm(duplicates.items(), desc="삭제"):
        # 첫 번째 파일 유지, 나머지 삭제
        for file_to_delete in files[1:]:
            try:
                file_to_delete.unlink()
                deleted_count += 1
            except Exception as e:
                print(f"\n[ERROR] 삭제 실패 {file_to_delete}: {e}")

    print(f"\n[OK] {deleted_count}개 파일 삭제 완료")


def main():
    parser = argparse.ArgumentParser(description="중복 이미지 제거")
    parser.add_argument(
        '--dir',
        default='./classified_full_v2',
        help='검색할 폴더'
    )
    parser.add_argument(
        '--remove',
        action='store_true',
        help='실제로 삭제 (기본: Dry Run)'
    )

    args = parser.parse_args()

    root_dir = Path(args.dir)

    if not root_dir.exists():
        print(f"[ERROR] 폴더 없음: {root_dir}")
        return

    print("=" * 60)
    print("[TOOL] 중복 이미지 제거")
    print("=" * 60)
    print(f"[DIR] {root_dir.resolve()}")
    print(f"[MODE] {'실제 삭제' if args.remove else 'Dry Run (미리보기)'}")
    print("=" * 60)

    duplicates, total_files = find_duplicates(root_dir)
    remove_duplicates(duplicates, dry_run=not args.remove)

    print("\n[DONE] 완료!")


if __name__ == "__main__":
    main()
