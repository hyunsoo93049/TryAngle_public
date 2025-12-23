"""
quick_check_reference.py
- precompute에서 생성한 .npz / .json을 불러와 내용 점검
- 임베딩 유무/차원, 깊이 통계, 구도요약 등을 콘솔에 보기 좋게 출력
"""

import os, json, argparse
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref_npz", required=True, help="precompute에서 저장된 .npz 경로")
    args = ap.parse_args()

    path_npz = args.ref_npz
    path_json = path_npz.replace(".npz", ".json")

    if not os.path.exists(path_npz):
        raise FileNotFoundError(path_npz)
    if not os.path.exists(path_json):
        raise FileNotFoundError(path_json)

    data = np.load(path_npz, allow_pickle=True)
    with open(path_json, "r", encoding="utf-8") as f:
        meta = json.load(f)

    print("=== NPZ ===")
    for k in data.files:
        arr = data[k]
        print(f"- {k}: shape={arr.shape}, dtype={arr.dtype}, nonzero={np.count_nonzero(arr)}")

    print("\n=== JSON(meta) ===")
    print(json.dumps(meta.get("meta", {}), ensure_ascii=False, indent=2))

    comp = meta.get("composition")
    if comp:
        print("\n=== Composition Summary ===")
        for k, v in comp.items():
            print(f"- {k}: {v}")

if __name__ == "__main__":
    main()
