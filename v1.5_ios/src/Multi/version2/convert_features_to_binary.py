import os
import time
import glob
import gc
import numpy as np
import psutil
import polars as pl
from typing import Optional, Any
from pathlib import Path

# ---- 설정 ----
PROJECT_ROOT = Path(__file__).resolve().parent
while PROJECT_ROOT != PROJECT_ROOT.parent and not ((PROJECT_ROOT / "data").exists() and (PROJECT_ROOT / "src").exists()):
    PROJECT_ROOT = PROJECT_ROOT.parent

BASE_DIR = PROJECT_ROOT / "src" / "Multi" / "version2" / "features"
CSV_PATH = PROJECT_ROOT / "features" / "clip_dino_midas_features.csv"
TMP_PARQDIR = BASE_DIR / "tmp_parquet_chunks"
FINAL_PARQ = BASE_DIR / "clip_dino_midas_features.parquet"
NPZ_DIR = BASE_DIR / "npz_shards"
BATCH_SIZE = 50  # 💡 자동 resume 시 안전한 배치 크기


def get_csv_info(csv_path: str):
    """CSV 헤더와 총 행 수 확인"""
    print(f"\n📂 CSV 파일: {csv_path}")
    print("⏳ 파일 분석 중...")
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        header_line = f.readline().strip()
        if not header_line:
            raise ValueError("CSV 파일이 비어있거나 헤더가 없습니다.")
        headers = [h.strip() for h in header_line.split(",")]
        total_rows = sum(1 for line in f if line.strip())
    print(f"✅ 컬럼 수: {len(headers)}개")
    print(f"✅ 총 행 수: {total_rows:,}행 (헤더 제외)")
    return headers, total_rows


def infer_schema_minimal(csv_path: str):
    """최소한의 행으로 스키마 추론"""
    print("⏳ 스키마 추론 중...")
    sample = pl.read_csv(csv_path, n_rows=10, infer_schema_length=10, low_memory=True)
    schema = sample.schema
    del sample
    gc.collect()
    return schema


def get_numeric_cols(schema: dict) -> list[str]:
    """숫자형 컬럼만 선별"""
    numeric_types = (
        pl.Int8, pl.Int16, pl.Int32, pl.Int64,
        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
        pl.Float32, pl.Float64
    )
    return [col for col, dtype in schema.items() if col != "filename" and dtype in numeric_types]


def save_chunk_atomic(df: pl.DataFrame, chunk_id: int, parquet_path: str, npz_path: str, numeric_cols: list[str]) -> bool:
    """원자적 청크 저장"""
    temp_parquet = parquet_path + ".tmp"
    temp_npz = npz_path + ".tmp"
    try:
        df.write_parquet(temp_parquet, compression="snappy", use_pyarrow=False)
        filenames = df["filename"].to_numpy()
        X = df.select(numeric_cols).to_numpy().astype(np.float16, copy=False)
        with open(temp_npz, "wb") as f:
            np.savez_compressed(f, X=X, filenames=filenames)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_parquet, parquet_path)
        os.replace(temp_npz, npz_path)
        return True
    except Exception as e:
        for tmp in (temp_parquet, temp_npz):
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except:
                    pass
        print(f"❌ 청크 {chunk_id} 저장 실패: {e}")
        return False


def cleanup_temp_files():
    """임시파일 정리"""
    for pattern in ["*.tmp", "*.tmp.npz"]:
        for p in glob.glob(os.path.join(str(NPZ_DIR), pattern)):
            try:
                os.remove(p)
            except:
                pass


def show_memory():
    """현재 메모리 사용량(MB) 출력"""
    mem = psutil.virtual_memory()
    return f"{mem.used / (1024**2):,.0f}MB / {mem.total / (1024**2):,.0f}MB"


def find_resume_point():
    """마지막으로 완성된 청크 다음 인덱스를 반환"""
    npz_files = sorted(glob.glob(os.path.join(str(NPZ_DIR), "shard_*.npz")))
    if not npz_files:
        return 0
    last_file = npz_files[-1]
    last_idx = int(os.path.basename(last_file).split("_")[1].split(".")[0])
    return last_idx


def main():
    print("=" * 70)
    print("CSV → Parquet + NPZ 변환 (Auto-Resume + Memory Monitor)")
    print("=" * 70)

    os.makedirs(BASE_DIR, exist_ok=True)
    os.makedirs(TMP_PARQDIR, exist_ok=True)
    os.makedirs(NPZ_DIR, exist_ok=True)
    cleanup_temp_files()

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"❌ CSV 파일 없음: {CSV_PATH}")

    headers, total_rows = get_csv_info(CSV_PATH)
    schema = infer_schema_minimal(CSV_PATH)
    numeric_cols = get_numeric_cols(schema)
    expected_feature_count = len(numeric_cols)
    total_chunks = (total_rows + BATCH_SIZE - 1) // BATCH_SIZE

    print(f"✅ 숫자형 컬럼 수: {expected_feature_count}")
    print(f"✅ 총 청크 수: {total_chunks}")

    # 자동 resume 위치 찾기
    resume_from = find_resume_point()
    if resume_from >= total_chunks:
        print("\n✅ 모든 청크 이미 완료됨. 종료합니다.")
        return

    print(f"\n🔁 이어서 실행: {resume_from + 1}번 청크부터 시작")
    start_time = time.time()

    for chunk_idx in range(resume_from, total_chunks):
        chunk_id = chunk_idx + 1
        skip_rows = chunk_idx * BATCH_SIZE
        parquet_path = os.path.join(TMP_PARQDIR, f"chunk_{chunk_id:05d}.parquet")
        npz_path = os.path.join(NPZ_DIR, f"shard_{chunk_id:05d}.npz")

        if os.path.exists(parquet_path) and os.path.exists(npz_path):
            print(f"⏭️ 청크 {chunk_id}/{total_chunks} - 이미 존재 (스킵)")
            continue

        try:
            df = pl.read_csv(
                CSV_PATH,
                skip_rows=skip_rows + (1 if chunk_idx > 0 else 0),
                n_rows=BATCH_SIZE,
                schema=schema if chunk_idx > 0 else None,
                has_header=(chunk_idx == 0),
                low_memory=True
            )
        except Exception as e:
            print(f"❌ 청크 {chunk_id} 읽기 실패: {e}")
            break

        if df.height == 0:
            print(f"✅ 청크 {chunk_id}: 데이터 없음 → 종료")
            break

        ok = save_chunk_atomic(df, chunk_id, parquet_path, npz_path, numeric_cols)
        del df
        gc.collect()

        mem_status = show_memory()
        if ok:
            print(f"✅ 청크 {chunk_id:3d}/{total_chunks} 완료 | 메모리: {mem_status}")
        else:
            print(f"❌ 청크 {chunk_id:3d} 실패 | 메모리: {mem_status}")
            break

    print("\n🎉 모든 청크 변환 종료")
    total_elapsed = time.time() - start_time
    H, rem = divmod(int(total_elapsed), 3600)
    M, S = divmod(rem, 60)
    print(f"⏱️ 총 소요 시간: {H:02d}:{M:02d}:{S:02d}")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ 사용자 중단 - 다음 실행 시 자동으로 이어서 처리됩니다.")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
