#!/usr/bin/env python3
import os
import csv
import zlib
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import accumulate
from typing import List, Tuple, Optional

import h5py
import numpy as np
# sys.path.append('/mnt/crb/code/opensdpc')
# import opensdpc_old as openslide

import openslide
from tqdm import tqdm

# ============================================================
# 常量
# ============================================================
PATCH_H = 224
PATCH_W = 224
PATCH_C = 3
PATCH_BYTES = PATCH_H * PATCH_W * PATCH_C

SHARD_SIZE_BYTES = 2 * 1024**4   # 2TB
SHARD_CAP = SHARD_SIZE_BYTES // PATCH_BYTES

INDEX_DTYPE = np.dtype([
    ("h5_id", np.int32),
    ("patch_id", np.int32),
    ("global_idx", np.int64),
])

# ============================================================
# done helpers（唯一的跨机协调）
# ============================================================
def make_done_path(done_root, svs_path, *, segment=None):
    """
    done_root/path/to/wsi.svs.{start}-{end}.done
    """
    rel = Path(svs_path).as_posix().lstrip("/").replace(":", "_")
    suffix = ""
    if segment is not None:
        start, end = segment
        suffix = f".{start:09d}-{end:09d}"
    p = Path(done_root) / f"{rel}{suffix}.done"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

def _try_acquire_lock(path: Path) -> bool:
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        return True
    except FileExistsError:
        return False

def _safe_unlink(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass

# ============================================================
# CSV
# ============================================================


def read_csv(csv_path: str, *, dedup_by: str = "pair", max_wsi=None) -> Tuple[List[str], List[str]]:
    """
    dedup_by:
      - "pair": 按 (h5_path, wsi_path) 这一对去重（推荐）
      - "h5":   只按 h5_path 去重（同一 h5 多行只保留第一次）
    """
    h5s: List[str] = []
    svss: List[str] = []
    seen = set()

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)

        for r in reader:
            h5_path = (r.get("h5_file_path") or r.get("h5_path") or r.get("h5") or "").strip()
            wsi_path = (r.get("wsi_path") or r.get("svs_path") or r.get("svs") or "").strip()

            if not h5_path or not wsi_path:
                continue

            # 可选：规范化路径，减少“同一路径不同写法”导致的重复
            h5_path = os.path.abspath(h5_path)
            wsi_path = os.path.abspath(wsi_path)

            # 可选：存在性检查
            if not os.path.isfile(h5_path) or not os.path.isfile(wsi_path):
                continue

            key = (h5_path, wsi_path) if dedup_by == "pair" else h5_path

            # 关键点：保序去重（只跳过重复，不做排序/重排）
            if key in seen:
                continue
            seen.add(key)
            if max_wsi is not None and len(svss) >= max_wsi:
                break

            h5s.append(h5_path)
            svss.append(wsi_path)

    return h5s, svss

# ============================================================
# subprocess helpers
# ============================================================
def _coords_len(coords_path: Path, K: int) -> Optional[int]:
    try:
        coords = np.load(coords_path, allow_pickle=False, mmap_mode="r")
        return min(len(coords), K)
    except Exception:
        return None


def _run_subprocess(cmd: List[str]) -> Tuple[bool, str]:
    result = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode == 0:
        return True, ""
    err = result.stderr.strip()
    if result.returncode < 0:
        rc = f"signal={-result.returncode}"
    else:
        rc = f"exitcode={result.returncode}"
    if err:
        return False, f"{err} ({rc})"
    return False, rc


def _run_scan_worker(task, script_path: Path) -> Tuple[bool, str]:
    _, h5_path, coords_path = task
    cmd = [
        sys.executable,
        str(script_path),
        "--worker-mode",
        "scan",
        "--h5_path",
        h5_path,
        "--coords_path",
        str(coords_path),
    ]
    return _run_subprocess(cmd)


# ============================================================
# scan H5（coords cache，多机安全）
# ============================================================
def scan_h5(h5s, K, workers, cache_dir):
    n_keep = [0] * len(h5s)
    coords_paths: List[Optional[Path]] = [None] * len(h5s)

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    for idx, h5_path in enumerate(h5s):
        coords_path = cache_dir / f"{idx:05d}.coords.npy"
        if coords_path.exists():
            k = _coords_len(coords_path, K)
            if k is not None:
                n_keep[idx] = k
                coords_paths[idx] = coords_path
                continue
        tasks.append((idx, h5_path, coords_path))

    if tasks:
        script_path = Path(__file__).resolve()
        max_workers = max(1, min(workers, len(tasks)))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {
                executor.submit(_run_scan_worker, task, script_path): task
                for task in tasks
            }
            for fut in tqdm(
                as_completed(future_to_task),
                total=len(tasks),
                desc="scan h5",
            ):
                idx, h5_path, coords_path = future_to_task[fut]
                try:
                    ok, err = fut.result()
                except Exception as exc:
                    ok = False
                    err = f"{type(exc).__name__}: {exc}"
                if not ok:
                    print(f"[WARN] scan h5_id={idx} error: {err} ({h5_path})")
                    n_keep[idx] = 0
                    coords_paths[idx] = None
                    continue
                k = _coords_len(coords_path, K)
                if k is None:
                    print(f"[WARN] scan h5_id={idx} error: coords cache read failed ({h5_path})")
                    n_keep[idx] = 0
                    coords_paths[idx] = None
                    continue
                n_keep[idx] = k
                coords_paths[idx] = coords_path

    return n_keep, coords_paths

# ============================================================
# big_index
# ============================================================
def build_big_index(n_keep, out_dir):
    total = sum(n_keep)
    offsets = list(accumulate([0] + n_keep))[:-1]

    index_path = out_dir / "big_index.npy"
    if index_path.exists():
        return index_path, offsets

    index = np.memmap(
        index_path, dtype=INDEX_DTYPE, mode="w+", shape=(total,)
    )

    cur = 0
    for h5_id, k in enumerate(n_keep):
        if k == 0:
            continue
        index[cur:cur+k]["h5_id"] = h5_id
        index[cur:cur+k]["patch_id"] = np.arange(k, dtype=np.int32)
        index[cur:cur+k]["global_idx"] = np.arange(cur, cur+k, dtype=np.int64)
        cur += k

    index.flush()
    return index_path, offsets

# ============================================================
# shard utils
# ============================================================
def open_shard(path, patch_count=SHARD_CAP):
    if patch_count <= 0:
        raise ValueError(f"patch_count must be > 0, got {patch_count}")
    shape = (patch_count, PATCH_H, PATCH_W, PATCH_C)
    expected_bytes = patch_count * PATCH_BYTES
    if not path.exists():
        mm = np.memmap(path, dtype=np.uint8, mode="w+", shape=shape)
        del mm
    else:
        actual_bytes = path.stat().st_size
        if actual_bytes != expected_bytes:
            os.truncate(path, expected_bytes)
    return np.memmap(path, dtype=np.uint8, mode="r+", shape=shape)

# ============================================================
# worker entrypoints
# ============================================================
def _worker_scan_main(h5_path: str, coords_path: str) -> int:
    try:
        with h5py.File(h5_path, "r") as f:
            if "coords" in f:
                coords = f["coords"][:]
            else:
                coords = np.empty((0, 2), dtype=np.int64)
        if coords is None:
            coords = np.empty((0, 2), dtype=np.int64)
        coords_path = Path(coords_path)
        coords_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = coords_path.with_suffix(".tmp.npy")
        np.save(tmp, coords)
        os.replace(tmp, coords_path)
        return 0
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


def _worker_extract_main(
    h5_path: str,
    svs_path: str,
    coords_path: str,
    K: int,
    patch_start: int,
    patch_end: int,
    shard_path: str,
    base: int,
) -> int:
    slide = None
    shard_mem = None
    try:
        coords = np.load(coords_path, allow_pickle=False)
        if coords is None or len(coords) == 0:
            return 0
        k = min(K, len(coords))
        if patch_start >= k:
            return 0
        patch_end = min(patch_end, k)

        seed = zlib.crc32(h5_path.encode()) & 0xFFFFFFFF
        rng = np.random.RandomState(seed)
        sel = rng.choice(len(coords), size=k, replace=False)
        sel = sel[patch_start:patch_end]
        if len(sel) == 0:
            return 0

        end = base + len(sel)
        if end > SHARD_CAP:
            raise ValueError(f"overflow {end}->{SHARD_CAP}")

        shard_mem = open_shard(Path(shard_path), patch_count=SHARD_CAP)
        slide = openslide.open_slide(svs_path)
        for j, ci in enumerate(sel):
            x, y = coords[ci]
            patch = slide.read_region((int(x), int(y)), 0, (PATCH_W, PATCH_H))
            shard_mem[base + j] = np.asarray(patch)[..., :3]
        shard_mem.flush()
        return 0
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        if slide is not None:
            try:
                slide.close()
            except Exception:
                pass
        if shard_mem is not None:
            try:
                shard_mem.flush()
            except Exception:
                pass
            del shard_mem


def _run_extract_worker(job, script_path: Path, K: int, shard_path: Path) -> Tuple[bool, str]:
    cmd = [
        sys.executable,
        str(script_path),
        "--worker-mode",
        "extract",
        "--h5_path",
        job["h5_path"],
        "--svs_path",
        job["svs_path"],
        "--coords_path",
        str(job["coords_path"]),
        "--k",
        str(K),
        "--patch_start",
        str(job["patch_start"]),
        "--patch_end",
        str(job["patch_end"]),
        "--shard_path",
        str(shard_path),
        "--base",
        str(job["base"]),
    ]
    return _run_subprocess(cmd)


# ============================================================
# shard worker（真正的多机线性加速点）
# ============================================================
def shard_worker(shard_id, tasks, h5s, svss, coords_paths, out_dir, K, workers, done_root):
    shard_path = out_dir / f"shard_{shard_id:05d}.npy"
    done_flag = shard_path.with_suffix(".npy.done")
    processing_flag = shard_path.with_suffix(".npy.processing")
    error_log = shard_path.with_suffix(".npy.errors.log")

    if done_flag.exists():
        if not _try_acquire_lock(processing_flag):
            print(f"[INFO] shard {shard_id} done, skip")
            return
        try:
            mm = open_shard(shard_path, patch_count=SHARD_CAP)
            del mm
        finally:
            _safe_unlink(processing_flag)
        print(f"[INFO] shard {shard_id} done, skip")
        return

    if not _try_acquire_lock(processing_flag):
        print(f"[INFO] shard {shard_id} is processing elsewhere, skip")
        return

    mm = open_shard(shard_path, patch_count=SHARD_CAP)
    del mm

    had_error = False
    had_pending = False
    jobs = []
    for h5_id, base, patch_start, patch_end in tasks:
        coords_path = coords_paths[h5_id]
        if coords_path is None or not Path(coords_path).exists():
            had_error = True
            msg = f"h5_id={h5_id} missing coords cache ({h5s[h5_id]})"
            print(f"[WARN] shard {shard_id} {msg}")
            with open(error_log, "a") as f:
                f.write(msg + "\n")
            continue

        done_path = make_done_path(done_root, svss[h5_id], segment=(patch_start, patch_end))
        processing_path = done_path.with_suffix(".processing")
        if done_path.exists():
            continue
        if not _try_acquire_lock(processing_path):
            had_pending = True
            continue

        jobs.append({
            "h5_id": h5_id,
            "h5_path": h5s[h5_id],
            "svs_path": svss[h5_id],
            "coords_path": coords_path,
            "base": base,
            "patch_start": patch_start,
            "patch_end": patch_end,
            "done_path": done_path,
            "processing_path": processing_path,
        })

    try:
        if jobs:
            script_path = Path(__file__).resolve()
            max_workers = max(1, min(workers, len(jobs)))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Run each h5 in its own subprocess to isolate C-level crashes.
                future_to_job = {
                    executor.submit(_run_extract_worker, job, script_path, K, shard_path): job
                    for job in jobs
                }
                for fut in tqdm(
                    as_completed(future_to_job),
                    total=len(jobs),
                    desc=f"shard {shard_id}",
                ):
                    job = future_to_job[fut]
                    try:
                        ok, err = fut.result()
                    except Exception as exc:
                        ok = False
                        err = f"{type(exc).__name__}: {exc}"
                    if ok:
                        job["done_path"].write_text("done\n")
                    else:
                        had_error = True
                        msg = f"h5_id={job['h5_id']} error: {err} ({job['h5_path']} | {job['svs_path']})"
                        print(f"[WARN] shard {shard_id} {msg}")
                        with open(error_log, "a") as f:
                            f.write(msg + "\n")
                    _safe_unlink(job["processing_path"])
    finally:
        if not had_error and not had_pending:
            done_flag.write_text("done\n")
        _safe_unlink(processing_flag)
# ============================================================
# main
# ============================================================
def build_dataset(csv_path, out_dir, K, workers, shared_dir=None, max_wsi=None):
    out_dir = Path(out_dir)
    shared_dir = Path(shared_dir or out_dir)
    workers = max(1, int(workers))

    data_root = shared_dir
    data_root.mkdir(parents=True, exist_ok=True)
    done_root = data_root / "wsi_done"
    done_root.mkdir(parents=True, exist_ok=True)
    scan_cache_dir = data_root / "scan_cache"
    scan_cache_dir.mkdir(parents=True, exist_ok=True)

    h5s, svss = read_csv(csv_path, max_wsi=max_wsi)
    print(f"[INFO] H5 files = {len(h5s)}")

    n_keep, coords_paths = scan_h5(h5s, K, workers, scan_cache_dir)
    index_path, offsets = build_big_index(n_keep, out_dir)

    total = sum(n_keep)
    n_shards = (total + SHARD_CAP - 1) // SHARD_CAP
    print(f"[INFO] shards = {n_shards}")

    shard_tasks = [[] for _ in range(n_shards)]
    for h5_id, start in enumerate(offsets):
        k = n_keep[h5_id]
        if k <= 0:
            continue
        end = start + k
        seg_start = start
        while seg_start < end:
            shard_id = seg_start // SHARD_CAP
            shard_end = min(end, (shard_id + 1) * SHARD_CAP)
            patch_start = seg_start - start
            patch_end = shard_end - start
            base = seg_start % SHARD_CAP
            if shard_id >= len(shard_tasks):
                shard_tasks.extend([] for _ in range(shard_id + 1 - len(shard_tasks)))
            shard_tasks[shard_id].append((h5_id, base, patch_start, patch_end))
            seg_start = shard_end
    n_shards = len(shard_tasks)
    # shard 数通常很小，直接串行 shard
    for sid in range(n_shards):
        if not shard_tasks[sid]:
            continue
        shard_worker(
            sid,
            shard_tasks[sid],
            h5s,
            svss,
            coords_paths,
            out_dir,
            K,
            workers,
            done_root,
        )

# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default='/mnt/crb/work/runcode/copy_svs/datawheel_rundata/patches_output/h5_wsi_index.csv')
    parser.add_argument("--k", type=int, default=500)
    parser.add_argument("--out_dir", default='/mnt/crb/work/runcode/copy_svs/datawheel_rundata/shared_datawheel_test')
    parser.add_argument("--workers", type=int, default=128)
    parser.add_argument("--shared", help="shared dir across machines (default = --out)")
    parser.add_argument("--max_wsi", type=int, default=None)
    parser.add_argument("--worker-mode", choices=["scan", "extract"], help=argparse.SUPPRESS)
    parser.add_argument("--h5_path", help=argparse.SUPPRESS)
    parser.add_argument("--svs_path", help=argparse.SUPPRESS)
    parser.add_argument("--coords_path", help=argparse.SUPPRESS)
    parser.add_argument("--patch_start", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--patch_end", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--shard_path", help=argparse.SUPPRESS)
    parser.add_argument("--base", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker_mode == "scan":
        if not args.h5_path or not args.coords_path:
            print("missing --h5_path/--coords_path", file=sys.stderr)
            raise SystemExit(2)
        raise SystemExit(_worker_scan_main(args.h5_path, args.coords_path))
    if args.worker_mode == "extract":
        missing = []
        if not args.h5_path:
            missing.append("--h5_path")
        if not args.svs_path:
            missing.append("--svs_path")
        if not args.coords_path:
            missing.append("--coords_path")
        if args.patch_start is None:
            missing.append("--patch_start")
        if args.patch_end is None:
            missing.append("--patch_end")
        if not args.shard_path:
            missing.append("--shard_path")
        if args.base is None:
            missing.append("--base")
        if missing:
            print(f"missing {', '.join(missing)}", file=sys.stderr)
            raise SystemExit(2)
        raise SystemExit(
            _worker_extract_main(
                args.h5_path,
                args.svs_path,
                args.coords_path,
                args.k,
                args.patch_start,
                args.patch_end,
                args.shard_path,
                args.base,
            )
        )

    build_dataset(
        csv_path=args.csv,
        out_dir=args.out_dir,
        K=args.k,
        workers=args.workers,
        shared_dir=args.shared,
        max_wsi=args.max_wsi,
    )
