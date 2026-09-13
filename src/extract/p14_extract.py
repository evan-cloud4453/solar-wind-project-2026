"""P14용 물리 피처 추출기.

P11의 피처를 호환되게 다시 저장하면서 다음을 추가한다.

* flare_area / flare_power: 두 EUV 채널에서 동시에 나타난 강한 고휘도 영역
* shape: 최대 CH 성분의 면적·연결성·원형도·종횡비·경계밀도

강도 깊이(다중 dark contour)와 CH 경계거리(theta_b)는 P11과 동일한 area/depth
배열에 이미 들어 있으므로 이 파일의 p14a 캐시 하나만으로 P14를 실행할 수 있다.
"""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import p11_extract as base


VERSION = "p14a"
FLARE_CUT = 2.50          # 원반 채널 중앙값의 2.5배 이상, 두 채널 동시 밝음
FLARE_EXCESS_CAP = 8.0    # 단일 포화 픽셀이 전체 피처를 지배하지 않게 제한
SHAPE_LEVEL = 1           # dark0.45: P11의 표준 CH 마스크

EXTRA_KEYS = ("flare_area", "flare_power", "shape")
SHAPE_COLUMNS = (
    "largest_frac", "component_count", "compactness", "elongation",
    "boundary_density", "core_fraction",
)


def _boundary(mask: np.ndarray) -> np.ndarray:
    """4-이웃 기준 내부 경계. 외부 원반 경계는 CH 경계로 세지 않는다."""
    interior = (np.roll(mask, 1, 0) & np.roll(mask, -1, 0)
                & np.roll(mask, 1, 1) & np.roll(mask, -1, 1))
    interior[0] = interior[-1] = False
    interior[:, 0] = interior[:, -1] = False
    return mask & ~interior


def _shape_features(ch_plane: np.ndarray, disk: np.ndarray, total_px: int,
                    core_mask: np.ndarray) -> np.ndarray:
    """면적만으로 구별 못 하는 큰 홀/조각난 홀/길쭉한 홀을 요약한다."""
    labels, n_found = base.connected_components(ch_plane)
    if labels is None or n_found <= 0:
        return np.zeros(len(SHAPE_COLUMNS), np.float32)

    flat = labels[disk]
    sizes = np.bincount(flat, minlength=n_found + 1).astype(np.float64)
    sizes[0] = 0.0
    largest = int(sizes.argmax())
    area = float(sizes[largest])
    if area <= 0.0:
        return np.zeros(len(SHAPE_COLUMNS), np.float32)

    member = labels == largest
    perimeter = float(_boundary(member).sum())
    compactness = float(4.0 * np.pi * area / max(perimeter * perimeter, 1.0))

    ys, xs = np.nonzero(member)
    if len(xs) >= 3:
        covariance = np.cov(np.stack((ys, xs)), bias=True)
        eig = np.clip(np.linalg.eigvalsh(covariance), 1e-6, None)
        elongation = float(np.sqrt(eig[-1] / eig[0]))
    else:
        elongation = 1.0

    core_in_largest = float((core_mask & member).sum())
    component_count = float((sizes / max(total_px, 1) > base.COMPONENT_MIN_FRAC).sum())
    return np.asarray((
        area / max(total_px, 1), component_count, compactness, elongation,
        perimeter / max(np.sqrt(area), 1.0), core_in_largest / area,
    ), np.float32)


# base.process 와 아래 process 가 같은 프레임을 두 번 읽지 않게 한 장만 들고 있는다.
# 프레임은 어느 쪽에서도 제자리 수정되지 않으므로 공유해도 안전하다.
_RAW_LOAD_PAIR = base.load_pair
_RAW_DETECT_DISK = base.detect_disk
_FRAME_CACHE = {"key": None, "frame": None, "disk": None}


def _load_pair_once(split: str, name: str) -> np.ndarray:
    key = (split, name)
    if _FRAME_CACHE["key"] != key:
        _FRAME_CACHE.update(key=key, frame=_RAW_LOAD_PAIR(split, name), disk=None)
    return _FRAME_CACHE["frame"]


def _detect_disk_once(frame: np.ndarray):
    if _FRAME_CACHE["disk"] is None:
        _FRAME_CACHE["disk"] = _RAW_DETECT_DISK(frame)
    return _FRAME_CACHE["disk"]


def process(args):
    """P11 결과 + P14 추가 피처를 한 프레임에서 추출한다."""
    p11 = base.process(args)
    split, name = args
    frame = base.load_pair(split, name)
    side = frame.shape[-1]
    center_y, center_x, radius = base.detect_disk(frame)
    r_used = radius * base.DISK_MARGIN
    disk, cell, _weight, _mu, _dy, _dx = base.geometry(side, center_y, center_x, r_used)

    on_disk = frame[:, disk]
    total_px = max(int(on_disk.shape[1]), 1)
    median = np.median(on_disk[:, ::4], axis=1, keepdims=True)
    normalized = on_disk / np.maximum(median, 1e-3)

    # 플레어는 밝은 활성영역과 분리하기 위해 더 높은 threshold와 excess를 쓴다.
    flare = np.logical_and(normalized[0] >= FLARE_CUT, normalized[1] >= FLARE_CUT)
    excess = np.clip(np.minimum(normalized[0], normalized[1]) - FLARE_CUT,
                     0.0, FLARE_EXCESS_CAP)
    flare_area = (np.bincount(cell[flare], minlength=base.FINE_CELLS)
                  / total_px).astype(np.float32)
    flare_power = (np.bincount(cell, weights=excess, minlength=base.FINE_CELLS)
                   / total_px).astype(np.float32)

    ch_plane = np.zeros((side, side), bool)
    body = np.logical_and(normalized[0] <= base.CH_CUTS[SHAPE_LEVEL],
                          normalized[1] <= base.CH_CUTS[SHAPE_LEVEL])
    core = np.logical_and(normalized[0] <= base.CH_CUTS[0],
                          normalized[1] <= base.CH_CUTS[0])
    ch_plane[disk] = body
    core_plane = np.zeros((side, side), bool)
    core_plane[disk] = core
    shape = _shape_features(ch_plane, disk, total_px, core_plane)
    return (*p11, flare_area, flare_power, shape)


def _initializer(data_root: str, backend: str) -> None:
    base._initializer(data_root, backend)
    base.load_pair = _load_pair_once      # base.process 안의 전역 조회도 함께 바뀐다
    base.detect_disk = _detect_disk_once


def _files(data_root: Path, split: str) -> list[str]:
    return base.split_files(data_root, split)


def run_split(data_root: Path, split: str, cache_dir: Path, workers: int, limit: int) -> None:
    path = cache_dir / f"{VERSION}_{split}.npz"
    files = _files(data_root, split)
    if limit:
        files = files[:limit]
    if path.exists() and not limit:
        with np.load(path) as cached:
            if cached["area"].shape[0] == len(files):
                print(f"[{split}] 캐시 재사용 {path.name} ({len(files):,}장)", flush=True)
                return

    n = len(files)
    out = {
        "area": np.zeros((n, base.N_LEVELS, base.FINE_CELLS), np.float32),
        "area_true": np.zeros((n, base.N_LEVELS, base.FINE_CELLS), np.float32),
        "depth_max": np.zeros((n, base.FINE_CELLS), np.float32),
        "depth_sum": np.zeros((n, base.FINE_CELLS), np.float32),
        "ratio": np.zeros((n, base.FINE_CELLS), np.float32),
        "frame": np.zeros((n, len(base.FRAME_COLUMNS)), np.float32),
        "flare_area": np.zeros((n, base.FINE_CELLS), np.float32),
        "flare_power": np.zeros((n, base.FINE_CELLS), np.float32),
        "shape": np.zeros((n, len(SHAPE_COLUMNS)), np.float32),
    }
    started = time.perf_counter()
    tasks = [(split, name) for name in files]

    def store(i: int, result) -> None:
        (out["area"][i], out["area_true"][i], out["depth_max"][i], out["depth_sum"][i],
         out["ratio"][i], out["frame"][i], out["flare_area"][i], out["flare_power"][i],
         out["shape"][i]) = result

    if workers <= 1:
        _initializer(str(data_root), base._BACKEND)
        iterator = map(process, tasks)
        for i, result in enumerate(iterator):
            store(i, result)
            base.report(split, i + 1, n, started)
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=_initializer,
                                 initargs=(str(data_root), base._BACKEND)) as pool:
            for i, result in enumerate(pool.map(process, tasks, chunksize=16)):
                store(i, result)
                base.report(split, i + 1, n, started)

    if limit:
        print(f"[{split}] 스모크 테스트 — 저장하지 않는다", flush=True)
        print(f"  flare 면적 {out['flare_area'].sum(1).mean():.5f} / "
              f"power {out['flare_power'].sum(1).mean():.5f} / "
              f"최대성분 {out['shape'][:, 0].mean():.4f}", flush=True)
        return

    temporary = path.with_name(path.stem + ".part.npz")
    np.savez(temporary, files=np.asarray(files), **out)
    os.replace(temporary, path)
    print(f"[{split}] 완료 {(time.perf_counter() - started) / 60:.1f}분 -> {path.name}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--splits", default="train,validation,test")
    parser.add_argument("--cache", default="work/cache")
    parser.add_argument("--backend", default="auto", choices=("auto", "cv2", "scipy", "numpy"))
    args = parser.parse_args()
    base._BACKEND = base.pick_backend(args.backend)
    data_root = base.find_data_root()
    cache_dir = Path(args.cache)
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"P14 추출 {VERSION} | 데이터 {data_root} | 백엔드 {base._BACKEND}", flush=True)
    for split in (s.strip() for s in args.splits.split(",") if s.strip()):
        run_split(data_root, split, cache_dir, args.workers, args.limit)
    print("전부 완료", flush=True)


if __name__ == "__main__":
    main()
