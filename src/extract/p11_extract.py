#!/usr/bin/env python
"""P11 확장 추출 — 대회 서버에서 백그라운드로 1회만 돌린다.

P9(`CH_CODE_VERSION="p7a"`) 추출을 그대로 포함하면서 다음을 추가로 낸다.

  area       (n, 4, 360)  원 픽셀수 면적비            <- p7a 와 동일 의미 (호환)
  area_true  (n, 4, 360)  μ 보정 진짜면적비            <- [L2] 정밀판
  depth_max  (n, 360)     셀 내 CH 최대 경계거리 θ_b   <- [L3] WSA 의 θ_b 대리값
  depth_sum  (n, 360)     셀 내 θ_b 합 / 원반픽셀수    <- 평균깊이 = depth_sum / area
  ratio      (n, 360)     셀 평균 I211 / I193          <- [L6] 온도 대리값
  frame      (n, 11)      프레임 스칼라 (아래 FRAME_COLUMNS)

이미지 순서는 P9 의 `train_files` (사슬 복원 순서) 와 **완전히 동일**하다.
노트북의 `train_map` 인덱스가 그대로 맞는다.

사용:
    python p11_extract.py --limit 40          # 스모크 테스트 (1분 내)
    nohup python p11_extract.py > extract.log 2>&1 &
    tail -f extract.log

주의: 워커를 4개 넘기지 말 것 (운영진 공지 — 다중 프로세스로 VM 다운 전례).
"""

import argparse
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

# ---- P9 와 반드시 일치시켜야 하는 상수 -------------------------------------
CHANNELS = ("193", "211")
IMAGE_COLUMNS = [f"image_{i:02d}" for i in range(20)]
FINE_LAT, FINE_LON = 12, 30
FINE_CELLS = FINE_LAT * FINE_LON
CH_CUTS = (0.30, 0.45, 0.60)
BRIGHT_CUT = 1.60
DISK_MARGIN = 0.95
LEVEL_NAMES = [f"dark{c}" for c in CH_CUTS] + ["bright"]
N_LEVELS = len(LEVEL_NAMES)

# ---- P11 신규 -------------------------------------------------------------
DEPTH_LEVEL = 1            # depth 를 재는 기준 마스크 = dark0.45 (CH_CUTS[1])
MU_FLOOR = 0.30            # 1/μ 가중의 상한 클램프 (최대 3.33배). 잘라내지는 않는다
COMPONENT_MIN_FRAC = 0.002  # 유효 연결성분 최소 면적비
VERSION = "p11a"

FRAME_COLUMNS = ["cy", "cx", "radius", "total_px", "total_true",
                 "big_frac", "big_frac_true", "n_components",
                 "big_lat_deg", "big_lon_deg", "big_depth_deg"]

_DATA_ROOT = None
_SIDE_CACHE = {}

# ---- 형태 백엔드: cv2 -> scipy -> numpy 순으로 고른다 ----------------------

def pick_backend(forced="auto"):
    """--backend 로 강제할 수 있게 한다. 대회 VM 에 무엇이 있는지 미리 알 수 없다."""
    if forced in ("cv2", "scipy", "numpy"):
        return forced
    try:
        import cv2  # noqa: F401
        return "cv2"
    except Exception:
        pass
    try:
        from scipy import ndimage  # noqa: F401
        return "scipy"
    except Exception:
        return "numpy"


_BACKEND = pick_backend()


def distance_transform(mask):
    """마스크 내부의 각 픽셀에서 가장 가까운 배경까지의 픽셀 거리."""
    if _BACKEND == "cv2":
        import cv2
        padded = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), np.uint8)
        padded[1:-1, 1:-1] = mask
        return cv2.distanceTransform(padded, cv2.DIST_L2, 5)[1:-1, 1:-1]
    if _BACKEND == "scipy":
        from scipy import ndimage
        return ndimage.distance_transform_edt(mask)
    # 순수 numpy 폴백 — 4연결 침식 반복. 해상도는 거칠지만 신호 성격은 같다.
    current = mask.copy()
    depth = np.zeros(mask.shape, np.float32)
    for _ in range(24):
        if not current.any():
            break
        current = (current
                   & np.roll(current, 1, 0) & np.roll(current, -1, 0)
                   & np.roll(current, 1, 1) & np.roll(current, -1, 1))
        current[0] = current[-1] = False
        current[:, 0] = current[:, -1] = False
        depth += current
    return depth


def connected_components(mask):
    """(라벨맵, 개수). 백엔드가 없으면 (None, 0) 을 돌려 형태 피처를 비운다."""
    if _BACKEND == "cv2":
        import cv2
        count, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
        return labels, count - 1
    if _BACKEND == "scipy":
        from scipy import ndimage
        labels, count = ndimage.label(mask, structure=np.ones((3, 3), bool))
        return labels, count
    return None, 0


# ---- 기하 ------------------------------------------------------------------

def load_pair(split, name):
    planes = []
    for channel in CHANNELS:
        with Image.open(_DATA_ROOT / split / channel / name) as image:
            planes.append(np.asarray(image.convert("L"), dtype=np.float32))
    return np.stack(planes)


def detect_disk(frame):
    """P9 `detect_disk` 와 동일. 플레어에 둔감한 원반 검출."""
    plane = frame.mean(axis=0)
    background = np.percentile(plane, 2.0)
    interior = np.percentile(plane, 70.0)
    mask = plane > background + 0.35 * (interior - background)
    ys, xs = np.nonzero(mask)
    return float(ys.mean()), float(xs.mean()), float(math.sqrt(mask.sum() / math.pi))


def geometry(side, center_y, center_x, radius):
    """셀 번호 · μ 가중 · 위경도. P9 `cell_geometry` 의 격자 정의를 그대로 쓴다."""
    key = (side, round(center_y), round(center_x), round(radius))
    entry = _SIDE_CACHE.get(key)
    if entry is not None:
        return entry
    _, cy, cx, r = key
    grid_y, grid_x = np.mgrid[0:side, 0:side].astype(np.float32)
    dy, dx = grid_y - cy, grid_x - cx
    rho2 = (dy * dy + dx * dx) / (r * r)
    disk = rho2 <= 1.0
    # r 은 이미 DISK_MARGIN 이 곱해진 값이다. μ 는 **진짜 태양 반지름** 기준이어야
    # 하므로 되돌려서 잰다. 이걸 빼면 원반 가장자리 보정이 과대평가된다.
    mu = np.sqrt(np.clip(1.0 - rho2 * DISK_MARGIN ** 2, 0.0, 1.0))
    weight = 1.0 / np.clip(mu, MU_FLOOR, 1.0)          # 잘라내지 않고 상한만 건다
    lat = np.clip((grid_y - (cy - r)) / (2 * r) * FINE_LAT, 0, FINE_LAT - 1e-4)
    lon = np.clip((grid_x - (cx - r)) / (2 * r) * FINE_LON, 0, FINE_LON - 1e-4)
    cell = (lat.astype(np.int32) * FINE_LON + lon.astype(np.int32))
    # 위경도용 정규화도 진짜 반지름 기준으로 되돌린다
    entry = (disk, cell[disk], weight[disk], mu,
             dy / r * DISK_MARGIN, dx / r * DISK_MARGIN)
    _SIDE_CACHE.clear()                                 # 프레임별 검출이라 1개만 유지
    _SIDE_CACHE[key] = entry
    return entry


def heliographic(dy_norm, dx_norm):
    """B0=0 가정의 일면 위경도(도). dy/dx 는 반지름으로 정규화된 값."""
    lat = math.asin(max(-1.0, min(1.0, dy_norm)))
    denominator = math.cos(lat)
    if denominator < 1e-3:
        return math.degrees(lat), 0.0
    lon = math.asin(max(-1.0, min(1.0, dx_norm / denominator)))
    return math.degrees(lat), math.degrees(lon)


# ---- 프레임 1장 ------------------------------------------------------------

def process(args):
    split, name = args
    frame = load_pair(split, name)
    side = frame.shape[-1]
    center_y, center_x, radius = detect_disk(frame)
    r_used = radius * DISK_MARGIN
    disk, cell, weight, mu, dy_norm, dx_norm = geometry(side, center_y, center_x, r_used)

    on_disk = frame[:, disk]
    total_px = max(on_disk.shape[1], 1)
    total_true = float(weight.sum()) or 1.0
    median = np.median(on_disk[:, ::4], axis=1, keepdims=True)
    normalized = on_disk / np.maximum(median, 1e-3)

    area = np.zeros((N_LEVELS, FINE_CELLS), np.float32)
    area_true = np.zeros((N_LEVELS, FINE_CELLS), np.float32)
    selections = []
    for level, cut in enumerate(CH_CUTS):
        selections.append(np.logical_and(normalized[0] <= cut, normalized[1] <= cut))
    selections.append(np.logical_and(normalized[0] >= BRIGHT_CUT,
                                     normalized[1] >= BRIGHT_CUT))
    for level, selection in enumerate(selections):
        area[level] = np.bincount(cell[selection], minlength=FINE_CELLS) / total_px
        area_true[level] = np.bincount(cell[selection], weights=weight[selection],
                                       minlength=FINE_CELLS) / total_true

    # 211/193 비 — 온도 대리값
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_pixels = np.where(on_disk[0] > 1.0, on_disk[1] / np.maximum(on_disk[0], 1.0), 0.0)
    counts = np.bincount(cell, minlength=FINE_CELLS).astype(np.float32)
    ratio = (np.bincount(cell, weights=ratio_pixels, minlength=FINE_CELLS)
             / np.maximum(counts, 1.0)).astype(np.float32)

    # ---- [L3] θ_b (경계거리) 와 형태 ---------------------------------------
    ch_plane = np.zeros((side, side), bool)
    ch_plane[disk] = selections[DEPTH_LEVEL]
    depth_pixels = distance_transform(ch_plane)[disk]
    # 투영 보정: 반경방향 국소 배율 1/μ. 원반 중심에서는 θ = d/R [rad] 로 정확하다.
    depth_deg = (depth_pixels * DISK_MARGIN / r_used
                 * (180.0 / math.pi) * weight).astype(np.float32)
    selection = selections[DEPTH_LEVEL]
    depth_max = np.zeros(FINE_CELLS, np.float32)
    np.maximum.at(depth_max, cell[selection], depth_deg[selection])
    depth_sum = (np.bincount(cell[selection], weights=depth_deg[selection],
                             minlength=FINE_CELLS) / total_px).astype(np.float32)

    labels, n_found = connected_components(ch_plane)
    big_frac = big_frac_true = big_lat = big_lon = big_depth = 0.0
    n_components = 0.0
    if labels is not None and n_found > 0:
        flat = labels[disk]
        sizes = np.bincount(flat, minlength=n_found + 1)
        sizes[0] = 0
        biggest = int(sizes.argmax())
        if sizes[biggest] > 0:
            member = flat == biggest
            big_frac = float(sizes[biggest]) / total_px
            big_frac_true = float(weight[member].sum()) / total_true
            big_depth = float(depth_deg[member].max())
            centroid_weight = weight[member]
            total_weight = float(centroid_weight.sum()) or 1.0
            big_lat, big_lon = heliographic(
                float((dy_norm[disk][member] * centroid_weight).sum()) / total_weight,
                float((dx_norm[disk][member] * centroid_weight).sum()) / total_weight)
        n_components = float((sizes / total_px > COMPONENT_MIN_FRAC).sum())
    else:
        big_frac = big_frac_true = np.nan
        n_components = np.nan

    scalars = np.array([center_y, center_x, radius, total_px, total_true,
                        big_frac, big_frac_true, n_components,
                        big_lat, big_lon, big_depth], np.float32)
    return area, area_true, depth_max, depth_sum, ratio, scalars


def _initializer(data_root, backend):
    global _DATA_ROOT, _BACKEND
    _DATA_ROOT = Path(data_root)
    _BACKEND = backend


# ---- 파일 목록 (P9 사슬 복원과 동일) ---------------------------------------

def reconstruct_frame_chains(inputs):
    images = inputs[IMAGE_COLUMNS].to_numpy()
    successor, predecessor, conflicts = {}, {}, 0
    for row in images:
        for current, following in zip(row[:-1], row[1:]):
            if successor.setdefault(current, following) != following:
                conflicts += 1
            if predecessor.setdefault(following, current) != current:
                conflicts += 1
    names = set(images.ravel().tolist())
    chains, visited = [], set()
    for head in sorted(names - set(predecessor)):
        chain, node = [], head
        while node is not None and node not in visited:
            visited.add(node); chain.append(node); node = successor.get(node)
        chains.append(chain)
    assert conflicts == 0 and not (names - visited), "사슬 복원 실패"
    return chains


def split_files(data_root, split):
    inputs = pd.read_csv(data_root / split / "inputs.csv")
    return [n for chain in reconstruct_frame_chains(inputs) for n in chain]


def find_data_root():
    candidates = [
        Path(os.getenv("SW_DATA_ROOT", "")) if os.getenv("SW_DATA_ROOT") else None,
        Path("public_dataset/competition_dataset_6h"),
        Path("/home/jovyan/public_dataset/competition_dataset_6h"),
        Path("public/public_dataset/competition_dataset_6h"),
        Path("/home/jovyan/public/public_dataset/competition_dataset_6h"),
        Path("dataset"), Path("/home/jovyan/dataset"),
    ]
    for candidate in candidates:
        if candidate is not None and (candidate / "train/inputs.csv").exists():
            return candidate
    raise FileNotFoundError("데이터 경로 없음 — SW_DATA_ROOT 를 지정할 것")


def run_split(data_root, split, cache_dir, workers, limit):
    path = cache_dir / f"{VERSION}_{split}.npz"
    files = split_files(data_root, split)
    if limit:
        files = files[:limit]
    if path.exists() and not limit:
        with np.load(path) as cached:
            if cached["area"].shape[0] == len(files):
                print(f"[{split}] 캐시 재사용 {path.name} ({len(files):,}장)", flush=True)
                return
        print(f"[{split}] 캐시 길이 불일치 -> 재계산", flush=True)

    n = len(files)
    out = {
        "area": np.zeros((n, N_LEVELS, FINE_CELLS), np.float32),
        "area_true": np.zeros((n, N_LEVELS, FINE_CELLS), np.float32),
        "depth_max": np.zeros((n, FINE_CELLS), np.float32),
        "depth_sum": np.zeros((n, FINE_CELLS), np.float32),
        "ratio": np.zeros((n, FINE_CELLS), np.float32),
        "frame": np.zeros((n, len(FRAME_COLUMNS)), np.float32),
    }
    started = time.perf_counter()
    tasks = [(split, name) for name in files]

    def store(i, result):
        (out["area"][i], out["area_true"][i], out["depth_max"][i],
         out["depth_sum"][i], out["ratio"][i], out["frame"][i]) = result

    if workers <= 1:
        _initializer(data_root, _BACKEND)
        for i, task in enumerate(tasks):
            store(i, process(task))
            report(split, i + 1, n, started)
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=_initializer,
                                 initargs=(str(data_root), _BACKEND)) as pool:
            for i, result in enumerate(pool.map(process, tasks, chunksize=16)):
                store(i, result)
                report(split, i + 1, n, started)

    if limit:
        print(f"[{split}] 스모크 테스트 — 저장하지 않는다", flush=True)
        summarize(out)
        return
    temporary = path.with_name(path.stem + ".part.npz")   # np.savez 는 .npz 를 강제한다
    np.savez(temporary, files=np.array(files), **out)
    os.replace(temporary, path)          # 중간에 끊겨도 반쪽 npz 가 남지 않는다
    print(f"[{split}] 완료 {(time.perf_counter() - started) / 60:.1f}분 -> {path.name} "
          f"({path.stat().st_size / 1024**2:.0f} MB)", flush=True)
    summarize(out)


def report(split, done, total, started):
    if done % 500 and done != total:
        return
    elapsed = time.perf_counter() - started
    print(f"  [{split}] {done:,}/{total:,} ({elapsed / 60:.1f}분 경과, "
          f"남은 {elapsed / done * (total - done) / 60:.1f}분)", flush=True)


def summarize(out):
    frame = out["frame"]
    radius = frame[:, 2]
    dark = out["area"][:, DEPTH_LEVEL].sum(axis=1)
    dark_true = out["area_true"][:, DEPTH_LEVEL].sum(axis=1)
    print(f"    반지름 {radius.mean():.1f} ± {radius.std():.1f}px "
          f"({radius.std() / max(radius.mean(), 1e-6):.2%})")
    print(f"    CH 면적비  원본 {dark.mean():.4f} / μ보정 {dark_true.mean():.4f} "
          f"(비 {dark_true.mean() / max(dark.mean(), 1e-9):.3f})")
    print(f"    θ_b 최대   {out['depth_max'].max(axis=1).mean():.2f}° "
          f"(전 프레임 최대 {out['depth_max'].max():.2f}°)")
    if np.isfinite(frame[:, 5]).any():
        print(f"    최대성분비 {np.nanmean(frame[:, 5]):.4f} / 성분수 "
              f"{np.nanmean(frame[:, 7]):.2f} / 211-193비 {out['ratio'].mean():.3f}")
    else:
        print(f"    형태 피처 없음 (백엔드 {_BACKEND}) / 211-193비 {out['ratio'].mean():.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="스모크 테스트 장수")
    parser.add_argument("--splits", default="train,validation,test")
    parser.add_argument("--cache", default="work/cache")
    parser.add_argument("--backend", default="auto",
                        choices=("auto", "cv2", "scipy", "numpy"))
    args = parser.parse_args()

    global _BACKEND
    _BACKEND = pick_backend(args.backend)
    data_root = find_data_root()
    cache_dir = Path(args.cache)
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"P11 추출 {VERSION} | 데이터 {data_root} | 형태 백엔드 {_BACKEND} | "
          f"워커 {args.workers}", flush=True)
    if _BACKEND == "numpy":
        print("  ⚠️ cv2/scipy 둘 다 없다. θ_b 는 침식 근사, 연결성분은 비운다.", flush=True)

    # test 를 마지막에 두어, 중간에 끊겨도 train/val 은 확보되게 한다
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        run_split(data_root, split, cache_dir, args.workers, args.limit)
    print("전부 완료", flush=True)


if __name__ == "__main__":
    sys.exit(main())
