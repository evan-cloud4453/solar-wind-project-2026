#!/usr/bin/env python
"""SWEEP 노트북 생성 — 밤새 돌려서 아침에 순위표를 받는다.

P3(Public 58.8028) 의 파이프라인을 **설정 하나짜리 노트북**에서
**설정 공간 전체를 훑는 탐색기**로 바꾼다. 바뀐 것은 세 가지다.

  1. 데이터가 GPU 에 상주한다. DataLoader 를 없애 1회 학습이 분 -> 초가 된다.
     (USE_CNN=False 인 P3 계열은 입력이 수십 MB 뿐이다)
  2. 코로나홀 피처를 **임계 5개 x 격자 8개 x 레벨 2개** 를 한 번에 뽑아 캐시한다.
     추출 1회에 모든 조합이 열린다.
  3. 판정은 P18 의 **사슬 grouped CV** + official validation **두 개를 나란히** 본다.
     하나만 믿지 않는다 (CODE_REPORT §1.5 의 교훈).

사용:
    python build_sweep_notebook.py              # -> code_sweep.ipynb
    python build_sweep_notebook.py --smoke      # 가짜 데이터로 엔진 자체 점검
"""

import argparse
import json
import sys
from pathlib import Path

OUTPUT = Path("code_sweep.ipynb")


def markdown_cell(text):
    return {"cell_type": "markdown", "metadata": {},
            "source": text.strip("\n").splitlines(keepends=True)}


def code_cell(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.strip("\n").splitlines(keepends=True)}


# =====================================================================
TITLE_MD = """
# 태양풍 속도 예측 — SWEEP (밤샘 탐색기)

**이 노트북은 제출물이 아니다.** 6시간 동안 설정 공간을 훑고, 아침에 읽을
순위표와 제출 후보 폴더를 남긴다. 제출용 노트북은 아침에
`python build_final_notebook.py work/sweep/final/<폴더>/config.json` 으로 만든다.

| 단계 | 하는 일 | 판정 |
|---|---|---|
| 0 | 데이터·이미지 캐시·코로나홀 피처 **일괄 추출** (임계 5 x 격자 8 x 레벨 2) | — |
| A | 한 번에 한 노브만 바꾼 사다리 + 무작위 탐색 | 사슬 CV (폴드 3 x 시드 1) |
| B | 상위 후보 정밀 재측정 | 사슬 CV (폴드 5 x 시드 2) **+ official val** |
| C | 최종 후보 전체 학습 x 시드 앙상블 -> `submission.csv` 생성 | val 1회 읽기 |
| D | `REPORT.md` · `results.csv` · 순위표 출력 | — |

## 아침에 볼 것

```
work/sweep/REPORT.md            <- 순위표 · 노브별 효과 · 다음 행동
work/sweep/results.csv          <- 평가한 모든 설정 (중간에 죽어도 남는다)
work/sweep/sweep.log            <- 브라우저가 끊겨도 남는 전체 로그
work/sweep/final/<rank>_<tag>/  <- 제출 후보: submission.csv · model.pth · config.json
```

## 먼저 읽을 경고

CODE_REPORT §1.5 가 실측한 대로 **로컬 지표의 분해능이 우리가 구분해야 할
차이보다 크다.** val 이 1위로 뽑은 P6 은 Public 4위였고, CV 는 P9-P10 의
2.56 km/s 차이를 0.013 으로, 부호까지 반대로 읽었다. 그러므로

* 이 노트북은 **1등 하나를 고르지 않는다.** CV 순위 · val 순위 · 두 지표가
  동시에 이기는지를 따로 찍는다. 아침에 고를 때 "둘 다 이긴 것" 을 먼저 본다.
* **차이가 표준오차 2배 안이면 같은 것으로 읽는다.** 순위표에 `2se` 열을 같이 찍는다.
* 시드 앙상블(같은 설정 시드 여러 개 평균)은 **탐색으로 얻은 이득이 아니라 분산
  감소**라 상대적으로 믿을 만하다. 그래서 C 단계는 모든 후보를 시드 앙상블로 만든다.
* P6 의 실패를 반복하지 않기 위해 **앙상블 가중치를 val 로 적합하지 않는다.** 균등 평균만 쓴다.

## 규정 점검

* validation 은 **학습에 쓰지 않는다.** 폴드는 전부 train 안에서만 잘린다.
* 외부 데이터·pretrained weight 없음. 이미지와 wind 만 쓴다.
* 제출 3종(`code.ipynb` · `model.pth` · `submission.csv`)은 C 단계 산출 폴더에서 만든다.
"""

# =====================================================================
KNOB_CELL = """
# ===== 실행 노브 — 여기만 보고 시작하면 된다 ==========================
BUDGET_HOURS = 6.0        # 전체 예산. 이 시간이 지나면 어느 단계든 멈추고 보고서를 쓴다
RESUME = True             # results.csv 가 있으면 이미 평가한 설정은 건너뛴다

# --- 단계별 계측 강도 -------------------------------------------------
A_FOLDS, A_SEEDS = 3, 1   # 훑기: 설정당 3회 학습
B_FOLDS, B_SEEDS = 5, 2   # 정밀: 설정당 10회 학습 + val 2회
B_TOP_K = 10              # A 상위 몇 개를 B 로 올릴지의 **목표치**.
                          # 예산이 모자라면 줄고, 남으면 최대 B_MAX_K 까지 늘린다
B_MAX_K = 40              # B 로 올릴 수 있는 최대 설정 수 (남는 시간을 놀리지 않기 위한 상한)
C_TOP_K = 4               # B 상위 몇 개를 최종 제출 후보로 만들지
C_SEEDS = 5               # 최종 후보 하나당 시드 몇 개를 앙상블할지
FOLD_SEED = 0             # 사슬 -> 폴드 배정 난수 (모든 설정이 같은 폴드를 쓴다)
SEARCH_SEED = 20260821    # 무작위 탐색 난수
EPOCH_SMOOTH = 3          # CV 곡선 이동평균. argmin 이 단발 노이즈를 집는 것 방지

# --- 탐색 범위 --------------------------------------------------------
RUN_OAT = True            # A-1: 한 번에 한 노브만 바꾸는 사다리 (해석 가능한 표가 나온다)
RUN_RANDOM = True         # A-2: 남은 예산 전부를 무작위 조합에 쓴다
MAX_RANDOM_CONFIGS = 400  # 안전 상한. 보통 예산이 먼저 끝난다

# --- 경로 ------------------------------------------------------------
IMAGE_SIZE = 128          # P3 캐시(work/cache/128px)를 그대로 재사용하려면 128 을 유지
SWEEP_DIR = "work/sweep"

# --- 안전장치 --------------------------------------------------------
# 6시간이 지나도 C 단계(제출물 생성)는 반드시 돌도록 예산에서 미리 떼어 둔다.
RESERVE_MINUTES = 8       # 보고서·그림·저장에 남겨 둘 시간
A_BUDGET_FRACTION = 0.62  # C 예약분을 뺀 시간 중 A(훑기)에 쓸 비율. 나머지가 B(정밀)
"""

# =====================================================================
ENGINE_MD = """
## 1. 엔진

정의만 있고 부작용이 없다. `build_final_notebook.py` 가 **이 셀을 글자 그대로**
제출 노트북에 옮겨 붙이므로, 여기서 학습한 모델은 제출 노트북에서 그대로 재현된다.
"""

ENGINE_SRC = r'''
"""SWEEP 엔진 — 데이터 · 코로나홀 피처 · 모델 · 학습 · 평가.

부작용 없는 정의만 둔다. 이 셀은 제출 노트북에도 그대로 들어간다.
"""
from pathlib import Path
import gc
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

AU_KM = 1.496e8
IMAGE_COLUMNS = [f"image_{i:02d}" for i in range(20)]
WIND_COLUMNS = [f"wind_{i:02d}" for i in range(20)]
TARGET_COLUMNS = [f"target_{i:02d}" for i in range(12)]
HORIZONS = np.arange(1, 13) * 6
STAT_NAMES = ["last", "mean4", "mean", "std", "min", "max", "slope",
              "last_minus_mean4", "range"]
NUM_STATS = len(STAT_NAMES)
CHANNELS = ("193", "211")
SPLITS = ("train", "validation", "test")

# 코로나홀 추출을 한 번에 끝내기 위한 사양. 탐색 공간의 모든 조합이 여기 들어 있어야 한다.
CH_SPEC = {
    "thresholds": [0.35, 0.40, 0.45, 0.50, 0.55],
    "grids": [[1, 1], [2, 3], [3, 3], [3, 5], [4, 3], [4, 5], [6, 3], [6, 5]],
    "bright_ratio": 1.6,
    "disk_margin": 0.95,
    "version": "sweep1",
}

# ---------------------------------------------------------------- 로깅
_LOG_PATH = None


def set_log(path):
    global _LOG_PATH
    _LOG_PATH = Path(path)
    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def log(*parts):
    text = " ".join(str(p) for p in parts)
    print(text, flush=True)
    if _LOG_PATH is not None:
        try:
            with open(_LOG_PATH, "a", encoding="utf-8") as handle:
                handle.write(text + "\n")
        except OSError:
            pass


def hms(seconds):
    seconds = max(0.0, float(seconds))
    return f"{int(seconds // 3600)}h{int(seconds % 3600 // 60):02d}m{int(seconds % 60):02d}s"


class Budget:
    """전체 예산과 단계 예산. 남은 시간을 보고 다음 학습을 시작할지 정한다."""

    def __init__(self, hours, reserve_minutes=8.0):
        self.started = time.time()
        self.end = self.started + hours * 3600.0
        self.reserve = reserve_minutes * 60.0
        self.phase_end = self.end

    def elapsed(self):
        return time.time() - self.started

    def left(self):
        return self.end - time.time() - self.reserve

    def open_phase(self, seconds):
        self.phase_end = min(self.end - self.reserve, time.time() + max(0.0, seconds))
        return self.phase_end

    def phase_left(self):
        return self.phase_end - time.time()

    def affords(self, seconds):
        return self.phase_left() > seconds and self.left() > seconds


# ------------------------------------------------------------ 데이터 로드
def find_data_root(extra=None):
    candidates = []
    if extra:
        candidates.append(Path(extra))
    if os.getenv("SW_DATA_ROOT"):
        candidates.append(Path(os.getenv("SW_DATA_ROOT")))
    candidates += [
        Path("public_dataset/competition_dataset_6h"),
        Path("/home/jovyan/public_dataset/competition_dataset_6h"),
        Path("public/public_dataset/competition_dataset_6h"),
        Path("/home/jovyan/public/public_dataset/competition_dataset_6h"),
        Path("dataset"),
        Path("/home/jovyan/dataset"),
    ]
    for candidate in candidates:
        if (candidate / "train/inputs.csv").exists():
            return candidate
    raise FileNotFoundError("데이터 경로를 찾지 못했습니다:\n"
                            + "\n".join(f"  - {c}" for c in candidates))


def forward_fill_rows(values):
    valid = np.isfinite(values)
    positions = np.where(valid, np.arange(values.shape[1])[None, :], 0)
    np.maximum.accumulate(positions, axis=1, out=positions)
    rows = np.arange(values.shape[0])[:, None]
    return np.where(valid.any(axis=1, keepdims=True), values[rows, positions], values)


def fill_wind(frame, fallback):
    values = frame[WIND_COLUMNS].to_numpy(np.float32)
    valid = np.isfinite(values).astype(np.float32)
    filled = forward_fill_rows(values)
    filled = forward_fill_rows(filled[:, ::-1])[:, ::-1]
    filled = np.where(np.isfinite(filled), filled, fallback)
    return np.ascontiguousarray(filled), np.ascontiguousarray(valid)


def load_tables(data_root):
    """P3 와 같은 규약으로 세 split 을 읽는다."""
    tables = {}
    inputs = {
        "train": pd.read_csv(data_root / "train/inputs.csv"),
        "validation": pd.read_csv(data_root / "validation/inputs.csv"),
        "test": pd.read_csv(data_root / "test/inputs.csv"),
    }
    train_targets = pd.read_csv(data_root / "train/targets.csv")
    val_targets = pd.read_csv(data_root / "validation/targets.csv")
    test_ids = pd.read_csv(data_root / "test/test_ids.csv")

    assert inputs["train"].sample_id.tolist() == train_targets.sample_id.tolist()
    assert inputs["validation"].sample_id.tolist() == val_targets.sample_id.tolist()
    assert inputs["test"].sample_id.tolist() == test_ids.sample_id.tolist()
    assert set(inputs["train"].sample_id).isdisjoint(inputs["validation"].sample_id)
    assert set(inputs["train"].sample_id).isdisjoint(inputs["test"].sample_id)
    assert not any(c.startswith("target_") for c in inputs["test"].columns)

    fallback = float(np.nanmedian(inputs["train"][WIND_COLUMNS].to_numpy(np.float32)))
    targets = {"train": train_targets[TARGET_COLUMNS].to_numpy(np.float32),
               "validation": val_targets[TARGET_COLUMNS].to_numpy(np.float32),
               "test": None}
    for split in SPLITS:
        wind, wind_valid = fill_wind(inputs[split], fallback)
        tables[split] = {"inputs": inputs[split], "wind": wind,
                         "wind_valid": wind_valid, "targets": targets[split]}
    tables["test_ids"] = test_ids
    return tables


def prepare_image_memmap(split, inputs, data_root, cache_root, image_size):
    """P3 와 완전히 같은 캐시 규약. 기존 work/cache/128px 를 그대로 재사용한다."""
    image_root = data_root / split
    folder = Path(cache_root) / f"{image_size}px"
    folder.mkdir(parents=True, exist_ok=True)
    array_path = folder / f"{split}_images.npy"
    metadata_path = folder / f"{split}_metadata.json"
    filenames = sorted(pd.unique(inputs[IMAGE_COLUMNS].to_numpy().ravel()).tolist())
    expected = {"image_size": image_size, "channels": list(CHANNELS), "filenames": filenames}
    shape = (len(filenames), len(CHANNELS), image_size, image_size)

    valid = False
    if array_path.exists() and metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            cached = np.load(array_path, mmap_mode="r")
            valid = (metadata == expected and cached.shape == shape
                     and cached.dtype == np.uint8)
        except (OSError, ValueError, json.JSONDecodeError):
            valid = False

    if not valid:
        array_temp = array_path.with_name(array_path.name + f".partial.{os.getpid()}")
        metadata_temp = metadata_path.with_name(metadata_path.name + f".partial.{os.getpid()}")
        resized = np.lib.format.open_memmap(array_temp, mode="w+", dtype=np.uint8, shape=shape)
        for index, filename in enumerate(filenames):
            for channel_index, channel in enumerate(CHANNELS):
                with Image.open(image_root / channel / filename) as image:
                    resized[index, channel_index] = np.asarray(
                        image.convert("L").resize((image_size, image_size),
                                                  Image.Resampling.BILINEAR), dtype=np.uint8)
            if (index + 1) % 2000 == 0 or index + 1 == len(filenames):
                log(f"  {split} resize {index + 1}/{len(filenames)}")
        resized.flush()
        del resized
        metadata_temp.write_text(json.dumps(expected, ensure_ascii=False) + "\n", encoding="utf-8")
        array_temp.replace(array_path)
        metadata_temp.replace(metadata_path)
        log(f"  이미지 캐시 생성: {array_path}")
    else:
        log(f"  이미지 캐시 재사용: {array_path}")

    array = np.load(array_path, mmap_mode="r")
    return array, {name: i for i, name in enumerate(filenames)}


def image_index_matrix(inputs, index):
    return np.asarray([[index[name] for name in row]
                       for row in inputs[IMAGE_COLUMNS].itertuples(index=False, name=None)],
                      dtype=np.int64)


# ------------------------------------------------- 원반 검출 · 코로나홀 추출
def detect_disk(image_array, image_size, sample_count=400):
    indexes = np.unique(np.linspace(0, len(image_array) - 1, sample_count).astype(int))
    mean_image = np.asarray(image_array[indexes], dtype=np.float64).mean(axis=(0, 1))
    mask = mean_image > mean_image.max() * 0.15
    ys, xs = np.nonzero(mask)
    return float(ys.mean()), float(xs.mean()), float(np.sqrt(mask.sum() / np.pi)), mean_image


def build_grid_binner(grid, disk_mask, disk_y, disk_x, effective_r, image_size):
    """P3 셀 8 과 같은 규칙 — 원반 외접 사각형의 균등 분할."""
    n_lat, n_lon = int(grid[0]), int(grid[1])
    grid_y, grid_x = np.mgrid[0:image_size, 0:image_size].astype(np.float64)
    lat = np.clip((grid_y - (disk_y - effective_r)) / (2 * effective_r) * n_lat, 0, n_lat - 1e-6)
    lon = np.clip((grid_x - (disk_x - effective_r)) / (2 * effective_r) * n_lon, 0, n_lon - 1e-6)
    cell = (lat.astype(np.int64) * n_lon + lon.astype(np.int64))[disk_mask]
    n_cells = n_lat * n_lon
    onehot = np.zeros((cell.size, n_cells), np.float32)
    onehot[np.arange(cell.size), cell] = 1.0
    counts = np.maximum(np.bincount(cell, minlength=n_cells).astype(np.float32), 1.0)
    return onehot, counts


def dark_key(threshold, grid):
    return f"dark_{float(threshold):.2f}_{int(grid[0])}x{int(grid[1])}"


def bright_key(grid):
    return f"bright_{int(grid[0])}x{int(grid[1])}"


def moment_key(threshold):
    return f"moment_{float(threshold):.2f}"


def spec_hash(spec, image_size):
    payload = json.dumps({"spec": spec, "image_size": image_size}, sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]


def extract_ch_store(split, image_array, spec, disk, cache_root, image_size, chunk=192):
    """임계 x 격자 x 레벨을 **한 번의 패스**로 전부 뽑는다. 결과는 npz 캐시."""
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    path = cache_root / f"sweepch_{split}_{spec_hash(spec, image_size)}_{image_size}.npz"
    n_frames = len(image_array)
    if path.exists():
        try:
            with np.load(path) as handle:
                store = {k: handle[k] for k in handle.files}
            if all(v.shape[0] == n_frames for v in store.values()):
                log(f"  CH 캐시 재사용: {path.name}  (배열 {len(store)}개)")
                return store
        except (OSError, ValueError):
            pass

    disk_y, disk_x, disk_r = disk
    effective_r = disk_r * spec["disk_margin"]
    grid_y, grid_x = np.mgrid[0:image_size, 0:image_size].astype(np.float64)
    radius_map = np.sqrt((grid_y - disk_y) ** 2 + (grid_x - disk_x) ** 2)
    disk_mask = radius_map <= effective_r
    n_pixels = int(disk_mask.sum())

    binners = {tuple(g): build_grid_binner(g, disk_mask, disk_y, disk_x, effective_r, image_size)
               for g in spec["grids"]}
    # 무게중심·퍼짐 계산용 좌표 (반지름으로 정규화, +x = 서쪽 림 방향)
    coord_x = ((grid_x - disk_x) / effective_r)[disk_mask].astype(np.float32)
    coord_y = ((grid_y - disk_y) / effective_r)[disk_mask].astype(np.float32)
    coord_r2 = (coord_x ** 2 + coord_y ** 2).astype(np.float32)

    store = {}
    for grid in binners:
        store[bright_key(grid)] = np.zeros((n_frames, grid[0] * grid[1]), np.float32)
    for threshold in spec["thresholds"]:
        for grid in binners:
            store[dark_key(threshold, grid)] = np.zeros((n_frames, grid[0] * grid[1]), np.float32)
        store[moment_key(threshold)] = np.zeros((n_frames, 4), np.float32)

    started = time.perf_counter()
    for start in range(0, n_frames, chunk):
        block = np.asarray(image_array[start:start + chunk], dtype=np.float32)
        stop = start + block.shape[0]
        on_disk = block[:, :, disk_mask]                              # (b, 2, npix)
        median = np.median(on_disk, axis=2, keepdims=True)
        relative = on_disk / np.maximum(median, 1e-6)                 # 원반 중앙값 대비 밝기

        bright = np.logical_and(relative[:, 0] > spec["bright_ratio"],
                                relative[:, 1] > spec["bright_ratio"]).astype(np.float32)
        for grid, (onehot, counts) in binners.items():
            store[bright_key(grid)][start:stop] = (bright @ onehot) / counts

        for threshold in spec["thresholds"]:
            # P3 규칙: 193 과 211 두 채널 모두에서 중앙값의 threshold 배보다 어두운 픽셀
            dark = np.logical_and(relative[:, 0] < threshold,
                                  relative[:, 1] < threshold).astype(np.float32)
            for grid, (onehot, counts) in binners.items():
                store[dark_key(threshold, grid)][start:stop] = (dark @ onehot) / counts
            mass = dark.sum(axis=1)
            safe = np.maximum(mass, 1e-6)
            center_x = (dark @ coord_x) / safe
            center_y = (dark @ coord_y) / safe
            second = (dark @ coord_r2) / safe
            spread = np.sqrt(np.maximum(second - center_x ** 2 - center_y ** 2, 0.0))
            depth = 1.0 - (dark * relative[:, 0]).sum(axis=1) / safe
            present = (mass > 0).astype(np.float32)
            store[moment_key(threshold)][start:stop] = np.stack(
                [center_x * present, center_y * present, spread * present, depth * present],
                axis=1).astype(np.float32)

        if (stop % (chunk * 20) == 0) or stop == n_frames:
            rate = stop / max(time.perf_counter() - started, 1e-6)
            log(f"  {split} CH 추출 {stop}/{n_frames}  ({rate:.0f} frame/s)")

    np.savez_compressed(path, **store)
    log(f"  CH 캐시 생성: {path.name}  (배열 {len(store)}개, "
        f"{path.stat().st_size / 1024 ** 2:.1f} MiB, {hms(time.perf_counter() - started)})")
    return store


# ------------------------------------------------------------ 사슬 · 폴드
def reconstruct_frame_chains(inputs):
    """프레임 후행관계로 연속 관측 구간을 복원한다 (P18 과 같은 규약)."""
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
            visited.add(node)
            chain.append(node)
            node = successor.get(node)
        chains.append(chain)
    assert conflicts == 0 and not (names - visited), "사슬 복원 실패"
    return chains


def window_positions(inputs, chains):
    position = {name: (c, o) for c, chain in enumerate(chains) for o, name in enumerate(chain)}
    first = inputs[IMAGE_COLUMNS[0]].to_numpy()
    chain_id = np.array([position[n][0] for n in first], np.int64)
    t_start = np.array([position[n][1] for n in first], np.int64)
    return chain_id, t_start


def chain_folds(chain_id, n_chains, n_folds, n_repeats, seed):
    """사슬을 통째로 폴드에 배정한다 -> 윈도우 중첩 누수가 구조적으로 0."""
    counts = np.bincount(chain_id, minlength=n_chains)
    rows = np.arange(len(chain_id))
    generator = np.random.default_rng(seed)
    folds = []
    for repeat in range(n_repeats):
        assignment = np.zeros(len(counts), np.int64)
        load = np.zeros(n_folds, np.int64)
        for chain in generator.permutation(len(counts)):
            fold = int(np.argmin(load))
            assignment[chain] = fold
            load[fold] += counts[chain]
        fold_of_row = assignment[chain_id]
        for fold in range(n_folds):
            held = fold_of_row == fold
            if not held.any() or held.all():
                continue
            folds.append({"repeat": repeat, "fold": fold,
                          "train": rows[~held], "evaluate": rows[held]})
    return folds


# ---------------------------------------------------------- 설정 -> 피처
def as_grid(cfg):
    return (int(cfg["ch_grid"][0]), int(cfg["ch_grid"][1]))


def feature_grid(cfg):
    """area 계열은 격자를 쓰지 않는다."""
    return (1, 1) if cfg["ch_mode"] in ("area", "area_shape") else as_grid(cfg)


def frame_features(cfg, store):
    grid = feature_grid(cfg)
    threshold = cfg["ch_threshold"]
    parts = [store[dark_key(threshold, grid)]]
    if cfg["ch_mode"] in ("area_shape", "grid_shape"):
        parts.append(store[moment_key(threshold)])
    if cfg["use_bright"]:
        parts.append(store[bright_key(grid)])
    return np.concatenate(parts, axis=1).astype(np.float32)


def ballistic_source(cfg, store):
    """탄도 정렬이 읽는 시계열 — 중앙자오선 셀 1개 또는 전 셀."""
    grid = feature_grid(cfg)
    matrix = store[dark_key(cfg["ch_threshold"], grid)]
    if cfg["ballistic_lon"] == "all":
        return matrix
    n_lat, n_lon = grid
    central = (n_lat // 2) * n_lon + (n_lon // 2)
    return matrix[:, central:central + 1]


def fixed_ballistic_index(speeds):
    table = np.zeros((12, len(speeds)), np.float32)
    for horizon in range(12):
        lead = (horizon + 1) * 6.0
        for index, speed in enumerate(speeds):
            table[horizon, index] = 19.0 + (lead - AU_KM / speed / 3600.0) / 6.0
    return table


def adaptive_ballistic_index(wind, tau_reference_hours, mean_speed):
    """[P9 R2] 관측된 최근 4스텝 평균 속도로 샘플별 전달시간을 만든다."""
    scale = tau_reference_hours / (AU_KM / max(mean_speed, 1.0) / 3600.0)
    speed = np.maximum(wind[:, -4:].mean(axis=1), 150.0)
    tau = scale * AU_KM / speed / 3600.0                       # (N,)
    lead = (np.arange(12, dtype=np.float32) + 1) * 6.0
    return (19.0 + (lead[None, :] - tau[:, None]) / 6.0).astype(np.float32)


def interpolate_fixed(sequence, index_vector):
    lower = np.floor(np.clip(index_vector, 0, 19)).astype(np.int64)
    upper = np.minimum(lower + 1, 19)
    weight = (np.clip(index_vector, 0, 19) - lower).astype(np.float32)
    return (sequence[:, lower] * (1.0 - weight)[None, :, None]
            + sequence[:, upper] * weight[None, :, None]).astype(np.float32)


def interpolate_adaptive(sequence, index_matrix):
    clipped = np.clip(index_matrix, 0, 19)
    lower = np.floor(clipped).astype(np.int64)
    upper = np.minimum(lower + 1, 19)
    weight = (clipped - lower).astype(np.float32)[:, :, None]
    depth = sequence.shape[2]
    low = np.take_along_axis(sequence, np.repeat(lower[:, :, None], depth, axis=2), axis=1)
    high = np.take_along_axis(sequence, np.repeat(upper[:, :, None], depth, axis=2), axis=1)
    return (low * (1.0 - weight) + high * weight).astype(np.float32)


def build_ballistic(cfg, sequence, wind, mean_speed):
    """반환 (N, 12, B). B=0 이면 탄도 피처를 끈 것."""
    count = sequence.shape[0]
    if cfg["ballistic"] == "off":
        return np.zeros((count, 12, 0), np.float32)
    if cfg["tau_mode"] == "fixed":
        table = fixed_ballistic_index(cfg["transit_speeds"])         # (12, S)
        parts = [interpolate_fixed(sequence, table[:, s]) for s in range(table.shape[1])]
        return np.concatenate(parts, axis=2)
    raw = adaptive_ballistic_index(wind, cfg["tau_reference_hours"], mean_speed)
    features = [interpolate_adaptive(sequence, raw)]
    if cfg["clip_flags"]:
        # [P9 R2] 적응형에서는 같은 horizon 이라도 샘플마다 창 이탈 여부가 달라 정보가 된다
        features.append(np.stack([(raw > 19.0), (raw < 0.0)], axis=2).astype(np.float32))
    return np.concatenate(features, axis=2)


def build_wind_stats(wind):
    centered = np.arange(20, dtype=np.float32) - 9.5
    denominator = float((centered ** 2).sum())
    last = wind[:, -1]
    mean4 = wind[:, -4:].mean(axis=1)
    slope = (wind - wind.mean(axis=1, keepdims=True)) @ centered / denominator
    return np.stack([last, mean4, wind.mean(axis=1), wind.std(axis=1), wind.min(axis=1),
                     wind.max(axis=1), slope, last - mean4,
                     wind.max(axis=1) - wind.min(axis=1)], axis=1).astype(np.float32)


def build_pack(cfg, split, tables, stores, index_matrices, device, mean_speed):
    """설정 하나에 대한 GPU 상주 텐서 묶음."""
    table = tables[split]
    wind = table["wind"]
    frames = frame_features(cfg, stores[split])
    matrix = index_matrices[split]
    ch = frames[matrix] if cfg["use_ch"] else np.zeros((len(matrix), 20, 0), np.float32)
    source = ballistic_source(cfg, stores[split])[matrix]
    ballistic = build_ballistic(cfg, source, wind, mean_speed)

    to_gpu = lambda a: torch.as_tensor(np.ascontiguousarray(a), device=device)
    pack = {
        "ch": to_gpu(ch.astype(np.float32)),
        "bal": to_gpu(ballistic),
        "wind": to_gpu(wind.astype(np.float32)),
        "wdiff": to_gpu(np.diff(wind, axis=1, prepend=wind[:, :1]).astype(np.float32)),
        "wvalid": to_gpu(table["wind_valid"].astype(np.float32)),
        "wstats": to_gpu(build_wind_stats(wind)),
        "last": to_gpu(wind[:, -1].astype(np.float32)),
        "targets": (to_gpu(table["targets"].astype(np.float32))
                    if table["targets"] is not None else None),
        "n": len(matrix),
        "ch_dim": ch.shape[2],
        "bal_dim": ballistic.shape[2],
        "device": device,
    }
    return pack


# ------------------------------------------------------------------ 통계
def fit_stats(cfg, pack, rows):
    """정규화 통계는 **그 폴드의 학습행에서만** 뽑는다."""
    index = torch.as_tensor(np.asarray(rows), device=pack["device"], dtype=torch.long)
    stats = {}
    if pack["ch_dim"] > 0:
        chunk = pack["ch"].index_select(0, index)
        stats["ch_mean"] = chunk.mean(dim=(0, 1))
        stats["ch_std"] = chunk.std(dim=(0, 1)) + 1e-6
        del chunk
    else:
        stats["ch_mean"] = torch.zeros(0, device=pack["device"])
        stats["ch_std"] = torch.ones(0, device=pack["device"])
    if pack["bal_dim"] > 0:
        chunk = pack["bal"].index_select(0, index)
        stats["bal_mean"] = chunk.mean()
        stats["bal_std"] = chunk.std() + 1e-8
        del chunk
    else:
        stats["bal_mean"] = torch.zeros((), device=pack["device"])
        stats["bal_std"] = torch.ones((), device=pack["device"])

    wind = pack["wind"].index_select(0, index)
    stats["wind_mean"] = wind.mean()
    stats["wind_std"] = wind.std() + 1e-6
    stats["diff_std"] = pack["wdiff"].index_select(0, index).std() + 1e-6
    wind_stats = pack["wstats"].index_select(0, index)
    stats["stat_mean"] = wind_stats.mean(dim=0)
    stats["stat_std"] = wind_stats.std(dim=0) + 1e-6

    targets = pack["targets"].index_select(0, index)
    last = pack["last"].index_select(0, index).unsqueeze(1)
    residual = targets - last
    stats["res_mean"] = residual.mean(dim=0)
    stats["res_std"] = residual.std(dim=0) + 1e-6
    stats["tgt_mean"] = targets.mean(dim=0)
    stats["tgt_std"] = targets.std(dim=0) + 1e-6
    stats["clip_low"] = float(targets.min().item() * 0.95)
    stats["clip_high"] = float(targets.max().item() * 1.05)
    return stats


def stats_to_cpu(stats):
    return {k: (v.detach().cpu().numpy().tolist() if torch.is_tensor(v) else v)
            for k, v in stats.items()}


def stats_from_cpu(payload, device):
    stats = {}
    for key, value in payload.items():
        if key in ("clip_low", "clip_high"):
            stats[key] = float(value)
        else:
            stats[key] = torch.as_tensor(np.asarray(value, dtype=np.float32), device=device)
    return stats


# ------------------------------------------------------------------ 모델
class SweepNet(nn.Module):
    """P3 구조의 일반화 — 노브로 폭·깊이·브랜치를 바꾼다.

    head 는 12 horizon 에 **가중치를 공유**하고 horizon embedding 과 그 horizon 의
    탄도 피처로만 구분한다 (P3 와 동일). 파라미터가 12배 적어 정규화 효과가 크다.
    """

    def __init__(self, cfg, ch_dim, bal_dim):
        super().__init__()
        self.cfg = cfg
        self.ch_dim = ch_dim
        self.bal_dim = bal_dim
        shared = 0

        self.wind_gru = nn.GRU(3, cfg["wind_hidden"], num_layers=2, batch_first=True)
        self.stats_encoder = nn.Sequential(
            nn.Linear(NUM_STATS, 128), nn.SELU(inplace=True),
            nn.Linear(128, 64), nn.SELU(inplace=True))
        shared += cfg["wind_hidden"] + 64

        self.use_ch = ch_dim > 0
        if self.use_ch:
            self.ch_gru = nn.GRU(ch_dim, cfg["gru_hidden"], num_layers=cfg["gru_layers"],
                                 batch_first=True, bidirectional=bool(cfg["gru_bidir"]))
            drop = cfg["ch_dropout"] if cfg["ch_dropout"] is not None else cfg["dropout"]
            self.ch_dropout = nn.Dropout(drop)
            shared += cfg["gru_hidden"] * (2 if cfg["gru_bidir"] else 1)

        self.horizon_embedding = nn.Parameter(torch.randn(12, cfg["horizon_embed"]) * 0.1)
        width = shared + cfg["horizon_embed"] + bal_dim
        layers = []
        for hidden in cfg["head_width"]:
            layers += [nn.Linear(width, hidden), nn.ReLU(inplace=True), nn.Dropout(cfg["dropout"])]
            width = hidden
        layers.append(nn.Linear(width, 1))
        self.head = nn.Sequential(*layers)
        self.shared_dim = shared

    def forward(self, ch, ballistic, wind_seq, wind_stats):
        _, wind_hidden = self.wind_gru(wind_seq)
        parts = [F.relu(wind_hidden[-1]), self.stats_encoder(wind_stats)]
        if self.use_ch:
            _, ch_hidden = self.ch_gru(ch)
            if self.cfg["gru_bidir"]:
                tail = torch.cat([ch_hidden[-2], ch_hidden[-1]], dim=1)
            else:
                tail = ch_hidden[-1]
            parts.append(self.ch_dropout(F.relu(tail)))
        shared = torch.cat(parts, dim=1)
        count = shared.shape[0]
        expanded = shared.unsqueeze(1).expand(count, 12, shared.shape[1])
        embedding = self.horizon_embedding.unsqueeze(0).expand(count, 12, -1)
        head_parts = [expanded, embedding]
        if self.bal_dim > 0:
            head_parts.append(ballistic)
        return self.head(torch.cat(head_parts, dim=2)).squeeze(-1)


class Ema:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items()}
        self.backup = None

    def update(self, model):
        for key, value in model.state_dict().items():
            shadow = self.shadow[key]
            if value.dtype.is_floating_point:
                shadow.mul_(self.decay).add_(value.detach().float(), alpha=1.0 - self.decay)
            else:
                shadow.copy_(value)

    def apply(self, model):
        self.backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict({k: v.to(self.backup[k].dtype) for k, v in self.shadow.items()})

    def restore(self, model):
        if self.backup is not None:
            model.load_state_dict(self.backup)
            self.backup = None


# ------------------------------------------------------------ 배치 · 손실
def make_inputs(cfg, pack, stats, index, training):
    ch = pack["ch"].index_select(0, index)
    if pack["ch_dim"] > 0:
        ch = (ch - stats["ch_mean"]) / stats["ch_std"]
    ballistic = pack["bal"].index_select(0, index)
    if pack["bal_dim"] > 0:
        ballistic = (ballistic - stats["bal_mean"]) / stats["bal_std"]
    wind = pack["wind"].index_select(0, index)
    wind_seq = torch.stack([(wind - stats["wind_mean"]) / stats["wind_std"],
                            pack["wdiff"].index_select(0, index) / stats["diff_std"],
                            pack["wvalid"].index_select(0, index)], dim=2)
    wind_stats = (pack["wstats"].index_select(0, index) - stats["stat_mean"]) / stats["stat_std"]
    if training and cfg["aug_ch_noise"] > 0:
        if pack["ch_dim"] > 0:
            ch = ch * (1.0 + torch.randn_like(ch) * cfg["aug_ch_noise"])
        if pack["bal_dim"] > 0:
            ballistic = ballistic * (1.0 + torch.randn_like(ballistic) * cfg["aug_ch_noise"])
    return ch, ballistic, wind_seq, wind_stats


def to_prediction(cfg, raw, pack, stats, index, clamp):
    if cfg["target"] == "residual":
        prediction = raw * stats["res_std"] + stats["res_mean"] + pack["last"].index_select(0, index).unsqueeze(1)
    else:
        prediction = raw * stats["tgt_std"] + stats["tgt_mean"]
    if clamp:
        prediction = prediction.clamp(stats["clip_low"], stats["clip_high"])
    return prediction


def horizon_weights(cfg, device):
    if cfg["loss_horizon_weight"] == "long":
        # 42h 이상 7개 horizon 이 전체 평균을 지배한다 (CODE_REPORT §P7)
        weight = torch.tensor([0.6] * 6 + [1.4] * 6, device=device)
    else:
        weight = torch.ones(12, device=device)
    return weight / weight.mean()


def make_loss(cfg, device):
    weight = horizon_weights(cfg, device)
    mode = cfg["loss"]

    def loss_fn(prediction, target):
        error = (prediction - target) / 100.0
        if mode == "metric":                      # 공식 지표 그대로 (P3)
            return (torch.sqrt((error ** 2).mean(dim=0) + 1e-8) * weight).mean()
        if mode == "mse":
            return ((error ** 2).mean(dim=0) * weight).mean()
        return (F.smooth_l1_loss(prediction / 100.0, target / 100.0, beta=0.5,
                                 reduction="none").mean(dim=0) * weight).mean()

    return loss_fn


@torch.no_grad()
def predict_rows(cfg, model, pack, stats, rows, batch=4096):
    model.eval()
    index_all = torch.as_tensor(np.asarray(rows), device=pack["device"], dtype=torch.long)
    outputs = []
    for start in range(0, index_all.numel(), batch):
        index = index_all[start:start + batch]
        ch, ballistic, wind_seq, wind_stats = make_inputs(cfg, pack, stats, index, False)
        raw = model(ch, ballistic, wind_seq, wind_stats)
        outputs.append(to_prediction(cfg, raw, pack, stats, index, True).float().cpu().numpy())
    return np.concatenate(outputs).astype(np.float64)


def official_rmse(y_true, y_pred):
    per_horizon = np.sqrt(np.mean((y_pred - y_true) ** 2, axis=0))
    return float(per_horizon.mean()), per_horizon


def run_training(cfg, pack, train_rows, evaluate_rows, seed, epochs, t_start=None,
                 stats=None, record_curve=True, t_max=None):
    """1회 학습. GPU 상주 텐서를 직접 인덱싱한다 (DataLoader 없음).

    evaluate_rows=None 이면 곡선을 재지 않고 학습만 한다 (최종 전체 학습).
    t_max 를 CV 때와 같게 두면 epoch k 의 LR 이 CV 때와 같아져 고른 epoch 을 그대로 옮길 수 있다.
    """
    device = pack["device"]
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    if stats is None:
        stats = fit_stats(cfg, pack, train_rows)
    model = SweepNet(cfg, pack["ch_dim"], pack["bal_dim"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                                  weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(t_max or epochs, 1))
    loss_fn = make_loss(cfg, device)
    ema = Ema(model, cfg["ema"]) if cfg["ema"] > 0 else None

    train_index = torch.as_tensor(np.asarray(train_rows), device=device, dtype=torch.long)
    if cfg["stride"] > 0 and t_start is not None:
        offsets = torch.as_tensor(t_start[np.asarray(train_rows)], device=device, dtype=torch.long)
    else:
        offsets = None
    generator = torch.Generator(device=device.type)
    generator.manual_seed(seed)
    batch_size = cfg["batch_size"]
    curve = np.zeros((epochs, 12), np.float64) if record_curve else None
    evaluate_targets = None
    if evaluate_rows is not None:
        evaluate_targets = pack["targets"].index_select(
            0, torch.as_tensor(np.asarray(evaluate_rows), device=device, dtype=torch.long)
        ).cpu().numpy().astype(np.float64)

    for epoch in range(epochs):
        model.train()
        if offsets is not None:
            selected = train_index[(offsets - epoch) % cfg["stride"] == 0]
        else:
            selected = train_index
        if selected.numel() < 2:
            continue
        permutation = selected[torch.randperm(selected.numel(), device=device,
                                              generator=generator)]
        for start in range(0, permutation.numel(), batch_size):
            index = permutation[start:start + batch_size]
            if index.numel() < 2:
                continue
            ch, ballistic, wind_seq, wind_stats = make_inputs(cfg, pack, stats, index, True)
            raw = model(ch, ballistic, wind_seq, wind_stats)
            prediction = to_prediction(cfg, raw, pack, stats, index, False)
            loss = loss_fn(prediction, pack["targets"].index_select(0, index))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if ema is not None:
                ema.update(model)
        scheduler.step()
        if record_curve and evaluate_rows is not None:
            if ema is not None:
                ema.apply(model)
            prediction = predict_rows(cfg, model, pack, stats, evaluate_rows)
            curve[epoch] = official_rmse(evaluate_targets, prediction)[1]
            if ema is not None:
                ema.restore(model)

    if ema is not None:
        ema.apply(model)                      # 최종 가중치는 EMA 로 고정한다
    return model, stats, curve


def smooth_curve(curve, window=3):
    if window <= 1:
        return curve
    left = window // 2
    padded = np.pad(curve, ((left, window - 1 - left), (0, 0)), mode="edge")
    return np.stack([padded[i:i + window].mean(axis=0) for i in range(len(curve))])


def evaluate_config(cfg, pack, folds, seeds, epochs, t_start, smooth=3, on_run=None):
    """사슬 CV. 반환값의 per_run 은 대응비교(paired) 에 쓴다."""
    runs, order = [], []
    for entry in folds:
        for seed in seeds:
            started = time.perf_counter()
            _, _, curve = run_training(cfg, pack, entry["train"], entry["evaluate"],
                                       seed, epochs, t_start=t_start)
            runs.append(curve)
            order.append((entry["repeat"], entry["fold"], seed))
            if on_run is not None:
                on_run(time.perf_counter() - started)
            gc.collect()
            if pack["device"].type == "cuda":
                torch.cuda.empty_cache()
    per_run = np.stack([smooth_curve(c, smooth).mean(axis=1) for c in runs])   # (R, epochs)
    mean_curve = per_run.mean(axis=0)
    best_epoch = int(np.argmin(mean_curve)) + 1
    at_best = per_run[:, best_epoch - 1]
    return {
        "score": float(mean_curve.min()),
        "best_epoch": best_epoch,
        "se": float(at_best.std(ddof=1) / math.sqrt(len(at_best))) if len(at_best) > 1 else float("nan"),
        "n_runs": len(runs),
        "per_run": at_best,
        "curve": mean_curve,
        "order": order,
    }


# ------------------------------------------------------------ 설정 유틸
def config_hash(cfg):
    payload = json.dumps(cfg, sort_keys=True, default=list)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8]


def normalize_config(cfg):
    """JSON 왕복해도 같은 해시가 나오도록 타입을 고정한다."""
    fixed = dict(cfg)
    fixed["ch_grid"] = [int(fixed["ch_grid"][0]), int(fixed["ch_grid"][1])]
    fixed["transit_speeds"] = [float(v) for v in fixed["transit_speeds"]]
    fixed["head_width"] = [int(v) for v in fixed["head_width"]]
    fixed["ch_threshold"] = float(fixed["ch_threshold"])
    return fixed


def feature_signature(cfg):
    """이 값이 같으면 피처 텐서를 다시 만들 필요가 없다."""
    return json.dumps({k: cfg[k] for k in
                       ("ch_mode", "ch_grid", "ch_threshold", "use_bright", "use_ch",
                        "ballistic", "ballistic_lon", "transit_speeds", "tau_mode",
                        "tau_reference_hours", "clip_flags")},
                      sort_keys=True, default=list)
'''

# =====================================================================
SETUP_MD = """
## 2. 준비 — 데이터 · 코로나홀 일괄 추출 · 폴드 · 속도 측정

CH 추출은 **한 번의 패스로 임계 5개 x 격자 8개 x 밝기 레벨까지 전부** 뽑아
`work/cache/sweepch_*.npz` 에 캐시한다. 이미지 캐시(`work/cache/128px`)는 P3 것을 그대로 쓴다.
"""

SETUP_SRC = """
SWEEP_ROOT = Path(SWEEP_DIR)
(SWEEP_ROOT / "final").mkdir(parents=True, exist_ok=True)
set_log(SWEEP_ROOT / "sweep.log")
BUDGET = Budget(BUDGET_HOURS, RESERVE_MINUTES)

log("=" * 78)
log(f"SWEEP 시작  {time.strftime('%Y-%m-%d %H:%M:%S')}  예산 {BUDGET_HOURS}시간"
    f"  (종료 예정 {time.strftime('%H:%M:%S', time.localtime(BUDGET.end))})")
log("=" * 78)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")
if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True
    log(f"PyTorch {torch.__version__} | GPU {torch.cuda.get_device_name(0)}")
else:
    log(f"PyTorch {torch.__version__} | !!! CPU 실행 — 탐색 규모를 크게 줄이세요 !!!")

DATA_ROOT = find_data_root()
CACHE_ROOT = Path("work/cache")
log(f"데이터: {DATA_ROOT.resolve()}")

TABLES = load_tables(DATA_ROOT)
log(f"샘플 수 — train {len(TABLES['train']['inputs']):,} / "
    f"val {len(TABLES['validation']['inputs']):,} / test {len(TABLES['test']['inputs']):,}")

IMAGE_ARRAYS, IMAGE_INDEX, INDEX_MATRICES = {}, {}, {}
for split in SPLITS:
    array, index = prepare_image_memmap(split, TABLES[split]["inputs"], DATA_ROOT,
                                        CACHE_ROOT, IMAGE_SIZE)
    IMAGE_ARRAYS[split] = array
    IMAGE_INDEX[split] = index
    INDEX_MATRICES[split] = image_index_matrix(TABLES[split]["inputs"], index)

DISK_Y, DISK_X, DISK_R, MEAN_IMAGE = detect_disk(IMAGE_ARRAYS["train"], IMAGE_SIZE)
log(f"원반 검출: center=({DISK_Y:.1f}, {DISK_X:.1f}) radius={DISK_R:.1f}px "
    f"(유효 {DISK_R * CH_SPEC['disk_margin']:.1f}px)")

STORES = {}
for split in SPLITS:
    STORES[split] = extract_ch_store(split, IMAGE_ARRAYS[split], CH_SPEC,
                                     (DISK_Y, DISK_X, DISK_R), CACHE_ROOT, IMAGE_SIZE)

# --- 사슬 · 폴드 (train 안에서만 자른다. validation 은 학습에 쓰지 않는다) ---
TRAIN_CHAINS = reconstruct_frame_chains(TABLES["train"]["inputs"])
TRAIN_CHAIN_ID, TRAIN_T_START = window_positions(TABLES["train"]["inputs"], TRAIN_CHAINS)
CHAIN_LENGTHS = np.array([len(c) for c in TRAIN_CHAINS], np.int64)
expected_windows = int(np.maximum(CHAIN_LENGTHS - 19, 0).sum())
assert expected_windows == len(TABLES["train"]["inputs"]), "사슬-윈도우 수 불일치"
log(f"사슬 {len(TRAIN_CHAINS)}개 · 고유 프레임 {int(CHAIN_LENGTHS.sum()):,} "
    f"· 자전 {CHAIN_LENGTHS.sum() * 6 / 24 / 27.2753:.1f}회전")

FOLDS_A = chain_folds(TRAIN_CHAIN_ID, len(TRAIN_CHAINS), A_FOLDS, 1, FOLD_SEED)
FOLDS_B = chain_folds(TRAIN_CHAIN_ID, len(TRAIN_CHAINS), B_FOLDS, 1, FOLD_SEED)
SEEDS_A = [777 + i for i in range(A_SEEDS)]
SEEDS_B = [777 + i for i in range(B_SEEDS)]
log(f"폴드 — A {len(FOLDS_A)}개 x 시드 {A_SEEDS} = {len(FOLDS_A) * A_SEEDS}회 학습/설정")
log(f"폴드 — B {len(FOLDS_B)}개 x 시드 {B_SEEDS} = {len(FOLDS_B) * B_SEEDS}회 학습/설정 + val {B_SEEDS}회")

MEAN_SPEED = float(TABLES["train"]["wind"].mean())
ALL_TRAIN_ROWS = np.arange(len(TABLES["train"]["inputs"]))
VAL_TARGETS = TABLES["validation"]["targets"].astype(np.float64)
VAL_PERSISTENCE = np.repeat(TABLES["validation"]["wind"][:, -1:], 12, axis=1).astype(np.float64)
PERSISTENCE_SCORE = official_rmse(VAL_TARGETS, VAL_PERSISTENCE)[0]
log(f"[기준선] val persistence 공식 RMSE = {PERSISTENCE_SCORE:.3f} km/s")
log(f"[참고] P1 val 68.408 / P3 val 64.203 (Public 58.8028, 현재 최고 제출)")

# --- 피처 팩 캐시 (같은 피처 설정이면 다시 만들지 않는다) ---
_PACK_CACHE = {"signature": None, "packs": {}}


def get_packs(cfg, splits=("train",)):
    signature = feature_signature(cfg)
    if _PACK_CACHE["signature"] != signature:
        for pack in _PACK_CACHE["packs"].values():
            for key in ("ch", "bal", "wind", "wdiff", "wvalid", "wstats", "last", "targets"):
                pack[key] = None
        _PACK_CACHE["packs"] = {}
        _PACK_CACHE["signature"] = signature
        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
    for split in splits:
        if split not in _PACK_CACHE["packs"]:
            _PACK_CACHE["packs"][split] = build_pack(cfg, split, TABLES, STORES,
                                                     INDEX_MATRICES, DEVICE, MEAN_SPEED)
    return {split: _PACK_CACHE["packs"][split] for split in splits}


log(f"준비 완료 — 경과 {hms(BUDGET.elapsed())}, 남은 예산 {hms(BUDGET.left())}")
"""

# =====================================================================
SPACE_MD = """
## 3. 탐색 공간

기준점은 **P3 그대로**(`BASE_CONFIG`)다. 모든 비교는 이 설정을 앵커로 한다.
공간에 넣은 값은 전부 근거가 있는 것들이다 — P4~P18 에서 실제로 시험됐거나,
CODE_REPORT 가 "재보자" 라고 남긴 것들.
"""

SPACE_SRC = '''
# ===== 기준 설정 — P3 (Public 58.8028) 를 글자 그대로 =====================
BASE_CONFIG = normalize_config({
    # --- 코로나홀 피처 ---
    "ch_mode": "grid",              # grid | area | area_shape | grid_shape
    "ch_grid": [3, 5],              # (위도, 경도) 구간 수
    "ch_threshold": 0.45,           # 원반 중앙값 대비 어두움 기준
    "use_bright": False,            # 활성영역(밝기 1.6배 이상) 면적 추가
    "use_ch": True,
    # --- 탄도 정렬 ---
    "ballistic": "on",              # on | off
    "ballistic_lon": "central",     # central | all
    "transit_speeds": [350.0, 500.0, 700.0],
    "tau_mode": "fixed",            # fixed | adaptive
    "tau_reference_hours": 108.0,   # adaptive 일 때만 의미 있음
    "clip_flags": False,
    # --- 모델 ---
    "gru_hidden": 64, "gru_layers": 2, "gru_bidir": False, "wind_hidden": 96,
    "dropout": 0.4, "ch_dropout": None, "head_width": [192, 96], "horizon_embed": 8,
    # --- 학습 ---
    "lr": 3e-4, "weight_decay": 1e-3, "batch_size": 64, "epochs": 60,
    "loss": "metric", "loss_horizon_weight": "uniform",
    "aug_ch_noise": 0.05, "stride": 0, "ema": 0.0, "target": "residual",
})

# ===== 노브별 후보값 — OAT 사다리와 무작위 탐색이 같은 표를 쓴다 ==========
SPACE = {
    "ch_threshold": [0.35, 0.40, 0.45, 0.50, 0.55],
    "ch_grid": [[1, 1], [2, 3], [3, 3], [3, 5], [4, 3], [4, 5], [6, 3], [6, 5]],
    "ch_mode": ["grid", "area", "area_shape", "grid_shape"],
    "use_bright": [False, True],
    "ballistic": ["on", "off"],
    "ballistic_lon": ["central", "all"],
    "transit_speeds": [[350.0, 500.0, 700.0], [385.0], [400.0, 500.0, 600.0],
                       [300.0, 400.0, 500.0, 650.0, 800.0], [250.0, 350.0, 450.0, 550.0, 700.0]],
    "tau_mode": ["fixed", "adaptive"],
    "tau_reference_hours": [96.0, 108.0, 120.0],
    "clip_flags": [False, True],
    "gru_hidden": [32, 64, 96, 128],
    "gru_layers": [1, 2],
    "gru_bidir": [False, True],
    "wind_hidden": [64, 96, 128],
    "dropout": [0.2, 0.3, 0.4, 0.5],
    "ch_dropout": [None, 0.15, 0.3],
    "head_width": [[96, 48], [128, 64], [192, 96], [256, 128], [192, 96, 48]],
    "horizon_embed": [4, 8, 16],
    "lr": [1e-4, 2e-4, 3e-4, 6e-4, 1e-3],
    "weight_decay": [1e-4, 3e-4, 1e-3, 3e-3],
    "batch_size": [64, 128, 256],
    "epochs": [40, 60, 90],
    "loss": ["metric", "mse", "huber"],
    "loss_horizon_weight": ["uniform", "long"],
    "aug_ch_noise": [0.0, 0.03, 0.05, 0.10],
    "stride": [0, 9],
    "ema": [0.0, 0.999],
    "target": ["residual", "absolute"],
}

# 사다리에서 뺄 노브 — 다른 노브가 켜져 있을 때만 의미가 있어서 단독 비교가 무의미하다
OAT_SKIP = {"tau_reference_hours", "clip_flags"}


def sample_config(rng, center=None, keep_probability=0.55):
    """중심 설정에서 일부 노브만 흔든다.

    노브가 27개라 완전 무작위는 대부분 쓰레기가 나온다. 기준(또는 사다리에서 나온
    탐욕 조합)을 중심에 두고 평균 12개 정도만 바꾸면, 같은 예산으로 훨씬 쓸모 있는
    영역을 훑는다. keep_probability=0 이면 완전 무작위와 같다.
    """
    cfg = dict(center or BASE_CONFIG)
    for key, values in SPACE.items():
        if rng.random() < keep_probability:
            continue
        cfg[key] = values[rng.integers(len(values))]
    return normalize_config(sanitize_config(cfg))


def sanitize_config(cfg):
    """모순되는 조합을 하나의 대표형으로 접는다 (해시 중복을 줄인다)."""
    cfg = dict(cfg)
    if cfg["ch_mode"] in ("area", "area_shape"):
        cfg["ch_grid"] = [1, 1]                 # 격자를 안 쓰는 모드
        cfg["ballistic_lon"] = "central"
    if not cfg["use_ch"]:
        cfg["ballistic"] = "off"
    if cfg["ballistic"] == "off":
        cfg["transit_speeds"] = [500.0]
        cfg["tau_mode"] = "fixed"
        cfg["clip_flags"] = False
        cfg["ballistic_lon"] = "central"
    if cfg["tau_mode"] == "fixed":
        cfg["tau_reference_hours"] = 108.0
        cfg["clip_flags"] = False
    else:
        cfg["transit_speeds"] = [500.0]          # adaptive 는 속도 목록을 쓰지 않는다
    if cfg["gru_layers"] == 1:
        pass
    return cfg


def oat_configs():
    """한 번에 한 노브만 바꾼다 — 결과 표가 '어떤 노브가 몇 km/s 짜리인지' 를 말해 준다."""
    seen = {config_hash(BASE_CONFIG)}
    items = []
    for key, values in SPACE.items():
        if key in OAT_SKIP:
            continue
        for value in values:
            if value == BASE_CONFIG[key]:
                continue
            cfg = dict(BASE_CONFIG)
            cfg[key] = value
            cfg = normalize_config(sanitize_config(cfg))
            tag = config_hash(cfg)
            if tag in seen:
                continue
            seen.add(tag)
            items.append((f"{key}={value}", cfg))
    # 브랜치 자체를 끄는 대조군 — "코로나홀이 실제로 일을 하는가" 를 한 줄로 답한다
    for label, patch in (("ablation: CH 브랜치 끔 (wind only)", {"use_ch": False}),
                         ("ablation: CH 만, 탄도 끔", {"ballistic": "off"})):
        cfg = normalize_config(sanitize_config({**BASE_CONFIG, **patch}))
        tag = config_hash(cfg)
        if tag not in seen:
            seen.add(tag)
            items.append((label, cfg))
    # adaptive 계열은 짝으로만 의미가 있어서 따로 몇 개를 붙인다
    for hours in SPACE["tau_reference_hours"]:
        for flags in (False, True):
            cfg = dict(BASE_CONFIG)
            cfg.update(tau_mode="adaptive", tau_reference_hours=hours, clip_flags=flags)
            cfg = normalize_config(sanitize_config(cfg))
            tag = config_hash(cfg)
            if tag not in seen:
                seen.add(tag)
                items.append((f"adaptive tau={hours:.0f}h flags={flags}", cfg))
    return items


BASE_TAG = config_hash(BASE_CONFIG)
OAT_ITEMS = oat_configs()
log(f"기준 설정 P3 = {BASE_TAG}")
log(f"OAT 사다리 {len(OAT_ITEMS)}개 · 무작위 탐색 상한 {MAX_RANDOM_CONFIGS}개")
'''

# =====================================================================
STAGE_A_MD = """
## 4. A 단계 — 훑기

설정당 `A_FOLDS x A_SEEDS` 회 학습한다. 순서는 **기준 -> OAT 사다리 -> 무작위**.
매 설정이 끝날 때마다 `results.csv` 에 append 하므로 중간에 죽어도 결과가 남는다.
"""

STAGE_A_SRC = '''
RESULTS_PATH = SWEEP_ROOT / "results.csv"
RESULT_ROWS = []
DONE = {}

if RESUME and RESULTS_PATH.exists():
    try:
        previous = pd.read_csv(RESULTS_PATH)
        for row in previous.to_dict("records"):
            RESULT_ROWS.append(row)
            DONE[(row["stage"], row["tag"])] = row
        log(f"이전 결과 {len(RESULT_ROWS)}행을 이어받았습니다 (RESUME=True)")
    except Exception as error:                       # noqa: BLE001
        log(f"결과 파일을 읽지 못해 새로 시작합니다: {error}")

RUN_TIMES = []


def note_run(seconds):
    RUN_TIMES.append(seconds)


def run_cost(n_runs):
    # 중앙값이 아니라 3사분위수를 쓴다. 예산 초과보다 조금 남기는 쪽이 안전하다.
    typical = float(np.quantile(RUN_TIMES, 0.75)) if RUN_TIMES else 25.0
    return typical * n_runs


def guarded(label, function, *args, **kwargs):
    """설정 하나가 터져도 밤샘 실행 전체가 멈추지 않게 한다."""
    try:
        return function(*args, **kwargs)
    except Exception as error:                        # noqa: BLE001
        import traceback
        log(f"  [!] '{label}' 실패 — 건너뜁니다: {type(error).__name__}: {error}")
        for line in traceback.format_exc().strip().splitlines()[-8:]:
            log("      " + line)
        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        return None


def record(row):
    RESULT_ROWS.append(row)
    DONE[(row["stage"], row["tag"])] = row
    frame = pd.DataFrame(RESULT_ROWS)
    frame.to_csv(RESULTS_PATH, index=False)


def measure(stage, label, cfg, folds, seeds):
    tag = config_hash(cfg)
    if (stage, tag) in DONE:
        return DONE[(stage, tag)]
    started = time.perf_counter()
    packs = get_packs(cfg, ("train",))
    result = evaluate_config(cfg, packs["train"], folds, seeds, cfg["epochs"],
                             TRAIN_T_START, EPOCH_SMOOTH, on_run=note_run)
    row = {"stage": stage, "tag": tag, "label": label, "cv": result["score"],
           "cv_se": result["se"], "best_epoch": result["best_epoch"],
           "n_runs": result["n_runs"], "seconds": time.perf_counter() - started,
           "ch_dim": packs["train"]["ch_dim"], "bal_dim": packs["train"]["bal_dim"],
           "config": json.dumps(cfg, sort_keys=True)}
    np.save(SWEEP_ROOT / f"per_run_{stage}_{tag}.npy", result["per_run"])
    record(row)
    return row


# --- 예산 배분 --------------------------------------------------------
probe_started = time.perf_counter()
BASE_ROW = measure("A", "P3 기준", BASE_CONFIG, FOLDS_A, SEEDS_A)
probe_seconds = time.perf_counter() - probe_started
per_config_a = max(probe_seconds, run_cost(len(FOLDS_A) * A_SEEDS))
per_config_b = run_cost(len(FOLDS_B) * B_SEEDS) + run_cost(B_SEEDS) * 1.2
per_config_c = run_cost(C_SEEDS) * 1.3
reserve_c = per_config_c * (C_TOP_K + 1) + 60.0

log(f"\\n[기준] P3 사슬 CV = {BASE_ROW['cv']:.3f} +- {BASE_ROW['cv_se']:.3f} km/s "
    f"(epoch {BASE_ROW['best_epoch']}, {BASE_ROW['n_runs']}회, {BASE_ROW['seconds']:.0f}s)")
log(f"[예산] 설정당 A {per_config_a:.0f}s · B {per_config_b:.0f}s · C {per_config_c:.0f}s"
    f" · C 예약 {hms(reserve_c)}")

available = max(BUDGET.left() - reserve_c, 0.0)
budget_a = available * A_BUDGET_FRACTION
budget_b = available - budget_a
planned_b = int(np.clip(budget_b // max(per_config_b, 1.0), 2, B_MAX_K))
if planned_b < B_TOP_K:
    log(f"[예산] B 목표 {B_TOP_K}개 -> 예산상 {planned_b}개만 가능")
log(f"[예산] A 에 {hms(budget_a)} (약 {int(budget_a // max(per_config_a, 1.0))}개 설정) · "
    f"B 에 {hms(budget_b)} (약 {planned_b}개 설정)")

# --- A-1 OAT 사다리 ---------------------------------------------------
BUDGET.open_phase(budget_a)
if RUN_OAT:
    log("\\n" + "-" * 78)
    log(f"A-1 OAT 사다리 — {len(OAT_ITEMS)}개 설정, 한 번에 한 노브만 바꾼다")
    log("-" * 78)
    for position, (label, cfg) in enumerate(OAT_ITEMS, 1):
        if not BUDGET.affords(per_config_a):
            log(f"  [예산] A-1 을 {position - 1}/{len(OAT_ITEMS)} 에서 끊습니다")
            break
        row = guarded(label, measure, "A", label, cfg, FOLDS_A, SEEDS_A)
        if row is None:
            continue
        delta = row["cv"] - BASE_ROW["cv"]
        log(f"  [{position:3d}/{len(OAT_ITEMS)}] {label:42s} CV {row['cv']:7.3f} "
            f"({delta:+6.3f}) ep{row['best_epoch']:3d} {row['seconds']:5.0f}s "
            f"| 남은 A {hms(BUDGET.phase_left())}")

# --- A-1b 사다리에서 나온 노브를 전부 켠 탐욕 조합 ---------------------
def greedy_from_oat(rows, base_cv):
    """노브별로 기준을 이긴 값이 있으면 전부 채택한 조합.

    상호작용을 무시한 낙관적 조합이라 좋게 나와도 그 자체를 믿지 않는다 —
    B 단계에서 같은 폴드로 다시 잰다. 여기서의 쓸모는 **탐색 중심을 하나 더 얻는 것**이다.
    """
    by_knob = {}
    for row in rows:
        label = str(row["label"])
        if "=" not in label:
            continue
        key = label.split("=")[0]
        if key not in SPACE:
            continue
        by_knob.setdefault(key, []).append((row["cv"], json.loads(row["config"])[key]))
    cfg = dict(BASE_CONFIG)
    adopted = []
    for key, items in by_knob.items():
        score, value = min(items, key=lambda item: item[0])
        if score < base_cv:
            cfg[key] = value
            adopted.append(f"{key}={value}")
    return normalize_config(sanitize_config(cfg)), adopted


GREEDY_CONFIG, ADOPTED = None, []
if RUN_OAT:
    oat_rows = [r for r in RESULT_ROWS if r["stage"] == "A" and "=" in str(r["label"])]
    if oat_rows:
        GREEDY_CONFIG, ADOPTED = greedy_from_oat(oat_rows, BASE_ROW["cv"])
        if config_hash(GREEDY_CONFIG) != BASE_TAG and BUDGET.affords(per_config_a):
            greedy_row = guarded("greedy", measure, "A", "greedy(OAT 채택)",
                                 GREEDY_CONFIG, FOLDS_A, SEEDS_A)
            log("")
            log(f"  [greedy] 채택 {len(ADOPTED)}개: {', '.join(ADOPTED[:8])}"
                f"{' ...' if len(ADOPTED) > 8 else ''}")
            if greedy_row is None:
                GREEDY_CONFIG = None
            else:
                log(f"  [greedy] CV {greedy_row['cv']:7.3f} "
                    f"({greedy_row['cv'] - BASE_ROW['cv']:+6.3f})")
                if greedy_row["cv"] > BASE_ROW["cv"]:
                    log("  [greedy] 기준보다 나쁘다 -> 노브가 서로 간섭한다는 뜻. "
                        "탐색 중심은 기준만 쓴다")
                    GREEDY_CONFIG = None

# --- A-2 조합 탐색 -----------------------------------------------------
if RUN_RANDOM:
    log("")
    log("-" * 78)
    log("A-2 조합 탐색 — 기준/탐욕 조합 주변을 예산이 끝날 때까지 흔든다")
    log("-" * 78)
    rng = np.random.default_rng(SEARCH_SEED)
    centers = [BASE_CONFIG] if GREEDY_CONFIG is None else [BASE_CONFIG, GREEDY_CONFIG]
    tried = 0
    while tried < MAX_RANDOM_CONFIGS and BUDGET.affords(per_config_a):
        center = centers[int(rng.integers(len(centers)))]
        cfg = sample_config(rng, center)
        tag = config_hash(cfg)
        if ("A", tag) in DONE:
            continue
        tried += 1
        row = guarded(tag, measure, "A", f"random#{tried}", cfg, FOLDS_A, SEEDS_A)
        if row is None:
            continue
        delta = row["cv"] - BASE_ROW["cv"]
        marker = "  <-- 기준보다 좋음" if delta < 0 else ""
        log(f"  [r{tried:3d}] {tag} CV {row['cv']:7.3f} ({delta:+6.3f}) "
            f"ep{row['best_epoch']:3d} {row['seconds']:5.0f}s "
            f"| 남은 A {hms(BUDGET.phase_left())}{marker}")

stage_a = pd.DataFrame([r for r in RESULT_ROWS if r["stage"] == "A"]).sort_values("cv")
log(f"\\nA 단계 종료 — 설정 {len(stage_a)}개 평가, 경과 {hms(BUDGET.elapsed())}")
log("\\n상위 12개 (사슬 CV 기준)")
for position, row in enumerate(stage_a.head(12).to_dict("records"), 1):
    log(f"  {position:2d}. {row['cv']:7.3f} ({row['cv'] - BASE_ROW['cv']:+6.3f})  "
        f"{row['tag']}  {row['label']}")
'''

# =====================================================================
STAGE_B_MD = """
## 5. B 단계 — 정밀 재측정 + official validation

A 는 폴드 3개 1시드라 순위 상위에 **운으로 올라온 설정**이 섞여 있다. B 는

* 폴드 `B_FOLDS` x 시드 `B_SEEDS` 로 다시 재고 (기준 설정과 **같은 폴드·시드**),
* 대응비교(paired) 표준오차로 기준 대비 차이를 판정하고,
* **train 전체로 학습해 official validation 을 시드 `B_SEEDS` 개 평균**으로 읽는다.

CV 와 val 은 서로 다른 것을 잰다. 둘 다 이기는 설정만 C 로 올린다.
"""

STAGE_B_SRC = '''
def measure_val(cfg, seeds, epochs, t_max=None):
    """train 전체로 학습해 official validation 을 읽는다. 시드별 예측도 돌려준다."""
    packs = get_packs(cfg, ("train", "validation"))
    scores, predictions = [], []
    for seed in seeds:
        started = time.perf_counter()
        model, stats, _ = run_training(cfg, packs["train"], ALL_TRAIN_ROWS, None, seed,
                                       epochs, t_start=TRAIN_T_START, record_curve=False,
                                       t_max=t_max)
        prediction = predict_rows(cfg, model, packs["validation"], stats,
                                  np.arange(packs["validation"]["n"]))
        predictions.append(prediction)
        scores.append(official_rmse(VAL_TARGETS, prediction)[0])
        note_run(time.perf_counter() - started)
        del model
        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
    ensemble = official_rmse(VAL_TARGETS, np.mean(predictions, axis=0))[0]
    return float(np.mean(scores)), float(np.std(scores)), ensemble, predictions


BUDGET.open_phase(max(BUDGET.left() - reserve_c, 0.0))
# A 가 일찍 끝났으면 그만큼 B 를 늘린다 — 남는 시간을 놀리지 않는다
planned_b = int(np.clip(BUDGET.phase_left() // max(per_config_b, 1.0), 2, B_MAX_K))
log("\\n" + "=" * 78)
log(f"B 단계 — 상위 {planned_b}개 정밀 재측정 (남은 예산 {hms(BUDGET.phase_left())})")
log("=" * 78)

candidate_rows = [r for r in RESULT_ROWS if r["stage"] == "A" and r["tag"] != BASE_TAG]
candidate_rows.sort(key=lambda r: r["cv"])
b_configs = [("P3 기준", BASE_CONFIG)]
for row in candidate_rows[:planned_b]:
    b_configs.append((row["label"], normalize_config(json.loads(row["config"]))))

B_ROWS = []
for position, (label, cfg) in enumerate(b_configs, 1):
    need = per_config_b
    if position > 1 and not BUDGET.affords(need):
        log(f"  [예산] B 를 {position - 1}/{len(b_configs)} 에서 끊습니다")
        break
    tag = config_hash(cfg)
    if RESUME and ("B", tag) in DONE:            # 재시작 — 이미 정밀 측정한 설정
        row = DONE[("B", tag)]
        B_ROWS.append(row)
        log(f"  [{position:2d}/{len(b_configs)}] {label:38s} (이전 결과 재사용) "
            f"CV {row['cv']:7.3f} | val {row['val']:7.3f}")
        continue
    started = time.perf_counter()


    def _measure_b(cfg=cfg):
        packs = get_packs(cfg, ("train",))
        cv = evaluate_config(cfg, packs["train"], FOLDS_B, SEEDS_B, cfg["epochs"],
                             TRAIN_T_START, EPOCH_SMOOTH, on_run=note_run)
        val = measure_val(cfg, SEEDS_B, cv["best_epoch"], t_max=cfg["epochs"])
        return packs, cv, val


    outcome = guarded(f"B/{label}", _measure_b)
    if outcome is None:
        continue
    packs, cv, (val_mean, val_sd, val_ensemble, _) = outcome
    row = {"stage": "B", "tag": tag, "label": label, "cv": cv["score"], "cv_se": cv["se"],
           "best_epoch": cv["best_epoch"], "n_runs": cv["n_runs"],
           "val": val_mean, "val_sd": val_sd, "val_ens": val_ensemble,
           "seconds": time.perf_counter() - started,
           "ch_dim": packs["train"]["ch_dim"], "bal_dim": packs["train"]["bal_dim"],
           "config": json.dumps(cfg, sort_keys=True)}
    np.save(SWEEP_ROOT / f"per_run_B_{tag}.npy", cv["per_run"])
    record(row)
    B_ROWS.append(row)
    log(f"  [{position:2d}/{len(b_configs)}] {label:38s} CV {row['cv']:7.3f}+-{row['cv_se']:.3f} "
        f"| val {val_mean:7.3f} (앙상블 {val_ensemble:7.3f}) | ep{row['best_epoch']:3d} "
        f"| {row['seconds']:5.0f}s | 남은 {hms(BUDGET.phase_left())}")

# --- 대응비교: 같은 폴드·시드에서 기준과의 차이 -------------------------
base_b = next((r for r in B_ROWS if r["tag"] == BASE_TAG), None)
if base_b is not None:
    base_runs = np.load(SWEEP_ROOT / f"per_run_B_{BASE_TAG}.npy")
    for row in B_ROWS:
        runs = np.load(SWEEP_ROOT / f"per_run_B_{row['tag']}.npy")
        if runs.shape != base_runs.shape:
            row["d_cv"], row["paired_se"] = float("nan"), float("nan")
            continue
        difference = runs - base_runs
        row["d_cv"] = float(difference.mean())
        row["paired_se"] = float(difference.std(ddof=1) / math.sqrt(len(difference))) \\
            if len(difference) > 1 else float("nan")
        row["d_val"] = float(row["val"] - base_b["val"])
    frame = pd.DataFrame(RESULT_ROWS)
    for row in B_ROWS:                      # 대응비교 결과를 results.csv 에 반영
        mask = (frame["stage"] == "B") & (frame["tag"] == row["tag"])
        for key in ("d_cv", "paired_se", "d_val"):
            frame.loc[mask, key] = row.get(key, float("nan"))
    frame.to_csv(RESULTS_PATH, index=False)
    RESULT_ROWS[:] = frame.to_dict("records")

log("\\nB 단계 결과 — 기준(P3) 대비 대응비교")
log(f"  {'설정':40s} {'CV':>8s} {'dCV':>7s} {'2se':>6s} {'val':>8s} {'dval':>7s}  판정")
for row in sorted(B_ROWS, key=lambda r: r["cv"]):
    two_se = 2 * row.get("paired_se", float("nan"))
    verdict = "기준" if row["tag"] == BASE_TAG else (
        "차이없음" if not (abs(row.get("d_cv", 0.0)) > two_se) else
        ("개선" if row.get("d_cv", 0.0) < 0 else "퇴보"))
    both = ""
    if base_b is not None and row["tag"] != BASE_TAG:
        if row.get("d_cv", 1.0) < 0 and row.get("d_val", 1.0) < 0:
            both = "  ** CV·val 동시 우세 **"
    log(f"  {row['label'][:40]:40s} {row['cv']:8.3f} {row.get('d_cv', float('nan')):+7.3f} "
        f"{two_se:6.3f} {row['val']:8.3f} {row.get('d_val', float('nan')):+7.3f}  {verdict}{both}")
'''

# =====================================================================
STAGE_C_MD = """
## 6. C 단계 — 제출 후보 만들기

B 상위 `C_TOP_K` 개(기준 P3 포함)를 **train 전체 x 시드 `C_SEEDS`** 로 학습하고
예측을 **균등 평균**한다. 가중치를 val 로 적합하지 않는 이유는 P6 이 그걸로 졌기 때문이다.

각 후보 폴더에 `submission.csv` · `model.pth` · `config.json` 이 들어간다.
`model.pth` 에는 시드별 state_dict 가 전부 들어 있어 제출 노트북이 같은 앙상블을 복원한다.
"""

STAGE_C_SRC = '''
def build_final(cfg, label, seeds, epochs, folder, t_max=None):
    """train 전체 x 시드 앙상블 -> val 읽기 + test 예측 + 저장."""
    folder.mkdir(parents=True, exist_ok=True)
    packs = get_packs(cfg, ("train", "validation", "test"))
    states, val_predictions, test_predictions, stats_payload = [], [], [], None
    for seed in seeds:
        started = time.perf_counter()
        model, stats, _ = run_training(cfg, packs["train"], ALL_TRAIN_ROWS, None, seed,
                                       epochs, t_start=TRAIN_T_START, record_curve=False,
                                       t_max=t_max)
        states.append({k: v.detach().cpu() for k, v in model.state_dict().items()})
        stats_payload = stats_to_cpu(stats)
        val_predictions.append(predict_rows(cfg, model, packs["validation"], stats,
                                            np.arange(packs["validation"]["n"])))
        test_predictions.append(predict_rows(cfg, model, packs["test"], stats,
                                             np.arange(packs["test"]["n"])))
        note_run(time.perf_counter() - started)
        del model
        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    val_single = float(np.mean([official_rmse(VAL_TARGETS, p)[0] for p in val_predictions]))
    val_mean_prediction = np.mean(val_predictions, axis=0)
    val_ensemble, per_horizon = official_rmse(VAL_TARGETS, val_mean_prediction)
    test_mean = np.mean(test_predictions, axis=0)

    submission = pd.DataFrame(test_mean, columns=TARGET_COLUMNS)
    submission.insert(0, "sample_id", TABLES["test"]["inputs"].sample_id.to_numpy())
    assert len(submission) == len(TABLES["test"]["inputs"])
    assert np.isfinite(test_mean).all()
    submission.to_csv(folder / "submission.csv", index=False)

    torch.save({"ensemble": states, "config": cfg, "stats": stats_payload,
                "seeds": list(seeds), "epochs": epochs, "t_max": t_max, "label": label,
                "val_official_rmse": val_ensemble, "engine": "sweep1"},
               folder / "model.pth")
    (folder / "config.json").write_text(json.dumps(
        {"config": cfg, "label": label, "seeds": list(seeds), "epochs": epochs,
         "t_max": t_max, "val_official_rmse": val_ensemble, "val_single_mean": val_single},
        ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame({"horizon_h": HORIZONS, "rmse": per_horizon,
                  "persistence": official_rmse(VAL_TARGETS, VAL_PERSISTENCE)[1]}
                 ).to_csv(folder / "val_metrics.csv", index=False)
    return {"val_single": val_single, "val_ens": val_ensemble,
            "test": test_mean, "val_pred": val_mean_prediction, "per_horizon": per_horizon}


log("\\n" + "=" * 78)
log(f"C 단계 — 제출 후보 생성 (남은 예산 {hms(BUDGET.left())})")
log("=" * 78)

ranked = sorted([r for r in RESULT_ROWS if r["stage"] == "B"], key=lambda r: r["cv"])
if not ranked:                                  # B 가 아예 못 돌았으면 A 결과를 쓴다
    ranked = sorted([r for r in RESULT_ROWS if r["stage"] == "A"], key=lambda r: r["cv"])
selection, seen_tags = [], set()
for row in ranked:
    if row["tag"] in seen_tags:
        continue
    seen_tags.add(row["tag"])
    selection.append(row)
    if len(selection) >= C_TOP_K:
        break
if BASE_TAG not in seen_tags:                   # 기준은 항상 후보에 넣는다
    base_row = next((r for r in RESULT_ROWS if r["tag"] == BASE_TAG), None)
    if base_row is not None:
        selection.append(base_row)

FINAL_ROWS = []
for position, row in enumerate(selection, 1):
    cfg = normalize_config(json.loads(row["config"]))
    epochs = int(row["best_epoch"])
    seeds = [777 + i for i in range(C_SEEDS)]
    need = run_cost(len(seeds)) * 1.2
    if position > 1 and BUDGET.left() < need:
        seeds = seeds[:max(2, int(BUDGET.left() // max(run_cost(1), 1.0)))]
        log(f"  [예산] 시드를 {len(seeds)}개로 줄입니다")
        if len(seeds) < 2:
            log("  [예산] C 단계를 여기서 끊습니다")
            break
    folder = SWEEP_ROOT / "final" / f"{position:02d}_{row['tag']}"
    if RESUME and (folder / "submission.csv").exists() and (folder / "model.pth").exists():
        log(f"  [{position}/{len(selection)}] {str(row['label'])[:38]:38s} "
            f"(이전 산출물 재사용) {folder.name}")
        continue
    started = time.perf_counter()
    outcome = guarded(f"C/{row['label']}", build_final, cfg, str(row["label"]), seeds,
                      epochs, folder, t_max=cfg["epochs"])
    if outcome is None:
        continue
    FINAL_ROWS.append({"stage": "C", "tag": row["tag"], "label": row["label"],
                       "cv": row["cv"], "cv_se": row.get("cv_se", float("nan")),
                       "best_epoch": epochs, "n_runs": len(seeds),
                       "val": outcome["val_single"], "val_ens": outcome["val_ens"],
                       "seconds": time.perf_counter() - started, "folder": str(folder),
                       "config": json.dumps(cfg, sort_keys=True)})
    record(FINAL_ROWS[-1])
    log(f"  [{position}/{len(selection)}] {str(row['label'])[:38]:38s} "
        f"val 단일평균 {outcome['val_single']:7.3f} -> 시드앙상블 {outcome['val_ens']:7.3f} "
        f"| CV {row['cv']:7.3f} | {folder.name} | {hms(time.perf_counter() - started)}")

# --- 남는 시간은 1위 후보의 시드를 늘리는 데 쓴다 -----------------------
# 탐색으로 1 km/s 를 더 깎는 것보다, 같은 설정의 시드를 늘려 분산을 줄이는 쪽이
# 훨씬 재현성 있는 이득이다 (시드 편차 0.78~0.90 km/s, CODE_REPORT §P6).
if FINAL_ROWS and BUDGET.left() > run_cost(2) * 1.4:
    top = FINAL_ROWS[0]
    extra = int(min(12 - C_SEEDS, BUDGET.left() // max(run_cost(1) * 1.4, 1.0)))
    if extra >= 2:
        cfg = normalize_config(json.loads(top["config"]))
        seeds = [777 + i for i in range(C_SEEDS + extra)]
        log(f"  [보강] 1위 후보의 시드를 {C_SEEDS} -> {len(seeds)}개로 늘립니다 "
            f"(남은 {hms(BUDGET.left())})")
        outcome = guarded("C/보강", build_final, cfg, str(top["label"]), seeds,
                          int(top["best_epoch"]), Path(top["folder"]), t_max=cfg["epochs"])
        if outcome is not None:
            top["n_runs"] = len(seeds)
            top["val"] = outcome["val_single"]
            top["val_ens"] = outcome["val_ens"]
            top["label"] = str(top["label"]) + f" (시드 {len(seeds)})"
            record(top)
            log(f"  [보강] val 시드앙상블 {outcome['val_ens']:7.3f} "
                f"-> {Path(top['folder']).name} 갱신")

# --- 서로 다른 설정끼리의 균등 앙상블 (가중치를 적합하지 않는다) --------
if len(FINAL_ROWS) >= 2:
    folders = [Path(r["folder"]) for r in FINAL_ROWS[:min(3, len(FINAL_ROWS))]]
    frames = [pd.read_csv(f / "submission.csv") for f in folders]
    ids = frames[0]["sample_id"].to_numpy()
    assert all((f["sample_id"].to_numpy() == ids).all() for f in frames)
    blended = np.mean([f[TARGET_COLUMNS].to_numpy() for f in frames], axis=0)
    folder = SWEEP_ROOT / "final" / "00_blend"
    folder.mkdir(parents=True, exist_ok=True)
    submission = pd.DataFrame(blended, columns=TARGET_COLUMNS)
    submission.insert(0, "sample_id", ids)
    submission.to_csv(folder / "submission.csv", index=False)
    (folder / "config.json").write_text(json.dumps(
        {"blend_of": [str(f) for f in folders], "weights": "uniform",
         "note": "제출하려면 각 폴더의 model.pth 를 모두 실어야 한다. 규정상 model.pth 는 1개 -> "
                 "제출용으로는 단일 설정 폴더를 쓰는 편이 안전하다."},
        ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"  [blend] 상위 {len(folders)}개 설정 균등 평균 -> {folder}")
'''

# =====================================================================
REPORT_MD = """
## 7. 보고서 — 아침에 읽을 것

`work/sweep/REPORT.md` 에 같은 내용이 저장된다. 노트북 출력이 날아가도 남는다.
"""

REPORT_SRC = '''
def knob_marginals(frame):
    """A 단계 결과에서 노브별 평균 효과를 뽑는다 — '무엇이 몇 km/s 짜리인가'."""
    records = []
    configs = [json.loads(c) for c in frame["config"]]
    for key in SPACE:
        values = {}
        for cfg, score in zip(configs, frame["cv"]):
            value = json.dumps(cfg.get(key), sort_keys=True)
            values.setdefault(value, []).append(score)
        usable = {v: s for v, s in values.items() if len(s) >= 3}
        if len(usable) < 2:
            continue
        best = min(usable.items(), key=lambda kv: np.mean(kv[1]))
        worst = max(usable.items(), key=lambda kv: np.mean(kv[1]))
        records.append({"knob": key, "spread": float(np.mean(worst[1]) - np.mean(best[1])),
                        "best": best[0], "best_cv": float(np.mean(best[1])),
                        "best_n": len(best[1]), "worst": worst[0],
                        "worst_cv": float(np.mean(worst[1]))})
    return pd.DataFrame(records).sort_values("spread", ascending=False)


results = pd.DataFrame(RESULT_ROWS)
results.to_csv(RESULTS_PATH, index=False)
stage_a_frame = results[results["stage"] == "A"].copy()
stage_b_frame = results[results["stage"] == "B"].copy()
stage_c_frame = results[results["stage"] == "C"].copy()
marginals = knob_marginals(stage_a_frame) if len(stage_a_frame) >= 12 else pd.DataFrame()

lines = []
lines.append("# SWEEP 결과 — " + time.strftime("%Y-%m-%d %H:%M:%S"))
lines.append("")
lines.append(f"- 총 소요 {hms(BUDGET.elapsed())} / 예산 {BUDGET_HOURS}시간")
lines.append(f"- 평가한 설정: A {len(stage_a_frame)}개 · B {len(stage_b_frame)}개 · "
             f"제출 후보 {len(stage_c_frame)}개")
lines.append(f"- 계측: A 폴드{A_FOLDS}x시드{A_SEEDS} · B 폴드{B_FOLDS}x시드{B_SEEDS} · "
             f"C 시드{C_SEEDS} 앙상블")
lines.append(f"- 기준(P3) 사슬 CV {BASE_ROW['cv']:.3f} · val persistence {PERSISTENCE_SCORE:.3f}")
lines.append("")
lines.append("## 1. 제출 후보 (C 단계)")
lines.append("")
lines.append("| # | 폴더 | 설정 | CV | val(단일평균) | val(시드앙상블) |")
lines.append("|---|---|---|---|---|---|")
for row in stage_c_frame.sort_values("val_ens").to_dict("records"):
    lines.append(f"| {Path(str(row['folder'])).name.split('_')[0]} | "
                 f"`{Path(str(row['folder'])).name}` | {row['label']} | {row['cv']:.3f} | "
                 f"{row['val']:.3f} | **{row['val_ens']:.3f}** |")
lines.append("")
lines.append("> val 은 1,199 샘플 = 자전 11.2회전뿐이라 1 km/s 안팎 차이는 운으로 설명된다.")
lines.append("> CV 와 val 이 **같은 방향으로** 기준을 이긴 설정을 먼저 본다.")
lines.append("")

if len(stage_b_frame):
    lines.append("## 2. B 단계 대응비교 (기준 P3 = 0)")
    lines.append("")
    lines.append("| 설정 | CV | dCV | 2se | 판정 | val | dval |")
    lines.append("|---|---|---|---|---|---|---|")
    for row in stage_b_frame.sort_values("cv").to_dict("records"):
        two_se = 2 * float(row.get("paired_se", float("nan")))
        d_cv = float(row.get("d_cv", float("nan")))
        verdict = "기준" if row["tag"] == BASE_TAG else (
            "차이없음" if not (abs(d_cv) > two_se) else ("**개선**" if d_cv < 0 else "퇴보"))
        lines.append(f"| {row['label']} | {row['cv']:.3f} | {d_cv:+.3f} | {two_se:.3f} | "
                     f"{verdict} | {row['val']:.3f} | {float(row.get('d_val', float('nan'))):+.3f} |")
    lines.append("")

if len(marginals):
    lines.append("## 3. 노브별 효과 (A 단계 평균, 낮을수록 좋음)")
    lines.append("")
    lines.append("| 노브 | 최선 값 | CV | n | 최악 값 | CV | 폭 |")
    lines.append("|---|---|---|---|---|---|---|")
    for row in marginals.head(15).to_dict("records"):
        lines.append(f"| `{row['knob']}` | `{row['best']}` | {row['best_cv']:.3f} | {row['best_n']} | "
                     f"`{row['worst']}` | {row['worst_cv']:.3f} | {row['spread']:.3f} |")
    lines.append("")
    lines.append("> 이 표는 **주변화(marginal)** 다. 다른 노브가 무작위로 섞인 채 평균낸 값이라")
    lines.append("> 상호작용을 보지 못한다. 사다리(OAT) 행과 어긋나면 OAT 를 믿는다.")
    lines.append("")

lines.append("## 4. A 단계 상위 20")
lines.append("")
lines.append("| # | tag | 설정 | CV | dCV(기준대비) | epoch |")
lines.append("|---|---|---|---|---|---|")
for position, row in enumerate(stage_a_frame.sort_values("cv").head(20).to_dict("records"), 1):
    lines.append(f"| {position} | `{row['tag']}` | {row['label']} | {row['cv']:.3f} | "
                 f"{row['cv'] - BASE_ROW['cv']:+.3f} | {int(row['best_epoch'])} |")
lines.append("")
lines.append("## 5. 아침에 할 일")
lines.append("")
lines.append("1. 위 1절 표에서 **CV·val 이 함께 기준을 이긴** 후보를 고른다. 없으면 P3 기준의")
lines.append("   시드 앙상블 폴더를 고른다 (앙상블은 탐색 이득이 아니라 분산 감소라 더 믿을 만하다).")
lines.append("2. 제출 노트북을 만든다.")
lines.append("")
lines.append("   ```bash")
lines.append("   python build_final_notebook.py work/sweep/final/<폴더>/config.json")
lines.append("   ```")
lines.append("")
lines.append("3. 생성된 `code_final.ipynb` 를 위에서부터 실행한다. `model.pth` 를 읽어")
lines.append("   `submission/` 에 3종(`code.ipynb`·`model.pth`·`submission.csv`)을 채운다.")
lines.append("4. 제출 전 행 수 3,868 · 컬럼명 · sample_id 일치를 노트북 마지막 셀에서 확인한다.")
lines.append("")
lines.append("## 6. 이 결과를 읽을 때의 한계")
lines.append("")
lines.append("- CV 절대값은 **낙관 편향**이 아니다(폴드마다 통계를 다시 뽑았다). 다만 epoch 을")
lines.append("  CV 곡선 최저점으로 골랐으므로 설정 수만큼의 선택 편향이 남는다. 설정 200개를")
lines.append("  훑으면 1위의 CV 는 실력보다 좋게 나온다 — 그래서 B 단계에서 다시 쟀다.")
lines.append("- val 은 학습에 쓰이지 않았지만, 후보를 고르는 데 썼으므로 Public 과 같지 않다.")
lines.append("- 시드 편차는 이 규모 모델에서 0.78~0.90 km/s 다 (CODE_REPORT §P6). 그보다 작은")
lines.append("  차이는 순위로 읽지 않는다.")

(SWEEP_ROOT / "REPORT.md").write_text("\\n".join(lines) + "\\n", encoding="utf-8")
log("\\n".join(lines))
log("")
log("=" * 78)
log(f"완료 — {hms(BUDGET.elapsed())} · 보고서 {SWEEP_ROOT / 'REPORT.md'}")
log("=" * 78)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(9, 4.5))
    scores = stage_a_frame.sort_values("cv")["cv"].to_numpy()
    axis.plot(np.arange(1, len(scores) + 1), scores, marker=".", linewidth=1)
    axis.axhline(BASE_ROW["cv"], color="crimson", linestyle="--", label="P3 baseline")
    axis.set_xlabel("rank"); axis.set_ylabel("chain CV official RMSE (km/s)")
    axis.grid(alpha=0.3); axis.legend()
    plt.tight_layout(); plt.savefig(SWEEP_ROOT / "leaderboard.png", dpi=140)
    log(f"그림 저장: {SWEEP_ROOT / 'leaderboard.png'}")
except Exception as error:                        # noqa: BLE001
    log(f"그림 생략: {error}")
'''


# =====================================================================
def build_notebook():
    cells = [
        markdown_cell(TITLE_MD),
        markdown_cell("## 0. 실행 노브"),
        code_cell(KNOB_CELL),
        markdown_cell(ENGINE_MD),
        code_cell(ENGINE_SRC),
        markdown_cell(SETUP_MD),
        code_cell(SETUP_SRC),
        markdown_cell(SPACE_MD),
        code_cell(SPACE_SRC),
        markdown_cell(STAGE_A_MD),
        code_cell(STAGE_A_SRC),
        markdown_cell(STAGE_B_MD),
        code_cell(STAGE_B_SRC),
        markdown_cell(STAGE_C_MD),
        code_cell(STAGE_C_SRC),
        markdown_cell(REPORT_MD),
        code_cell(REPORT_SRC),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    try:
        import nbformat
        notebook = nbformat.from_dict(notebook)
        _, notebook = nbformat.validator.normalize(notebook)
    except ImportError:
        pass
    return notebook


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    notebook = build_notebook()
    args.out.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
                        encoding="utf-8")
    code_count = sum(1 for c in notebook["cells"] if c["cell_type"] == "code")
    print(f"wrote {args.out}  (셀 {len(notebook['cells'])}개, 코드 {code_count}개)")


if __name__ == "__main__":
    main()
