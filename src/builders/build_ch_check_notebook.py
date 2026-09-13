"""Build ch_mask_check.ipynb: diagnose and fix the coronal-hole mask.

The notebook is a *diagnostic*, not part of any submission. It

1. reproduces the current P3/P17 CH rule verbatim (both 193 and 211 below
   CH_THRESHOLD_RATIO x on-disk median, inside a disk whose radius is derived from
   the area of a 15%-of-max threshold),
2. shows the two ways that rule fails on real AIA frames -- an over-estimated disk
   radius that lets the off-limb sky into the mask, and a median reference that
   tracks the activity level, and
3. implements a v2 detector (limb-gradient circle fit -> temporal radial flattening
   -> quiet-sun mode reference -> component filter) and puts it next to v1.

It reads the raw image files directly, so it needs no work/cache/*.npy.

Plot labels are ASCII on purpose: the competition server has no Korean matplotlib
font, so Korean titles render as tofu boxes. Prose and printed output stay Korean.
"""

from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
TARGET = HERE / "ch_mask_check.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.strip("\n").splitlines(True)}


CELLS = [

md(r'''
# 코로나홀 마스킹 점검 · 진단 · 수정

P3 / P17 의 코로나홀 규칙을 눈으로 확인하고, 어긋난 부분을 고친 규칙과 나란히 놓는
노트북입니다. 제출물이 아니고 학습도 하지 않습니다.

## 지금 파이프라인이 쓰는 규칙 (v1)

```
원반   : 평균 영상에서 (최대값의 15% 초과) 픽셀 -> 그 넓이로 반지름 역산, x 0.95
코로나홀: 193 < 0.45 x (193 원반 중앙값)  AND  211 < 0.45 x (211 원반 중앙값)
```

## 관측된 두 가지 고장

**(A) 원반이 실제보다 크게 잡혀서 원반 밖 하늘이 마스크에 들어온다.**
반지름을 `sqrt(넓이/pi)` 로 역산하는데, 그 넓이에 림 위쪽의 확산 코로나·스트리머가
함께 세어집니다. EUV 에서 원반 밖은 어둡지만 0 은 아니라서 15% 문턱을 넘깁니다.
그래서 나온 반지름은 실제 광구 림보다 크고, `DISK_MARGIN = 0.95` 로도 다 못 뺍니다.
남은 고리는 진짜 하늘이라 중앙값의 45% 보다 당연히 어둡고, **두 채널 모두에서**
어두우니 AND 도 통과합니다 -> 림을 따라 초승달 모양 마스크.

**(B) 기준값(중앙값)이 활동도에 끌려간다.**
`0.45 x 중앙값` 에서 중앙값은 그 프레임 전체에서 나옵니다. 밝은 활동영역·플레어가
원반의 상당 부분을 덮으면 중앙값이 올라가고, 절대 문턱도 같이 올라가서
**평범한 조용한 코로나가 "코로나홀" 로 넘어옵니다.** 반대로 조용한 날은 문턱이
내려가 진짜 홀을 놓칩니다. 즉 마스크 면적이 코로나홀이 아니라 활동도를 따라갑니다.

셀 6·7 이 A 와 B 를 각각 수치로 확인하고, 셀 8 이 고친 규칙을 정의합니다.

## v2 에서 바꾸는 것

| | v1 | v2 |
|---|---|---|
| 원반 | 밝은 픽셀 넓이 -> 반지름 | **림 기울기 최대점을 원으로 최소제곱 적합** |
| 중심 | 밝은 픽셀 무게중심 | 같은 원 적합 (림 밖 비대칭 발광에 안 흔들림) |
| 밝기 기준 | 프레임 전체 중앙값 | **반경별 시간중앙값으로 평탄화** 후 **조용한 코로나 최빈값** |
| 유효 영역 | 반지름 95% | 반지름 **90%** (림 근처 투영 왜곡 제외) |
| 후처리 | 없음 | 아주 작은 조각 제거 |

> 그림 라벨은 영문입니다. 서버 matplotlib 에 한글 폰트가 없어 한글 제목은 깨집니다.
'''),

md("## 1. 설정"),

code(r'''
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from PIL import Image

# --- v1: 지금 파이프라인이 쓰는 값 (code_p17.ipynb 설정 셀과 같아야 합니다) ---
CHANNELS = ("193", "211")
CH_THRESHOLD_RATIO = 0.45   # 원반 중앙값 대비 이 비율보다 어두우면 코로나홀
DISK_MARGIN = 0.95          # 림 밝아짐 회피용 반지름 축소
CH_GRID = (3, 5)            # P3 격자
PIPE_SIZE = 128             # 모델이 실제로 보는 해상도

# --- v2: 고친 규칙 ---
V2_RATIO = 0.45             # 평탄화 후 "조용한 코로나 최빈값" 대비 비율. 셀 10 에서 재조정
V2_CORE_FRACTION = 0.90     # 이 반지름 비율 안쪽만 코로나홀로 인정
V2_MIN_AREA = 0.0015        # 원반 넓이 대비 이보다 작은 조각은 버림
RADIAL_BINS = 64            # 반경 방향 평탄화 구간 수

# --- 이 노트북 전용 ---
SPLIT = "train"             # "train" / "validation" / "test"
N_SAMPLES = 5               # 눈으로 볼 프레임 수
RENDER_SIZE = 256           # 보기 해상도. PIPE_SIZE 로 두면 파이프라인과 완전히 동일
STACK_COUNT = 200           # 원반 적합 + 반경 프로파일에 쓸 프레임 수
TREND_COUNT = 300           # 진단(반경별 검출률 · 활동도 상관)에 쓸 프레임 수
SEED = 777                  # 다른 프레임을 보려면 이 값을 바꿉니다
SWEEP_RATIOS = (0.30, 0.35, 0.40, 0.45, 0.50, 0.60)

random.seed(SEED)
rng = np.random.default_rng(SEED)

DATA_ROOT_CANDIDATES = [
    Path(os.getenv("SW_DATA_ROOT", "")) if os.getenv("SW_DATA_ROOT") else None,
    Path("public_dataset/competition_dataset_6h"),
    Path("/home/jovyan/public_dataset/competition_dataset_6h"),
    Path("public/public_dataset/competition_dataset_6h"),
    Path("/home/jovyan/public/public_dataset/competition_dataset_6h"),
    Path("dataset"),
    Path("/home/jovyan/dataset"),
]
DATA_ROOT = None
for candidate in DATA_ROOT_CANDIDATES:
    if candidate is not None and (candidate / "train/inputs.csv").exists():
        DATA_ROOT = candidate
        break
if DATA_ROOT is None:
    searched = "\n".join(f"  - {c}" for c in DATA_ROOT_CANDIDATES if c is not None)
    raise FileNotFoundError("데이터 경로를 찾지 못했습니다:\n" + searched)

IMAGE_ROOT = DATA_ROOT / SPLIT
print(f"데이터: {DATA_ROOT.resolve()}")
print(f"v1 규칙: 두 채널 모두 중앙값의 {CH_THRESHOLD_RATIO:.0%} 미만, "
      f"반지름 {DISK_MARGIN:.0%} 이내")
print(f"v2 규칙: 평탄화 후 최빈값의 {V2_RATIO:.0%} 미만, "
      f"반지름 {V2_CORE_FRACTION:.0%} 이내, 조각 {V2_MIN_AREA:.2%} 미만 제거")
'''),

md("## 2. 원본에서 프레임 읽기 (캐시 불필요)"),

code(r'''
IMAGE_COLUMNS = [f"image_{index:02d}" for index in range(20)]
inputs = pd.read_csv(DATA_ROOT / f"{SPLIT}/inputs.csv")
FILENAMES = sorted(pd.unique(inputs[IMAGE_COLUMNS].to_numpy().ravel()).tolist())
print(f"{SPLIT} 고유 이미지 {len(FILENAMES):,}장")


def load_frame(filename, size):
    """원본 파일 -> (2, size, size) uint8. 파이프라인과 같은 BILINEAR 축소."""
    frame = np.empty((len(CHANNELS), size, size), dtype=np.uint8)
    for channel_index, channel in enumerate(CHANNELS):
        with Image.open(IMAGE_ROOT / channel / filename) as image:
            frame[channel_index] = np.asarray(
                image.convert("L").resize((size, size), Image.Resampling.BILINEAR),
                dtype=np.uint8)
    return frame


def load_stack(indexes, size):
    """(n, 2, size, size) uint8. 200장 x 256px 이면 26MB 정도라 그냥 들고 있습니다."""
    stack = np.empty((len(indexes), len(CHANNELS), size, size), dtype=np.uint8)
    for position, index in enumerate(indexes):
        stack[position] = load_frame(FILENAMES[index], size)
    return stack


with Image.open(IMAGE_ROOT / CHANNELS[0] / FILENAMES[0]) as probe:
    print(f"원본 해상도 {probe.size[0]}x{probe.size[1]} -> "
          f"보기 {RENDER_SIZE}px / 모델 {PIPE_SIZE}px")

STACK_INDEXES = np.unique(np.linspace(0, len(FILENAMES) - 1, STACK_COUNT).astype(int))
STACK = load_stack(STACK_INDEXES, RENDER_SIZE)
MEAN_IMAGE = STACK.astype(np.float64).mean(axis=(0, 1))
print(f"기준 스택 {len(STACK)}장 적재 ({STACK.nbytes / 1e6:.0f} MB)")

SAMPLE_NAMES = [FILENAMES[i] for i in
                sorted(rng.choice(len(FILENAMES), size=N_SAMPLES, replace=False))]
print("\n뽑은 프레임:")
for name in SAMPLE_NAMES:
    print(f"  {name}")
'''),

md(r'''## 3. v1 — 지금 파이프라인이 하는 그대로

`code_p17.ipynb` 셀 8 · `SolarWindDataset._coronal_hole_mask` 를 글자 그대로
옮긴 것입니다. 아래 두 함수가 고장의 출처입니다.'''),

code(r'''
def detect_disk_v1(mean_image):
    """파이프라인 셀 8 과 같은 규칙. 밝은 픽셀의 '넓이' 에서 반지름을 역산합니다."""
    bright = mean_image > mean_image.max() * 0.15
    ys, xs = np.nonzero(bright)
    return float(ys.mean()), float(xs.mean()), float(np.sqrt(bright.sum() / np.pi))


def make_geometry(size, center_y, center_x, radius, margin):
    grid_y, grid_x = np.mgrid[0:size, 0:size].astype(np.float64)
    r_map = np.sqrt((grid_y - center_y) ** 2 + (grid_x - center_x) ** 2)
    effective_r = radius * margin
    disk_mask = r_map <= effective_r

    n_lat, n_lon = CH_GRID
    lat = np.clip((grid_y - (center_y - effective_r)) / (2 * effective_r) * n_lat, 0, n_lat - 1e-6)
    lon = np.clip((grid_x - (center_x - effective_r)) / (2 * effective_r) * n_lon, 0, n_lon - 1e-6)
    cell_id = lat.astype(np.int64) * n_lon + lon.astype(np.int64)
    counts = np.maximum(np.bincount(cell_id[disk_mask], minlength=n_lat * n_lon), 1.0)
    return {"size": size, "y": center_y, "x": center_x, "r": radius, "eff_r": effective_r,
            "r_map": r_map, "mask": disk_mask, "cell_id": cell_id, "counts": counts}


def coronal_hole_mask_v1(frame, geometry, ratio=CH_THRESHOLD_RATIO):
    """(2, H, W) uint8 -> (H, W) bool. 파이프라인과 같은 식."""
    median = np.median(frame[:, geometry["mask"]].astype(np.float32), axis=1)
    dark = frame.astype(np.float32) < (ratio * median)[:, None, None]
    return np.logical_and(dark[0], dark[1]) & geometry["mask"]


def grid_area(mask, geometry):
    n_lat, n_lon = CH_GRID
    flat = np.bincount(geometry["cell_id"][mask], minlength=n_lat * n_lon)
    return (flat / geometry["counts"]).reshape(n_lat, n_lon)


V1_Y, V1_X, V1_R = detect_disk_v1(MEAN_IMAGE)
GEO_V1 = make_geometry(RENDER_SIZE, V1_Y, V1_X, V1_R, DISK_MARGIN)
print(f"v1 원반: center=({V1_Y:.1f}, {V1_X:.1f}) radius={V1_R:.1f}px "
      f"-> 유효 {GEO_V1['eff_r']:.1f}px  (프레임의 {2 * V1_R / RENDER_SIZE:.0%})")
'''),

md(r'''## 4. v1 원반이 진짜 림에 맞는가

가운데 그림이 핵심입니다. 실제 림은 밝기가 **급락하는 지점**입니다. 그 급락점보다
빨간 원(v1 유효 반지름)이 바깥에 있으면, 그 사이는 전부 하늘이고 마스크에 들어옵니다.'''),

code(r'''
grid_y, grid_x = np.mgrid[0:RENDER_SIZE, 0:RENDER_SIZE].astype(np.float64)
r_v1 = np.sqrt((grid_y - V1_Y) ** 2 + (grid_x - V1_X) ** 2)
edges = np.arange(0, V1_R * 1.35, 1.0)
profile = np.array([MEAN_IMAGE[(r_v1 >= lo) & (r_v1 < lo + 1)].mean() for lo in edges])
gradient = np.gradient(profile)
limb_guess = edges[np.argmin(gradient)]

figure, axes = plt.subplots(1, 3, figsize=(14, 4.4))
axes[0].imshow(MEAN_IMAGE, cmap="gray")
axes[0].add_patch(plt.Circle((V1_X, V1_Y), GEO_V1["eff_r"], fill=False, color="red", linewidth=1.5))
axes[0].add_patch(plt.Circle((V1_X, V1_Y), limb_guess, fill=False, color="lime",
                             linewidth=1.2, linestyle="--"))
axes[0].set_title(f"{SPLIT} mean image\nred = v1 effective, green = brightness cliff")
axes[0].set_xticks([]); axes[0].set_yticks([])

axes[1].plot(edges, profile, color="0.2")
axes[1].axvline(GEO_V1["eff_r"], color="red", label=f"v1 effective ({DISK_MARGIN:.2f} R)")
axes[1].axvline(V1_R, color="orange", linestyle=":", label="v1 raw R (area-derived)")
axes[1].axvline(limb_guess, color="green", linestyle="--", label="steepest drop = true limb")
axes[1].set_xlabel("radius (px)"); axes[1].set_ylabel("mean brightness")
axes[1].set_title("radial profile"); axes[1].legend(fontsize=8)

axes[2].plot(edges, gradient, color="0.2")
axes[2].axvline(GEO_V1["eff_r"], color="red"); axes[2].axvline(limb_guess, color="green",
                                                              linestyle="--")
axes[2].set_xlabel("radius (px)"); axes[2].set_ylabel("d(brightness)/dr")
axes[2].set_title("gradient (limb = the minimum)")
plt.tight_layout(); plt.show()

overshoot = GEO_V1["eff_r"] / limb_guess
print(f"진짜 림 ~= {limb_guess:.1f}px, v1 유효 반지름 = {GEO_V1['eff_r']:.1f}px "
      f"({overshoot:.2f}배)")
if overshoot > 1.01:
    ring = (r_v1 <= GEO_V1["eff_r"]) & (r_v1 > limb_guess)
    print(f"-> v1 원반 안에 하늘이 {ring.sum() / GEO_V1['mask'].sum():.1%} 섞여 있습니다. "
          "고장 (A) 확인.")
else:
    print("-> 반지름은 문제없습니다. 고장 (A) 는 이 데이터에서 발생하지 않습니다.")
'''),

md(r'''## 5. v1 로 5장 병렬 비교

빨간 영역이 림을 따라 초승달로 붙으면 고장 (A) 입니다.'''),

code(r'''
frames = {name: load_frame(name, RENDER_SIZE) for name in SAMPLE_NAMES}
masks_v1 = {name: coronal_hole_mask_v1(frames[name], GEO_V1) for name in SAMPLE_NAMES}

RED = ListedColormap([(1, 0, 0, 0.0), (1, 0.15, 0.15, 0.55)])
GREEN = ListedColormap([(0, 0, 0, 0.0), (0.2, 1.0, 0.3, 0.55)])

figure, axes = plt.subplots(N_SAMPLES, 5, figsize=(16, 3.2 * N_SAMPLES))
axes = np.atleast_2d(axes)
for row, name in enumerate(SAMPLE_NAMES):
    frame, mask = frames[name], masks_v1[name]
    median = np.median(frame[:, GEO_V1["mask"]].astype(np.float32), axis=1)
    area = mask.sum() / max(GEO_V1["mask"].sum(), 1)

    for channel_index in (0, 1):
        axis = axes[row, channel_index]
        axis.imshow(frame[channel_index], cmap="gray")
        axis.add_patch(plt.Circle((V1_X, V1_Y), GEO_V1["eff_r"], fill=False,
                                  color="deepskyblue", linewidth=0.8))
        axis.set_title(f"{CHANNELS[channel_index]} A  median={median[channel_index]:.0f}  "
                       f"thr={CH_THRESHOLD_RATIO * median[channel_index]:.0f}", fontsize=9)

    axes[row, 2].imshow(mask, cmap="gray")
    axes[row, 2].set_title(f"v1 mask   area={area:.2%} of disk", fontsize=9)

    axes[row, 3].imshow(frame[0], cmap="gray")
    axes[row, 3].imshow(mask, cmap=RED, interpolation="nearest")
    axes[row, 3].add_patch(plt.Circle((V1_X, V1_Y), GEO_V1["eff_r"], fill=False,
                                      color="deepskyblue", linewidth=0.8))
    axes[row, 3].set_title("overlay (red = v1 CH)", fontsize=9)

    cells = grid_area(mask, GEO_V1)
    image = axes[row, 4].imshow(cells, cmap="magma", vmin=0, vmax=max(cells.max(), 1e-3))
    for i in range(CH_GRID[0]):
        for j in range(CH_GRID[1]):
            axes[row, 4].text(j, i, f"{cells[i, j]:.02f}", ha="center", va="center", fontsize=8,
                              color="white" if cells[i, j] < cells.max() * 0.6 else "black")
    axes[row, 4].set_title("P3 grid area ratio", fontsize=9)
    figure.colorbar(image, ax=axes[row, 4], fraction=0.046)

    axes[row, 0].set_ylabel(name, fontsize=8)
    for axis in axes[row]:
        axis.set_xticks([]); axis.set_yticks([])
plt.tight_layout(); plt.show()
'''),

md(r'''## 6. 진단 — 고장 (A): 마스크가 반경 어디에 몰려 있나

여러 프레임에서 "이 반경의 픽셀이 코로나홀로 찍힐 확률" 을 그립니다.
코로나홀은 원반 전역에 생기므로 **평평하거나 완만해야** 정상입니다.
`r/R = 1` 근처에서 치솟으면 마스크가 잡고 있는 것은 코로나홀이 아니라 하늘입니다.

같은 루프에서 프레임별 통계도 모아 다음 셀(고장 B)에 씁니다.'''),

code(r'''
TREND_INDEXES = np.unique(np.linspace(0, len(FILENAMES) - 1, TREND_COUNT).astype(int))
GEO_V1_PIPE = make_geometry(PIPE_SIZE, *(np.array([V1_Y, V1_X, V1_R]) * PIPE_SIZE / RENDER_SIZE),
                            DISK_MARGIN)
r_norm_pipe = GEO_V1_PIPE["r_map"] / GEO_V1_PIPE["eff_r"]
N_RBINS = 40
r_bin = np.clip((r_norm_pipe * N_RBINS).astype(int), 0, N_RBINS - 1)
r_bin_flat = r_bin[GEO_V1_PIPE["mask"]]
bin_pixels = np.maximum(np.bincount(r_bin_flat, minlength=N_RBINS), 1)

# 고장 (B) 를 (A) 와 떼어 보기 위한 "정상 기하 위의 v1 규칙". 하늘 고리가 면적을
# 지배해 버리면 활동도 신호가 묻히므로, 기울기로 추정한 진짜 림 안쪽 80% 로 자릅니다.
CORE_GUESS = GEO_V1_PIPE["r_map"] <= 0.8 * limb_guess * PIPE_SIZE / RENDER_SIZE

hit_counts = np.zeros(N_RBINS, dtype=np.float64)
trend = []
for index in TREND_INDEXES:
    frame = load_frame(FILENAMES[index], PIPE_SIZE).astype(np.float32)
    mask = coronal_hole_mask_v1(frame, GEO_V1_PIPE)
    hit_counts += np.bincount(r_bin[mask], minlength=N_RBINS)
    on_disk = frame[:, GEO_V1_PIPE["mask"]]
    median = np.median(on_disk, axis=1)
    core_values = frame[:, CORE_GUESS]
    core_median = np.median(core_values, axis=1)
    core_mask = ((core_values < (CH_THRESHOLD_RATIO * core_median)[:, None]).all(axis=0))
    trend.append((mask.sum() / GEO_V1_PIPE["mask"].sum(),          # v1 CH 면적 (있는 그대로)
                  core_mask.mean(),                                # v1 규칙, 정상 기하 위
                  float(core_median[0]),                           # 중앙값 = 기준값
                  float((core_values[0] > 2.0 * core_median[0]).mean())))  # 활동영역 비율
trend = np.array(trend)
detection_rate = hit_counts / (bin_pixels * len(TREND_INDEXES))
centers = (np.arange(N_RBINS) + 0.5) / N_RBINS

figure, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(centers, detection_rate * 100, marker="o", markersize=3, color="crimson")
axes[0].axvline(V2_CORE_FRACTION / DISK_MARGIN, color="green", linestyle="--",
                label=f"v2 core cut ({V2_CORE_FRACTION:.2f} R)")
axes[0].set_xlabel("r / effective R"); axes[0].set_ylabel("P(flagged as CH)  [%]")
axes[0].set_title(f"v1 detection rate vs radius  ({len(TREND_INDEXES)} frames)")
axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

inner = detection_rate[:int(N_RBINS * 0.8)].mean()
outer = detection_rate[int(N_RBINS * 0.9):].mean()
axes[1].bar(["inner 0-80%", "outer 90-100%"], [inner * 100, outer * 100],
            color=["0.5", "crimson"])
axes[1].set_ylabel("P(flagged as CH)  [%]")
axes[1].set_title(f"edge / inner ratio = {outer / max(inner, 1e-9):.1f}x")
plt.tight_layout(); plt.show()

print(f"안쪽(0~80% R) 검출률 {inner:.2%},  바깥 고리(90~100% R) 검출률 {outer:.2%}")
if outer > inner * 2:
    print("-> 바깥 고리가 안쪽보다 2배 넘게 찍힙니다. 고장 (A) 확정: 마스크의 상당 부분이 하늘입니다.")
else:
    print("-> 반경 방향으로 크게 치우치지 않았습니다.")
'''),

md(r'''## 7. 진단 — 고장 (B): 마스크 면적이 활동도를 따라가는가

가로축은 **활동영역 비율**(중앙값의 2배보다 밝은 원반 픽셀의 비율), 세로축은 코로나홀
면적입니다. 물리적으로 이 둘은 특별한 관계가 없어야 합니다.

여기서는 **v1 임계값 규칙을 정상 기하(진짜 림의 80% 안쪽) 위에서** 돌린 면적을 봅니다.
있는 그대로의 v1 면적은 하늘 고리가 지배해서 (A) 가 (B) 를 가려 버리기 때문입니다.
**양의 상관이 뚜렷하면 "밝은 곳이 많을수록 코로나홀도 많다" 는 뜻이고, 그건
코로나홀이 아니라 중앙값 기준선이 끌려 올라간 결과입니다.**'''),

code(r'''
area_v1, area_core, median_core, active = trend.T

figure, axes = plt.subplots(1, 3, figsize=(15, 4))
correlation = np.corrcoef(active, area_core)[0, 1]
axes[0].scatter(active * 100, area_core * 100, s=14, alpha=0.6, color="crimson")
axes[0].set_xlabel("active-region fraction [%]")
axes[0].set_ylabel("CH area, v1 rule on correct disk [%]")
axes[0].set_title(f"(B) CH area vs activity   r = {correlation:+.2f}")
axes[0].grid(alpha=0.3)

axes[1].scatter(active * 100, median_core, s=14, alpha=0.6, color="tab:blue")
axes[1].set_xlabel("active-region fraction [%]"); axes[1].set_ylabel("on-disk median (193)")
axes[1].set_title("the mechanism: activity pushes the reference\n"
                  f"r = {np.corrcoef(active, median_core)[0, 1]:+.2f}")
axes[1].grid(alpha=0.3)

axes[2].plot(area_core * 100, color="crimson", linewidth=1)
axis_twin = axes[2].twinx()
axis_twin.plot(active * 100, color="0.4", linewidth=1)
axes[2].set_xlabel("frame index (time order)")
axes[2].set_ylabel("CH area [%]", color="crimson")
axis_twin.set_ylabel("active fraction [%]", color="0.4")
axes[2].set_title("do they move together?")
plt.tight_layout(); plt.show()

print(f"활동영역 비율 vs 중앙값 기준선 상관 r = {np.corrcoef(active, median_core)[0, 1]:+.2f}"
      "  (기준선이 활동도에 끌려가는 정도)")
print(f"활동영역 비율 vs CH 면적       상관 r = {correlation:+.2f}")
if correlation > 0.3:
    print("-> 뚜렷한 양의 상관. 고장 (B) 확정: 밝은 프레임에서 조용한 코로나가 홀로 넘어옵니다.")
elif correlation < -0.3:
    print("-> 뚜렷한 음의 상관. 밝은 프레임에서 진짜 홀을 놓치고 있습니다 (B 의 반대 방향).")
else:
    print("-> 활동도와의 결합은 약합니다. 고장 (B) 는 이 데이터에서 두드러지지 않습니다.")
print(f"\n참고: 있는 그대로의 v1 면적 평균 {area_v1.mean():.2%} vs "
      f"정상 기하 위 {area_core.mean():.2%} "
      f"— 차이가 곧 하늘 고리가 만든 가짜 면적입니다.")
'''),

md(r'''## 8. v2 — 고친 규칙

세 단계입니다.

1. **림 적합.** 720개 방향으로 광선을 쏘아 밝기 기울기가 가장 급한 반지름을 찾고,
   그 점들에 원을 최소제곱 적합합니다. 넓이 역산이 아니라 **모서리** 를 보므로
   림 밖 확산 코로나가 반지름을 부풀리지 못합니다. 중심도 같이 나옵니다.
2. **반경 평탄화.** 200장 스택에서 `프로파일[채널, 반경] = 그 고리 밝기의 시간중앙값`
   을 만들고 픽셀마다 나눕니다. 림 밝아짐과 중심-림 밝기 변화가 사라져 원반 전역에
   같은 문턱을 쓸 수 있습니다. **시간중앙값이라 한 프레임의 홀이 자기 기준을
   지우지 않습니다** (극지방 홀 보호).
3. **조용한 코로나 최빈값 기준.** 평탄화한 원반 값의 히스토그램 봉우리를 기준으로
   씁니다. 최빈값은 활동영역이 원반의 몇 %를 덮든 거의 안 움직입니다 — 중앙값과 달리.

여기에 유효 영역을 `0.90 R` 로 좁히고, 아주 작은 조각을 버립니다.'''),

code(r'''
def _fit_circle(ys, xs):
    """Kasa 원 적합. x^2+y^2 = 2ax + 2by + c 를 선형 최소제곱으로 풉니다."""
    design = np.stack([2 * xs, 2 * ys, np.ones_like(xs)], axis=1)
    target = xs ** 2 + ys ** 2
    (cx, cy, c), *_ = np.linalg.lstsq(design, target, rcond=None)
    return float(cy), float(cx), float(np.sqrt(max(c + cx ** 2 + cy ** 2, 1e-6)))


def fit_limb(mean_image, seed_y, seed_x, seed_r, n_rays=720, iterations=2):
    """방향별로 밝기 기울기가 가장 급한 반지름 = 림. 그 점들에 원을 적합합니다."""
    size = mean_image.shape[0]
    center_y, center_x, radius = seed_y, seed_x, seed_r
    for _ in range(iterations):
        angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
        steps = np.arange(0.55 * radius, 1.35 * radius, 0.25)
        ys = center_y + steps[None, :] * np.sin(angles)[:, None]
        xs = center_x + steps[None, :] * np.cos(angles)[:, None]
        inside = (ys >= 0) & (ys <= size - 1) & (xs >= 0) & (xs <= size - 1)
        samples = mean_image[np.clip(np.rint(ys), 0, size - 1).astype(int),
                             np.clip(np.rint(xs), 0, size - 1).astype(int)]
        samples = np.where(inside, samples, np.nan)
        # 반경 방향 3점 이동평균으로 노이즈를 죽인 뒤 기울기를 봅니다.
        kernel = np.ones(3) / 3.0
        smooth = np.apply_along_axis(lambda row: np.convolve(row, kernel, "same"), 1, samples)
        gradient = np.diff(smooth, axis=1)
        usable = np.isfinite(gradient).all(axis=1)
        if usable.sum() < n_rays * 0.5:
            gradient = np.nan_to_num(gradient, nan=0.0)
            usable = np.ones(n_rays, dtype=bool)
        index = np.argmin(np.where(np.isfinite(gradient), gradient, 0.0), axis=1)
        depth = -np.take_along_axis(gradient, index[:, None], axis=1).ravel()
        found = steps[index] + 0.5 * (steps[1] - steps[0])
        # 급락이 얕은 방향과, 중앙값에서 크게 벗어난 점을 버립니다.
        keep = usable & (depth > np.nanmedian(depth) * 0.25)
        deviation = np.abs(found - np.median(found[keep]))
        keep &= deviation < 4.0 * (np.median(deviation[keep]) + 1e-6)
        center_y, center_x, radius = _fit_circle(
            center_y + found[keep] * np.sin(angles[keep]),
            center_x + found[keep] * np.cos(angles[keep]))
    return center_y, center_x, radius, int(keep.sum())


def radial_profile(stack, geometry, n_bins=RADIAL_BINS):
    """프로파일[채널, 반경구간] = 그 고리 밝기의 (프레임별 중앙값)의 시간중앙값."""
    r_norm = geometry["r_map"] / geometry["r"]
    bins = np.clip((r_norm * n_bins).astype(int), 0, n_bins - 1)
    on_disk = r_norm <= 1.0
    per_frame = np.zeros((len(stack), len(CHANNELS), n_bins), dtype=np.float32)
    for bin_index in range(n_bins):
        selector = on_disk & (bins == bin_index)
        if not selector.any():
            per_frame[:, :, bin_index] = np.nan
            continue
        per_frame[:, :, bin_index] = np.median(stack[:, :, selector], axis=2)
    profile = np.nanmedian(per_frame, axis=0)                    # (2, n_bins)
    # 빈 구간은 이웃으로 메웁니다.
    for channel_index in range(len(CHANNELS)):
        row = profile[channel_index]
        bad = ~np.isfinite(row) | (row <= 0)
        if bad.any():
            row[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(~bad), row[~bad])
    # 밝기 단위 그대로 돌려줍니다. 나눈 결과가 곧 "조용한 코로나 대비 몇 배" 가 됩니다.
    return profile.astype(np.float32), bins


def quiet_level(values, bins=256, span=(0.0, 3.0)):
    """평탄화한 원반 밝기의 히스토그램 봉우리 = 조용한 코로나 수준."""
    histogram, edges = np.histogram(values, bins=bins, range=span)
    kernel = np.ones(9) / 9.0
    smooth = np.convolve(histogram.astype(np.float64), kernel, "same")
    centers = 0.5 * (edges[:-1] + edges[1:])
    return float(centers[int(np.argmax(smooth))])


def connected_components(mask):
    """(라벨맵, 개수). scipy 가 없으면 순수 numpy 로 전파합니다."""
    try:
        from scipy import ndimage
        return ndimage.label(mask, structure=np.ones((3, 3), bool))
    except Exception:
        labels = np.zeros(mask.shape, dtype=np.int32)
        labels[mask] = np.arange(1, int(mask.sum()) + 1)
        while True:
            merged = labels.copy()
            for axis in (0, 1):
                for shift in (1, -1):
                    merged = np.maximum(merged, np.roll(labels, shift, axis))
            merged = np.maximum(merged, np.roll(np.roll(labels, 1, 0), 1, 1))
            merged = np.maximum(merged, np.roll(np.roll(labels, 1, 0), -1, 1))
            merged = np.maximum(merged, np.roll(np.roll(labels, -1, 0), 1, 1))
            merged = np.maximum(merged, np.roll(np.roll(labels, -1, 0), -1, 1))
            merged[~mask] = 0
            if np.array_equal(merged, labels):
                break
            labels = merged
        unique = np.unique(labels[labels > 0])
        remap = np.zeros(labels.max() + 1, dtype=np.int32)
        remap[unique] = np.arange(1, len(unique) + 1)
        return remap[labels], len(unique)


def build_geometry_v2(size, stack, mean_image):
    seed_y, seed_x, seed_r = detect_disk_v1(mean_image)
    fit_y, fit_x, fit_r, kept = fit_limb(mean_image, seed_y, seed_x, seed_r)
    geometry = make_geometry(size, fit_y, fit_x, fit_r, V2_CORE_FRACTION)
    geometry["core_mask"] = geometry["mask"]
    geometry["fit_rays"] = kept
    profile, bins = radial_profile(stack, geometry)
    geometry["profile"] = profile
    geometry["profile_map"] = profile[:, bins]                   # (2, H, W)
    return geometry


def flatten(frame, geometry):
    return frame.astype(np.float32) / np.maximum(geometry["profile_map"], 1e-3)


def coronal_hole_mask_v2(frame, geometry, ratio=V2_RATIO, min_area=V2_MIN_AREA):
    core = geometry["core_mask"]
    flat = flatten(frame, geometry)
    reference = np.array([quiet_level(flat[c][core]) for c in range(len(CHANNELS))],
                         dtype=np.float32)
    dark = flat < (ratio * reference)[:, None, None]
    mask = np.logical_and(dark[0], dark[1]) & core
    if min_area > 0 and mask.any():
        labels, count = connected_components(mask)
        if count:
            sizes = np.bincount(labels.ravel())
            keep = sizes >= min_area * core.sum()
            keep[0] = False
            mask = keep[labels]
    return mask


GEO_V2 = build_geometry_v2(RENDER_SIZE, STACK, MEAN_IMAGE)
print(f"v2 원반: center=({GEO_V2['y']:.1f}, {GEO_V2['x']:.1f}) radius={GEO_V2['r']:.1f}px "
      f"({GEO_V2['fit_rays']}/720 방향 채택)")
print(f"    v1 radius {V1_R:.1f}px -> v2 {GEO_V2['r']:.1f}px "
      f"({GEO_V2['r'] / V1_R - 1:+.1%}),  유효 반지름 "
      f"{GEO_V1['eff_r']:.1f} -> {GEO_V2['eff_r']:.1f}px")
example = flatten(frames[SAMPLE_NAMES[0]], GEO_V2)
print(f"    평탄화 후 조용한 코로나 최빈값: "
      f"193 {quiet_level(example[0][GEO_V2['core_mask']]):.3f}, "
      f"211 {quiet_level(example[1][GEO_V2['core_mask']]):.3f}  (1.0 에 가까우면 정상)")
'''),

md(r'''## 9. v1 vs v2 병렬 비교

마지막 열에서 **빨강은 v1 만 잡은 것**(대부분 하늘 고리여야 합니다),
**초록은 v2 만 잡은 것**, 흰색은 둘 다입니다.'''),

code(r'''
masks_v2 = {name: coronal_hole_mask_v2(frames[name], GEO_V2) for name in SAMPLE_NAMES}

figure, axes = plt.subplots(N_SAMPLES, 5, figsize=(16, 3.2 * N_SAMPLES))
axes = np.atleast_2d(axes)
for row, name in enumerate(SAMPLE_NAMES):
    frame = frames[name]
    old, new = masks_v1[name], masks_v2[name]
    flat = flatten(frame, GEO_V2)

    axes[row, 0].imshow(frame[0], cmap="gray")
    axes[row, 0].add_patch(plt.Circle((V1_X, V1_Y), GEO_V1["eff_r"], fill=False,
                                      color="red", linewidth=0.9))
    axes[row, 0].add_patch(plt.Circle((GEO_V2["x"], GEO_V2["y"]), GEO_V2["eff_r"],
                                      fill=False, color="lime", linewidth=0.9))
    axes[row, 0].set_title("193 A   red = v1 disk, green = v2 disk", fontsize=9)

    image = axes[row, 1].imshow(np.where(GEO_V2["core_mask"], flat[0], np.nan),
                                cmap="coolwarm", vmin=0, vmax=2)
    axes[row, 1].set_title("193 flattened (quiet sun ~ 1.0)", fontsize=9)
    figure.colorbar(image, ax=axes[row, 1], fraction=0.046)

    axes[row, 2].imshow(frame[0], cmap="gray")
    axes[row, 2].imshow(old, cmap=RED, interpolation="nearest")
    axes[row, 2].set_title(f"v1   area={old.sum() / GEO_V1['mask'].sum():.2%}", fontsize=9)

    axes[row, 3].imshow(frame[0], cmap="gray")
    axes[row, 3].imshow(new, cmap=GREEN, interpolation="nearest")
    axes[row, 3].set_title(f"v2   area={new.sum() / GEO_V2['core_mask'].sum():.2%}", fontsize=9)

    difference = np.zeros(old.shape + (3,), dtype=np.float32)
    difference[..., 0] = old & ~new
    difference[..., 1] = new & ~old
    difference[old & new] = 1.0
    axes[row, 4].imshow(difference)
    union = np.logical_or(old, new).sum()
    axes[row, 4].set_title(f"IoU(v1, v2) = {np.logical_and(old, new).sum() / max(union, 1):.2f}\n"
                           "red: v1 only, green: v2 only", fontsize=9)

    axes[row, 0].set_ylabel(name, fontsize=8)
    for axis in axes[row]:
        axis.set_xticks([]); axis.set_yticks([])
plt.tight_layout(); plt.show()

print(f"{'frame':<28}{'v1 area':>9}{'v2 area':>9}{'v1 only':>9}{'v2 only':>9}")
for name in SAMPLE_NAMES:
    old, new = masks_v1[name], masks_v2[name]
    total = GEO_V2["core_mask"].sum()
    print(f"{name:<28}{old.sum() / GEO_V1['mask'].sum():>8.2%}{new.sum() / total:>9.2%}"
          f"{(old & ~new).sum() / max(old.sum(), 1):>8.0%}"
          f"{(new & ~old).sum() / max(new.sum(), 1):>8.0%}")
'''),

md(r'''## 10. v2 재검증 — 두 고장이 실제로 사라졌나

셀 6·7 과 같은 진단을 v2 로 다시 돌립니다. **반경 곡선이 평평해지고 활동도 상관이
0 근처로 내려와야** 고친 것입니다. 임계값 스윕도 같이 봅니다.'''),

code(r'''
STACK_PIPE = load_stack(STACK_INDEXES, PIPE_SIZE)
GEO_V2_PIPE = build_geometry_v2(PIPE_SIZE, STACK_PIPE,
                                STACK_PIPE.astype(np.float64).mean(axis=(0, 1)))
core_pipe = GEO_V2_PIPE["core_mask"]
r_bin_v2 = np.clip((GEO_V2_PIPE["r_map"] / GEO_V2_PIPE["eff_r"] * N_RBINS).astype(int),
                   0, N_RBINS - 1)
bin_pixels_v2 = np.maximum(np.bincount(r_bin_v2[core_pipe], minlength=N_RBINS), 1)

hit_v2 = np.zeros(N_RBINS, dtype=np.float64)
trend_v2 = []
for index in TREND_INDEXES:
    frame = load_frame(FILENAMES[index], PIPE_SIZE)
    mask = coronal_hole_mask_v2(frame, GEO_V2_PIPE)
    hit_v2 += np.bincount(r_bin_v2[mask], minlength=N_RBINS)
    on_disk = frame[0][core_pipe].astype(np.float32)
    median = float(np.median(on_disk))
    trend_v2.append((mask.sum() / core_pipe.sum(), float((on_disk > 2.0 * median).mean())))
trend_v2 = np.array(trend_v2)
rate_v2 = hit_v2 / (bin_pixels_v2 * len(TREND_INDEXES))

figure, axes = plt.subplots(1, 3, figsize=(16, 4))
axes[0].plot(centers, detection_rate * 100, marker="o", markersize=3,
             color="crimson", label="v1")
axes[0].plot(centers, rate_v2 * 100, marker="o", markersize=3, color="green", label="v2")
axes[0].set_xlabel("r / effective R"); axes[0].set_ylabel("P(flagged as CH) [%]")
axes[0].set_title("(A) detection rate vs radius"); axes[0].legend(fontsize=8)
axes[0].grid(alpha=0.3)

correlation_v2 = np.corrcoef(trend_v2[:, 1], trend_v2[:, 0])[0, 1]
axes[1].scatter(active * 100, area_v1 * 100, s=10, alpha=0.5, color="crimson",
                label=f"v1  r={correlation:+.2f}")
axes[1].scatter(trend_v2[:, 1] * 100, trend_v2[:, 0] * 100, s=10, alpha=0.5,
                color="green", label=f"v2  r={correlation_v2:+.2f}")
axes[1].set_xlabel("active-region fraction [%]"); axes[1].set_ylabel("CH area [%]")
axes[1].set_title("(B) CH area vs activity"); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

areas = np.zeros((N_SAMPLES, len(SWEEP_RATIOS)))
for row, name in enumerate(SAMPLE_NAMES):
    for column, ratio in enumerate(SWEEP_RATIOS):
        areas[row, column] = (coronal_hole_mask_v2(frames[name], GEO_V2, ratio=ratio).sum()
                              / GEO_V2["core_mask"].sum())
    axes[2].plot(SWEEP_RATIOS, areas[row] * 100, marker="o", markersize=4, label=name[:16])
axes[2].axvline(V2_RATIO, color="crimson", linestyle="--", label=f"V2_RATIO {V2_RATIO}")
axes[2].set_xlabel("threshold ratio (vs quiet-sun mode)"); axes[2].set_ylabel("v2 CH area [%]")
axes[2].set_title("v2 threshold sweep"); axes[2].legend(fontsize=7); axes[2].grid(alpha=0.3)
plt.tight_layout(); plt.show()

inner_v2 = rate_v2[:int(N_RBINS * 0.8)].mean()
outer_v2 = rate_v2[int(N_RBINS * 0.9):].mean()
print(f"(A) 바깥/안쪽 검출률 비  v1 {outer / max(inner, 1e-9):>6.1f}x  "
      f"->  v2 {outer_v2 / max(inner_v2, 1e-9):>6.1f}x   (1 에 가까울수록 좋음)")
print(f"(B) 활동도 상관        v1 {correlation:>+6.2f}   ->  v2 {correlation_v2:>+6.2f}"
      "   (0 에 가까울수록 좋음)")
print(f"\nv2 평균 CH 면적 {trend_v2[:, 0].mean():.2%} "
      f"(v1 {area_v1.mean():.2%}),  프레임간 표준편차 {trend_v2[:, 0].std():.2%}")
print("면적이 1% 아래로 눌렸으면 V2_RATIO 를 0.50~0.55 로 올리고, "
      "10% 를 넘으면 0.35~0.40 으로 내리세요.")
'''),

md(r'''## 11. 그래서 파이프라인은 어떻게 고치나

**셀 10 의 두 숫자가 개선된 것을 확인한 뒤에** `code_p17.ipynb` 를 고칩니다.
순서가 중요합니다 — `V2_RATIO` 는 이 노트북에서 정하고 넘기는 값입니다.

### 바꿀 곳

| 파일 | 위치 | 내용 |
|---|---|---|
| `build_p17_notebook.py` | 설정 셀 | `DISK_MARGIN = 0.95` -> `0.90`, `CH_THRESHOLD_RATIO` -> 셀 10 에서 고른 값 |
| " | 셀 8 `detect_disk` | 넓이 역산 -> 이 노트북의 `fit_limb` + `_fit_circle` |
| " | 셀 8 | `radial_profile` 로 `PROFILE` 을 만들고 `work/cache/profile_*.npy` 로 캐시 |
| " | 셀 8 `compute_ch_grid` | `on_disk` 를 평탄화하고 기준을 `np.median` -> `quiet_level` 로 |
| " | 셀 4 `cached_disk_median` | **`median_*` 캐시를 `quiet_*` 캐시로** (평탄화 후 최빈값 저장) |
| " | `_coronal_hole_mask` | `images` 를 `PROFILE` 로 나눈 뒤 `quiet` 와 비교, `core_mask` 적용 |

P3 (`Trial/P3/code_p3.ipynb`) 도 같은 셀 8 을 쓰므로 같이 고쳐야 비교가 성립합니다.

### 반드시 지울 캐시

```bash
rm -f work/cache/ch_*.npy work/cache/median_*.npy
```

`ch_*` 파일명에는 임계 비율이 들어가지만 `median_*` 에는 들어가지 않습니다.
지우지 않으면 옛 기하로 만든 값이 그대로 재사용됩니다.

### 다시 돌려야 하는 것

* **P3 의 64.203 은 이 마스크 위에서 나온 점수입니다.** 격자 피처의 상당 부분이
  하늘 고리 면적이었다면, P3 는 코로나홀이 아니라 *원반이 프레임에서 차지하는 위치의
  미세한 흔들림* 을 학습했을 수 있습니다. 고친 뒤 P3 를 한 번 다시 돌려서
  점수가 어떻게 움직이는지 봐야 합니다.
* **P17 의 보조 감독 라벨도 같은 마스크입니다.** `RUN_P17.md` 4절의 2번 대조군
  (`AUX_CH_WEIGHT = 0`) 은 마스크를 고친 뒤에 다시 돌려야 의미가 있습니다.
  고장난 라벨로 얻은 "보조 감독은 일하지 않는다" 는 결론은 무효입니다.

### v2 로도 남는 한계

* **필라멘트를 못 가른다.** 필라멘트도 193·211 에서 어둡습니다. 자기장 단극성으로
  갈라야 하는데 HMI 자기장이 데이터에 없습니다. 가늘고 길쭉한 조각은 필라멘트일
  가능성이 높으니, 필요하면 세장비(elongation) 로 한 번 더 거르세요.
* **림 근처는 여전히 못 믿는다.** `V2_CORE_FRACTION = 0.90` 으로 잘라냈지만
  그 안쪽 가장자리도 시선 방향 투영 때문에 면적이 왜곡됩니다.
* **`V2_RATIO` 는 여전히 손으로 고른 상수입니다.** 셀 10 스윕에서 면적 곡선의
  무릎을 골랐다면 그럭저럭 안정적이지만, 물리 상수는 아닙니다.
'''),
]


def main() -> None:
    notebook = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    TARGET.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
                      encoding="utf-8")
    print(f"wrote {TARGET}  ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
