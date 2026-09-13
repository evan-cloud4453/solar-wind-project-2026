#!/usr/bin/env python
"""Final1 — 단순화 노선. `code_final1.ipynb` 를 만든다.

    python build_final1_notebook.py

설계 원칙은 네 줄이다.

1. 코로나홀이 전부다. 태양 원반 **내부**에서만, 193 ∪ 211 **합집합**으로 어두운 영역을
   잡고, 위도가 높을수록(극지방) 가중치를 낮춘다.
2. 오늘 보이는 코로나홀은 **며칠 뒤** 지구에 도착한다. horizon 마다 "그 바람이 태양을
   떠난 시각" 의 프레임을 직접 집어서(gather) 헤드에 넣는다.
3. 어떤 물리 피처를 써야 할지 모른다 → 프레임마다 작은 **CNN** 이 스스로 뽑게 한다.
4. 나머지는 전부 뺀다. 튜닝 sweep · 앙상블 가중 적합 · 3D CNN · 적응형 τ 없음.
"""

import json
from pathlib import Path


def markdown_cell(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code_cell(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(True)}


# ══════════════════════════════════════════════════════════════════ 0. 머리말
INTRO_MD = """
# 태양풍 속도 예측 — **Final1 (단순화 노선)**

앞선 P1~P18 · SWEEP 은 "무엇을 믿고 고를 것인가" 를 파느라 커졌다. Final1 은 그걸 전부
버리고 **네 문장**만 코드로 옮긴 것이다.

| # | 원칙 | 코드에서의 위치 |
|---|---|---|
| 1 | 코로나홀이 절대적으로 중요하다. 원반 **내부**만 보고(플레어·림 오인 방지), 193Å·211Å 의 **합집합**으로 어두운 영역을 잡고, **극지방일수록 가중치를 낮춘다** | 셀 2 `build_frames` |
| 2 | 오늘 보이는 코로나홀은 **며칠 뒤** 지구에 도착한다 | 셀 2-1 진단 · 셀 3 `alignment_index` |
| 3 | 어떤 물리 피처가 옳은지 모른다 → **CNN** 이 프레임에서 직접 뽑는다 | 셀 3 `FrameEncoder` |
| 4 | 나머지는 뺀다 | sweep · NNLS 앙상블 · 3D CNN · 적응형 τ **없음** |

### 모델 한 장 요약

```
프레임 20장 ──▶ [작은 2D CNN] ──▶ 프레임 특징 20 x D
                                     │
   horizon j 마다 "그 바람이 태양을 떠난 시각" 의 프레임을 집는다 (전달시간 3종)
                                     ▼
        [ 정렬 프레임 3 x D | 전체 평균 D | 풍속 20 | horizon 임베딩 ] ──▶ MLP ──▶ Δv
                                                                              │
                                              예측 = 마지막 관측 풍속 + Δv ◀───┘
```

전달시간 정렬이 이 노트북의 핵심이다. horizon j(=+6(j+1)h)에 지구에 닿는 바람은
`6(j+1) − 전달시간` 시각에 태양을 떠났고, 전달시간이 3~5일이면 그 시각은 **입력 120시간
창 안**이다. 즉 우리가 이미 들고 있는 사진 안에 정답의 원인이 찍혀 있다.

| 전달시간 | 대응 속도 | horizon 0(+6h) → 프레임 | horizon 11(+72h) → 프레임 |
|---|---|---|---|
| 120h (5일) | ~350 km/s | 0 | 11 |
| 96h (4일) | ~430 km/s | 4 | 15 |
| 72h (3일) | ~580 km/s | 8 | 19 |

### 실행 순서

1. 위에서부터 전부 실행한다.
2. 셀 2-1 의 진단 출력(코로나홀 면적 ↔ 타깃 상관의 시차 봉우리)이 3~5일에 서면
   원칙 1·2 가 이 데이터에서 실제로 성립한다는 뜻이다.
3. 마지막 셀이 `submission/` 에 `code.ipynb` · `model.pth` · `submission.csv` 를 채운다.
   **Ctrl+S 로 저장한 뒤** 마지막 셀을 한 번 더 실행해야 `code.ipynb` 가 최신본이 된다.

> validation 은 **학습에도 epoch 선택에도 쓰지 않는다.** epoch 은 train 내부 시간 홀드아웃으로
> 고르고, validation 은 마지막에 한 번 읽기만 한다 (P6 이 val 로 골랐다가 진 전례 때문).
"""

# ══════════════════════════════════════════════════════════════════ 1. 노브
KNOBS_SRC = '''
# ===== 노브 — 이 노트북에서 사람이 만지는 값은 여기가 전부다 ==============
from pathlib import Path

IMAGE_SIZE      = 128          # 코로나홀을 뽑을 해상도 (기존 work/cache/128px 재사용)
FRAME_SIZE      = 64           # CNN 이 먹는 해상도 (128 에서 2x2 평균 축소)

# --- 원칙 1: 코로나홀 ---
CH_THRESHOLD    = 0.45         # 원반 중앙값 대비 이 배수보다 어두우면 코로나홀
CH_UNION        = True         # True = 193 ∪ 211 (합집합). False 면 교집합(옛 P3 규칙)
DISK_MARGIN     = 0.95         # 원반 반지름의 몇 %까지만 본다 (림 밝아짐·플레어 배제)
LAT_SIGMA_DEG   = 40.0         # 위도 가중 exp(-0.5*(lat/sigma)^2). 극지방 코로나홀을 깎는다

# --- 원칙 2: 전달시간 ---
TRANSIT_HOURS   = (72, 96, 120)   # 3일 / 4일 / 5일 가설. horizon 마다 이 시각의 프레임을 집는다

# --- 원칙 3: CNN ---
FEATURE_DIM     = 96           # 프레임 하나가 남기는 특징 차원
WIND_DIM        = 64
HEAD_WIDTH      = 256
DROPOUT         = 0.1

# --- 학습 ---
MAX_EPOCHS      = 30           # 홀드아웃 곡선을 여기까지 그리고 최저점을 고른다
BATCH_SIZE      = 32
LEARNING_RATE   = 3e-4
WEIGHT_DECAY    = 1e-4
SEEDS           = (777, 1234, 2026)   # 시드 앙상블(균등 평균) — 분산만 줄인다
HOLDOUT_FRACTION = 0.15        # train 뒤쪽 15% 를 epoch 선택용으로 뗀다
RESIDUAL_SCALE  = 100.0        # 네트워크는 (타깃 - 마지막풍속)/100 을 예측한다
SPEED_RANGE     = (200.0, 1200.0)

CACHE_ROOT      = Path("work/cache")
CACHE_ROOT.mkdir(parents=True, exist_ok=True)
SUBMISSION_DIR  = Path("submission")
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
'''

# ══════════════════════════════════════════════════════════ 2. 공통 · 데이터
DATA_SRC = '''
# ===== 공통 유틸 · 데이터 로드 ==========================================
import gc, json, math, os, random, shutil, time

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F

IMAGE_COLUMNS  = [f"image_{i:02d}"  for i in range(20)]
WIND_COLUMNS   = [f"wind_{i:02d}"   for i in range(20)]
TARGET_COLUMNS = [f"target_{i:02d}" for i in range(12)]
CHANNELS = ("193", "211")
SPLITS   = ("train", "validation", "test")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")
if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True
print("PyTorch", torch.__version__, "|",
      torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "CPU")


def find_data_root():
    candidates = [Path(os.getenv("SW_DATA_ROOT"))] if os.getenv("SW_DATA_ROOT") else []
    candidates += [Path("public_dataset/competition_dataset_6h"),
                   Path("/home/jovyan/public_dataset/competition_dataset_6h"),
                   Path("public/public_dataset/competition_dataset_6h"),
                   Path("/home/jovyan/public/public_dataset/competition_dataset_6h"),
                   Path("dataset"), Path("/home/jovyan/dataset")]
    for candidate in candidates:
        if (candidate / "train/inputs.csv").exists():
            return candidate
    raise FileNotFoundError("데이터 경로를 찾지 못했습니다: "
                            + ", ".join(str(c) for c in candidates))


DATA_ROOT = find_data_root()
print("데이터:", DATA_ROOT.resolve())

INPUTS = {s: pd.read_csv(DATA_ROOT / s / "inputs.csv") for s in SPLITS}
TARGETS = {
    "train": pd.read_csv(DATA_ROOT / "train/targets.csv")[TARGET_COLUMNS].to_numpy(np.float32),
    "validation": pd.read_csv(DATA_ROOT / "validation/targets.csv")[TARGET_COLUMNS].to_numpy(np.float32),
    "test": None,
}
TEST_IDS = pd.read_csv(DATA_ROOT / "test/test_ids.csv")

assert INPUTS["test"].sample_id.tolist() == TEST_IDS.sample_id.tolist()
assert not any(c.startswith("target_") for c in INPUTS["test"].columns)
for split in SPLITS:
    assert INPUTS[split].sample_id.is_unique

WIND = {s: INPUTS[s][WIND_COLUMNS].to_numpy(np.float32) for s in SPLITS}
for split in SPLITS:              # 감사 결과 결측은 없다. 그래도 방어만 해 둔다
    bad = ~np.isfinite(WIND[split])
    if bad.any():
        WIND[split][bad] = np.nanmedian(WIND["train"])
        print(f"  [!] {split} wind 결측 {int(bad.sum())}개를 중앙값으로 채움")

WIND_MEAN = float(WIND["train"].mean())
WIND_STD  = float(WIND["train"].std())
print(f"샘플 수 — train {len(INPUTS['train']):,} / val {len(INPUTS['validation']):,} "
      f"/ test {len(INPUTS['test']):,}")
print(f"풍속 평균 {WIND_MEAN:.1f} ± {WIND_STD:.1f} km/s")


def prepare_image_memmap(split, inputs, image_size):
    """P3 이래 써 온 캐시 규약 그대로 — work/cache/128px 가 있으면 재사용한다."""
    folder = CACHE_ROOT / f"{image_size}px"
    folder.mkdir(parents=True, exist_ok=True)
    array_path = folder / f"{split}_images.npy"
    metadata_path = folder / f"{split}_metadata.json"
    filenames = sorted(pd.unique(inputs[IMAGE_COLUMNS].to_numpy().ravel()).tolist())
    expected = {"image_size": image_size, "channels": list(CHANNELS), "filenames": filenames}
    shape = (len(filenames), len(CHANNELS), image_size, image_size)

    ok = False
    if array_path.exists() and metadata_path.exists():
        try:
            cached = np.load(array_path, mmap_mode="r")
            ok = (json.loads(metadata_path.read_text(encoding="utf-8")) == expected
                  and cached.shape == shape and cached.dtype == np.uint8)
        except (OSError, ValueError, json.JSONDecodeError):
            ok = False

    if not ok:
        temp = array_path.with_name(array_path.name + f".partial.{os.getpid()}")
        resized = np.lib.format.open_memmap(temp, mode="w+", dtype=np.uint8, shape=shape)
        for index, filename in enumerate(filenames):
            for channel_index, channel in enumerate(CHANNELS):
                with Image.open(DATA_ROOT / split / channel / filename) as image:
                    resized[index, channel_index] = np.asarray(
                        image.convert("L").resize((image_size, image_size),
                                                  Image.Resampling.BILINEAR), dtype=np.uint8)
            if (index + 1) % 2000 == 0 or index + 1 == len(filenames):
                print(f"  {split} resize {index + 1}/{len(filenames)}", flush=True)
        resized.flush()
        del resized
        metadata_path.write_text(json.dumps(expected, ensure_ascii=False) + "\\n",
                                 encoding="utf-8")
        temp.replace(array_path)
        print(f"  이미지 캐시 생성: {array_path}")
    else:
        print(f"  이미지 캐시 재사용: {array_path}")
    return np.load(array_path, mmap_mode="r"), {n: i for i, n in enumerate(filenames)}


IMAGES, FRAME_INDEX = {}, {}
for split in SPLITS:
    array, index = prepare_image_memmap(split, INPUTS[split], IMAGE_SIZE)
    IMAGES[split] = array
    FRAME_INDEX[split] = np.asarray(
        [[index[name] for name in row]
         for row in INPUTS[split][IMAGE_COLUMNS].itertuples(index=False, name=None)],
        dtype=np.int64)                     # (샘플, 20) — 프레임 배열에서의 위치
    print(f"  {split}: 고유 프레임 {len(array):,}장")
'''

# ═══════════════════════════════════════════════════ 3. 코로나홀 · 프레임 텐서
CH_SRC = '''
# ===== 원칙 1 — 코로나홀 마스크: 원반 내부 · 합집합 · 위도 가중 ============
def detect_disk(image_array, image_size, sample_count=400):
    """train 평균 영상에서 태양 원반의 중심과 반지름을 잡는다."""
    picks = np.unique(np.linspace(0, len(image_array) - 1, sample_count).astype(int))
    mean_image = np.asarray(image_array[picks], dtype=np.float64).mean(axis=(0, 1))
    mask = mean_image > mean_image.max() * 0.15
    ys, xs = np.nonzero(mask)
    return float(ys.mean()), float(xs.mean()), float(np.sqrt(mask.sum() / np.pi)), mean_image


DISK_Y, DISK_X, DISK_R, MEAN_IMAGE = detect_disk(IMAGES["train"], IMAGE_SIZE)
EFFECTIVE_R = DISK_R * DISK_MARGIN
print(f"원반: center=({DISK_Y:.1f}, {DISK_X:.1f})  radius={DISK_R:.1f}px  "
      f"유효반경={EFFECTIVE_R:.1f}px")

_yy, _xx = np.mgrid[0:IMAGE_SIZE, 0:IMAGE_SIZE].astype(np.float64)
RADIUS_MAP = np.sqrt((_yy - DISK_Y) ** 2 + (_xx - DISK_X) ** 2)
DISK_MASK = RADIUS_MAP <= EFFECTIVE_R                     # 원반 내부에서만 본다

# 위도 가중: y 변위 → 겉보기 위도. 극지방 코로나홀은 지구에 잘 닿지 않으므로 깎는다.
LATITUDE = np.degrees(np.arcsin(np.clip((_yy - DISK_Y) / DISK_R, -1.0, 1.0)))
LAT_WEIGHT = (np.exp(-0.5 * (LATITUDE / LAT_SIGMA_DEG) ** 2) * DISK_MASK).astype(np.float32)
print("위도 가중: " + " / ".join(
    f"±{d}° {math.exp(-0.5 * (d / LAT_SIGMA_DEG) ** 2):.2f}" for d in (0, 30, 45, 60, 80)))


def build_frames(split, image_array, chunk=256):
    """프레임마다 CNN 입력 3채널 + 코로나홀 면적 스칼라를 만든다.

    채널 0,1 : 193Å·211Å 를 **원반 중앙값 대비 상대밝기**로 (노출 변화에 둔감)
    채널 2   : 코로나홀 마스크 x 위도 가중  ← 원칙 1
    """
    key = (f"{CH_THRESHOLD:.2f}_{'U' if CH_UNION else 'I'}_{LAT_SIGMA_DEG:.0f}"
           f"_{DISK_MARGIN:.2f}_{FRAME_SIZE}")
    frame_path = CACHE_ROOT / f"final1_{split}_{key}_frames.npy"
    area_path  = CACHE_ROOT / f"final1_{split}_{key}_area.npy"
    n_frames = len(image_array)
    if frame_path.exists() and area_path.exists():
        frames, area = np.load(frame_path), np.load(area_path)
        if len(frames) == n_frames:
            print(f"  {split} 프레임 캐시 재사용: {frame_path.name}")
            return frames, area

    assert IMAGE_SIZE % FRAME_SIZE == 0
    factor = IMAGE_SIZE // FRAME_SIZE
    frames = np.zeros((n_frames, 3, FRAME_SIZE, FRAME_SIZE), np.uint8)
    area = np.zeros((n_frames, 2), np.float32)     # [위도가중 면적, 생면적]
    weight_on_disk = LAT_WEIGHT[DISK_MASK]
    started = time.perf_counter()

    for start in range(0, n_frames, chunk):
        block = np.asarray(image_array[start:start + chunk], dtype=np.float32)
        stop = start + len(block)
        median = np.median(block[:, :, DISK_MASK], axis=2)              # (b, 2)
        relative = block / np.maximum(median[:, :, None, None], 1e-6)   # (b, 2, H, W)
        relative *= DISK_MASK                                           # 원반 밖은 0

        dark_193 = relative[:, 0] < CH_THRESHOLD
        dark_211 = relative[:, 1] < CH_THRESHOLD
        # 합집합: 둘 중 한 파장에서만 어두워도 코로나홀로 본다 (교집합이면 놓친다)
        dark = np.logical_or(dark_193, dark_211) if CH_UNION else np.logical_and(dark_193, dark_211)
        dark = (dark & DISK_MASK).astype(np.float32)
        coronal_hole = dark * LAT_WEIGHT                                # 원칙 1의 위도 가중

        area[start:stop, 0] = ((dark[:, DISK_MASK] * weight_on_disk).sum(1)
                               / weight_on_disk.sum())
        area[start:stop, 1] = dark[:, DISK_MASK].mean(1)

        stack = np.stack([np.clip(relative[:, 0] / 2.0, 0, 1),
                          np.clip(relative[:, 1] / 2.0, 0, 1),
                          coronal_hole], axis=1)                        # (b, 3, H, W)
        small = stack.reshape(len(block), 3, FRAME_SIZE, factor,
                              FRAME_SIZE, factor).mean((3, 5))
        frames[start:stop] = np.clip(small * 255.0 + 0.5, 0, 255).astype(np.uint8)

        if stop == n_frames:
            print(f"  {split} 프레임 {n_frames:,}장 "
                  f"({n_frames / max(time.perf_counter() - started, 1e-6):.0f} frame/s)")

    np.save(frame_path, frames)
    np.save(area_path, area)
    return frames, area


FRAMES, CH_AREA = {}, {}
for split in SPLITS:
    FRAMES[split], CH_AREA[split] = build_frames(split, IMAGES[split])
print(f"프레임 텐서 합계 {sum(f.nbytes for f in FRAMES.values()) / 1024 ** 2:.0f} MiB (uint8)")
'''

DIAG_SRC = '''
# ===== 원칙 2 확인 — 코로나홀이 며칠 뒤에 도착하는가 ======================
# train 에서 "프레임 i 의 위도가중 코로나홀 면적" 과 "horizon j 의 타깃 속도" 의 상관을
# 전부 재 본다. 봉우리가 3~5일 시차에 서면 전달시간 정렬(다음 셀)의 근거가 된다.
area_sequence = CH_AREA["train"][:, 0][FRAME_INDEX["train"]]         # (샘플, 20)
correlation = np.zeros((20, 12), np.float32)
for i in range(20):
    for j in range(12):
        correlation[i, j] = np.corrcoef(area_sequence[:, i], TARGETS["train"][:, j])[0, 1]

print("행 = 프레임 관측시각(T0 기준) · 열 = horizon · 값 = 상관계수\\n")
print("        " + "".join(f"{(j + 1) * 6:>6d}h" for j in range(12)))
for i in range(20):
    print(f"T-{(19 - i) * 6:3d}h " + "".join(f"{correlation[i, j]:>7.2f}" for j in range(12)))

best_frame = correlation.argmax(axis=0)
transit_estimate = np.array([(19 - int(i)) * 6 for i in best_frame]) + np.arange(1, 13) * 6
print("\\nhorizon 별 최고상관 프레임 -> 태양에서 지구까지 걸린 시간(전달시간 추정)")
print("  horizon(h) : " + " ".join(f"{(j + 1) * 6:>4d}" for j in range(12)))
print("  전달시간(h) : " + " ".join(f"{v:>4d}" for v in transit_estimate))
print(f"\\n  중앙값 {np.median(transit_estimate):.0f}h "
      f"({np.median(transit_estimate) / 24:.1f}일)  ->  TRANSIT_HOURS = {TRANSIT_HOURS} "
      f"가 이 범위를 덮는지 본다")
print("  참고: 1 AU 를 350 km/s 로 가면 119h(5.0일) · 430 km/s 면 97h(4.0일) · "
      "580 km/s 면 72h(3.0일)")
'''

PLOT_SRC = '''
# ===== 눈으로 한 번 — 마스킹이 원반 안에만 잡혔는지 ======================
import matplotlib.pyplot as plt

probe = int(len(IMAGES["train"]) // 2)
raw = np.asarray(IMAGES["train"][probe], dtype=np.float32)
median = np.median(raw[:, DISK_MASK], axis=1)[:, None, None]
relative_probe = raw / np.maximum(median, 1e-6) * DISK_MASK
dark_probe = (((relative_probe[0] < CH_THRESHOLD) | (relative_probe[1] < CH_THRESHOLD))
              & DISK_MASK)

# 한글 폰트가 없는 서버가 많아 제목은 영문으로 둔다
panels = [(raw[0], "193A raw", "gray"),
          (raw[1], "211A raw", "gray"),
          (dark_probe.astype(float), f"CH union inside disk (th={CH_THRESHOLD})", "magma"),
          (dark_probe * LAT_WEIGHT, f"x latitude weight (sigma={LAT_SIGMA_DEG:.0f}deg)", "magma")]
figure, axes = plt.subplots(1, 4, figsize=(16, 4.2))
for axis, (image, title, cmap) in zip(axes, panels):
    axis.imshow(image, cmap=cmap)
    axis.set_title(title, fontsize=10)
    axis.add_patch(plt.Circle((DISK_X, DISK_Y), EFFECTIVE_R, fill=False, color="cyan", lw=1))
    axis.set_xticks([]); axis.set_yticks([])
plt.tight_layout(); plt.show()

print(f"이 프레임의 코로나홀 면적 — 생 {dark_probe.mean() * 100:.2f}% / "
      f"위도가중 {CH_AREA['train'][probe, 0] * 100:.2f}%")
print(f"train 전체 위도가중 면적: 평균 {CH_AREA['train'][:, 0].mean() * 100:.2f}% "
      f"(min {CH_AREA['train'][:, 0].min() * 100:.2f}% / "
      f"max {CH_AREA['train'][:, 0].max() * 100:.2f}%)")
'''

# ══════════════════════════════════════════════════════════════ 4. 모델
MODEL_SRC = '''
# ===== 원칙 2·3 — 전달시간 정렬 + 프레임 CNN =============================
def alignment_index(transit_hours):
    """horizon j 의 바람이 태양을 떠난 시각에 가장 가까운 프레임 번호.

    프레임 i 는 T-(19-i)x6h 에 관측됐고, horizon j 는 T+(j+1)x6h 에 지구에 닿는다.
    출발 시각 = 도착 - 전달시간  ->  i = 19 + ((j+1)*6 - transit) / 6
    """
    index = np.zeros((12, len(transit_hours)), np.int64)
    for k, transit in enumerate(transit_hours):
        for j in range(12):
            index[j, k] = int(np.clip(round(19 + ((j + 1) * 6 - transit) / 6.0), 0, 19))
    return index


ALIGN_INDEX = alignment_index(TRANSIT_HOURS)
print("horizon -> 정렬 프레임 번호 (전달시간 "
      + ", ".join(f"{t}h" for t in TRANSIT_HOURS) + ")")
for j in range(12):
    print(f"  +{(j + 1) * 6:2d}h : " + " ".join(f"{int(v):2d}" for v in ALIGN_INDEX[j]))


class FrameEncoder(nn.Module):
    """프레임 한 장 -> 특징 벡터. 물리 피처를 사람이 고르지 않는다 (원칙 3)."""

    def __init__(self, dim):
        super().__init__()
        layers, in_channels = [], 3
        for out_channels in (32, 64, 96, 128):
            layers += [nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False),
                       nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]
            in_channels = out_channels
        self.body = nn.Sequential(*layers)          # 64 -> 32 -> 16 -> 8 -> 4
        self.project = nn.Linear(in_channels * 2, dim)

    def forward(self, x):
        x = self.body(x)
        # 평균과 최대를 함께 넘긴다 — 면적(평균)과 가장 짙은 홀(최대)이 둘 다 신호다
        pooled = torch.cat([x.mean((2, 3)), x.amax((2, 3))], dim=1)
        return F.relu(self.project(pooled))


class Final1Net(nn.Module):
    def __init__(self, align_index):
        super().__init__()
        self.register_buffer("align", torch.as_tensor(align_index, dtype=torch.long))
        self.n_align = int(align_index.shape[1])
        self.encoder = FrameEncoder(FEATURE_DIM)
        self.wind_encoder = nn.Sequential(
            nn.Linear(20, 128), nn.ReLU(inplace=True),
            nn.Linear(128, WIND_DIM), nn.ReLU(inplace=True))
        self.horizon_embed = nn.Embedding(12, 16)
        head_in = FEATURE_DIM * (self.n_align + 1) + WIND_DIM + 16
        self.head = nn.Sequential(
            nn.Linear(head_in, HEAD_WIDTH), nn.ReLU(inplace=True), nn.Dropout(DROPOUT),
            nn.Linear(HEAD_WIDTH, HEAD_WIDTH // 2), nn.ReLU(inplace=True), nn.Dropout(DROPOUT),
            nn.Linear(HEAD_WIDTH // 2, 1))

    def forward(self, frames, wind):
        """frames (B,20,3,H,W) 0~1 실수 · wind (B,20) km/s  ->  (B,12) km/s"""
        batch = frames.shape[0]
        features = self.encoder(frames.flatten(0, 1)).view(batch, 20, FEATURE_DIM)

        aligned = features[:, self.align.reshape(-1)]                 # (B, 12*K, D)
        aligned = aligned.reshape(batch, 12, self.n_align * FEATURE_DIM)
        summary = features.mean(1, keepdim=True).expand(-1, 12, -1)   # 창 전체 요약

        wind_normalized = (wind - WIND_MEAN) / WIND_STD
        wind_features = self.wind_encoder(wind_normalized).unsqueeze(1).expand(-1, 12, -1)
        horizon = self.horizon_embed.weight.unsqueeze(0).expand(batch, -1, -1)

        delta = self.head(torch.cat([aligned, summary, wind_features, horizon],
                                    dim=2)).squeeze(2)
        # 예측 = 마지막 관측 풍속 + Δv. 잔차만 배우게 하는 것이 P1 이래 가장 안전했다.
        return wind[:, -1:] + delta * RESIDUAL_SCALE


_probe = Final1Net(ALIGN_INDEX)
print(f"\\n파라미터 {sum(p.numel() for p in _probe.parameters()):,}개 "
      f"(CNN {sum(p.numel() for p in _probe.encoder.parameters()):,})")
del _probe
'''

# ══════════════════════════════════════════════════════════════ 5. 학습
TRAIN_SRC = '''
# ===== 학습 — 프레임을 GPU 에 상주시키고 인덱싱만 한다 ===================
def to_device_frames(array):
    """가능하면 GPU 에 올린다. 크면 CPU 에 두고 배치마다 옮긴다."""
    tensor = torch.from_numpy(np.ascontiguousarray(array))
    if DEVICE.type == "cuda":
        free, _ = torch.cuda.mem_get_info()
        if tensor.numel() < free * 0.35:
            return tensor.to(DEVICE)
        return tensor.pin_memory()
    return tensor


GPU_FRAMES = {s: to_device_frames(FRAMES[s]) for s in SPLITS}
GPU_INDEX  = {s: torch.as_tensor(FRAME_INDEX[s], device=DEVICE) for s in SPLITS}
GPU_WIND   = {s: torch.as_tensor(WIND[s], device=DEVICE) for s in SPLITS}
GPU_TARGET = {s: torch.as_tensor(TARGETS[s], device=DEVICE)
              for s in SPLITS if TARGETS[s] is not None}
print("프레임 상주 위치:", {s: str(GPU_FRAMES[s].device) for s in SPLITS})


def take_batch(split, rows):
    index = GPU_INDEX[split][rows]                          # (B, 20)
    store = GPU_FRAMES[split]
    if store.device.type == DEVICE.type:
        frames = store[index.reshape(-1)]
    else:
        frames = store[index.reshape(-1).cpu()].to(DEVICE, non_blocking=True)
    frames = frames.view(len(rows), 20, 3, FRAME_SIZE, FRAME_SIZE).float().div_(255.0)
    return frames, GPU_WIND[split][rows]


def official_rmse(y_true, y_pred):
    """horizon 별 RMSE 를 먼저 내고 평균 — 대회 지표."""
    per_horizon = np.sqrt(np.mean((y_pred - y_true) ** 2, axis=0))
    return float(per_horizon.mean()), per_horizon


@torch.no_grad()
def predict(model, split, rows=None, batch=128):
    model.eval()
    rows = torch.arange(len(INPUTS[split]), device=DEVICE) if rows is None else rows
    outputs = []
    for start in range(0, len(rows), batch):
        frames, wind = take_batch(split, rows[start:start + batch])
        with torch.autocast(DEVICE.type, enabled=DEVICE.type == "cuda"):
            outputs.append(model(frames, wind).float())
    return torch.cat(outputs).clamp(*SPEED_RANGE).cpu().numpy().astype(np.float64)


def train_once(seed, train_rows, epochs, evaluate=None, log_every=1):
    """1회 학습. evaluate=(split, rows) 를 주면 epoch 마다 공식 지표 곡선을 남긴다."""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    model = Final1Net(ALIGN_INDEX).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)
    # T_max 를 MAX_EPOCHS 로 고정해 두면 홀드아웃에서 고른 epoch 의 LR 이 전체 학습에서도 같다
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
    scaler = torch.amp.GradScaler(DEVICE.type, enabled=DEVICE.type == "cuda")
    loss_function = nn.HuberLoss(delta=1.0)
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    curve = []

    for epoch in range(1, epochs + 1):
        model.train()
        started = time.perf_counter()
        order = train_rows[torch.randperm(len(train_rows), generator=generator,
                                          device=DEVICE)]
        total, seen = 0.0, 0
        for start in range(0, len(order), BATCH_SIZE):
            rows = order[start:start + BATCH_SIZE]
            frames, wind = take_batch("train", rows)
            target = GPU_TARGET["train"][rows]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(DEVICE.type, enabled=DEVICE.type == "cuda"):
                prediction = model(frames, wind)
                # 잔차 단위로 재서 loss 스케일을 1 근처로 맞춘다
                loss = loss_function((prediction - wind[:, -1:]) / RESIDUAL_SCALE,
                                     (target - wind[:, -1:]) / RESIDUAL_SCALE)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer); scaler.update()
            total += float(loss) * len(rows); seen += len(rows)
        scheduler.step()

        message = f"  epoch {epoch:3d}/{epochs}  loss {total / seen:.4f}"
        if evaluate is not None:
            evaluate_split, evaluate_rows = evaluate
            truth = TARGETS[evaluate_split][evaluate_rows.cpu().numpy()].astype(np.float64)
            score, _ = official_rmse(truth, predict(model, evaluate_split, evaluate_rows))
            curve.append(score)
            message += f"  홀드아웃 RMSE {score:7.3f}"
        if epoch % log_every == 0 or epoch == epochs:
            print(message + f"  ({time.perf_counter() - started:.0f}s)", flush=True)
    return model, curve


# --- epoch 선택: train 뒤쪽을 시간 블록으로 뗀다 (validation 은 건드리지 않는다) ---
n_train = len(INPUTS["train"])
split_at = int(n_train * (1.0 - HOLDOUT_FRACTION))
FIT_ROWS  = torch.arange(0, split_at - 20, device=DEVICE)   # 창 20개 겹침만큼 간격을 둔다
HOLD_ROWS = torch.arange(split_at, n_train, device=DEVICE)
print(f"epoch 선택용 분할 — 학습 {len(FIT_ROWS):,}행 / 홀드아웃 {len(HOLD_ROWS):,}행 "
      f"(사이 20행은 입력 창이 겹치므로 버린다)")

started = time.perf_counter()
_, HOLDOUT_CURVE = train_once(SEEDS[0], FIT_ROWS, MAX_EPOCHS, evaluate=("train", HOLD_ROWS))
smooth = np.convolve(HOLDOUT_CURVE, np.ones(3) / 3, mode="valid")
BEST_EPOCH = int(np.argmin(smooth)) + 2                     # 3점 이동평균의 최저점
print(f"\\n홀드아웃 최저 {min(HOLDOUT_CURVE):.3f} km/s "
      f"@ epoch {int(np.argmin(HOLDOUT_CURVE)) + 1}  ->  평활 후 채택 epoch = {BEST_EPOCH}"
      f"  ({time.perf_counter() - started:.0f}s)")

# --- 채택 epoch 으로 train 전체를 시드마다 다시 학습 (균등 평균 앙상블) ---
ALL_ROWS = torch.arange(n_train, device=DEVICE)
MODELS = []
for seed in SEEDS:
    print(f"\\n[전체 학습] seed {seed} · epoch {BEST_EPOCH}")
    model, _ = train_once(seed, ALL_ROWS, BEST_EPOCH, log_every=5)
    MODELS.append(model)
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
'''

# ══════════════════════════════════════════════════════════════ 6. 평가
EVAL_SRC = '''
# ===== validation — 여기서 처음이자 마지막으로 읽는다 ====================
val_predictions = [predict(model, "validation") for model in MODELS]
VAL_PREDICTION = np.mean(val_predictions, axis=0)
VAL_TRUTH = TARGETS["validation"].astype(np.float64)
VAL_PERSISTENCE = np.repeat(WIND["validation"][:, -1:], 12, axis=1).astype(np.float64)

val_score, val_per_horizon = official_rmse(VAL_TRUTH, VAL_PREDICTION)
persistence_score, persistence_per_horizon = official_rmse(VAL_TRUTH, VAL_PERSISTENCE)

print(f"official validation RMSE (시드 {len(MODELS)}개 앙상블) : {val_score:8.3f} km/s")
print(f"persistence 기준선                                    : {persistence_score:8.3f} km/s")
for index, model_prediction in enumerate(val_predictions):
    print(f"  · seed {SEEDS[index]} 단독 : {official_rmse(VAL_TRUTH, model_prediction)[0]:.3f}")
print("[참고] P1 val 68.408 (Public 62.34) · P3 val 64.203 (Public 58.80, 현재 최고)")

print("\\nhorizon    +6h  +12h  +18h  +24h  +30h  +36h  +42h  +48h  +54h  +60h  +66h  +72h")
print("Final1  " + "".join(f"{v:6.1f}" for v in val_per_horizon))
print("persist " + "".join(f"{v:6.1f}" for v in persistence_per_horizon))
print("차이    " + "".join(f"{a - b:+6.1f}" for a, b in zip(val_per_horizon,
                                                            persistence_per_horizon)))

if val_score >= persistence_score:
    print("\\n>>> 경고: persistence 를 이기지 못했습니다. 제출하지 마세요.")
else:
    print(f"\\npersistence 대비 {persistence_score - val_score:.2f} km/s 개선")
'''

# ══════════════════════════════════════════════════════════════ 7. 제출물
SUBMIT_SRC = '''
# ===== 제출물 3종 ======================================================
TEST_PREDICTION = np.mean([predict(model, "test") for model in MODELS], axis=0)
assert np.isfinite(TEST_PREDICTION).all()

submission = pd.DataFrame(TEST_PREDICTION, columns=TARGET_COLUMNS)
submission.insert(0, "sample_id", INPUTS["test"].sample_id.to_numpy())
submission.to_csv(SUBMISSION_DIR / "submission.csv", index=False)

torch.save({
    "ensemble": [{k: v.detach().cpu() for k, v in m.state_dict().items()} for m in MODELS],
    "seeds": list(SEEDS), "epochs": BEST_EPOCH, "align_index": ALIGN_INDEX,
    "wind_mean": WIND_MEAN, "wind_std": WIND_STD, "disk": [DISK_Y, DISK_X, DISK_R],
    "holdout_curve": HOLDOUT_CURVE, "val_official_rmse": val_score,
    "config": {"image_size": IMAGE_SIZE, "frame_size": FRAME_SIZE,
               "ch_threshold": CH_THRESHOLD, "ch_union": CH_UNION,
               "disk_margin": DISK_MARGIN, "lat_sigma_deg": LAT_SIGMA_DEG,
               "transit_hours": list(TRANSIT_HOURS), "feature_dim": FEATURE_DIM,
               "batch_size": BATCH_SIZE, "lr": LEARNING_RATE, "engine": "final1"},
}, SUBMISSION_DIR / "model.pth")

# 실행 중인 노트북을 복사한다. **반드시 Ctrl+S 로 저장한 뒤** 이 셀을 실행할 것.
for candidate in ("code_final1.ipynb", "code.ipynb"):
    if Path(candidate).exists():
        shutil.copyfile(candidate, SUBMISSION_DIR / "code.ipynb")
        print(f"code.ipynb 복사: {candidate}")
        break
else:
    print("!! 노트북 파일을 찾지 못했습니다. 저장(Ctrl+S) 후 이 셀을 다시 실행하세요.")

print("\\n=== 제출 점검 ===")
ok = True
for name in ("code.ipynb", "model.pth", "submission.csv"):
    path = SUBMISSION_DIR / name
    if path.exists():
        print(f"  [O] {name:16s} {path.stat().st_size / 1024 ** 2:8.2f} MiB")
    else:
        print(f"  [X] {name:16s} 없음"); ok = False

check = pd.read_csv(SUBMISSION_DIR / "submission.csv")
print(f"  행 수      : {len(check):,} (기대 {len(TEST_IDS):,})")
print(f"  컬럼       : {check.columns.tolist() == ['sample_id'] + TARGET_COLUMNS}")
print(f"  결측       : {int(check[TARGET_COLUMNS].isna().sum().sum())}")
print(f"  sample_id  : 유일={check.sample_id.is_unique}, "
      f"test_ids 일치={sorted(check.sample_id) == sorted(TEST_IDS.sample_id)}")
print(f"  값 범위    : [{check[TARGET_COLUMNS].to_numpy().min():.1f}, "
      f"{check[TARGET_COLUMNS].to_numpy().max():.1f}] km/s")
print(f"  val RMSE   : {val_score:.3f} (persistence {persistence_score:.3f})")
print("\\n제출 준비 완료" if ok else "\\n빠진 파일이 있습니다")
'''


CELLS = [
    markdown_cell(INTRO_MD),
    markdown_cell("## 0. 노브"),
    code_cell(KNOBS_SRC),
    markdown_cell("## 1. 데이터 로드 · 이미지 캐시"),
    code_cell(DATA_SRC),
    markdown_cell("## 2. 원칙 1 — 코로나홀 마스크 (원반 내부 · 합집합 · 위도 가중)"),
    code_cell(CH_SRC),
    markdown_cell("### 2-1. 원칙 2 확인 — 시차가 실제로 존재하는가"),
    code_cell(DIAG_SRC),
    markdown_cell("### 2-2. 마스킹 육안 점검"),
    code_cell(PLOT_SRC),
    markdown_cell("## 3. 원칙 2·3 — 전달시간 정렬 + 프레임 CNN"),
    code_cell(MODEL_SRC),
    markdown_cell("## 4. 학습 (epoch 은 train 내부 홀드아웃으로 고른다)"),
    code_cell(TRAIN_SRC),
    markdown_cell("## 5. validation 평가"),
    code_cell(EVAL_SRC),
    markdown_cell("## 6. 제출물 생성 · 점검"),
    code_cell(SUBMIT_SRC),
]


def build(output=Path("code_final1.ipynb")):
    notebook = {"cells": CELLS,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                            "name": "python3"},
                             "language_info": {"name": "python", "version": "3.11"}},
                "nbformat": 4, "nbformat_minor": 5}
    try:
        import nbformat
        notebook = nbformat.from_dict(notebook)
        _, notebook = nbformat.validator.normalize(notebook)
    except ImportError:
        pass
    Path(output).write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
                            encoding="utf-8")
    return output


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    path = build()
    print(f"wrote {path}")
    print("노트북을 열어 위에서부터 실행하면 submission/ 에 3종이 채워집니다.")
