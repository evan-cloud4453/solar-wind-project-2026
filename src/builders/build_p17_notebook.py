"""Build P17: U-Net frame encoder + coronal-hole auxiliary supervision, no ballistic features.

P17 keeps P3's data half verbatim (config prefix, CSV load, image memmap, disk
detection + CH grid) and replaces the whole modelling half:

* the hand-made 3x5 CH area vector is demoted to an off-by-default control branch;
  coronal holes now enter the model as an **auxiliary segmentation target** that the
  U-Net decoder has to predict, so the encoder is forced to represent CH structure
  while still choosing its own features,
* Inception3D -> a per-frame 2D **U-Net** (shared weights across the 20 frames),
* the image LSTM -> a **GRU**, matching the wind branch,
* every ballistic-alignment feature is removed (TRANSIT_SPEEDS, BALLISTIC_INDEX,
  compute_ballistic, the per-horizon ballistic head input).

P3 cell 8 is reused as-is: it detects the solar disk and produces the CH grid, and
P17 needs the disk geometry and the same threshold rule to build its pseudo-labels.
"""

from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "Trial" / "P3" / "code_p3.ipynb"
TARGET = HERE / "code_p17.ipynb"
TRIAL_TARGET = HERE / "Trial" / "P17" / "code_p17.ipynb"

CONFIG_CELL = 2
METRIC_CELL = 16


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.strip("\n").splitlines(True)}


HEADER = r'''# 태양풍 속도 예측 — P17: U-Net 영상 브랜치 + 코로나홀 보조 감독

P17 은 P3 를 베이스로 하되 **코로나홀을 손으로 요약하지 않는다.** 코로나홀이 태양풍
속도에 영향을 준다는 사실만 남기고, *그 코로나홀을 어떤 숫자로 접을지* 는 신경망이
정하게 한다.

| | P3 | P17 |
|---|---|---|
| 코로나홀 | 3×5 격자 면적 15개 실수 → GRU | U-Net 디코더가 **CH 마스크를 맞히도록 보조 감독**, 그 확률맵으로 인코더 특징을 모음 |
| 영상 인코더 | Inception3D (기본 꺼짐) | **U-Net** (항상 켬) |
| 영상 시계열 | LSTM | **GRU** (바람 브랜치와 동일) |
| 탄도 정렬 피처 | horizon별 3속도 역산 | **제거** |

## 코로나홀이 들어가는 방식

P3 의 임계값 규칙(193 · 211 두 채널 모두에서 원반 중앙값의 45% 미만)은 그대로 쓰되,
그 결과를 **모델 입력이 아니라 정답 라벨로** 쓴다.

```
프레임 --U-Net 인코더--> bottleneck --U-Net 디코더--> CH 확률맵  <--(BCE)-- 임계값 마스크
                            |                            |
                            +------ CH 확률로 가중 평균 ---+--> 프레임 임베딩 --GRU--> 예측
```

세 가지가 동시에 일어난다.

1. **인코더가 코로나홀을 표현하도록 강제된다.** 디코더가 마스크를 복원해야 하므로
   bottleneck 에 CH 위치·모양 정보가 남는다. 격자로 접었을 때 버려지던 경계 모양,
   비대칭, 크기 분포가 살아 있다.
2. **무엇을 볼지는 모델이 정한다.** 회귀에 쓰이는 값은 격자 면적이 아니라
   `CH 확률 가중 평균 특징 + 전역 평균 특징` 이다. 임계값 마스크는 어디를 보라는
   힌트일 뿐, 최종 피처를 지정하지 않는다.
3. **과적합 경로가 좁아진다.** 영상 브랜치가 자유롭게 아무 패턴이나 외우는 대신
   물리적으로 정의된 보조 과제를 함께 풀어야 한다.

> **주의 — 이 데이터에서 영상 브랜치는 네 번 실패했다** (P2·P6·P10, `CODE_REPORT.md` §5.2).
> 고유 이미지 10,139장이 실질 자전 135회분뿐인 것이 원인으로 지목됐다. P17 의 보조 감독은
> 정확히 그 실패를 겨눈 대응책이지만, 검증된 처방은 아니다. **`AUX_CH_WEIGHT = 0`
> 으로 한 번 더 돌려서** 보조 감독이 실제로 값을 하는지 확인하고 판단할 것.
> 또한 로컬 val 의 계측 노이즈 바닥은 약 3 km/s 다. 1~2 km/s 차이는 읽지 말 것.

`USE_CH_GRID = True` 로 두면 P3 의 격자 브랜치를 함께 켤 수 있다. 기본은 꺼져 있다.
'''


CONFIG = r'''
IMAGE_SIZE = 128
CHANNELS = ("193", "211")
FRAME_STRIDE = 1           # 2 로 올리면 20 -> 10 프레임. U-Net 비용이 정확히 절반
BATCH_SIZE = 32            # OOM 이면 16 -> 8 순으로 내린다
EPOCHS = 60
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-3
GRAD_CLIP = 1.0
SCHEDULER_PATIENCE = 3
EARLY_STOP_PATIENCE = 10
NUM_WORKERS = 4
LOSS_EPSILON = 1e-8
LOSS_SCALE = 100.0
LOSS_MODE = "metric"

# --- 브랜치 스위치 (ablation 용) ---------------------------------------
USE_UNET = True        # U-Net 영상 브랜치. P17 의 본체
USE_CH_AUX = True      # 코로나홀 마스크 보조 감독. 끄면 디코더는 비지도 attention
USE_CH_GRID = False    # P3 의 손수 만든 3x5 격자 브랜치. 대조용으로만
# 탄도 정렬 피처는 P17 에서 완전히 제거했다 (TRANSIT_SPEEDS / BALLISTIC_* 없음).

# --- 코로나홀 (P3 와 같은 규칙. P17 에서는 입력이 아니라 라벨) ---------
CH_GRID = (3, 5)           # USE_CH_GRID 와 pos_weight 추정에만 쓰인다
CH_THRESHOLD_RATIO = 0.45  # 원반 중앙값 대비 이 비율보다 어두우면 코로나홀
DISK_MARGIN = 0.95         # 림 밝아짐(limb brightening) 회피용 반지름 축소

# --- U-Net -------------------------------------------------------------
UNET_BASE = 24             # 최상위 채널 수. 32 로 올리면 용량이 약 1.8배
UNET_STEM_STRIDE = 2       # 128 -> 64 에서 U-Net 시작. 마스크 해상도도 64
MASK_SIZE = IMAGE_SIZE // UNET_STEM_STRIDE
IMAGE_EMBED = 96           # 프레임당 임베딩 차원
IMAGE_GRU_HIDDEN = 128     # 영상 GRU (P3 의 LSTM 자리)
AUX_CH_WEIGHT = 0.10       # 보조 BCE 손실 가중치. 0 이면 감독 없음
AUX_OFF_DISK_WEIGHT = 0.25 # 원반 밖 픽셀의 BCE 가중치 (배경도 0 으로 눌러 둔다)
# 보조 감독이 실제로 켜지는 조건. AUX_CH_WEIGHT=0 ablation 을 여기서 한 번에 끕니다.
AUX_ACTIVE = bool(USE_UNET and USE_CH_AUX and AUX_CH_WEIGHT > 0)

DROPOUT = 0.4
HORIZON_EMBED = 8
AUGMENT = True
AUG_BRIGHTNESS = 0.10
AUG_SHIFT_PIXELS = 6       # MASK_FACTOR 배수로 굴린다 (마스크 정합 유지)
AUG_NOISE_STD = 0.02
AUG_ERASE_PROB = 0.3
AUG_CH_NOISE = 0.05        # USE_CH_GRID 일 때 격자 피처에 주는 곱셈 노이즈

assert IMAGE_SIZE % (UNET_STEM_STRIDE * 8) == 0, "U-Net 4단 다운샘플이 나누어떨어져야 합니다"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"
PIN_MEMORY = DEVICE.type == "cuda"
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")
if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True
else:
    print("WARNING: CUDA Unavailable")

print("PyTorch:", torch.__version__, "| device:", DEVICE)
if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
print("data:", DATA_ROOT.resolve())
print(f"branches: UNET={USE_UNET} CH_AUX={USE_CH_AUX} CH_GRID={USE_CH_GRID} BALLISTIC=제거됨")
print(f"U-Net base={UNET_BASE} mask={MASK_SIZE}px  frames={len(range(0, 20, FRAME_STRIDE))} "
      f"aux={'on' if AUX_ACTIVE else 'off'}")
'''


DISK_MARKDOWN = r'''## 3. 태양 원반 검출 · 코로나홀 마스크 — **라벨 생성기**

셀 내용은 P3 와 한 글자도 다르지 않다. **쓰는 곳이 다르다.**

| 산출물 | P3 에서의 용도 | P17 에서의 용도 |
|---|---|---|
| `DISK_MASK`, `DISK_Y/X/R` | 격자 분할 | U-Net 보조 라벨의 유효 영역, BCE 가중치 |
| CH 임계값 규칙 (193 AND 211) | 격자 면적 계산 | **디코더가 맞혀야 할 정답 마스크** |
| `train_ch` 등 격자 면적 | 모델 입력 | `USE_CH_GRID` 대조군 + 보조 손실 `pos_weight` 추정 |

즉 P17 은 이 임계값을 **모델에 먹이지 않는다.** 어디를 봐야 하는지 알려 주는 데만
쓰고, 그 영역에서 무엇을 읽을지는 U-Net 이 학습으로 정한다.
'''


STATS = r'''
FRAME_INDEX = np.arange(0, 20, FRAME_STRIDE)
N_FRAMES = len(FRAME_INDEX)
MASK_FACTOR = IMAGE_SIZE // MASK_SIZE


def block_mean(array, factor):
    # (..., H, W) -> (..., H//factor, W//factor). 마스크를 면적 비율로 줄입니다.
    shape = array.shape[:-2] + (array.shape[-2] // factor, factor,
                                array.shape[-1] // factor, factor)
    return array.reshape(shape).mean(axis=(-3, -1))


# 마스크 해상도에서의 원반 점유율. BCE 가중치와 시각화에 씁니다.
DISK_WEIGHT = block_mean(DISK_MASK.astype(np.float32), MASK_FACTOR)


def compute_image_stats(array, chunk=256):
    total = np.zeros(len(CHANNELS), np.float64)
    total_square = np.zeros(len(CHANNELS), np.float64)
    count = 0
    for start in range(0, len(array), chunk):
        block = np.asarray(array[start:start + chunk], dtype=np.float64) / 255.0
        total += block.sum(axis=(0, 2, 3))
        total_square += (block ** 2).sum(axis=(0, 2, 3))
        count += block.shape[0] * block.shape[2] * block.shape[3]
    mean = total / count
    return mean.astype(np.float32), np.sqrt(
        np.maximum(total_square / count - mean ** 2, 1e-12)).astype(np.float32)


IMAGE_MEAN, IMAGE_STD = compute_image_stats(train_image_array)
WIND_MEAN = float(train_wind.mean())
WIND_STD = float(train_wind.std() + 1e-6)
DIFF_STD = float(np.diff(train_wind, axis=1, prepend=train_wind[:, :1]).std() + 1e-6)
CH_MEAN = train_ch.mean(axis=0).astype(np.float32)
# P3 는 +1e-8 이었는데, CH 가 한 번도 안 걸린 셀이 있으면 그 열이 1e8 배로 튄다.
# 실데이터에서는 모든 셀에 분산이 있어 아래 하한은 무해하다 (USE_CH_GRID 대조군 보호용).
CH_STD = np.maximum(train_ch.std(axis=0), 1e-3).astype(np.float32)

train_residual = train_targets - train_wind[:, -1:]
RESIDUAL_MEAN = train_residual.mean(axis=0).astype(np.float32)
RESIDUAL_STD = (train_residual.std(axis=0) + 1e-6).astype(np.float32)
CLIP_LOW = float(train_targets.min() * 0.95)
CLIP_HIGH = float(train_targets.max() * 1.05)

# 보조 손실의 양성 가중치. CH 는 원반 면적의 몇 % 뿐이라 그냥 두면 전부 0 으로 눌립니다.
_ch_positive_rate = float(train_ch.mean())
AUX_POS_WEIGHT = float(np.clip((1.0 - _ch_positive_rate) / max(_ch_positive_rate, 1e-3), 1.0, 10.0))


def image_index_matrix(inputs, image_index):
    return np.asarray([
        [image_index[name] for name in row]
        for row in inputs[IMAGE_COLUMNS].itertuples(index=False, name=None)
    ], dtype=np.int32)


def compute_disk_median(image_array, chunk=256):
    # 이미지·채널별 원반 내부 중앙값. 마스크 라벨을 매 배치에서 즉석으로 만들기 위한 상수입니다.
    result = np.zeros((len(image_array), len(CHANNELS)), dtype=np.float32)
    for start in range(0, len(image_array), chunk):
        block = np.asarray(image_array[start:start + chunk], dtype=np.float32)
        result[start:start + chunk] = np.median(block[:, :, DISK_MASK], axis=2)
    return result


def cached_disk_median(split, image_array):
    path = CACHE_ROOT / f"median_{split}_{IMAGE_SIZE}_{DISK_MARGIN}.npy"
    if path.exists():
        median = np.load(path)
        if median.shape == (len(image_array), len(CHANNELS)):
            print(f"reusing median cache: {path.name}")
            return median.astype(np.float32)
    median = compute_disk_median(image_array)
    np.save(path, median)
    print(f"created median cache: {path.name}  shape={median.shape}")
    return median


train_median = cached_disk_median("train", train_image_array)
val_median = cached_disk_median("validation", val_image_array)
test_median = cached_disk_median("test", test_image_array)

print(f"\n프레임: 20 -> {N_FRAMES}개 (stride {FRAME_STRIDE}), 인덱스 {FRAME_INDEX.tolist()}")
print(f"마스크: {MASK_SIZE}x{MASK_SIZE} (원본 {IMAGE_SIZE}px 를 {MASK_FACTOR}배 축소), "
      f"원반 점유 {DISK_WEIGHT.mean():.1%}")
print(f"CH 양성 비율 {_ch_positive_rate:.4f} -> AUX_POS_WEIGHT = {AUX_POS_WEIGHT:.2f}")
print(f"residual std by horizon: {np.round(RESIDUAL_STD, 1)}")
print(f"clip range: [{CLIP_LOW:.1f}, {CLIP_HIGH:.1f}] km/s")
'''


DATASET = r'''
STAT_NAMES = ["last", "mean4", "mean", "std", "min", "max", "slope", "last_minus_mean4", "range"]
_TIME_CENTERED = np.arange(20, dtype=np.float32) - 9.5
_TIME_DENOMINATOR = float((_TIME_CENTERED ** 2).sum())


def build_wind_stats(wind):
    last = wind[:, -1]
    mean4 = wind[:, -4:].mean(axis=1)
    slope = (wind - wind.mean(axis=1, keepdims=True)) @ _TIME_CENTERED / _TIME_DENOMINATOR
    return np.stack([last, mean4, wind.mean(axis=1), wind.std(axis=1), wind.min(axis=1),
                     wind.max(axis=1), slope, last - mean4,
                     wind.max(axis=1) - wind.min(axis=1)], axis=1).astype(np.float32)


train_stats_raw = build_wind_stats(train_wind)
STATS_MEAN = train_stats_raw.mean(axis=0).astype(np.float32)
STATS_STD = (train_stats_raw.std(axis=0) + 1e-6).astype(np.float32)
NUM_STATS = len(STAT_NAMES)


class SolarWindDataset(Dataset):
    """영상은 uint8 그대로 넘기고 정규화는 GPU 에서 합니다 (전송량 1/4).

    코로나홀 마스크 라벨은 여기서 즉석으로 만듭니다. 미리 캐시한 원반 중앙값
    (`disk_median`) 덕분에 임계값 비교 한 번이면 끝나서 loader 병목이 되지 않습니다.
    """

    def __init__(self, image_array, image_index, inputs, wind, wind_valid,
                 ch_grid, disk_median, targets=None, training=False):
        self.training = training
        self.image_array = image_array
        self.image_indexes = image_index_matrix(inputs, image_index)[:, FRAME_INDEX]
        self.sample_ids = inputs.sample_id.to_numpy()
        self.disk_median = disk_median
        self.last_wind = np.ascontiguousarray(wind[:, -1]).astype(np.float32)
        self.wind_seq = np.stack([
            (wind - WIND_MEAN) / WIND_STD,
            np.diff(wind, axis=1, prepend=wind[:, :1]) / DIFF_STD,
            wind_valid,
        ], axis=2).astype(np.float32)
        self.wind_stats = ((build_wind_stats(wind) - STATS_MEAN) / STATS_STD).astype(np.float32)
        # 격자 브랜치는 대조군일 때만 메모리를 씁니다.
        self.ch_seq = (((ch_grid[self.image_indexes] - CH_MEAN) / CH_STD).astype(np.float32)
                       if USE_CH_GRID else None)
        self.targets = targets.astype(np.float32) if targets is not None else None

    def __len__(self):
        return len(self.sample_ids)

    def _coronal_hole_mask(self, images, indexes):
        # P3 와 같은 규칙: 193 과 211 두 채널 모두에서 원반 중앙값의 CH_THRESHOLD_RATIO 미만.
        threshold = CH_THRESHOLD_RATIO * self.disk_median[indexes]        # (F, 2)
        dark = images < threshold[:, :, None, None]
        hole = np.logical_and(dark[:, 0], dark[:, 1]) & DISK_MASK         # (F, H, W)
        return block_mean(hole.astype(np.float32), MASK_FACTOR)           # (F, M, M) 면적 비율

    def __getitem__(self, item):
        indexes = self.image_indexes[item]
        if USE_UNET:
            images = np.asarray(self.image_array[indexes], dtype=np.uint8)   # (F, 2, H, W)
            ch_mask = (self._coronal_hole_mask(images, indexes) if AUX_ACTIVE
                       else np.zeros((1, 1, 1), dtype=np.float32))
        else:
            images = np.zeros((1, 1, 1, 1), dtype=np.uint8)
            ch_mask = np.zeros((1, 1, 1), dtype=np.float32)

        if USE_CH_GRID:
            ch_seq = self.ch_seq[item]
            if self.training and AUGMENT and AUG_CH_NOISE > 0:
                ch_seq = ch_seq * (1.0 + np.random.normal(
                    0, AUG_CH_NOISE, ch_seq.shape)).astype(np.float32)
        else:
            ch_seq = np.zeros((1, 1), dtype=np.float32)

        result = {
            "images": torch.from_numpy(np.ascontiguousarray(images)),
            "ch_mask": torch.from_numpy(np.ascontiguousarray(ch_mask)),
            "wind_seq": torch.from_numpy(self.wind_seq[item]),
            "wind_stats": torch.from_numpy(self.wind_stats[item]),
            "ch_seq": torch.from_numpy(np.ascontiguousarray(ch_seq)),
            "last_wind": torch.tensor(self.last_wind[item]),
            "sample_id": self.sample_ids[item],
        }
        if self.targets is not None:
            result["target"] = torch.from_numpy(self.targets[item])
        return result


def seed_worker(worker_id):
    worker_seed = (SEED + worker_id) % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_loader(dataset, shuffle):
    options = dict(dataset=dataset, batch_size=BATCH_SIZE, shuffle=shuffle,
                   num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=False,
                   worker_init_fn=seed_worker,
                   generator=torch.Generator().manual_seed(SEED))
    if NUM_WORKERS > 0:
        options.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(**options)


train_dataset = SolarWindDataset(train_image_array, train_image_index, train_inputs,
                                 train_wind, train_wind_valid, train_ch, train_median,
                                 train_targets, training=True)
val_dataset = SolarWindDataset(val_image_array, val_image_index, val_inputs,
                               val_wind, val_wind_valid, val_ch, val_median, val_targets)
train_loader = make_loader(train_dataset, shuffle=True)
val_loader = make_loader(val_dataset, shuffle=False)

batch = next(iter(train_loader))
if USE_UNET:
    assert batch["images"].shape[1:] == (N_FRAMES, len(CHANNELS), IMAGE_SIZE, IMAGE_SIZE)
    assert batch["images"].dtype == torch.uint8
if AUX_ACTIVE:
    assert batch["ch_mask"].shape[1:] == (N_FRAMES, MASK_SIZE, MASK_SIZE)
    print(f"CH 마스크 라벨 — 평균 면적 비율 {batch['ch_mask'].mean():.4f}, "
          f"최대 {batch['ch_mask'].max():.3f}")
print({k: tuple(v.shape) for k, v in batch.items() if torch.is_tensor(v)})
'''


MODEL_MARKDOWN = r'''## 6. 모델 — U-Net 프레임 인코더 + GRU

```
images (B, F, 2, 128, 128)
   └─ U-Net (프레임마다 가중치 공유, B·F 를 배치축으로 접어서 한 번에)
        ├─ 인코더 64 → 32 → 16 → 8            (base, 2·base, 4·base, 8·base)
        ├─ 디코더 8 → 16 → 32 → 64 + skip     → CH 로짓맵 (B·F, 1, 64, 64)
        └─ 임베딩 = [bottleneck 전역평균 ‖ CH확률 가중평균]  → Linear → 96차
   └─ GRU(96 → 128)  ← P3 의 LSTM 자리
wind_seq  → GRU(3 → 96, 2층)          (P3 와 동일)
wind_stats→ MLP(9 → 64)               (P3 와 동일)
   └─ concat → horizon 12개에 가중치 공유하는 head → 잔차 12개
```

**CH 확률 가중 평균**이 이 구조의 핵심이다. 디코더가 내놓은 확률맵을 bottleneck
해상도(8×8)로 줄여 가중치로 쓰므로, 회귀에 흘러가는 특징이 코로나홀 쪽으로 쏠린다.
가중치는 학습되는 값이라 "코로나홀 면적"이라는 한 가지 요약에 갇히지 않는다.

bottleneck 에는 학습되는 **위치 임베딩**을 더한다. 원반 위 어디가 중요한지(동/서
림, 중앙자오선)를 손으로 정하지 않고 모델이 고르게 하려는 것이다 — P10 이 중앙자오선
밴드로 좁혔다가 2.98 잃었던 것에 대한 대응이다.
'''


MODEL = r'''
def upsample2(tensor):
    return F.interpolate(tensor, scale_factor=2, mode="bilinear", align_corners=False)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.SiLU(inplace=True))

    def forward(self, x):
        return self.block(x)


class UNetFrameEncoder(nn.Module):
    """프레임 한 장 -> (임베딩, 코로나홀 로짓맵). 20 프레임이 같은 가중치를 씁니다."""

    def __init__(self, base=UNET_BASE):
        super().__init__()
        widths = (base, base * 2, base * 4, base * 8)
        self.stem = nn.Sequential(
            nn.Conv2d(len(CHANNELS), widths[0], 3, stride=UNET_STEM_STRIDE,
                      padding=1, bias=False),
            nn.BatchNorm2d(widths[0]), nn.SiLU(inplace=True))
        self.pool = nn.MaxPool2d(2)
        self.encode1 = DoubleConv(widths[0], widths[0])
        self.encode2 = DoubleConv(widths[0], widths[1])
        self.encode3 = DoubleConv(widths[1], widths[2])
        self.bottleneck = DoubleConv(widths[2], widths[3])
        self.decode3 = DoubleConv(widths[3] + widths[2], widths[2])
        self.decode2 = DoubleConv(widths[2] + widths[1], widths[1])
        self.decode1 = DoubleConv(widths[1] + widths[0], widths[0])
        self.ch_head = nn.Conv2d(widths[0], 1, 1)
        # 원반 위 위치를 모델이 스스로 구분할 수 있게 하는 학습 가능한 위치 임베딩
        self.position = nn.Parameter(torch.zeros(1, widths[3], MASK_SIZE // 8, MASK_SIZE // 8))
        self.project = nn.Sequential(
            nn.Linear(widths[3] * 2, IMAGE_EMBED), nn.SiLU(inplace=True))

    def forward(self, x):
        first = self.encode1(self.stem(x))                    # (N, base,   M,   M)
        second = self.encode2(self.pool(first))               # (N, 2base,  M/2, M/2)
        third = self.encode3(self.pool(second))               # (N, 4base,  M/4, M/4)
        deep = self.bottleneck(self.pool(third)) + self.position   # (N, 8base, M/8, M/8)
        up3 = self.decode3(torch.cat([upsample2(deep), third], dim=1))
        up2 = self.decode2(torch.cat([upsample2(up3), second], dim=1))
        up1 = self.decode1(torch.cat([upsample2(up2), first], dim=1))
        logits = self.ch_head(up1)                            # (N, 1, M, M)

        deep32 = deep.float()
        weight = F.adaptive_avg_pool2d(torch.sigmoid(logits.float()), deep32.shape[-2:])
        # CH 확률로 가중한 평균과 전역 평균을 함께 넘깁니다.
        focused = (deep32 * weight).sum(dim=(2, 3)) / (weight.sum(dim=(2, 3)) + 1e-3)
        return self.project(torch.cat([deep32.mean(dim=(2, 3)), focused], dim=1)), logits


class SolarWindP17(nn.Module):
    def __init__(self):
        super().__init__()
        shared_dim = 0

        self.wind_gru = nn.GRU(3, 96, num_layers=2, batch_first=True)
        self.stats_encoder = nn.Sequential(
            nn.Linear(NUM_STATS, 128), nn.SELU(inplace=True),
            nn.Linear(128, 64), nn.SELU(inplace=True))
        shared_dim += 96 + 64

        if USE_CH_GRID:
            self.ch_gru = nn.GRU(N_CELLS, 64, num_layers=2, batch_first=True)
            self.ch_dropout = nn.Dropout(DROPOUT)
            shared_dim += 64

        if USE_UNET:
            self.frame_encoder = UNetFrameEncoder()
            # P3 의 image_lstm 자리. GRU 로 바꿨습니다.
            self.image_gru = nn.GRU(IMAGE_EMBED, IMAGE_GRU_HIDDEN, batch_first=True)
            self.image_dropout = nn.Dropout(DROPOUT)
            shared_dim += IMAGE_GRU_HIDDEN

        self.horizon_embedding = nn.Parameter(torch.randn(12, HORIZON_EMBED) * 0.1)
        head_input = shared_dim + HORIZON_EMBED
        # head 는 12 horizon 에 동일 가중치로 적용됩니다 (nn.Linear 는 마지막 축에만 작용).
        self.head = nn.Sequential(
            nn.Linear(head_input, 192), nn.ReLU(inplace=True), nn.Dropout(DROPOUT),
            nn.Linear(192, 96), nn.ReLU(inplace=True), nn.Dropout(DROPOUT),
            nn.Linear(96, 1))

        self.register_buffer("residual_mean", torch.as_tensor(RESIDUAL_MEAN))
        self.register_buffer("residual_std", torch.as_tensor(RESIDUAL_STD))
        print(f"shared_dim={shared_dim}  head_input={head_input}")

    def forward(self, images, wind_seq, wind_stats, ch_seq):
        _, wind_hidden = self.wind_gru(wind_seq)
        parts = [F.relu(wind_hidden[-1]), self.stats_encoder(wind_stats)]
        ch_logits = None

        if USE_CH_GRID:
            _, ch_hidden = self.ch_gru(ch_seq)
            parts.append(self.ch_dropout(F.relu(ch_hidden[-1])))

        if USE_UNET:
            batch_size, frames = images.shape[0], images.shape[1]
            flat = images.reshape(batch_size * frames, *images.shape[2:])
            embed, logits = self.frame_encoder(flat)
            _, image_hidden = self.image_gru(embed.reshape(batch_size, frames, IMAGE_EMBED))
            parts.append(self.image_dropout(F.relu(image_hidden[-1])))
            ch_logits = logits.reshape(batch_size, frames, MASK_SIZE, MASK_SIZE)

        shared = torch.cat(parts, dim=1)                                   # (B, D)
        batch_size = shared.shape[0]
        expanded = shared.unsqueeze(1).expand(batch_size, 12, shared.shape[1])
        embedding = self.horizon_embedding.unsqueeze(0).expand(batch_size, 12, HORIZON_EMBED)
        z = self.head(torch.cat([expanded, embedding], dim=2)).squeeze(-1)  # (B, 12)
        return z * self.residual_std + self.residual_mean, ch_logits


def build_model():
    return SolarWindP17().to(DEVICE)


model = build_model()
_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("trainable parameters:", f"{_total:,}")
if USE_UNET:
    _unet = sum(p.numel() for p in model.frame_encoder.parameters())
    print(f"  그 중 U-Net: {_unet:,} ({_unet / _total:.0%})")
'''


AUX = r'''

# --- 보조 손실 · GPU 정규화 · 증강 ------------------------------------
IMAGE_MEAN_GPU = torch.as_tensor(IMAGE_MEAN, device=DEVICE).view(1, 1, len(CHANNELS), 1, 1)
IMAGE_STD_GPU = torch.as_tensor(IMAGE_STD, device=DEVICE).view(1, 1, len(CHANNELS), 1, 1)
# 원반 밖도 0 으로 눌러 둡니다. 안 그러면 attention 이 배경 구석으로 새어 나갑니다.
AUX_PIXEL_WEIGHT = torch.as_tensor(
    DISK_WEIGHT + AUX_OFF_DISK_WEIGHT * (1.0 - DISK_WEIGHT), device=DEVICE)
AUX_POS_WEIGHT_GPU = torch.tensor(AUX_POS_WEIGHT, device=DEVICE)


def normalize_images(raw):
    images = raw.to(DEVICE, non_blocking=PIN_MEMORY).float().div_(255.0)
    return (images - IMAGE_MEAN_GPU) / IMAGE_STD_GPU


def aux_ch_loss(logits, target):
    # 소프트 타깃(셀별 CH 면적 비율)에 대한 가중 BCE.
    weight = AUX_PIXEL_WEIGHT.expand_as(target)
    loss = F.binary_cross_entropy_with_logits(
        logits.float(), target, weight=weight,
        pos_weight=AUX_POS_WEIGHT_GPU, reduction="sum")
    return loss / weight.sum()


def augment_batch(images, ch_mask):
    # GPU 에서 수행합니다. numpy 증강은 CPU 병목으로 epoch 시간이 4배 늘었습니다.
    # 기하 변환은 마스크 라벨에 **같은 양** 적용해야 정합이 깨지지 않습니다.
    if AUG_SHIFT_PIXELS >= MASK_FACTOR:
        limit = AUG_SHIFT_PIXELS // MASK_FACTOR
        shift_y = int(torch.randint(-limit, limit + 1, (1,)).item())
        shift_x = int(torch.randint(-limit, limit + 1, (1,)).item())
        images = torch.roll(images, shifts=(shift_y * MASK_FACTOR, shift_x * MASK_FACTOR),
                            dims=(3, 4))
        if ch_mask is not None:
            ch_mask = torch.roll(ch_mask, shifts=(shift_y, shift_x), dims=(2, 3))
    count = images.shape[0]
    scale = 1.0 + (torch.rand(count, 1, 1, 1, 1, device=images.device) * 2 - 1) * AUG_BRIGHTNESS
    offset = torch.randn(count, 1, 1, 1, 1, device=images.device) * AUG_BRIGHTNESS
    images = images * scale + offset
    if AUG_NOISE_STD > 0:
        images = images + torch.randn_like(images) * AUG_NOISE_STD
    if AUG_ERASE_PROB > 0:
        size = max(MASK_FACTOR * 2, IMAGE_SIZE // 8)
        selected = torch.rand(count, device=images.device) < AUG_ERASE_PROB
        if bool(selected.any()):
            # 마스크와 정확히 대응하도록 MASK_FACTOR 배수 위치에서만 지웁니다.
            cells = (IMAGE_SIZE - size) // MASK_FACTOR + 1
            top = int(torch.randint(0, cells, (1,)).item()) * MASK_FACTOR
            left = int(torch.randint(0, cells, (1,)).item()) * MASK_FACTOR
            images[selected, :, :, top:top + size, left:left + size] = 0.0
            if ch_mask is not None:
                # 지운 자리는 평균 밝기가 되므로 코로나홀이 아닙니다 -> 라벨도 0.
                ch_mask = ch_mask.clone()
                ch_mask[selected, :, top // MASK_FACTOR:(top + size) // MASK_FACTOR,
                        left // MASK_FACTOR:(left + size) // MASK_FACTOR] = 0.0
    return images, ch_mask

'''


TRAIN = r'''
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

model = build_model()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=SCHEDULER_PATIENCE, min_lr=1e-6)
scaler = torch.amp.GradScaler(DEVICE.type, enabled=USE_AMP)

checkpoint_path = OUTPUT_DIR / "best_model.pth"
if checkpoint_path.exists():
    checkpoint_path.unlink()
best_val_score = float("inf")
epochs_without_improvement = 0
history = []

CONFIG = {"image_size": IMAGE_SIZE, "channels": list(CHANNELS), "frame_stride": FRAME_STRIDE,
          "use_unet": USE_UNET, "use_ch_aux": USE_CH_AUX, "use_ch_grid": USE_CH_GRID,
          "unet_base": UNET_BASE, "unet_stem_stride": UNET_STEM_STRIDE, "mask_size": MASK_SIZE,
          "image_embed": IMAGE_EMBED, "image_gru_hidden": IMAGE_GRU_HIDDEN,
          "aux_ch_weight": AUX_CH_WEIGHT, "aux_pos_weight": AUX_POS_WEIGHT,
          "ch_grid": list(CH_GRID), "ch_threshold_ratio": CH_THRESHOLD_RATIO,
          "disk": [DISK_Y, DISK_X, DISK_R], "seed": SEED,
          "image_mean": IMAGE_MEAN.tolist(), "image_std": IMAGE_STD.tolist(),
          "wind_mean": WIND_MEAN, "wind_std": WIND_STD, "diff_std": DIFF_STD,
          "ch_mean": CH_MEAN.tolist(), "ch_std": CH_STD.tolist(),
          "stats_mean": STATS_MEAN.tolist(), "stats_std": STATS_STD.tolist(),
          "residual_mean": RESIDUAL_MEAN.tolist(), "residual_std": RESIDUAL_STD.tolist(),
          "clip_low": CLIP_LOW, "clip_high": CLIP_HIGH,
          "initialization": "random_from_scratch"}


def run_epoch(loader, training):
    model.train(training)
    squared_error_sum = np.zeros(12, dtype=np.float64)
    aux_sum, batch_count, sample_count = 0.0, 0, 0
    for batch in loader:
        images = normalize_images(batch["images"]) if USE_UNET else None
        ch_mask = (batch["ch_mask"].to(DEVICE, non_blocking=PIN_MEMORY)
                   if AUX_ACTIVE else None)
        wind_seq = batch["wind_seq"].to(DEVICE, non_blocking=PIN_MEMORY)
        wind_stats = batch["wind_stats"].to(DEVICE, non_blocking=PIN_MEMORY)
        ch_seq = batch["ch_seq"].to(DEVICE, non_blocking=PIN_MEMORY)
        last_wind = batch["last_wind"].to(DEVICE, non_blocking=PIN_MEMORY)
        target = batch["target"].to(DEVICE, non_blocking=PIN_MEMORY)

        if training and USE_UNET and AUGMENT:
            images, ch_mask = augment_batch(images, ch_mask)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.amp.autocast(DEVICE.type, enabled=USE_AMP):
                residual, ch_logits = model(images, wind_seq, wind_stats, ch_seq)
            prediction = residual.float() + last_wind.unsqueeze(1)
            loss = metric_loss(prediction, target)
            if AUX_ACTIVE and ch_logits is not None:
                auxiliary = aux_ch_loss(ch_logits, ch_mask)
                loss = loss + AUX_CH_WEIGHT * auxiliary
                aux_sum += float(auxiliary.detach())
                batch_count += 1
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()

        error = (prediction.detach() - target).double()
        squared_error_sum += torch.sum(error ** 2, dim=0).cpu().numpy()
        sample_count += error.shape[0]
    per_horizon = np.sqrt(squared_error_sum / sample_count)
    return float(per_horizon.mean()), per_horizon, aux_sum / max(batch_count, 1)


for epoch in range(1, EPOCHS + 1):
    started = time.perf_counter()
    train_score, _, train_aux = run_epoch(train_loader, training=True)
    with torch.no_grad():
        val_score, _, val_aux = run_epoch(val_loader, training=False)
    scheduler.step(val_score)
    learning_rate = optimizer.param_groups[0]["lr"]
    elapsed = time.perf_counter() - started
    history.append({"epoch": epoch, "train_rmse": train_score, "val_rmse": val_score,
                    "train_aux": train_aux, "val_aux": val_aux,
                    "learning_rate": learning_rate, "seconds": elapsed})
    marker = ""
    if val_score < best_val_score:
        best_val_score = val_score
        epochs_without_improvement = 0
        torch.save({"model_state_dict": model.state_dict(), "epoch": epoch,
                    "val_official_rmse": val_score, **CONFIG}, checkpoint_path)
        marker = "  <- best"
    else:
        epochs_without_improvement += 1
    print(f"epoch={epoch:03d} train={train_score:7.3f} val={val_score:7.3f} "
          f"aux={train_aux:6.4f}/{val_aux:6.4f} lr={learning_rate:.2e} "
          f"{elapsed:6.1f}s{marker}", flush=True)
    if epochs_without_improvement >= EARLY_STOP_PATIENCE:
        print("early stopping")
        break

history_frame = pd.DataFrame(history)
history_frame.to_csv(OUTPUT_DIR / "history.csv", index=False)
figure, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history_frame.epoch, history_frame.train_rmse, label="train")
axes[0].plot(history_frame.epoch, history_frame.val_rmse, label="validation")
axes[0].axhline(persistence_score, color="gray", linestyle="--", label="persistence")
axes[0].set_xlabel("epoch"); axes[0].set_ylabel("official RMSE (km/s)")
axes[0].grid(alpha=0.3); axes[0].legend()
axes[1].plot(history_frame.epoch, history_frame.train_aux, label="train")
axes[1].plot(history_frame.epoch, history_frame.val_aux, label="validation")
axes[1].set_xlabel("epoch"); axes[1].set_ylabel("CH 보조 BCE")
axes[1].set_title("보조 감독이 실제로 학습되는지 확인")
axes[1].grid(alpha=0.3); axes[1].legend()
plt.tight_layout(); plt.savefig(OUTPUT_DIR / "learning_curve.png", dpi=140); plt.show()

checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
model.load_state_dict(checkpoint["model_state_dict"])
print(f"best epoch {checkpoint['epoch']}  val official RMSE {checkpoint['val_official_rmse']:.3f}")
'''


VALIDATE = r'''
@torch.no_grad()
def predict(loader):
    model.eval()
    predictions, sample_ids = [], []
    for batch in loader:
        images = normalize_images(batch["images"]) if USE_UNET else None
        wind_seq = batch["wind_seq"].to(DEVICE, non_blocking=PIN_MEMORY)
        wind_stats = batch["wind_stats"].to(DEVICE, non_blocking=PIN_MEMORY)
        ch_seq = batch["ch_seq"].to(DEVICE, non_blocking=PIN_MEMORY)
        last_wind = batch["last_wind"].to(DEVICE, non_blocking=PIN_MEMORY)
        with torch.amp.autocast(DEVICE.type, enabled=USE_AMP):
            residual, _ = model(images, wind_seq, wind_stats, ch_seq)
        prediction = (residual.float() + last_wind.unsqueeze(1)).clamp(CLIP_LOW, CLIP_HIGH)
        predictions.append(prediction.cpu().numpy())
        sample_ids.extend(batch["sample_id"])
    return np.concatenate(predictions).astype(np.float64), sample_ids


validation_prediction, validation_ids = predict(val_loader)
assert validation_ids == val_inputs.sample_id.tolist()
model_score, _ = official_rmse(val_targets, validation_prediction)
validation_metrics = metrics_by_horizon(val_targets, validation_prediction, val_persistence)
validation_metrics.to_csv(OUTPUT_DIR / "validation_metrics.csv", index=False)

print(f"P17 OFFICIAL_VALIDATION_RMSE     : {model_score:8.3f} km/s")
print(f"pooled RMSE (전체 원소)          : {pooled_rmse(val_targets, validation_prediction):8.3f} km/s")
print(f"persistence 공식 RMSE            : {persistence_score:8.3f} km/s")
print(f"persistence 대비 개선             : {persistence_score - model_score:8.3f} km/s"
      f"  ({(persistence_score - model_score) / persistence_score:.1%})")
print("\n[참고] P3 val = 64.203 (Public 58.8028) / P1 val = 68.408")
print("[주의] val 계측 노이즈 바닥은 약 3 km/s 다. 그 안의 차이는 순위 근거가 못 된다.")
if model_score >= persistence_score:
    print("\n>>> 경고: persistence 미달. 제출하지 마세요.")

figure, axis = plt.subplots(figsize=(7, 4))
axis.plot(validation_metrics.horizon_h, validation_metrics.rmse, marker="o", label="model")
axis.plot(validation_metrics.horizon_h, validation_metrics.persistence_rmse,
          marker="s", linestyle="--", label="persistence")
axis.set_xlabel("forecast horizon (h)"); axis.set_ylabel("RMSE (km/s)")
axis.grid(alpha=0.3); axis.legend()
plt.tight_layout(); plt.show()

# 보조 감독이 실제로 코로나홀을 찾고 있는지 눈으로 확인합니다.
if AUX_ACTIVE:
    sample = next(iter(val_loader))
    with torch.no_grad(), torch.amp.autocast(DEVICE.type, enabled=USE_AMP):
        _, sample_logits = model(normalize_images(sample["images"]),
                                 sample["wind_seq"].to(DEVICE),
                                 sample["wind_stats"].to(DEVICE),
                                 sample["ch_seq"].to(DEVICE))
    probability = torch.sigmoid(sample_logits.float())[0, -1].cpu().numpy()
    label = sample["ch_mask"][0, -1].numpy()
    figure, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].imshow(sample["images"][0, -1, 0].numpy(), cmap="gray")
    axes[0].set_title("입력 193 (T0 프레임)")
    axes[1].imshow(label, cmap="magma", vmin=0, vmax=1)
    axes[1].set_title("임계값 라벨 (P3 규칙)")
    axes[2].imshow(probability, cmap="magma", vmin=0, vmax=1)
    axes[2].set_title("U-Net 예측 CH 확률")
    for axis in axes:
        axis.set_xticks([]); axis.set_yticks([])
    plt.tight_layout(); plt.show()
    print(f"라벨 평균 {label.mean():.4f} / 예측 평균 {probability.mean():.4f}")

validation_metrics
'''


SUBMIT = r'''
del train_loader, val_loader, train_dataset, val_dataset
gc.collect()
if DEVICE.type == "cuda":
    torch.cuda.empty_cache()

test_dataset = SolarWindDataset(test_image_array, test_image_index, test_inputs,
                                test_wind, test_wind_valid, test_ch, test_median,
                                targets=None)
test_loader = make_loader(test_dataset, shuffle=False)
test_prediction, predicted_ids = predict(test_loader)

assert predicted_ids == test_inputs.sample_id.tolist()
assert test_prediction.shape == (len(test_inputs), 12)
assert np.isfinite(test_prediction).all()

submission = pd.DataFrame(test_prediction, columns=TARGET_COLUMNS)
submission.insert(0, "sample_id", predicted_ids)
submission.to_csv(SUBMISSION_DIR / "submission.csv", index=False)
shutil.copyfile(checkpoint_path, SUBMISSION_DIR / "model.pth")

# 규정: "code.ipynb 에서 model.pth 를 불러와 추론이 가능해야 함" 을 문자 그대로 충족
saved = torch.load(SUBMISSION_DIR / "model.pth", map_location=DEVICE, weights_only=True)
model.load_state_dict(saved["model_state_dict"])
print("reloaded from submission/model.pth")

print("saved:", (SUBMISSION_DIR / "submission.csv").resolve(), submission.shape)
print(submission[TARGET_COLUMNS].describe().loc[["mean", "std", "min", "max"]].round(1))
del test_dataset, test_loader
gc.collect()
if DEVICE.type == "cuda":
    torch.cuda.empty_cache()
submission.head()
'''


def build() -> None:
    notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
    source = notebook["cells"]

    # P3 설정 셀의 앞부분(imports · seed · DATA_ROOT 탐색 · 디렉터리)은 그대로 쓴다.
    config_source = "".join(source[CONFIG_CELL]["source"])
    anchor = "\nIMAGE_SIZE = 128"
    if anchor not in config_source or "outputs_p3" not in config_source:
        raise RuntimeError("P3 config anchor not found")
    prefix = config_source[:config_source.index(anchor)].replace("outputs_p3", "outputs_p17")

    # P3 지표 셀에서 augment_batch(P17 이 새로 쓴다)만 들어내고 나머지는 유지한다.
    metric_source = "".join(source[METRIC_CELL]["source"])
    for mark in ("def augment_batch", "val_persistence = "):
        if mark not in metric_source:
            raise RuntimeError(f"P3 metric anchor {mark!r} not found")
    metric_source = (metric_source[:metric_source.index("def augment_batch")].rstrip("\n")
                     + "\n" + AUX + "\n"
                     + metric_source[metric_source.index("val_persistence = "):])

    cells = [
        md(HEADER),
        dict(source[1]), code(prefix + CONFIG.lstrip("\n")),
        dict(source[3]), dict(source[4]),
        dict(source[5]), dict(source[6]),
        md(DISK_MARKDOWN), dict(source[8]),
        md("## 4. 정규화 통계 · 코로나홀 마스크 라벨 준비"), code(STATS),
        md("## 5. Dataset"), code(DATASET),
        md(MODEL_MARKDOWN), code(MODEL),
        md("## 7. 지표 · 손실 · 증강"), code(metric_source),
        md("## 8. 학습"), code(TRAIN),
        md("## 9. Validation 평가"), code(VALIDATE),
        md("## 10. Test 추론 · 제출 파일 생성"), code(SUBMIT),
        dict(source[23]), dict(source[24]),
    ]

    notebook["cells"] = cells
    TARGET.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    TRIAL_TARGET.parent.mkdir(parents=True, exist_ok=True)
    TRIAL_TARGET.write_text(TARGET.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"created {TARGET.name} and {TRIAL_TARGET.relative_to(HERE)} ({len(cells)} cells)")


if __name__ == "__main__":
    build()
