# =====================================================================
# EDA — 결측 실태 · 시간 순서 검증
#
# 서버 Jupyter 에 셀 하나로 붙여넣어 실행한다.
# 출력은 print 요약뿐이다. 원본 데이터를 복사·압축·분할 저장하지 않는다.
#
# 답하려는 것:
#   A. 결측이 정말 없는가 (NaN 이 아닌 형태로 숨어 있지 않은가)
#   B. 태양 이미지가 시간 순서로 학습에 들어가는가 (방향·정렬 오프셋까지)
#
# 2026-08-20 실행 결과 요약 (전문은 CODE_REPORT.md §5.6):
#   결측 없음 · 이미지 완전 · _19 = T0 확정 · 정렬 오프셋 없음(물리 봉우리 +96h)
#   부산물 — valid 플래그는 상수(죽은 채널), 186h 음의 봉우리는 아티팩트
# =====================================================================
from pathlib import Path
import os

import numpy as np
import pandas as pd
from PIL import Image

DATA_ROOT_CANDIDATES = [
    Path(os.getenv("SW_DATA_ROOT", "")) if os.getenv("SW_DATA_ROOT") else None,
    Path("public_dataset/competition_dataset_6h"),
    Path("/home/jovyan/public_dataset/competition_dataset_6h"),
    Path("public/public_dataset/competition_dataset_6h"),
    Path("/home/jovyan/public/public_dataset/competition_dataset_6h"),
    Path("dataset"), Path("/home/jovyan/dataset"),
]
DATA_ROOT = None
for candidate in DATA_ROOT_CANDIDATES:
    if candidate is not None and (candidate / "train/inputs.csv").exists():
        DATA_ROOT = candidate
        break
if DATA_ROOT is None:
    raise FileNotFoundError("데이터 경로 없음")

IMAGE_COLUMNS = [f"image_{i:02d}" for i in range(20)]
WIND_COLUMNS = [f"wind_{i:02d}" for i in range(20)]
TARGET_COLUMNS = [f"target_{i:02d}" for i in range(12)]
CHANNELS = ("193", "211")
AU_KM = 1.496e8

SPLITS = {
    "train": ("train", "train/targets.csv"),
    "validation": ("validation", "validation/targets.csv"),
    "test": ("test", None),
}
inputs_by_split, targets_by_split = {}, {}
for name, (folder, target_path) in SPLITS.items():
    inputs_by_split[name] = pd.read_csv(DATA_ROOT / folder / "inputs.csv")
    targets_by_split[name] = (pd.read_csv(DATA_ROOT / target_path)[TARGET_COLUMNS]
                              .to_numpy(np.float64) if target_path else None)

print("data:", DATA_ROOT.resolve())
print({k: len(v) for k, v in inputs_by_split.items()})


# =====================================================================
# A1. 결측 — NaN 이 아닌 형태까지 뒤진다
# =====================================================================
# 태양풍 속도의 물리적 범위는 대략 250~900 km/s 다. 이를 벗어나면 결측 sentinel 을 의심한다.
PLAUSIBLE_LOW, PLAUSIBLE_HIGH = 200.0, 1200.0
SENTINEL_CANDIDATES = (0.0, -1.0, -999.0, 999.0, -9999.0, 9999.0, 99999.0, 999.9)

print("\n" + "=" * 70)
print("A1. 결측 실태 — NaN · sentinel · 물리 범위")
print("=" * 70)

for name, inputs in inputs_by_split.items():
    wind = inputs[WIND_COLUMNS].to_numpy(np.float64)
    flat = wind.ravel()
    print(f"\n--- {name} (wind {wind.shape}) ---")
    print(f"  NaN            : {int(np.isnan(flat).sum())}")
    print(f"  inf            : {int(np.isinf(flat).sum())}")
    finite = flat[np.isfinite(flat)]
    print(f"  범위           : {finite.min():.2f} ~ {finite.max():.2f}")
    print(f"  분위 (1/50/99) : {np.percentile(finite, 1):.1f} / "
          f"{np.percentile(finite, 50):.1f} / {np.percentile(finite, 99):.1f}")
    outside = int(((finite < PLAUSIBLE_LOW) | (finite > PLAUSIBLE_HIGH)).sum())
    print(f"  물리범위 밖    : {outside}  ({outside / finite.size:.4%})")
    hits = [(s, int((finite == s).sum())) for s in SENTINEL_CANDIDATES]
    hits = [(s, c) for s, c in hits if c]
    print(f"  sentinel 후보  : {hits if hits else '없음'}")
    # 값이 몇 개의 고유값으로 뭉쳐 있는지 — 상위 빈도값이 튀면 fill 흔적이다
    values, counts = np.unique(np.round(finite, 4), return_counts=True)
    top = np.argsort(counts)[::-1][:5]
    print(f"  최빈값 상위 5  : "
          + ", ".join(f"{values[i]:.2f}×{counts[i]}" for i in top))

    targets = targets_by_split[name]
    if targets is not None:
        tflat = targets.ravel()
        touts = int(((tflat < PLAUSIBLE_LOW) | (tflat > PLAUSIBLE_HIGH)).sum())
        print(f"  [target] NaN {int(np.isnan(tflat).sum())} | "
              f"범위 {np.nanmin(tflat):.1f}~{np.nanmax(tflat):.1f} | 물리범위 밖 {touts}")

print("\n>>> sentinel 후보에 유의미한 개수가 잡히거나 물리범위 밖이 존재하면,")
print("    현재 fill_wind() 의 np.isfinite 기반 결측 처리는 그것을 '정상값'으로 학습시키고 있다.")


# =====================================================================
# A2. 주최측이 이미 채워서 준 흔적 — 계단(반복값) 구간
# =====================================================================
# NaN 이 0개여도, 주최측이 forward-fill 로 메워서 줬다면 시간축에 '계단'이 남는다.
# 그건 우리 눈에는 정상값이지만 실제로는 관측이 아니다.
print("\n" + "=" * 70)
print("A2. 반복값(계단) 구간 — 이미 채워져 온 데이터인가")
print("=" * 70)


def reconstruct_frame_chains(inputs):
    """이미지 파일명만으로 프레임 시간축을 복원한다 (eda_p7 과 동일 알고리즘)."""
    images = inputs[IMAGE_COLUMNS].to_numpy()
    successor, predecessor = {}, {}
    for row in images:
        for current, following in zip(row[:-1], row[1:]):
            successor.setdefault(current, following)
            predecessor.setdefault(following, current)
    names = set(images.ravel().tolist())
    chains, visited = [], set()
    for head in sorted(names - set(predecessor)):
        chain, node = [], head
        while node is not None and node not in visited:
            visited.add(node)
            chain.append(node)
            node = successor.get(node)
        chains.append(chain)
    return chains


def frame_wind_map(inputs):
    """프레임 파일명 -> 그 프레임 시각의 태양풍 값. 샘플 간 불일치도 함께 센다."""
    images = inputs[IMAGE_COLUMNS].to_numpy()
    wind = inputs[WIND_COLUMNS].to_numpy(np.float64)
    accumulator = {}
    for row_images, row_wind in zip(images, wind):
        for name, value in zip(row_images, row_wind):
            accumulator.setdefault(name, []).append(value)
    mapping, disagree = {}, 0
    for name, values in accumulator.items():
        array = np.asarray(values)
        mapping[name] = float(np.nanmedian(array))
        if np.nanmax(array) - np.nanmin(array) > 1e-6:
            disagree += 1
    return mapping, disagree


def run_lengths(series):
    """인접 동일값 구간의 길이 분포."""
    if len(series) < 2:
        return np.array([])
    breaks = np.flatnonzero(np.diff(series) != 0)
    edges = np.concatenate([[-1], breaks, [len(series) - 1]])
    return np.diff(edges)


for name, inputs in inputs_by_split.items():
    chains = reconstruct_frame_chains(inputs)
    mapping, disagree = frame_wind_map(inputs)
    lengths = []
    for chain in chains:
        series = np.array([mapping[n] for n in chain])
        lengths.append(run_lengths(series))
    lengths = np.concatenate([l for l in lengths if len(l)])
    total_frames = sum(len(c) for c in chains)
    repeated = int(lengths[lengths >= 2].sum())
    print(f"\n--- {name} ---")
    print(f"  고유 프레임 {total_frames:,} | 사슬 {len(chains)}개 | "
          f"샘플 간 wind 불일치 프레임 {disagree}")
    print(f"  반복 구간에 속한 프레임 : {repeated:,}  ({repeated / total_frames:.2%})")
    print(f"  최장 반복 길이          : {int(lengths.max())} 스텝 "
          f"({int(lengths.max()) * 6}h)")
    histogram = {int(k): int(v) for k, v in
                 zip(*np.unique(lengths[lengths >= 2], return_counts=True))}
    print(f"  반복 길이 분포 (2 이상) : {dict(list(histogram.items())[:8])}")

print("\n>>> 🔴 '반복 비율' 은 판별 지표가 아니다. wind 는 1 km/s 정수로 양자화돼 있어서")
print("    인접 동일값의 기저율이 2% 안팎으로 원래 높다 (6h 변화 std 26.5 -> P(delta=0) ~ 1.5%).")
print(">>> 봐야 할 것은 **런 길이 분포의 꼬리**다.")
print("    forward-fill 이라면 실제 L1 공백(수 시간~수 일)을 메운 것이므로 4·5·10 스텝 런이 나온다.")
print("    최장 런이 3 이하이고 길이-3 이 한 자릿수면 자연스러운 양자화이고 결측은 없는 것이다.")
print(">>> 2026-08-20 실측: train {2: 223, 3: 1} -> fill 아님 확정. CODE_REPORT §5.6")


# =====================================================================
# A3. valid 플래그가 살아 있는가 — 죽은 입력 채널 점검
# =====================================================================
print("\n" + "=" * 70)
print("A3. wind_valid 채널이 정보를 담고 있는가")
print("=" * 70)
for name, inputs in inputs_by_split.items():
    wind = inputs[WIND_COLUMNS].to_numpy(np.float32)
    valid = np.isfinite(wind).astype(np.float32)
    print(f"  {name:11s} valid 평균 {valid.mean():.6f} | "
          f"0 인 원소 {int((valid == 0).sum())} | "
          f"{'상수 -> GRU 3번째 채널은 무의미' if valid.min() == 1.0 else '정보 있음'}")


# =====================================================================
# B1. 이미지 시간 순서 — 방향 검증
# =====================================================================
# image_00 이 가장 과거인가, 가장 미래인가?
# wind_19 를 persistence 로 썼을 때가 wind_00 보다 나아야 image/wind 인덱스가 증가=미래다.
print("\n" + "=" * 70)
print("B1. 인덱스 방향 — 00 이 과거이고 19 가 T0 인가")
print("=" * 70)
for name in ("train", "validation"):
    wind = inputs_by_split[name][WIND_COLUMNS].to_numpy(np.float64)
    targets = targets_by_split[name]
    for column in (0, 9, 19):
        prediction = np.repeat(wind[:, column:column + 1], 12, axis=1)
        rmse = float(np.sqrt(np.mean((prediction - targets) ** 2, axis=0)).mean())
        first_horizon = float(np.sqrt(np.mean((prediction[:, 0] - targets[:, 0]) ** 2)))
        print(f"  {name:11s} wind_{column:02d} persistence : "
              f"공식 RMSE {rmse:7.2f} | 6h RMSE {first_horizon:7.2f}")
print("\n>>> wind_19 가 가장 낮아야 정상이다. wind_00 이 더 낮으면 인덱스가 역순이고,")
print("    탄도 정렬(19 = T0 가정)과 잔차 예측(target - wind_19)이 전부 뒤집혀 있다는 뜻이다.")


# =====================================================================
# B2. 이미지축과 wind축의 정렬 오프셋
# =====================================================================
# 사슬 검증은 '같은 방향·같은 stride' 까지만 증명한다. 일정한 오프셋은 통과해 버린다.
# 이미지에서 뽑은 코로나홀 면적과 wind 를 직접 맞춰 lag 을 재면 오프셋이 드러난다.
#
# 전달 시간 tau = 1AU/v 이므로 corr 봉우리는 tau (평균 속도 기준 약 100h) 에 서야 한다.
# 봉우리가 그보다 6h 의 정수배만큼 벗어나 있으면 그만큼 인덱스가 어긋나 있는 것이다.
print("\n" + "=" * 70)
print("B2. 이미지 <-> wind 정렬 오프셋 (샘플링 검사)")
print("=" * 70)

DISK_MARGIN = 0.95
CH_CUT = 0.45
MAX_FRAMES = 1500        # 가장 긴 사슬에서 이만큼만 읽는다 (전체를 읽지 않는다)

train_inputs = inputs_by_split["train"]
train_chains = reconstruct_frame_chains(train_inputs)
longest = max(train_chains, key=len)[:MAX_FRAMES]
print(f"  가장 긴 사슬에서 {len(longest):,} 프레임만 읽는다 "
      f"(train 고유 이미지의 {len(longest) / sum(len(c) for c in train_chains):.0%})")


def load_pair(name):
    planes = []
    for channel in CHANNELS:
        with Image.open(DATA_ROOT / "train" / channel / name) as image:
            planes.append(np.asarray(image.convert("L"), dtype=np.float32))
    return np.stack(planes)


reference = load_pair(longest[len(longest) // 2])
mean_plane = reference.mean(axis=0)
mask = mean_plane > mean_plane.max() * 0.15
ys, xs = np.nonzero(mask)
center_y, center_x = float(ys.mean()), float(xs.mean())
effective = float(np.sqrt(mask.sum() / np.pi)) * DISK_MARGIN
grid_y, grid_x = np.mgrid[0:reference.shape[1], 0:reference.shape[2]].astype(np.float64)
disk = np.sqrt((grid_y - center_y) ** 2 + (grid_x - center_x) ** 2) <= effective
# 중앙 자오선 ±15% 밴드만 본다 (지구를 마주보는 경도)
meridian = disk & (np.abs(grid_x - center_x) <= effective * 0.15)

ch_series = np.zeros(len(longest), np.float32)
for index, name in enumerate(longest):
    frame = load_pair(name)
    median = np.median(frame[:, disk], axis=1)
    dark = (frame[0] <= CH_CUT * median[0]) & (frame[1] <= CH_CUT * median[1])
    ch_series[index] = (dark & meridian).sum() / meridian.sum()
    if (index + 1) % 500 == 0:
        print(f"    {index + 1}/{len(longest)}", flush=True)

wind_map, _ = frame_wind_map(train_inputs)
wind_series = np.array([wind_map[n] for n in longest])

print(f"\n  CH 면적 평균 {ch_series.mean():.4f} | wind 평균 {wind_series.mean():.1f} km/s")
print("\n  lag [h]   corr(CH(t), wind(t+lag))   함의 속도")

# 🔴 abs() 로 최대를 찾으면 안 된다. 코로나홀이 크면 태양풍이 빠르므로 물리 상관은 **양수**다.
# 범위 끝단의 음의 봉우리(186h 부근)는 저주파 추세 아티팩트이고, P7 EDA 에서도 같은 값이
# 나왔다가 폐기됐다. 부호와 탐색 범위를 모두 물리로 제한한다. (CODE_REPORT §5.6)
SEARCH_LOW_HOURS, SEARCH_HIGH_HOURS = 48, 144    # v = 866 ~ 289 km/s
best = (0, -1.0)
for steps in range(0, 34):                       # 0h ~ 198h (표는 전 범위를 찍는다)
    if steps == 0:
        a, b = ch_series, wind_series
    else:
        a, b = ch_series[:-steps], wind_series[steps:]
    if a.std() < 1e-12 or b.std() < 1e-12:
        continue
    correlation = float(np.corrcoef(a, b)[0, 1])
    lag_hours = steps * 6
    if SEARCH_LOW_HOURS <= lag_hours <= SEARCH_HIGH_HOURS and correlation > best[1]:
        best = (lag_hours, correlation)
    if steps % 2 == 0:
        speed = f"{AU_KM / (lag_hours * 3600.0):7.0f} km/s" if steps else "      —"
        window = " *" if SEARCH_LOW_HOURS <= lag_hours <= SEARCH_HIGH_HOURS else ""
        print(f"   {lag_hours:4d}      {correlation:+.4f}              {speed}{window}")

theoretical = AU_KM / wind_series.mean() / 3600.0
print(f"\n  (* = 물리 탐색 범위 {SEARCH_LOW_HOURS}~{SEARCH_HIGH_HOURS}h)")
if best[0]:
    print(f"  >>> 물리 봉우리 lag = {best[0]} h (corr {best[1]:+.4f}) "
          f"-> 함의 속도 {AU_KM / (best[0] * 3600.0):.0f} km/s")
print(f"  >>> train 평균 속도 {wind_series.mean():.0f} km/s 의 이론 전달 시간 = {theoretical:.0f} h")
print("\n>>> 봉우리가 이론 전달 시간보다 다소 짧은 것은 정상이다 — 상관을 만드는 주체가")
print("    코로나홀發 고속풍이라 평균보다 빠른 쪽으로 당겨진다.")
print(">>> 18h 이상 계통적으로 어긋나면 그 차이 / 6 만큼 인덱스가 밀려 있다는 뜻이고,")
print("    탄도 정렬의 기준점(19 = T0)을 그만큼 보정해야 한다.")
print(">>> 봉우리가 넓고 corr 이 약하므로 1스텝(6h) 오프셋은 이 검사로 배제되지 않는다.")


# =====================================================================
# B3. 윈도우 내부의 중복 프레임 · 파일 존재 여부
# =====================================================================
print("\n" + "=" * 70)
print("B3. 이미지 자체의 결측 — 중복 프레임 · 누락 파일")
print("=" * 70)
for name, inputs in inputs_by_split.items():
    images = inputs[IMAGE_COLUMNS].to_numpy()
    unique_per_row = np.array([len(set(row)) for row in images])
    folder = {"train": "train", "validation": "validation", "test": "test"}[name]
    names = pd.unique(images.ravel())
    missing = [n for n in names[:2000]
               if not (DATA_ROOT / folder / CHANNELS[0] / n).exists()]
    print(f"  {name:11s} 20개 전부 고유한 행 {int((unique_per_row == 20).sum()):,}"
          f" / {len(inputs):,}"
          f" | 중복 있는 행 {int((unique_per_row < 20).sum()):,}"
          f" | 파일 누락(앞 2000개 검사) {len(missing)}")
print("\n>>> 중복 있는 행이 존재하면 그 윈도우는 같은 관측을 두 번 쓰고 있다 = 이미지 결측을")
print("    주최측이 복제로 메운 것이다. GRU 가 '변화 없음' 으로 오해하므로 플래그가 필요하다.")

print("\n" + "=" * 70)
print("검사 종료 — 위 출력만 공유하면 된다 (원본 반출 아님)")
print("=" * 70)
