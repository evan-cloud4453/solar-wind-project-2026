"""Build P16: P3 features, polynomial ridge regression instead of the GRU.

P16 reuses P3 cells 0-10 verbatim (config, data, image memmap, CH grid, ballistic
setup) and P3's official metric, then replaces the whole torch model/training half
with a closed-form polynomial ridge fit.  The USE_* toggles match P15 exactly, so a
P16 run and a P15 run of the same combination are measuring the same feature set.

P3's cell 12 is dropped along with the rest of the torch half, so the two pieces of
it that P16 still needs (STAT_NAMES, build_wind_stats) are restated in the design
cell rather than dragged in with the Dataset class.
"""

from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "Trial" / "P3" / "code_p3.ipynb"
TARGET = HERE / "code_p16.ipynb"
TRIAL_TARGET = HERE / "Trial" / "P16" / "code_p16.ipynb"

# P3 cells kept as-is: 0 header, 1-2 config, 3-4 data, 5-6 memmap, 7-8 CH grid, 9-10 stats.
KEEP_THROUGH = 10
METRIC_CELL = 16
CHECKLIST_CELL = 24


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.strip("\n").splitlines(True)}


HEADER = r'''# 태양풍 속도 예측 — P16: P3 피처 + 다항 릿지회귀

P16 은 P3 의 **피처는 그대로 두고 학습기만 바꾼 대조군**이다. 3×5 코로나홀 격자, 탄도
정렬, 결측 처리, 공식 검증 지표는 P3 와 동일하고, GRU 대신 닫힌 형태의 다항 릿지회귀를
푼다. `USE_*` 토글은 P15 와 이름·의미가 같으므로 두 노트북의 같은 조합은 같은 피처
집합을 재고 있다.

이게 쓸모 있는 이유는 두 가지다.

1. **피처 선별이 몇 초로 끝난다.** 릿지는 정규방정식 한 번이라 GRU 60 epoch 대신
   조합당 수십 초다. P15 에서 어떤 토글을 살릴지 먼저 여기서 훑고, 살아남은 조합만
   GRU 로 확인하면 된다.
2. **신호와 용량을 분리한다.** P16 이 persistence 를 못 이기면 그 피처에는 선형·2차
   범위의 신호가 없다는 뜻이고, P16 은 이기는데 P15 가 못 이기면 학습 쪽 문제다.

P16 은 P15 를 대체하지 않는다. 20 프레임 시계열을 요약통계 몇 개로 접어 버리므로 GRU 가
쓰는 시간 구조 정보를 상당히 버린다. 제출 후보가 아니라 진단 도구로 읽는 게 맞다.

플레어·형태·강도깊이 토글은 `work/cache/p14a_*.npz` 캐시를 필요로 한다. 캐시가 없으면
「P14 캐시 준비」 셀이 같은 폴더의 `p14_extract.py` 를 직접 실행해 만든다.
'''


CONFIG = r'''

# ============================= P16 설정 ====================================
# P15 와 같은 토글. 같은 조합끼리 직접 비교할 수 있게 이름을 맞춘다.
USE_FLARE = False
USE_CH_SHAPE = False
USE_ROTATION = False
USE_INTENSITY_DEPTH = False
P16_CACHE_VERSION = "p14a"

# 핵심 피처에만 다항 전개를 건다. 1 이면 순수 선형회귀가 된다.
P16_DEGREE = 2
# 릿지 계수는 validation 공식 RMSE 로 고르고, 12 horizon 이 하나의 alpha 를 공유한다.
P16_ALPHAS = (0.1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0, 30000.0)
# 20 프레임 시계열을 접는 방법. 줄이면 설계행렬이 그만큼 좁아진다.
P16_SUMMARIES = ("last", "mean", "mean4", "slope")
P16_CHUNK = 4096          # 시퀀스 요약을 나눠 처리하는 행 수 (메모리 상한용)
P16_REQUESTED_SWITCHES = {
    "flare": USE_FLARE,
    "shape": USE_CH_SHAPE,
    "rotation": USE_ROTATION,
    "intensity_depth": USE_INTENSITY_DEPTH,
}
'''


BOOTSTRAP = r'''
# ===================== P16: P14 물리 피처 캐시 준비 =========================
# work/cache/p14a_{train,validation,test}.npz 가 없으면 이 셀이 직접 만든다.
# 노트북과 같은 폴더에 p14_extract.py 와 p11_extract.py 가 함께 있어야 한다.
# 캐시가 이미 있으면 아무 일도 하지 않고 즉시 넘어간다.
import subprocess
import sys

P16_AUTO_EXTRACT = True     # False 로 두면 캐시가 없어도 추출하지 않는다
P16_EXTRACT_WORKERS = 4     # 운영진 공지: 워커 4개를 넘기지 말 것
P16_EXTRACT_SPLITS = ("train", "validation", "test")


def p16_cache_path(split):
    return CACHE_ROOT / f"{P16_CACHE_VERSION}_{split}.npz"


def p16_missing_splits():
    return [split for split in P16_EXTRACT_SPLITS if not p16_cache_path(split).exists()]


def p16_run_extract(splits):
    """p14_extract.py 를 자식 프로세스로 돌리고 진행 로그를 그대로 흘린다."""
    scripts = [Path("p14_extract.py"), Path("p11_extract.py")]
    absent = [str(script.resolve()) for script in scripts if not script.exists()]
    if absent:
        print("추출 스크립트가 없다:")
        for item in absent:
            print("   ", item)
        print("  -> 이 노트북과 같은 폴더에 p11_extract.py 와 p14_extract.py 를 올린 뒤 다시 실행할 것")
        return False

    command = [sys.executable, "-u", "p14_extract.py",
               "--workers", str(P16_EXTRACT_WORKERS),
               "--splits", ",".join(splits),
               "--cache", str(CACHE_ROOT)]
    # 자식이 데이터 경로를 다시 추측하지 않도록 노트북이 찾은 경로를 그대로 넘긴다.
    environment = dict(os.environ, SW_DATA_ROOT=str(DATA_ROOT.resolve()))
    print("실행:", " ".join(command), flush=True)
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               env=environment, text=True, encoding="utf-8",
                               errors="replace", bufsize=1)
    for line in process.stdout:
        print(line, end="", flush=True)
    status = process.wait()
    if status != 0:
        print(f"p14_extract.py 가 실패했다 (exit {status}).", flush=True)
    return status == 0


_p16_missing = p16_missing_splits()
if not _p16_missing:
    print("P14 캐시 확인 완료:")
    for split in P16_EXTRACT_SPLITS:
        _path = p16_cache_path(split)
        print(f"  {split:<10} {_path.resolve()} ({_path.stat().st_size / 1024 ** 2:.0f} MB)")
elif not P16_AUTO_EXTRACT:
    print("P14 캐시 없음:", ", ".join(_p16_missing), "| P16_AUTO_EXTRACT=False 이므로 건너뛴다")
else:
    print("P14 캐시 없음 ->", ", ".join(_p16_missing), "추출을 시작한다.")
    print("전체 split 은 수십 분에서 몇 시간까지 걸린다. 터미널을 쓸 수 있으면 대신")
    print("  nohup python p14_extract.py --workers 4 > extract.log 2>&1 &")
    print("로 돌리고 이 셀을 다시 실행하는 편이 낫다.", flush=True)
    p16_run_extract(_p16_missing)
    _p16_missing = p16_missing_splits()
    print("아직 없는 split:", ", ".join(_p16_missing) if _p16_missing else "없음")
'''


FEATURES = r'''
# ===================== P16: 물리 피처 로드 (P15 와 동일) ====================
# p14a 의 파일 순서는 시간 사슬 순서이고 P3 memmap 은 파일명 정렬 순서다.
# 반드시 filename 으로 재매핑한다. 배열 위치가 우연히 같다고 가정하지 않는다.

P16_CACHE_KEYS = ("area", "depth_max", "depth_sum", "flare_area", "flare_power", "shape")


def p16_load(split, image_index):
    path = CACHE_ROOT / f"{P16_CACHE_VERSION}_{split}.npz"
    if not path.exists():
        print(f"  ⚠️ 캐시 파일 없음: {path.resolve()}")
        return None
    with np.load(path) as blob:
        if any(key not in blob for key in P16_CACHE_KEYS) or "files" not in blob:
            print(f"  ⚠️ {path.name}: P14 키 없음 -> p14_extract.py 를 실행할 것")
            return None
        source_index = {str(name): i for i, name in enumerate(blob["files"])}
        names = list(image_index)
        if set(names) != set(source_index):
            print(f"  ⚠️ {path.name}: filename 집합 불일치 -> P14 피처를 쓰지 않는다")
            return None
        order = np.asarray([source_index[name] for name in names], np.int64)
        return {key: np.asarray(blob[key][order], np.float32) for key in P16_CACHE_KEYS}


P16 = {
    "train": p16_load("train", train_image_index),
    "validation": p16_load("validation", val_image_index),
    "test": p16_load("test", test_image_index),
}
P16_AVAILABLE = all(value is not None for value in P16.values())
if not P16_AVAILABLE:
    print("P14 cache unavailable: flare/shape/intensity-depth disabled; rotation is still available.")
    print("  해결: 노트북 폴더에서 아래를 실행한 다음 위의 「P14 캐시 준비」 셀부터 다시 실행할 것")
    print("    python p14_extract.py --workers 4")
    USE_FLARE = USE_CH_SHAPE = USE_INTENSITY_DEPTH = False
else:
    print("P14 cache available: P3-compatible feature remapping complete.")


def p16_aggregate(fine, channels=1):
    """P14 12x30 fine grid -> P3 3x5 grid. 값은 원반 면적비이므로 합으로 접는다."""
    block = fine.reshape(len(fine), channels, 12, 30)
    return block.reshape(len(fine), channels, 3, 4, 5, 6).sum(axis=(3, 5)).reshape(len(fine), -1)


def p16_static(split, base_ch):
    """이미지 1장당 피처 (n_images, F). 첫 15 열은 언제나 P3 원본 격자다."""
    pieces = [base_ch.astype(np.float32)]
    if not P16_AVAILABLE:
        return np.concatenate(pieces, axis=1)
    cached = P16[split]
    if USE_FLARE:
        pieces += [p16_aggregate(cached["flare_area"]), p16_aggregate(cached["flare_power"])]
    if USE_CH_SHAPE:
        pieces += [p16_aggregate(cached["depth_max"]), p16_aggregate(cached["depth_sum"]),
                   cached["shape"]]
    if USE_INTENSITY_DEPTH:
        pieces += [p16_aggregate(cached["area"][:, :3], channels=3),
                   p16_aggregate(cached["depth_max"]), p16_aggregate(cached["depth_sum"])]
    return np.concatenate(pieces, axis=1).astype(np.float32)


def p16_rotation(raw_ch):
    """P3 3x5 격자만으로 만드는 자전 상태. P15 의 p15_rotation 과 같은 정의다."""
    area = raw_ch.reshape(len(raw_ch), 20, 3, 5).sum(axis=2)
    position = np.arange(5, dtype=np.float32) - 2.0
    centrality = 1.0 - np.abs(position) / 2.0
    core = np.einsum("btl,l->bt", area, centrality)
    side = np.einsum("btl,l->bt", area, position / 2.0)
    d_core = np.diff(core, axis=1, prepend=core[:, :1])
    d_side = np.diff(side, axis=1, prepend=side[:, :1])
    return np.stack((core, side, d_core, d_side), axis=-1).astype(np.float32)


P16_STATIC = {
    "train": p16_static("train", train_ch),
    "validation": p16_static("validation", val_ch),
    "test": p16_static("test", test_ch),
}
P16_INDEXES = {
    "train": image_index_matrix(train_inputs, train_image_index),
    "validation": image_index_matrix(val_inputs, val_image_index),
    "test": image_index_matrix(test_inputs, test_image_index),
}
P16_RAW_CH = {"train": train_ch, "validation": val_ch, "test": test_ch}
P16_WIND = {"train": train_wind, "validation": val_wind, "test": test_wind}
P16_SPLITS = ("train", "validation", "test")


def p16_sequence(split, rows=slice(None)):
    """P15 가 GRU 에 넣는 것과 같은 (n, 20, F) 입력 시퀀스. rows 로 잘라 쓸 수 있다."""
    indexes = P16_INDEXES[split][rows]
    sequence = P16_STATIC[split][indexes]
    if USE_ROTATION:
        sequence = np.concatenate((sequence, p16_rotation(P16_RAW_CH[split][indexes])), axis=2)
    return sequence


P16_ACTIVE_SWITCHES = {
    "flare": USE_FLARE, "shape": USE_CH_SHAPE,
    "rotation": USE_ROTATION, "intensity_depth": USE_INTENSITY_DEPTH,
}
P16_FEATURE_DIM = P16_STATIC["train"].shape[1] + (4 if USE_ROTATION else 0)
P16_EXPECTED_DIM = (15 + (30 if USE_FLARE else 0) + (36 if USE_CH_SHAPE else 0)
                    + (75 if USE_INTENSITY_DEPTH else 0) + (4 if USE_ROTATION else 0))
assert P16_FEATURE_DIM == P16_EXPECTED_DIM, (P16_FEATURE_DIM, P16_EXPECTED_DIM)
print("P16 requested switches:", P16_REQUESTED_SWITCHES)
print("P16 active switches:   ", P16_ACTIVE_SWITCHES)
if P16_REQUESTED_SWITCHES != P16_ACTIVE_SWITCHES:
    print("WARNING: 캐시가 없어 일부 토글이 꺼졌다. p14_extract.py 를 돌리고 다시 실행할 것.")
print(f"P16 프레임당 피처 F = {P16_FEATURE_DIM}")
'''


DESIGN = r'''
# ===================== P16: 설계행렬 ======================================
# 다항 전개를 전부에 걸면 항이 폭발한다 (F=160 이면 요약 후 640열, 2차만 205,000 항).
# 그래서 두 층으로 나눈다.
#   core   : 물리적으로 상호작용이 의미 있는 소수 피처. 여기에만 다항 전개를 건다.
#   linear : 나머지 넓은 CH/물리 피처. 1차항으로만 붙인다.
# core 구성은 토글과 무관하게 고정이라 조합을 바꿔도 비교축이 흔들리지 않는다.

from itertools import combinations_with_replacement

_P16_TIME = np.arange(20, dtype=np.float64) - 9.5
_P16_TIME_DENOMINATOR = float((_P16_TIME ** 2).sum())

# P3 cell 12 에 있던 바람 통계. P16 은 그 셀(Dataset)을 쓰지 않으므로 여기 다시 둔다.
STAT_NAMES = ["last", "mean4", "mean", "std", "min", "max", "slope", "last_minus_mean4", "range"]


def build_wind_stats(wind):
    last = wind[:, -1]
    mean4 = wind[:, -4:].mean(axis=1)
    slope = (wind - wind.mean(axis=1, keepdims=True)) @ _P16_TIME / _P16_TIME_DENOMINATOR
    return np.stack([last, mean4, wind.mean(axis=1), wind.std(axis=1), wind.min(axis=1),
                     wind.max(axis=1), slope, last - mean4,
                     wind.max(axis=1) - wind.min(axis=1)], axis=1).astype(np.float64)


def p16_summaries(sequence):
    """(n, 20, F) -> (n, len(P16_SUMMARIES) * F). summary-major, feature-minor 순서."""
    sequence = sequence.astype(np.float64)
    pieces = []
    for name in P16_SUMMARIES:
        if name == "last":
            pieces.append(sequence[:, -1, :])
        elif name == "mean":
            pieces.append(sequence.mean(axis=1))
        elif name == "mean4":
            pieces.append(sequence[:, -4:, :].mean(axis=1))
        elif name == "slope":
            centered = sequence - sequence.mean(axis=1, keepdims=True)
            pieces.append(np.tensordot(centered, _P16_TIME, axes=([1], [0]))
                          / _P16_TIME_DENOMINATOR)
        else:
            raise ValueError(f"알 수 없는 요약 {name}")
    return np.concatenate(pieces, axis=1)


def p16_summarize_split(split):
    """토글을 다 켜면 (n, 20, 160) 이 1GB 를 넘는다. 행으로 잘라 상한을 건다."""
    total = len(P16_INDEXES[split])
    parts = [p16_summaries(p16_sequence(split, slice(start, start + P16_CHUNK)))
             for start in range(0, total, P16_CHUNK)]
    return np.concatenate(parts, axis=0)


def p16_fold3(series):
    """(n, 20) -> (n, 3): last, mean, slope. core 는 P16_SUMMARIES 설정과 무관하다."""
    series = series.astype(np.float64)
    centered = series - series.mean(axis=1, keepdims=True)
    return np.stack([series[:, -1], series.mean(axis=1),
                     centered @ _P16_TIME / _P16_TIME_DENOMINATOR], axis=1)


def p16_summary_names(feature_names):
    return [f"{name}[{summary}]" for summary in P16_SUMMARIES for name in feature_names]


def p16_feature_names():
    names = [f"ch{cell:02d}" for cell in range(N_CELLS)]
    if USE_FLARE:
        names += [f"flare_area{c:02d}" for c in range(N_CELLS)]
        names += [f"flare_power{c:02d}" for c in range(N_CELLS)]
    if USE_CH_SHAPE:
        names += [f"theta_max{c:02d}" for c in range(N_CELLS)]
        names += [f"theta_sum{c:02d}" for c in range(N_CELLS)]
        names += ["largest_frac", "component_count", "compactness",
                  "elongation", "boundary_density", "core_fraction"]
    if USE_INTENSITY_DEPTH:
        for level in ("dark30", "dark45", "dark60"):
            names += [f"{level}_{c:02d}" for c in range(N_CELLS)]
        names += [f"d_theta_max{c:02d}" for c in range(N_CELLS)]
        names += [f"d_theta_sum{c:02d}" for c in range(N_CELLS)]
    if USE_ROTATION:
        names += ["rot_core", "rot_side", "rot_d_core", "rot_d_side"]
    return names


# ---- core: 바람 통계 9 + CH 총면적 3 + 중앙자오선 3 + horizon 별 탄도 3 = 18 ----
P16_CORE_NAMES = (
    [f"wind_{name}" for name in STAT_NAMES]
    + ["ch_total_last", "ch_total_mean", "ch_total_slope"]
    + ["ch_central_last", "ch_central_mean", "ch_central_slope"]
    + [f"ballistic_v{int(speed)}" for speed in TRANSIT_SPEEDS]
)


def p16_core_static(split):
    """horizon 과 무관한 core 부분. 탄도 3열은 뒤에서 horizon 별로 붙인다."""
    base = P16_RAW_CH[split][P16_INDEXES[split]]      # (n, 20, N_CELLS) 원본 P3 격자
    return np.concatenate([build_wind_stats(P16_WIND[split]),
                           p16_fold3(base.sum(axis=2)),
                           p16_fold3(base[:, :, CENTRAL_CELL])], axis=1)


def p16_standardize(matrix, statistics=None):
    """train 에서만 통계를 내고, 분산이 0 인 열은 아예 버린다 (전부 0 인 플레어 셀 등)."""
    if statistics is None:
        deviation = matrix.std(axis=0)
        keep = deviation > 1e-12
        statistics = (keep, matrix[:, keep].mean(axis=0), deviation[keep])
    keep, mean, deviation = statistics
    return (matrix[:, keep] - mean) / deviation, statistics


def p16_polynomial(matrix, degree):
    """상수항 없는 다항 전개. 절편은 따로 두므로 1 로만 된 열을 만들지 않는다."""
    if degree <= 1:
        return matrix
    columns = [matrix]
    for order in range(2, degree + 1):
        for combination in combinations_with_replacement(range(matrix.shape[1]), order):
            columns.append(matrix[:, combination].prod(axis=1, keepdims=True))
    return np.concatenate(columns, axis=1)


def p16_polynomial_names(names, degree):
    terms = list(names)
    for order in range(2, degree + 1):
        for combination in combinations_with_replacement(range(len(names)), order):
            terms.append("*".join(names[index] for index in combination))
    return terms


# ---- 넓은 선형 블록 ----------------------------------------------------------
P16_FEATURE_NAMES = p16_feature_names()
_linear_raw = {split: p16_summarize_split(split) for split in P16_SPLITS}
P16_LINEAR = {}
P16_LINEAR["train"], P16_LINEAR_STATS = p16_standardize(_linear_raw["train"])
for split in ("validation", "test"):
    P16_LINEAR[split], _ = p16_standardize(_linear_raw[split], P16_LINEAR_STATS)
P16_LINEAR_NAMES = [name for name, keep
                    in zip(p16_summary_names(P16_FEATURE_NAMES), P16_LINEAR_STATS[0]) if keep]
_dropped = int((~P16_LINEAR_STATS[0]).sum())
del _linear_raw

# ---- core: horizon 별 탄도 3열을 붙여 표준화해 둔다 ---------------------------
_core_static = {split: p16_core_static(split) for split in P16_SPLITS}
_ballistic = {split: compute_ballistic(P16_RAW_CH[split], P16_INDEXES[split])
              for split in P16_SPLITS}                       # (n, 12, n_speeds)

P16_CORE, P16_CORE_STATS = {}, {}
for horizon in range(12):
    raw = {split: np.concatenate(
        [_core_static[split], _ballistic[split][:, horizon, :].astype(np.float64)], axis=1)
        for split in P16_SPLITS}
    P16_CORE[("train", horizon)], P16_CORE_STATS[horizon] = p16_standardize(raw["train"])
    for split in ("validation", "test"):
        P16_CORE[(split, horizon)], _ = p16_standardize(raw[split], P16_CORE_STATS[horizon])
del _core_static, _ballistic

_core_kept = int(P16_CORE_STATS[0][0].sum())
P16_TERM_COUNT = len(p16_polynomial_names([str(i) for i in range(_core_kept)], P16_DEGREE))
P16_WIDTH = P16_TERM_COUNT + P16_LINEAR["train"].shape[1]
print(f"core {_core_kept}열 -> {P16_DEGREE}차 전개 {P16_TERM_COUNT}항")
print(f"linear 블록 {P16_LINEAR['train'].shape[1]}열 "
      f"(요약 {len(P16_SUMMARIES)} x F {P16_FEATURE_DIM} 중 상수열 {_dropped}개 제거)")
print(f"설계행렬 폭 D = {P16_WIDTH}, train 표본 n = {len(train_targets):,} "
      f"(n/D = {len(train_targets) / P16_WIDTH:.1f})")
if len(train_targets) < 5 * P16_WIDTH:
    print("  주의: n/D 가 5 미만이다. P16_DEGREE 를 낮추거나 P16_SUMMARIES 를 줄일 것.")
'''


FIT = r'''
# ===================== P16: 릿지 적합 · alpha 선택 =========================
# horizon 마다 별도 회귀를 푼다. 탄도 피처가 horizon 별로 다르기 때문이다.
# 목표는 P3 와 같은 잔차 = target - 마지막 관측 풍속.
# 설계행렬은 열마다 train 평균이 0 이라 절편은 잔차 평균 하나로 끝난다.
# alpha 는 12 horizon 이 공유한다. horizon 마다 따로 고르면 validation 에 더 과적합된다.

P16_RESIDUAL = train_targets - train_wind[:, -1:]


def p16_design(split, horizon, statistics=None):
    expanded = p16_polynomial(P16_CORE[(split, horizon)], P16_DEGREE)
    expanded, statistics = p16_standardize(expanded, statistics)
    return np.concatenate([expanded, P16_LINEAR[split]], axis=1), statistics


def p16_ridge(gram, moment, alpha):
    regularized = gram.copy()
    regularized.flat[:: gram.shape[0] + 1] += alpha
    return np.linalg.solve(regularized, moment)


P16_GRAM, P16_MOMENT, P16_INTERCEPT, P16_POLY_STATS, P16_VAL_DESIGN = {}, {}, {}, {}, {}
_started = time.perf_counter()
for horizon in range(12):
    design, P16_POLY_STATS[horizon] = p16_design("train", horizon)
    residual = P16_RESIDUAL[:, horizon].astype(np.float64)
    P16_INTERCEPT[horizon] = float(residual.mean())
    P16_GRAM[horizon] = design.T @ design
    P16_MOMENT[horizon] = design.T @ (residual - P16_INTERCEPT[horizon])
    del design
    # validation 설계행렬은 alpha 탐색 내내 재사용하므로 들고 있는다 (표본이 적다).
    P16_VAL_DESIGN[horizon], _ = p16_design("validation", horizon, P16_POLY_STATS[horizon])
_widths = {len(P16_MOMENT[h]) for h in range(12)}
assert len(_widths) == 1, f"horizon 별 설계행렬 폭이 다르다: {_widths}"
print(f"정규방정식 구성 {time.perf_counter() - _started:.1f}초 (12 horizon, 폭 {_widths.pop()})")


def p16_predict(designs, weights, last_wind):
    prediction = np.zeros((len(last_wind), 12), np.float64)
    for horizon in range(12):
        prediction[:, horizon] = designs[horizon] @ weights[horizon] + P16_INTERCEPT[horizon]
    return np.clip(prediction + last_wind[:, None], CLIP_LOW, CLIP_HIGH)


P16_SWEEP = []
for alpha in P16_ALPHAS:
    weights = {h: p16_ridge(P16_GRAM[h], P16_MOMENT[h], alpha) for h in range(12)}
    score, _ = official_rmse(val_targets, p16_predict(P16_VAL_DESIGN, weights, val_wind[:, -1]))
    P16_SWEEP.append({"alpha": alpha, "validation_rmse": score})
    print(f"  alpha={alpha:>9.1f}  validation 공식 RMSE = {score:8.3f} km/s")

P16_SWEEP = pd.DataFrame(P16_SWEEP)
P16_BEST_ALPHA = float(P16_SWEEP.loc[P16_SWEEP.validation_rmse.idxmin(), "alpha"])
P16_WEIGHTS = {h: p16_ridge(P16_GRAM[h], P16_MOMENT[h], P16_BEST_ALPHA) for h in range(12)}
print(f"\n선택된 alpha = {P16_BEST_ALPHA:g}")
if P16_BEST_ALPHA in (min(P16_ALPHAS), max(P16_ALPHAS)):
    print("  주의: alpha 가 격자 끝에서 골라졌다. P16_ALPHAS 범위를 넓힐 것.")
print("  alpha 를 validation 으로 골랐으므로 아래 RMSE 는 낙관적으로 편향된다.")
print("  조합 간 비교에는 문제없지만, 절대 성능은 test 로만 판단할 것.")
'''


VALIDATE = r'''
# ===================== P16: Validation 평가 ================================
validation_prediction = p16_predict(P16_VAL_DESIGN, P16_WEIGHTS, val_wind[:, -1])
model_score, _ = official_rmse(val_targets, validation_prediction)
validation_metrics = metrics_by_horizon(val_targets, validation_prediction, val_persistence)
validation_metrics.to_csv(OUTPUT_DIR / "validation_metrics.csv", index=False)

print(f"공식 RMSE (mean of horizon RMSE) : {model_score:8.3f} km/s")
print(f"pooled RMSE (전체 원소)          : {pooled_rmse(val_targets, validation_prediction):8.3f} km/s")
print(f"persistence 공식 RMSE            : {persistence_score:8.3f} km/s")
print(f"persistence 대비 개선             : {persistence_score - model_score:8.3f} km/s"
      f"  ({(persistence_score - model_score) / persistence_score:.1%})")
if model_score >= persistence_score:
    print("\n>>> persistence 미달. 이 피처 조합에는 선형·다항 범위의 신호가 없다.")

figure, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(validation_metrics.horizon_h, validation_metrics.rmse, marker="o", label="P16 ridge")
axes[0].plot(validation_metrics.horizon_h, validation_metrics.persistence_rmse,
             marker="s", linestyle="--", label="persistence")
axes[0].set_xlabel("forecast horizon (h)"); axes[0].set_ylabel("RMSE (km/s)")
axes[0].grid(alpha=0.3); axes[0].legend()
axes[1].semilogx(P16_SWEEP.alpha, P16_SWEEP.validation_rmse, marker="o")
axes[1].axvline(P16_BEST_ALPHA, color="red", linestyle="--", linewidth=1)
axes[1].set_xlabel("ridge alpha"); axes[1].set_ylabel("validation 공식 RMSE")
axes[1].grid(alpha=0.3)
plt.tight_layout(); plt.show()

# ---- 어떤 항이 실제로 일하고 있는지 ------------------------------------------
# 설계행렬이 열별로 표준화되어 있으므로 계수 크기를 그대로 비교할 수 있다.
_core_names_kept = [name for name, keep in zip(P16_CORE_NAMES, P16_CORE_STATS[0][0]) if keep]
P16_TERM_NAMES = (
    [name for name, keep in zip(p16_polynomial_names(_core_names_kept, P16_DEGREE),
                                P16_POLY_STATS[0][0]) if keep]
    + P16_LINEAR_NAMES
)
assert len(P16_TERM_NAMES) == len(P16_WEIGHTS[0]), (len(P16_TERM_NAMES), len(P16_WEIGHTS[0]))
_importance = np.mean([np.abs(P16_WEIGHTS[h]) for h in range(12)], axis=0)
P16_TOP_TERMS = (pd.DataFrame({"term": P16_TERM_NAMES, "mean_abs_coef": _importance})
                 .sort_values("mean_abs_coef", ascending=False).head(20).reset_index(drop=True))
P16_TOP_TERMS.to_csv(OUTPUT_DIR / "top_terms.csv", index=False)
print("\n계수 크기 상위 20항 (12 horizon 평균, 표준화 기준):")
print(P16_TOP_TERMS.to_string(index=False, float_format=lambda value: f"{value:8.3f}"))

print("\nP16 switches:", P16_ACTIVE_SWITCHES)
print(f"P16 degree={P16_DEGREE} alpha={P16_BEST_ALPHA:g} width={P16_WIDTH}")
print(f"P16 OFFICIAL_VALIDATION_RMSE = {model_score:.4f} km/s")
validation_metrics
'''


SUBMIT = r'''
# ===================== P16: Test 추론 · 제출 파일 ==========================
_test_design = {}
for horizon in range(12):
    _test_design[horizon], _ = p16_design("test", horizon, P16_POLY_STATS[horizon])
test_prediction = p16_predict(_test_design, P16_WEIGHTS, test_wind[:, -1])

assert test_prediction.shape == (len(test_inputs), 12)
assert np.isfinite(test_prediction).all()

submission = pd.DataFrame(test_prediction, columns=TARGET_COLUMNS)
submission.insert(0, "sample_id", test_inputs.sample_id.tolist())
submission.to_csv(SUBMISSION_DIR / "submission.csv", index=False)

# 규정: "code.ipynb 에서 model.pth 를 불러와 추론이 가능해야 함".
# 릿지는 가중치·표준화 통계·절편이 전부이므로 그대로 직렬화한다.
checkpoint = {
    "kind": "p16_polynomial_ridge",
    "degree": P16_DEGREE,
    "alpha": P16_BEST_ALPHA,
    "switches": P16_ACTIVE_SWITCHES,
    "summaries": list(P16_SUMMARIES),
    "term_names": P16_TERM_NAMES,
    "intercept": {h: P16_INTERCEPT[h] for h in range(12)},
    "weights": {h: P16_WEIGHTS[h] for h in range(12)},
    "core_stats": {h: P16_CORE_STATS[h] for h in range(12)},
    "poly_stats": {h: P16_POLY_STATS[h] for h in range(12)},
    "linear_stats": P16_LINEAR_STATS,
    "clip": (CLIP_LOW, CLIP_HIGH),
    "validation_rmse": model_score,
}
checkpoint_path = OUTPUT_DIR / "p16_ridge.pth"
torch.save(checkpoint, checkpoint_path)
shutil.copyfile(checkpoint_path, SUBMISSION_DIR / "model.pth")

# 저장한 파일에서 되읽어 같은 예측이 나오는지 실제로 확인한다.
reloaded = torch.load(SUBMISSION_DIR / "model.pth", map_location="cpu", weights_only=False)
_reloaded_prediction = p16_predict(_test_design, reloaded["weights"], test_wind[:, -1])
assert np.allclose(_reloaded_prediction, test_prediction), "model.pth 재현 실패"
print("reloaded from submission/model.pth — test 예측 재현 확인")

print("saved:", (SUBMISSION_DIR / "submission.csv").resolve(), submission.shape)
print(submission[TARGET_COLUMNS].describe().loc[["mean", "std", "min", "max"]].round(1))
del _test_design
gc.collect()
submission.head()
'''


def build() -> None:
    notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
    source_cells = notebook["cells"]

    cells = [md(HEADER)] + [dict(cell) for cell in source_cells[1:KEEP_THROUGH + 1]]

    # P3 config stays byte-identical apart from the output directory and the P16 block.
    config_source = "".join(cells[2]["source"])
    if "outputs_p3" not in config_source:
        raise RuntimeError("P3 config anchor not found")
    cells[2] = code(config_source.replace("outputs_p3", "outputs_p16") + CONFIG)

    # P3's metric cell, minus the two torch-only helpers P16 has no use for.
    metric_source = "".join(source_cells[METRIC_CELL]["source"])
    for anchor in ("def metric_loss", "val_persistence = "):
        if anchor not in metric_source:
            raise RuntimeError(f"P3 metric anchor {anchor!r} not found")
    metric_source = (metric_source[:metric_source.index("def metric_loss")]
                     + metric_source[metric_source.index("val_persistence = "):])

    cells += [
        md("## 5. P14 물리 피처 캐시 준비 (없으면 여기서 생성)"), code(BOOTSTRAP),
        md("## 6. 물리 피처 로드 — P3 파일 순서로 재매핑"), code(FEATURES),
        md("## 7. 지표 — P3 와 동일한 공식 RMSE"), code(metric_source),
        md("## 8. 설계행렬 — core 다항 전개 + 넓은 선형 블록"), code(DESIGN),
        md("## 9. 릿지 적합 · alpha 선택"), code(FIT),
        md("## 10. Validation 평가"), code(VALIDATE),
        md("## 11. Test 추론 · 제출 파일 생성"), code(SUBMIT),
        md("## 12. 제출 전 체크리스트"), dict(source_cells[CHECKLIST_CELL]),
    ]

    notebook["cells"] = cells
    TARGET.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    TRIAL_TARGET.parent.mkdir(parents=True, exist_ok=True)
    TRIAL_TARGET.write_text(TARGET.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"created {TARGET.name} and {TRIAL_TARGET.relative_to(HERE)} ({len(cells)} cells)")


if __name__ == "__main__":
    build()
