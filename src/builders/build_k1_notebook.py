#!/usr/bin/env python
"""code_p9.ipynb -> code_K1.ipynb 생성기 (개발 도구, 제출물 아님).

P9 의 셀 0~16 을 **한 글자도 고치지 않고** 가져온 뒤 K1 레이어를 붙인다.

K1 이 P11 에서 바꾼 것
  [P0] 멤버 학습 길이   epochs=1 고정 -> val 곡선 최저점 자동 선택
  [P1] 릿지 회귀        GRU 와 구조가 다른 저분산 추정기를 멤버로 추가
  [P2] AR + 경도 미분   bright 레벨 사용 + fine 30열에서 서/동 가장자리 기울기
  [P3] τ 재설계         과거 관측 속도 -> **소스 시각의 CH 면적** 기반 고정점
  [P4] 경계 선명도      k1_extract.py 의 Sobel 경계 강도 (재추출 필요)
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "Trial/P9/code_p9.ipynb"
TARGET = HERE / "code_K1.ipynb"
KEEP_THROUGH = 16


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.strip("\n").splitlines(True)}


# ============================================================================
HEADER = """
# 태양풍 속도 예측 — K1

P9 노트북의 셀 0~16 을 **한 글자도 고치지 않고** 쓰고, 그 뒤에서 필요한 함수만 재정의한다.

## P11 이 진 이유와 K1 이 바꾼 것

P11-S1(앙상블만) 은 **59.7332** 로 P3(58.8028) 에 졌다. 앙상블 기계는 정상 작동했고
(멤버 평균 val 67.46 -> 65.62) 멤버간 불일치도 22.53 RMS 로 충분했다. **재료가 나빴다.**
`A0` 는 "P3 재현" 이라고 불렀지만 재현된 것은 피처 설정뿐이고 학습 레시피가 아니었다 —
P3 는 60 epoch + early stopping, P11 멤버는 1~2 epoch. val 이 64.203 대 66.608 로 벌어져 있었다.

| | 레버 | 왜 |
|---|---|---|
| **P0** | **멤버 학습 길이 자동 선택** | `epochs=1` 은 P9/P10 의 CV 최소점에서 온 값이고, 그 CV 는 §1.5 가 이미 반증한 계측기다. val 곡선을 직접 훑어 최저점을 잡는다 |
| **P1** | **릿지 회귀 멤버** | 유효 표본 ≈ 480개(20프레임 슬라이딩 윈도우)에 GRU 파라미터 31만개. 저분산 추정기가 맞고, 무엇보다 **GRU 와 오차가 다르다** — 평균의 이득은 개별 성능이 아니라 오차상관 ρ 가 정한다 |
| **P2** | **AR 레벨 + fine 격자 경도 미분** | `bright` 는 이미 캐시에 있는데 P3 계열이 안 쓰고 있었다. 경도 미분은 **fine 30열**에서 잡아 서/동 가장자리로 나눠야 새 정보가 된다 (3열 미분은 입력의 선형결합이라 정보 0) |
| **P3** | **τ = AU / (a + b·A)** | P9 [R2] 는 *과거 결과*(최근 관측 속도)로 *미래 원인*의 전달시간을 재단했다. 전달시간은 **그 스트림의 소스 면적**이 정한다. 소스 시각이 τ 에 의존하므로 고정점 2회로 푼다 |
| **P4** | **CH 경계 선명도** | 자기장 확장인자의 시각적 대리. μ 로 정규화해야 자전만으로 흐려지는 가짜 변동이 안 생긴다 |

## 실행 순서

1. (선택) `python k1_extract.py` 를 백그라운드로 — [P4] 와 θ_b 를 낸다.
   없으면 `p11a` 캐시를 쓰고, 그것도 없으면 해당 레버만 자동으로 꺼진다.
2. 위에서부터 순서대로. **바꾸는 것은 K1 설정 셀의 `STAGE` 한 줄뿐이다.**
3. 마지막 두 셀(재현성 검증 · 제출 점검)이 통과해야 제출한다.

> **CV 사다리는 돌리지 않는다.** 셀 16 은 릿지 λ 선택과 [L5] 가 쓰는 `FOLDS` 정의 때문에
> 남겨 둔 것이고 `run_cv` 는 호출하지 않는다.
>
> **재현성.** `model.pth` 하나에 GRU 멤버 K개의 state_dict · 정규화 통계 · 릿지 계수 ·
> τ 계수 · 설정이 전부 들어간다. "재현성 검증" 셀이 그 파일만으로 `submission.csv` 를
> 다시 만들어 RMS 차이를 잰다.
"""

LADDER_NOTE = """
> K1 에서는 이 계측기를 **돌리지 않는다.** 릿지 λ 선택과 [L5] 가 쓰는 `FOLDS` 정의만 가져간다.
"""

# ============================================================================
CELL_CONFIG = r'''
# ============================  K1 레이어 시작  =============================
# 위 셀(P9 원본)은 한 글자도 고치지 않았다. 아래에서 함수를 재정의해 덮어쓴다.

K1_VERSION = "k1a"              # 경계 선명도까지 (k1_extract.py)
P11_VERSION = "p11a"            # θ_b·형태까지 (p11_extract.py) — k1a 없을 때 대체

# ---- [P0] 멤버 학습 길이 ---------------------------------------------------
EPOCH_POLICY   = "auto"         # "auto" = val 곡선 최저점 / "fixed" = MEMBER_EPOCHS
MEMBER_EPOCHS  = (1, 2)         # EPOCH_POLICY="fixed" 일 때만 쓴다
EPOCH_PROBE_MAX = 24            # auto 탐색 상한 (config 하나가 약 1분)
EPOCH_DELTAS   = (0, -3, 3)     # 최저점 주변으로 멤버를 흩뜨린다 (공짜 다양성)
EPOCH_SMOOTH   = 3              # 곡선 평활 창 — 단발 노이즈를 argmin 이 집는 것을 막는다

# ---- [P1] 릿지 회귀 --------------------------------------------------------
USE_RIDGE      = True
RIDGE_WEIGHT   = 0.5            # 최종 = (1-w)·GRU평균 + w·릿지. **적합하지 않는다**
RIDGE_OFFSETS  = (-4, 0, 4)     # 탄도 소스 시각 주변 샘플 점 (6h 스텝)
RIDGE_LAMBDAS  = np.logspace(-2.0, 4.0, 25)

# ---- [P2] 기하 · AR · 경도 미분 -------------------------------------------
AREA_SOURCE     = "mu_approx"   # "raw" | "mu_approx" | "true"(캐시 필요)
EQUAL_ANGLE_LON = True          # 경도 셀을 등각으로 (x 등간격 -> sinφ 등간격)
MU_FLOOR        = 0.30
USE_GRADIENT    = False         # fine 30열의 서/동 가장자리 기울기를 채널로

# ---- [P3] τ 재설계 ---------------------------------------------------------
TAU_MODE       = "fixed"        # "fixed" | "area"
TAU_ITERATIONS = 2              # 고정점 반복 (τ -> 소스시각 -> 면적 -> v -> τ)

# ---- [P4] 경계 선명도 / [L3] 물리 피처 (캐시 필요) -------------------------
USE_SHARP          = False      # Sobel 경계 강도 (k1a 필요)
USE_DEPTH          = False      # 셀별 θ_b — WSA 경계거리 항
USE_RATIO          = False      # 셀별 211/193 비
USE_FRAME          = False      # 프레임 형태 스칼라
DEPTH_IN_BALLISTIC = False
FRAME_USE = ["big_frac_true", "n_components", "big_lat_deg", "big_lon_deg", "big_depth_deg"]

# ---- [L5] 수축 보정 --------------------------------------------------------
FIT_SHRINKAGE = False
SHRINK_CLIP   = (0.70, 1.00)
LAM_H = np.ones(12, np.float64)

# ---- 앙상블 ----------------------------------------------------------------
STAGE = "B"                     # A~F. **제출할 때 여기만 바꾼다** (아래 표 참고)
MEMBER_CODES = ("A0", "F3")
ENSEMBLE_SEEDS = (777, 778, 779)
ENSEMBLE_SEEDS_FINAL = (777, 778, 779, 780, 781)

# ---- 캐시 적재 (k1a -> p11a -> 없음) ---------------------------------------
CACHE_FRAME_COLUMNS = ["cy", "cx", "radius", "total_px", "total_true",
                       "big_frac", "big_frac_true", "n_components",
                       "big_lat_deg", "big_lon_deg", "big_depth_deg"]
CACHE_KEYS = ("area", "area_true", "depth_max", "depth_sum", "ratio", "frame")


def load_cache(version, split, files):
    path = CACHE_ROOT / f"{version}_{split}.npz"
    if not path.exists():
        return None
    with np.load(path) as blob:
        if [str(n) for n in blob["files"]] != list(files):
            print(f"  ⚠️ {path.name}: 파일 목록이 다르다 -> 쓰지 않는다")
            return None
        loaded = {k: np.asarray(blob[k], np.float32) for k in CACHE_KEYS if k in blob}
        if "sharp" in blob:
            loaded["sharp"] = np.asarray(blob["sharp"], np.float32)
        return loaded


EXTRA = {}
EXTRA_VERSION = None
for _version in (K1_VERSION, P11_VERSION):
    _found = {}
    for _split, _files in (("train", train_files), ("validation", val_files),
                           ("test", test_files)):
        _loaded = load_cache(_version, _split, _files)
        if _loaded is not None:
            _found[_split] = _loaded
    if len(_found) == 3:
        EXTRA, EXTRA_VERSION = _found, _version
        break

EXTRA_AVAILABLE = EXTRA_VERSION is not None
SHARP_AVAILABLE = EXTRA_AVAILABLE and "sharp" in EXTRA["train"]
print(f"확장 캐시: {EXTRA_VERSION or '없음'} (경계 선명도 {'있음' if SHARP_AVAILABLE else '없음'})")

if EXTRA_AVAILABLE:
    # 확장 캐시의 `area` 는 p7a 와 **같은 알고리즘·같은 파일 순서**로 낸 것이다.
    # 어긋나면 나머지 피처도 다른 프레임을 가리키고 있다는 뜻이라 여기서 멈춘다.
    _gap = float(np.abs(EXTRA["train"]["area"] - train_area).max())
    print(f"  p7a 대조: 면적 최대 오차 {_gap:.2e} (0 이어야 정상)")
    assert _gap < 1e-5, "확장 캐시가 p7a 와 다르다 — 파일 순서 / 원반 검출을 확인할 것"
    del _gap
    _frame = EXTRA["train"]["frame"]
    _live = [c for c in FRAME_USE
             if np.isfinite(_frame[:, CACHE_FRAME_COLUMNS.index(c)]).any()
             and float(np.nanstd(_frame[:, CACHE_FRAME_COLUMNS.index(c)])) > 1e-6]
    if _live != FRAME_USE:
        print(f"  상수/결측 프레임 스칼라 제외: {[c for c in FRAME_USE if c not in _live]}")
        FRAME_USE = _live
    del _frame, _live
else:
    print("  -> [P4]·[L3] 을 끄고 캐시 없이 되는 레버만 쓴다 (P0·P1·P2·P3 는 전부 동작한다).")
    if AREA_SOURCE == "true":
        AREA_SOURCE = "mu_approx"
    USE_SHARP = USE_DEPTH = USE_RATIO = USE_FRAME = DEPTH_IN_BALLISTIC = False
'''

# ============================================================================
CELL_FEATURES = r'''
from collections import namedtuple

K1Grid = namedtuple("K1Grid", "seq cells summary cm_area")


def mu_weights(fine_lat=FINE_LAT, fine_lon=FINE_LON, mu_floor=None, side=512):
    """fine 셀별 평균 1/μ. 캐시된 픽셀면적에 곱하면 근사 진짜면적이 된다.

    fine 격자는 반지름 r_used = R·DISK_MARGIN 의 외접 사각형을 등분한 것이므로,
    μ 는 **진짜 반지름 R** 기준으로 되돌려서 재야 한다.
    """
    mu_floor = MU_FLOOR if mu_floor is None else mu_floor
    grid = np.linspace(-1.0, 1.0, side)
    Y, X = np.meshgrid(grid, grid, indexing="ij")
    rho2 = X ** 2 + Y ** 2
    on_disk = rho2 <= 1.0
    mu = np.sqrt(np.clip(1.0 - rho2 * DISK_MARGIN ** 2, 0.0, 1.0))
    weight = 1.0 / np.clip(mu, mu_floor, 1.0)
    li = np.clip(((Y + 1) / 2 * fine_lat).astype(int), 0, fine_lat - 1)
    lj = np.clip(((X + 1) / 2 * fine_lon).astype(int), 0, fine_lon - 1)
    cell = li * fine_lon + lj
    numerator = np.bincount(cell[on_disk], weights=weight[on_disk],
                            minlength=fine_lat * fine_lon)
    denominator = np.bincount(cell[on_disk], minlength=fine_lat * fine_lon)
    return (numerator / np.maximum(denominator, 1)).astype(np.float32)


MU_W = mu_weights()
# fine 경도 열 중심의 일면 경도(도). x 등간격은 경도 등간격이 아니다 (sinφ = x/R).
LON_CENTERS = np.degrees(np.arcsin(np.clip(
    (((np.arange(FINE_LON) + 0.5) / FINE_LON) * 2.0 - 1.0) * DISK_MARGIN, -1.0, 1.0)))
CENTRAL_BAND = np.abs(LON_CENTERS) <= 30.0          # 중앙자오선 ±30°


def lon_columns(fine_lon, grid_lon, equal_angle):
    """fine 경도 열 -> coarse 열 배정. 등각이면 sinφ 경계로 나눈다."""
    if not equal_angle:
        return np.minimum((np.arange(fine_lon) * grid_lon) // fine_lon, grid_lon - 1)
    return np.clip(((LON_CENTERS + 90.0) / 180.0 * grid_lon).astype(int), 0, grid_lon - 1)


def edge_gradients(fine):
    """[P2] fine (n, FINE_CELLS) -> (서쪽 성분, 동쪽 성분).

    fine 격자의 열 인덱스가 커지는 방향 = 이미지 오른쪽 = **태양 서쪽**이다.
    자전은 동->서이므로 CH 의 서쪽 경계가 중앙자오선을 **먼저** 지나고, 거기서 나온
    고속류가 앞서 나간 저속류를 따라잡아 압축(CIR)이 생긴다. 후행(동쪽) 경계는
    반대로 희박화 영역이다. **부호를 뭉개면 두 물리가 상쇄되므로 나눠서 낸다.**

    3열 격자에서의 미분은 모델 입력의 선형결합이라 정보가 0이다. 여기서는 모델이
    보지 않는 **fine 30열**에서 잡고, 클리핑(비선형)으로 요약한 뒤 집계한다.

    > 방향 가정이 반대여도 두 채널의 역할이 바뀔 뿐 손해는 없다.
    """
    block = fine.reshape(len(fine), FINE_LAT, FINE_LON)
    gradient = np.empty_like(block)
    gradient[:, :, 1:-1] = (block[:, :, 2:] - block[:, :, :-2]) / 2.0
    gradient[:, :, 0] = block[:, :, 1] - block[:, :, 0]
    gradient[:, :, -1] = block[:, :, -1] - block[:, :, -2]
    west = np.clip(-gradient, 0.0, None).reshape(len(fine), -1)
    east = np.clip(gradient, 0.0, None).reshape(len(fine), -1)
    return west.astype(np.float32), east.astype(np.float32)


def _reduce(block, axis, how):
    if how == "sum":
        return block.sum(axis)
    if how == "mean":
        return block.mean(axis)
    return block.max(axis)


def aggregate_fine(fine, grid_lat, grid_lon, fold, lon_map, how="sum"):
    """fine (n, FINE_CELLS) -> (n, USED_LAT · grid_lon).

    위도는 P9 그대로 균등 블록이다 (y 등간격 = sinθ 등간격 = **등면적 밴드**).
    경도만 lon_map 으로 묶는다. 접기는 P9 와 같은 순서·같은 방향이다.
    """
    n = len(fine)
    block = fine.reshape(n, FINE_LAT, FINE_LON)
    block = block.reshape(n, grid_lat, FINE_LAT // grid_lat, FINE_LON)
    block = _reduce(block, 2, how)
    out = np.empty((n, grid_lat, grid_lon), np.float32)
    for column in range(grid_lon):
        out[:, :, column] = _reduce(block[:, :, lon_map == column], 2, how)
    if fold:
        flipped = out[:, ::-1, :]
        out = (np.maximum(out, flipped) if how == "max"
               else (out + flipped) / 2.0 if how == "mean" else out + flipped)
        out = out[:, : (grid_lat + 1) // 2, :]
    return np.ascontiguousarray(out.reshape(n, -1), dtype=np.float32)


def level_fine(split, area_raw, name):
    """레벨 이름 -> fine 격자 (n, FINE_CELLS). AREA_SOURCE 가 여기서 반영된다."""
    level = LEVEL_NAMES.index(name)
    if AREA_SOURCE == "true" and EXTRA_AVAILABLE:
        return EXTRA[split]["area_true"][:, level]
    fine = area_raw[:, level]
    return fine * MU_W if AREA_SOURCE == "mu_approx" else fine


def channel_fine(split, area_raw, name):
    """채널 이름 -> (fine 격자, 집계 방식)."""
    if name in LEVEL_NAMES:
        return level_fine(split, area_raw, name), "sum"
    if name in ("grad_west", "grad_east"):
        west, east = edge_gradients(level_fine(split, area_raw, USE_LEVELS[0]))
        return (west if name == "grad_west" else east), "sum"
    if name == "depth_max":
        return EXTRA[split]["depth_max"], "max"      # 깊이는 더하면 안 된다
    if name == "depth_mass":
        return EXTRA[split]["depth_sum"], "sum"      # Σθ_b — 크기까지 반영한 양
    if name == "ratio":
        return EXTRA[split]["ratio"], "mean"
    if name == "sharp":
        return EXTRA[split]["sharp"], "mean"         # 경계 강도는 평균이 맞다
    raise KeyError(name)


def frame_features(split):
    if not (USE_FRAME and EXTRA_AVAILABLE and FRAME_USE):
        return None
    columns = [CACHE_FRAME_COLUMNS.index(c) for c in FRAME_USE]
    return np.nan_to_num(EXTRA[split]["frame"][:, columns], nan=0.0).astype(np.float32)


# ---- 릿지용 이미지 요약 -----------------------------------------------------
def image_summary(split, area_raw):
    """이미지 1장당 스칼라 요약 (n_images, M). 릿지 피처와 τ 추정의 재료다.

    GRU 는 격자를 통째로 받지만 릿지는 **압축된 소수의 물리량**을 받아야 한다.
    유효 표본이 480개(20프레임 슬라이딩 윈도우)라 피처를 늘릴 여지가 없다.
    """
    dark = level_fine(split, area_raw, "dark0.45")
    bright = level_fine(split, area_raw, "bright")
    west, east = edge_gradients(dark)
    n = len(dark)

    def band(fine, columns=None, rows=None):
        block = fine.reshape(n, FINE_LAT, FINE_LON)
        if rows is not None:
            block = block[:, rows]
        if columns is not None:
            block = block[:, :, columns]
        return block.sum(axis=(1, 2)).astype(np.float32)

    third = FINE_LAT // 3
    columns = {
        "ch_all":  band(dark),
        "ch_cm":   band(dark, CENTRAL_BAND),
        "ch_north": band(dark, CENTRAL_BAND, slice(0, third)),
        "ch_eq":   band(dark, CENTRAL_BAND, slice(third, 2 * third)),
        "ch_south": band(dark, CENTRAL_BAND, slice(2 * third, None)),
        "ar_all":  band(bright),
        "ar_cm":   band(bright, CENTRAL_BAND),
        "grad_west": band(west, CENTRAL_BAND),
        "grad_east": band(east, CENTRAL_BAND),
    }
    if USE_DEPTH and EXTRA_AVAILABLE:
        depth = EXTRA[split]["depth_max"].reshape(n, FINE_LAT, FINE_LON)
        columns["depth_cm"] = depth[:, :, CENTRAL_BAND].max(axis=(1, 2)).astype(np.float32)
    if USE_SHARP and SHARP_AVAILABLE:
        sharp = EXTRA[split]["sharp"].reshape(n, FINE_LAT, FINE_LON)
        columns["sharp_cm"] = sharp[:, :, CENTRAL_BAND].mean(axis=(1, 2)).astype(np.float32)
    return list(columns), np.stack(list(columns.values()), axis=1).astype(np.float32)


def build_grid(split, area_raw):
    """-> K1Grid(seq, cells, summary, cm_area)."""
    stacked, ballistic = [], []
    for name in CH_CHANNELS:
        fine, how = channel_fine(split, area_raw, name)
        coarse = aggregate_fine(fine, GRID_LAT, GRID_LON, FOLD_LATITUDE, LON_MAP, how)
        stacked.append(coarse)
        if name in BALLISTIC_CHANNELS:
            ballistic.append(coarse)
    sequence = np.stack(stacked, axis=1).reshape(len(area_raw), -1)
    frame = frame_features(split)
    if frame is not None:
        sequence = np.concatenate([sequence, frame], axis=1)
    names, summary = image_summary(split, area_raw)
    globals()["SUMMARY_COLUMNS"] = names
    return K1Grid(seq=np.ascontiguousarray(sequence, np.float32),
                  cells=np.ascontiguousarray(np.stack(ballistic, axis=1), np.float32),
                  summary=summary,
                  cm_area=np.ascontiguousarray(summary[:, names.index("ch_cm")]))


def configure(**overrides):
    """P9 의 configure 를 대체한다. 멤버 전환은 이 함수로만 한다."""
    globals().update(overrides)
    global GRID_LAT, GRID_LON, USED_LAT, N_CELLS, CENTRAL_LON, EQUATOR_ROW
    global LEVEL_INDEX, N_USED_LEVELS, CH_SEQ_DIM, CH_CHANNELS, BALLISTIC_CHANNELS
    global train_grid, val_grid, test_grid, LON_MAP
    global AREA_LAT_ROWS, N_AREA_OFFSETS, BALLISTIC_DIM, N_GATHER

    GRID_LAT, GRID_LON = CH_GRID
    assert FINE_LAT % GRID_LAT == 0, "위도 격자가 fine 격자를 나누지 못한다"
    USED_LAT = (GRID_LAT + 1) // 2 if FOLD_LATITUDE else GRID_LAT
    N_CELLS = USED_LAT * GRID_LON
    CENTRAL_LON = GRID_LON // 2
    EQUATOR_ROW = USED_LAT - 1 if FOLD_LATITUDE else GRID_LAT // 2
    LEVEL_INDEX = [LEVEL_NAMES.index(name) for name in USE_LEVELS]
    N_USED_LEVELS = len(LEVEL_INDEX)

    LON_MAP = lon_columns(FINE_LON, GRID_LON, EQUAL_ANGLE_LON)
    assert len(np.unique(LON_MAP)) == GRID_LON, "빈 경도 열이 생겼다 — GRID_LON 을 줄일 것"

    CH_CHANNELS = list(USE_LEVELS)
    if USE_GRADIENT:
        CH_CHANNELS += ["grad_west", "grad_east"]
    if USE_DEPTH and EXTRA_AVAILABLE:
        CH_CHANNELS += ["depth_max", "depth_mass"]
    if USE_RATIO and EXTRA_AVAILABLE:
        CH_CHANNELS += ["ratio"]
    if USE_SHARP and SHARP_AVAILABLE:
        CH_CHANNELS += ["sharp"]
    BALLISTIC_CHANNELS = list(USE_LEVELS)
    if USE_GRADIENT:
        BALLISTIC_CHANNELS = BALLISTIC_CHANNELS + ["grad_west"]
    if DEPTH_IN_BALLISTIC and USE_DEPTH and EXTRA_AVAILABLE:
        BALLISTIC_CHANNELS = BALLISTIC_CHANNELS + ["depth_max"]

    train_grid = build_grid("train", train_area)
    val_grid = build_grid("validation", val_area)
    test_grid = build_grid("test", test_area)

    CH_SEQ_DIM = int(train_grid.seq.shape[1])
    AREA_LAT_ROWS = list(range(USED_LAT)) if BALLISTIC_LAT == "profile" else [EQUATOR_ROW]
    N_AREA_OFFSETS = (len(BALLISTIC_OFFSETS) if BALLISTIC_SOURCE == "window"
                      else len(TRANSIT_SPEEDS))
    _, n_lons = ballistic_columns()
    BALLISTIC_DIM = (N_AREA_OFFSETS * len(AREA_LAT_ROWS) * n_lons
                     * len(BALLISTIC_CHANNELS))
    N_GATHER = len(GATHER_OFFSETS) if ADAPTIVE_GATHER else len(TRANSIT_SPEEDS)


def flatten_ch(grid, indexes):
    """(n, 20) 인덱스 -> (n, 20, CH_SEQ_DIM)."""
    return grid.seq[indexes].astype(np.float32)


configure()
print(f"K1 격자 {GRID_LAT}x{GRID_LON} -> 셀 {N_CELLS} | 채널 {CH_CHANNELS}")
print(f"  ch_seq {CH_SEQ_DIM} / 탄도 {BALLISTIC_DIM} (채널 {BALLISTIC_CHANNELS})")
print(f"  면적 {AREA_SOURCE} / 등각경도 {EQUAL_ANGLE_LON} / 요약열 {SUMMARY_COLUMNS}")

_uniform = lon_columns(FINE_LON, GRID_LON, False)
_raw = aggregate_fine(train_area[:, LEVEL_INDEX[0]], GRID_LAT, GRID_LON,
                      FOLD_LATITUDE, _uniform, "sum")
_fix = aggregate_fine(train_area[:, LEVEL_INDEX[0]] * MU_W, GRID_LAT, GRID_LON,
                      FOLD_LATITUDE, lon_columns(FINE_LON, GRID_LON, True), "sum")
print(f"\n[P2] 총 CH 면적 원본 {_raw.sum(1).mean():.4f} -> 보정 {_fix.sum(1).mean():.4f}")
print(f"     프레임간 상대변동 {_raw.sum(1).std() / _raw.sum(1).mean():.4f} -> "
      f"{_fix.sum(1).std() / _fix.sum(1).mean():.4f}  (작을수록 가짜 변조가 줄었다)")
_w, _e = edge_gradients(level_fine("train", train_area, "dark0.45"))
print(f"[P2] 경도 기울기 서쪽 {_w.sum(1).mean():.5f} / 동쪽 {_e.sum(1).mean():.5f} "
      f"(비대칭이면 선행/후행 가장자리가 구분되고 있다는 뜻)")
del _raw, _fix, _uniform, _w, _e
'''

# ============================================================================
CELL_TAU = r'''
# ==========================  [P3] τ 재설계  ================================
# P9 [R2] 는 **최근 4스텝 관측 속도**로 τ 를 정했다. 그건 *이미 지나간 스트림*의
# 속도라, 고속류 직전 저속 구간에서는 τ 를 정반대로 민다. 그래서 졌다.
#
# 전달시간을 정하는 것은 **그 스트림 자신의 속도**이고, 경험적으로 속도는 중앙자오선
# 부근 CH 면적에 비례한다 (Rotter et al. 2012).  v ≈ a + b·A
# 그런데 A 를 읽을 소스 시각이 τ 에 의존한다 -> **암시적 방정식**이라 고정점으로 푼다.
#
#   τ⁰ = 108h  ->  소스 인덱스  ->  A  ->  v = a+b·A  ->  τ¹ = AU/v  ->  ...
#
# a, b 는 **train 에서만** 2 파라미터 최소제곱으로 적합한다 (과적합 여지가 없다).

TAU_A, TAU_B = float(train_wind.mean()), 0.0


def summary_sequence(grid, indexes):
    """(n, 20, M) — 샘플별 20프레임의 요약열."""
    return grid.summary[indexes]


def cm_area_sequence(grid, indexes):
    """(n, 20, 1) — 중앙자오선 CH 면적."""
    return grid.cm_area[indexes][:, :, None]


def fixed_index(n_samples, hours):
    """τ 가 상수일 때의 기준 인덱스. (n, 12)"""
    raw = LAST_INDEX + (HORIZONS[None, :] - float(hours)) / 6.0
    return np.repeat(raw, n_samples, axis=0)


def source_index(grid, indexes, wind):
    """[P3] 클리핑 전 소스 인덱스. (n, 12)"""
    raw = fixed_index(len(indexes), REFERENCE_TRANSIT_HOURS)
    if TAU_MODE != "area":
        return raw
    sequence = cm_area_sequence(grid, indexes)
    for _ in range(TAU_ITERATIONS):
        area = pick_per_sample(sequence, np.clip(raw, 0.0, LAST_INDEX
                                                ).astype(np.float32))[:, :, 0]
        speed = np.clip(TAU_A + TAU_B * area, SPEED_FLOOR, SPEED_CEIL)
        tau = TAU_SCALE * AU_KM / speed / 3600.0
        raw = LAST_INDEX + (HORIZONS[None, :] - tau) / 6.0
    return raw


def fit_tau_relation():
    """v = a + b·A 를 train 에서 적합한다. 파라미터 2개."""
    sequence = cm_area_sequence(train_grid, train_index_matrix)
    index = np.clip(fixed_index(len(train_inputs), REFERENCE_TRANSIT_HOURS),
                    0.0, LAST_INDEX).astype(np.float32)
    area = pick_per_sample(sequence, index)[:, :, 0].ravel()
    design = np.stack([np.ones_like(area), area], axis=1)
    solution, *_ = np.linalg.lstsq(design, train_targets.ravel(), rcond=None)
    correlation = float(np.corrcoef(area, train_targets.ravel())[0, 1])
    return float(solution[0]), float(solution[1]), correlation


TAU_A, TAU_B, TAU_CORR = fit_tau_relation()
_area = pick_per_sample(cm_area_sequence(train_grid, train_index_matrix),
                        np.clip(fixed_index(len(train_inputs), REFERENCE_TRANSIT_HOURS),
                                0.0, LAST_INDEX).astype(np.float32))[:, :, 0]
_speed = np.clip(TAU_A + TAU_B * _area, SPEED_FLOOR, SPEED_CEIL)
_tau = TAU_SCALE * AU_KM / _speed / 3600.0
print(f"[P3] v = {TAU_A:.1f} + {TAU_B:.1f}·A   (면적↔속도 상관 {TAU_CORR:+.4f})")
print(f"     함의 τ 분위수 {np.percentile(_tau, [5, 25, 50, 75, 95]).round(1)} h "
      f"(5/25/50/75/95%) — 고정값은 {REFERENCE_TRANSIT_HOURS:.0f}h")
print(f"     TAU_MODE = {TAU_MODE}")
del _area, _speed, _tau


# ---- 아래 4개는 P9 원본을 K1 시그니처로 재정의한 것이다 --------------------
def area_indices(grid, indexes, wind):
    if BALLISTIC_SOURCE == "window":
        raw = source_index(grid, indexes, wind)[:, :, None] \
            + np.asarray(BALLISTIC_OFFSETS, np.float64)[None, None, :]
        return np.clip(raw, 0.0, LAST_INDEX).astype(np.float32)
    return fixed_speed_indices(len(indexes), TRANSIT_SPEEDS)


def gather_indices(grid, indexes, wind):
    if ADAPTIVE_GATHER:
        raw = source_index(grid, indexes, wind)[:, :, None] \
            + np.asarray(GATHER_OFFSETS, np.float64)[None, None, :]
        return np.clip(raw, 0.0, LAST_INDEX).astype(np.float32)
    return fixed_speed_indices(len(indexes), TRANSIT_SPEEDS)


def transit_flags(grid, indexes, wind):
    """관측창 이탈 플래그. (n, 12, FLAG_DIM)"""
    raw = source_index(grid, indexes, wind)
    over = np.clip(raw - LAST_INDEX, 0.0, None) / 8.0
    under = np.clip(-raw, 0.0, None) / 8.0
    return np.stack([(raw > LAST_INDEX).astype(np.float32),
                     (raw < 0.0).astype(np.float32),
                     over.astype(np.float32), under.astype(np.float32)],
                    axis=2).astype(np.float32)


def ballistic_area(grid, indexes, index):
    """탄도 소스 시각의 셀 값. index (n, 12, K) -> (n, 12, K·D)"""
    columns, _ = ballistic_columns()
    sequence = grid.cells[indexes][:, :, :, columns].reshape(len(indexes), 20, -1)
    n_samples, n_horizon, n_offset = index.shape
    picked = pick_per_sample(sequence, index.reshape(n_samples, n_horizon * n_offset))
    return picked.reshape(n_samples, n_horizon, -1).astype(np.float32)


def build_ballistic(grid, indexes, wind):
    return ballistic_area(grid, indexes, area_indices(grid, indexes, wind))


def build_arrays(inputs, index_matrix, wind, wind_valid, grid, stats, targets=None):
    """P9 원본과 같되 gather/flags 가 grid·indexes 를 받는다."""
    arrays = {
        "wind_seq": np.stack([
            (wind - stats["wind_mean"]) / stats["wind_std"],
            np.diff(wind, axis=1, prepend=wind[:, :1]) / stats["diff_std"],
            wind_valid], axis=2).astype(np.float32),
        "wind_stats": ((build_wind_stats(wind) - stats["stats_mean"])
                       / stats["stats_std"]).astype(np.float32),
        "ch_seq": ((flatten_ch(grid, index_matrix) - stats["ch_mean"])
                   / stats["ch_std"]).astype(np.float32),
        "ballistic": ((build_ballistic(grid, index_matrix, wind) - stats["ballistic_mean"])
                      / stats["ballistic_std"]).astype(np.float32),
        "gather_idx": gather_indices(grid, index_matrix, wind),
        "flags": transit_flags(grid, index_matrix, wind),
        "last_wind": np.ascontiguousarray(wind[:, -1]).astype(np.float32),
    }
    if targets is not None:
        arrays["target"] = targets.astype(np.float32)
    arrays["sample_ids"] = inputs.sample_id.to_numpy()
    return arrays
'''

# ============================================================================
CELL_RIDGE = r'''
# ==========================  [P1] 릿지 회귀  ===============================
# 왜 이게 1순위인가
#   · train 9,607 샘플은 **20프레임 슬라이딩 윈도우**라 인접 샘플이 19/20 프레임을
#     공유한다 (CODE_REPORT: "1 epoch ≈ 독립 데이터 20회 통과").
#     -> **유효 표본 ≈ 480개.** GRU 파라미터는 약 31만개다.
#   · 지금까지의 증상(epoch 1 최적·시드 요동·CV 무력화)이 전부 여기서 나온다.
#   · 그리고 결정적으로 **릿지는 GRU 와 오차가 다르다.** 평균의 이득은 개별 성능이 아니라
#     오차상관 ρ 가 정한다. 실측 ρ 는 GRU 계열끼리 0.85~0.92 에 몰려 있다.
#   · 덤: 릿지는 추정기 분산이 몇 자릿수 낮아 **CV 가 다시 작동한다.** λ 를 CV 로 고른다.

RIDGE_BASE_NAMES = ["last", "mean4", "mean", "std", "min", "max", "slope",
                    "last_minus_mean4", "range", "valid_frac",
                    "d1", "d2", "d3", "d4", "d5"]


def ridge_base(wind, wind_valid):
    """바람 시계열만으로 되는 피처. (n, 15)  — 지속성·추세·변동폭"""
    statistics = build_wind_stats(wind)
    valid = wind_valid.mean(axis=1, keepdims=True)
    recent = wind[:, -6:-1] - wind[:, -1:]          # 최근 5스텝의 마지막 대비 편차
    return np.concatenate([statistics, valid, recent], axis=1).astype(np.float32)


def ridge_design(grid, indexes, wind, wind_valid):
    """-> (n, 12, D). horizon 마다 **다른 소스 시각**의 요약을 읽는다."""
    base = ridge_base(wind, wind_valid)                       # (n, 15)
    sequence = summary_sequence(grid, indexes)                # (n, 20, M)
    raw = source_index(grid, indexes, wind)                   # (n, 12)
    picked = []
    for offset in RIDGE_OFFSETS:
        index = np.clip(raw + offset, 0.0, LAST_INDEX).astype(np.float32)
        picked.append(pick_per_sample(sequence, index))       # (n, 12, M)
    window = np.stack([sequence[:, -1, 0], sequence[:, :, 0].mean(axis=1),
                       sequence[:, -1, 0] - sequence[:, 0, 0]], axis=1)   # (n, 3)
    n = len(indexes)
    parts = [np.repeat(base[:, None, :], 12, axis=1),
             np.repeat(window[:, None, :], 12, axis=1)] + picked
    return np.concatenate(parts, axis=2).astype(np.float64)


def ridge_column_names():
    names = list(RIDGE_BASE_NAMES) + ["ch_all_T0", "ch_all_mean", "ch_all_slope"]
    for offset in RIDGE_OFFSETS:
        names += [f"{c}@{offset:+d}" for c in SUMMARY_COLUMNS]
    return names


def ridge_solve(X, y, lambdas):
    """중심화·표준화된 X 에 대한 닫힌형. -> (len(lambdas), D)"""
    gram = X.T @ X
    rhs = X.T @ y
    eye = np.eye(X.shape[1])
    return np.stack([np.linalg.solve(gram + lam * eye, rhs) for lam in lambdas])


def ridge_fit(rows, lambdas):
    """train 의 부분집합으로 적합. -> dict(mean, std, coef, intercept) — λ 축 유지"""
    X = RIDGE_X_TRAIN[rows]                                   # (n, 12, D)
    residual = train_targets[rows] - train_wind[rows][:, -1:]  # 지속성 잔차를 맞춘다
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8
    coef = np.zeros((len(lambdas), 12, X.shape[2]))
    intercept = np.zeros((len(lambdas), 12))
    for h in range(12):
        Z = (X[:, h] - mean[h]) / std[h]
        target = residual[:, h]
        offset = target.mean()
        coef[:, h] = ridge_solve(Z, target - offset, lambdas)
        intercept[:, h] = offset
    return {"mean": mean, "std": std, "coef": coef, "intercept": intercept}


def ridge_apply(model, X, last_wind, pick=None):
    """-> (n, 12) 예측. pick 은 horizon 별 λ 인덱스."""
    n = len(X)
    out = np.zeros((n, 12))
    for h in range(12):
        index = 0 if pick is None else int(pick[h])
        Z = (X[:, h] - model["mean"][h]) / model["std"][h]
        out[:, h] = Z @ model["coef"][index, h] + model["intercept"][index, h]
    return out + last_wind[:, None]


RIDGE = None
RIDGE_VAL = RIDGE_TEST = None
if USE_RIDGE:
    RIDGE_X_TRAIN = ridge_design(train_grid, train_index_matrix, train_wind, train_wind_valid)
    RIDGE_X_VAL = ridge_design(val_grid, val_index_matrix, val_wind, val_wind_valid)
    RIDGE_X_TEST = ridge_design(test_grid, test_index_matrix, test_wind, test_wind_valid)
    RIDGE_NAMES = ridge_column_names()
    assert RIDGE_X_TRAIN.shape[2] == len(RIDGE_NAMES)

    # ---- λ 선택: 시간블록 폴드 CV. 릿지는 분산이 작아 이 계측기가 실제로 작동한다 ----
    scores = np.zeros((len(RIDGE_LAMBDAS), 12))
    counts = np.zeros(12)
    for train_rows, evaluate_rows in FOLDS:
        fold_model = ridge_fit(train_rows, RIDGE_LAMBDAS)
        for h in range(12):
            Z = (RIDGE_X_TRAIN[evaluate_rows, h] - fold_model["mean"][h]) / fold_model["std"][h]
            truth = train_targets[evaluate_rows, h] - train_wind[evaluate_rows][:, -1]
            for i in range(len(RIDGE_LAMBDAS)):
                prediction = Z @ fold_model["coef"][i, h] + fold_model["intercept"][i, h]
                scores[i, h] += float(np.mean((prediction - truth) ** 2)) * len(evaluate_rows)
            counts[h] += len(evaluate_rows)
    scores = np.sqrt(scores / np.maximum(counts, 1))
    RIDGE_PICK = scores.argmin(axis=0)
    RIDGE_CV = float(scores.min(axis=0).mean())

    RIDGE = ridge_fit(np.arange(len(train_inputs)), RIDGE_LAMBDAS)
    RIDGE["pick"] = RIDGE_PICK
    RIDGE["lambdas"] = np.asarray(RIDGE_LAMBDAS)
    RIDGE["names"] = RIDGE_NAMES

    RIDGE_VAL = ridge_apply(RIDGE, RIDGE_X_VAL, val_wind[:, -1], RIDGE_PICK)
    RIDGE_TEST = ridge_apply(RIDGE, RIDGE_X_TEST, test_wind[:, -1], RIDGE_PICK)
    RIDGE_VAL_SCORE, RIDGE_VAL_HORIZON = official_rmse(val_targets, RIDGE_VAL)

    print(f"[P1] 릿지 피처 {RIDGE_X_TRAIN.shape[2]}개 / 유효표본 ≈ "
          f"{len(train_inputs) // 20}  (표본당 피처 "
          f"{RIDGE_X_TRAIN.shape[2] / max(len(train_inputs) // 20, 1):.2f})")
    print(f"     λ (horizon별) = {np.round(RIDGE['lambdas'][RIDGE_PICK], 2).tolist()}")
    print(f"     train CV {RIDGE_CV:.3f}  ->  val {RIDGE_VAL_SCORE:.3f}")
    print(f"     horizon별 val {RIDGE_VAL_HORIZON.round(2).tolist()}")

    _weight = np.abs(RIDGE["coef"][RIDGE_PICK[-1], -1])
    _order = np.argsort(_weight)[::-1][:8]
    print("     72h 에서 큰 계수 8개:",
          ", ".join(f"{RIDGE_NAMES[i]}({RIDGE['coef'][RIDGE_PICK[-1], -1, i]:+.1f})"
                    for i in _order))
    del _weight, _order
else:
    print("[P1] 릿지 건너뜀 (USE_RIDGE=False)")
'''

# ============================================================================
CELL_TRAINER = r'''
def train_full(stats, epochs, seed, evaluate=None):
    """train 전체로 고정 epoch 학습. LR 궤적은 P9 최종학습과 같다 (T_max=CV_EPOCHS).

    evaluate 가 주어지면 매 epoch val RMSE 를 기록해 곡선으로 돌려준다 ([P0] 진단).
    """
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    loader = make_batcher(train_inputs, train_index_matrix, train_wind, train_wind_valid,
                          train_grid, stats, train_targets,
                          shuffle=True, seed=seed, training=True)
    model = build_model(stats)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CV_EPOCHS)
    scaler = torch.amp.GradScaler(DEVICE.type, enabled=USE_AMP)
    curve = []
    for _ in range(epochs):
        model.train()
        for batch in loader:
            moved = {k: batch[k].to(DEVICE, non_blocking=PIN_MEMORY)
                     for k in BATCH_KEYS + ("target",)}
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(DEVICE.type, enabled=USE_AMP):
                residual = model(moved["wind_seq"], moved["wind_stats"], moved["ch_seq"],
                                 moved["ballistic"], moved["gather_idx"], moved["flags"])
            loss = metric_loss(residual.float() + moved["last_wind"].unsqueeze(1),
                               moved["target"])
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer); scaler.update()
        scheduler.step()
        if evaluate is not None:
            prediction, _ = predict_with(model, evaluate,
                                         stats["clip_low"], stats["clip_high"])
            curve.append(official_rmse(val_targets, prediction)[0])
            print(f"      epoch {len(curve):2d}  val {curve[-1]:7.3f}", flush=True)
    loader.release()
    return (model, np.asarray(curve)) if evaluate is not None else model


def predict_split(model, inputs, index_matrix, wind, valid, grid, stats, targets=None):
    batcher = make_batcher(inputs, index_matrix, wind, valid, grid, stats, targets,
                           shuffle=False)
    prediction, ids = predict_with(model, batcher, stats["clip_low"], stats["clip_high"])
    batcher.release()
    return prediction, ids


def smooth(curve, window=EPOCH_SMOOTH):
    """epoch 축 이동평균. argmin 이 단발 노이즈를 집는 것을 막는다."""
    if window <= 1 or len(curve) < window:
        return curve
    left = window // 2
    padded = np.pad(curve, (left, window - 1 - left), mode="edge")
    return np.array([padded[i:i + window].mean() for i in range(len(curve))])


def probe_epochs(settings, seed=777, max_epochs=None):
    """[P0] val 곡선을 훑어 최저점 epoch 를 돌려준다.

    ⚠️ val 로 epoch 를 고르는 것이다. **모델 순위**를 val 로 매기는 것과는 다른 문제이고
    (§1.5 가 반증한 것은 후자다), 최고 제출인 P3 자체가 val early stopping 으로
    epoch 를 골랐다. 같은 config 안에서 학습 길이를 정하는 데는 이게 최선의 신호다.
    """
    max_epochs = max_epochs or EPOCH_PROBE_MAX
    configure(**settings)
    stats = fit_stats(np.arange(len(train_inputs)))
    batcher = make_batcher(val_inputs, val_index_matrix, val_wind, val_wind_valid,
                           val_grid, stats, val_targets, shuffle=False)
    model, curve = train_full(stats, max_epochs, seed, evaluate=batcher)
    batcher.release()
    del model
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    best = int(np.argmin(smooth(curve))) + 1
    return best, curve
'''

# ============================================================================
CELL_MEMBERS = r'''
# ---- P9 사다리 설정 (셀 18 에서 그대로 옮겨 왔다) --------------------------
ABLATION_LADDER = [
    ("A0. P3 재현", dict(
        CH_GRID=(3, 5), FOLD_LATITUDE=False, USE_LEVELS=("dark0.45",),
        TRANSIT_SPEEDS=(350.0, 500.0, 700.0), BALLISTIC_SOURCE="speeds",
        BALLISTIC_LAT="equator", BALLISTIC_LON="central", BALLISTIC_OFFSETS=(0,),
        ADAPTIVE_AREA=False, ADAPTIVE_GATHER=False, USE_TRANSIT_FLAGS=False,
        USE_CH_GATHER=False, USE_BALLISTIC_WINDOW=True, CH_BIDIRECTIONAL=False)),
    ("B0. + 탄도속도 재설정", dict(TRANSIT_SPEEDS=(315.0, 385.0, 500.0))),
    ("C0. + 격자축·적도대칭", dict(CH_GRID=(6, 3), FOLD_LATITUDE=True)),
    ("D0. + 활성영역 레벨", dict(USE_LEVELS=("dark0.45", "bright"))),
    ("E0. + 탄도창(시각 4점)", dict(BALLISTIC_SOURCE="window",
                                    BALLISTIC_OFFSETS=(-4, -2, 0, 2))),
    ("E1. + 위도 프로파일", dict(BALLISTIC_LAT="profile")),
    ("E2. + 경도 전체 [R1]", dict(BALLISTIC_LON="all")),
    ("F1. + gather(단방향)", dict(USE_CH_GATHER=True)),
    ("F2. + 양방향 GRU", dict(CH_BIDIRECTIONAL=True)),
    ("F3. + gather 속도 확장", dict(TRANSIT_SPEEDS=(315.0, 345.0, 385.0,
                                                    435.0, 500.0, 600.0))),
]


def config_at(code):
    settings = {}
    for label, overrides in ABLATION_LADDER:
        settings.update(overrides)
        if label.split(".")[0] == code:
            return dict(settings)
    raise KeyError(code)


# ---- 레버 묶음 --------------------------------------------------------------
GEOMETRY_ON  = dict(AREA_SOURCE="true" if EXTRA_AVAILABLE else "mu_approx",
                    EQUAL_ANGLE_LON=True)
GEOMETRY_OFF = dict(AREA_SOURCE="raw", EQUAL_ANGLE_LON=False)
AR_ON        = dict(USE_LEVELS=("dark0.45", "bright"))     # A0 계열에 AR 을 켠다
GRADIENT_ON  = dict(USE_GRADIENT=True)
GRADIENT_OFF = dict(USE_GRADIENT=False)
PHYSICS_ON   = dict(USE_DEPTH=True, USE_SHARP=True, USE_FRAME=True, DEPTH_IN_BALLISTIC=True)
PHYSICS_OFF  = dict(USE_DEPTH=False, USE_SHARP=False, USE_FRAME=False,
                    DEPTH_IN_BALLISTIC=False)

# STAGE 하나가 전부 정한다. 손으로 켤 스위치는 없다.
#   A : P11-S1 의 정정판 — epoch 만 고친 GRU 앙상블. 기준선 재측정
#   B : A + 릿지 50:50            <- **최우선. 기대값이 제일 크다**
#   C : B + AR + 경도 미분        (재추출 불필요)
#   D : C + τ 면적 기반
#   E : D + θ_b · 경계 선명도     (k1a 캐시 필요)
#   F : 최종 — 시드 5개로 확대
STAGE_TABLE = {
    "A": dict(ridge=False, geometry=GEOMETRY_OFF, ar=False, gradient=False,
              tau="fixed", physics=False, shrink=False, seeds="normal"),
    "B": dict(ridge=True,  geometry=GEOMETRY_OFF, ar=False, gradient=False,
              tau="fixed", physics=False, shrink=False, seeds="normal"),
    "C": dict(ridge=True,  geometry=GEOMETRY_ON,  ar=True,  gradient=True,
              tau="fixed", physics=False, shrink=False, seeds="normal"),
    "D": dict(ridge=True,  geometry=GEOMETRY_ON,  ar=True,  gradient=True,
              tau="area",  physics=False, shrink=False, seeds="normal"),
    "E": dict(ridge=True,  geometry=GEOMETRY_ON,  ar=True,  gradient=True,
              tau="area",  physics=True,  shrink=True,  seeds="normal"),
    "F": dict(ridge=True,  geometry=GEOMETRY_ON,  ar=True,  gradient=True,
              tau="area",  physics=True,  shrink=True,  seeds="final"),
}
STAGE_SPEC = STAGE_TABLE[STAGE]


def member_settings(code, spec):
    settings = {**config_at(code), **PHYSICS_OFF, **GRADIENT_OFF, **spec["geometry"]}
    if spec["ar"]:
        settings.update(AR_ON)
    if spec["gradient"]:
        settings.update(GRADIENT_ON)
    if spec["physics"]:
        settings.update(PHYSICS_ON)
    return settings


def members_for(stage):
    spec = STAGE_TABLE[stage]
    seeds = ENSEMBLE_SEEDS_FINAL if spec["seeds"] == "final" else ENSEMBLE_SEEDS
    members = []
    for code in MEMBER_CODES:
        settings = member_settings(code, spec)
        for i, seed in enumerate(seeds):
            epochs = (max(1, BEST_EPOCH[code] + EPOCH_DELTAS[i % len(EPOCH_DELTAS)])
                      if EPOCH_POLICY == "auto"
                      else MEMBER_EPOCHS[i % len(MEMBER_EPOCHS)])
            members.append(dict(name=f"{code}_s{seed}_e{epochs}", settings=settings,
                                seed=seed, epochs=epochs))
    return members


# ---- [P0] epoch 자동 선택 ---------------------------------------------------
# P11 이 진 이유가 여기다. epochs=1 은 §1.5 가 반증한 CV 에서 온 값이었다.
BEST_EPOCH = {}
PROBE_CURVES = {}
if EPOCH_POLICY == "auto":
    VERBOSE_MODEL = False
    for _code in MEMBER_CODES:
        print(f"[P0] epoch 탐색 — {_code} (최대 {EPOCH_PROBE_MAX})", flush=True)
        BEST_EPOCH[_code], PROBE_CURVES[_code] = probe_epochs(
            member_settings(_code, STAGE_SPEC))
        _curve = PROBE_CURVES[_code]
        print(f"     -> 최저 {_curve.min():.3f} @ epoch {BEST_EPOCH[_code]} "
              f"| epoch 1 은 {_curve[0]:.3f} (차이 {_curve[0] - _curve.min():+.3f})")
    VERBOSE_MODEL = True
    print("\n기준선: P3 val 64.203 / P9(=F3, epoch 1) val 66.978")
    del _code, _curve
else:
    print(f"[P0] EPOCH_POLICY='fixed' -> MEMBER_EPOCHS={MEMBER_EPOCHS}")

# ---- STAGE 반영 -------------------------------------------------------------
USE_RIDGE = STAGE_SPEC["ridge"] and RIDGE is not None
TAU_MODE = STAGE_SPEC["tau"]
FIT_SHRINKAGE = STAGE_SPEC["shrink"]
if STAGE_SPEC["physics"] and not EXTRA_AVAILABLE:
    print("⚠️ 확장 캐시가 없다 -> [P4]·[L3] 은 꺼진 채로 돈다.")
MEMBERS = members_for(STAGE)

print(f"\nSTAGE = {STAGE} | GRU 멤버 {len(MEMBERS)}개 | 릿지 {USE_RIDGE}"
      f"(w={RIDGE_WEIGHT}) | 면적 {STAGE_SPEC['geometry']['AREA_SOURCE']} · "
      f"AR {STAGE_SPEC['ar']} · 경도미분 {STAGE_SPEC['gradient']} · "
      f"τ {TAU_MODE} · 물리 {STAGE_SPEC['physics']} · 수축 {FIT_SHRINKAGE}")
for m in MEMBERS:
    print(f"  {m['name']:<16s} seed {m['seed']} epoch {m['epochs']}")
'''

# ============================================================================
CELL_TRAIN = r'''
# ============================  GRU 멤버 학습  ==============================
VERBOSE_MODEL = False
MEMBER_STATES, MEMBER_STATS, VAL_PREDICTIONS, TEST_PREDICTIONS = [], [], [], []
_started = time.perf_counter()

for index, member in enumerate(MEMBERS):
    configure(**member["settings"])
    stats = fit_stats(np.arange(len(train_inputs)))          # train 행만 — val 미사용
    model = train_full(stats, member["epochs"], member["seed"])

    val_prediction, val_ids = predict_split(model, val_inputs, val_index_matrix, val_wind,
                                            val_wind_valid, val_grid, stats, val_targets)
    test_prediction, test_ids = predict_split(model, test_inputs, test_index_matrix,
                                              test_wind, test_wind_valid, test_grid, stats)
    assert val_ids == val_inputs.sample_id.tolist()
    assert test_ids == test_inputs.sample_id.tolist()

    MEMBER_STATES.append({k: v.detach().cpu() for k, v in model.state_dict().items()})
    MEMBER_STATS.append(stats)
    VAL_PREDICTIONS.append(val_prediction)
    TEST_PREDICTIONS.append(test_prediction)
    score, _ = official_rmse(val_targets, val_prediction)
    print(f"  [{index + 1}/{len(MEMBERS)}] {member['name']:<16s} "
          f"ch_seq {CH_SEQ_DIM:3d} 탄도 {BALLISTIC_DIM:3d} | val {score:7.3f} | "
          f"{(time.perf_counter() - _started) / 60:.1f}분 경과", flush=True)
    del model
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
VERBOSE_MODEL = True

GRU_VAL = np.mean(VAL_PREDICTIONS, axis=0)
GRU_TEST = np.mean(TEST_PREDICTIONS, axis=0)

_member_scores = [official_rmse(val_targets, p)[0] for p in VAL_PREDICTIONS]
_gru_score, _ = official_rmse(val_targets, GRU_VAL)
_pairs = [(a - b) ** 2 for i, a in enumerate(TEST_PREDICTIONS)
          for b in TEST_PREDICTIONS[i + 1:]]
_disagreement = float(np.sqrt(np.mean(_pairs))) if _pairs else 0.0
_mean_member = float(np.mean(_member_scores))
print(f"\nGRU 멤버 val 평균 {_mean_member:.3f} (최저 {min(_member_scores):.3f}) "
      f"-> 평균 {_gru_score:.3f}")
print(f"멤버간 test 예측 불일치 {_disagreement:.2f} RMS km/s")
# 기준은 **최고 멤버**가 아니라 **멤버 평균**이다. 볼록성이 보장하는 것은 거기까지다.
if len(MEMBERS) > 1 and _gru_score > _mean_member:
    print("🔴 평균이 멤버 평균보다 나쁘다 — 예측 정렬/통계/클립을 의심할 것")
assert len(MEMBERS) == 1 or _gru_score <= _mean_member + 0.5, \
    "앙상블이 멤버 평균보다 크게 나쁘다 — 파이프라인이 깨졌다"
'''

# ============================================================================
CELL_BLEND = r'''
# =====================  [P1] 오차상관 측정 · 혼합  =========================
# 평균의 이득을 정하는 것은 **개별 성능이 아니라 오차상관 ρ** 다.
# ρ 는 제출을 쓰지 않고 val 에서 **정확히** 잰다 (타깃이 있으니 추정이 아니라 측정이다).

def error_correlation(a, b, truth):
    ea, eb = (a - truth).ravel(), (b - truth).ravel()
    return float(np.corrcoef(ea, eb)[0, 1])


if USE_RIDGE:
    RHO = error_correlation(GRU_VAL, RIDGE_VAL, val_targets)
    _gru = official_rmse(val_targets, GRU_VAL)[0]
    _ridge = official_rmse(val_targets, RIDGE_VAL)[0]
    BLEND_VAL = (1.0 - RIDGE_WEIGHT) * GRU_VAL + RIDGE_WEIGHT * RIDGE_VAL
    BLEND_TEST = (1.0 - RIDGE_WEIGHT) * GRU_TEST + RIDGE_WEIGHT * RIDGE_TEST
    _blend = official_rmse(val_targets, BLEND_VAL)[0]
    print(f"[P1] GRU {_gru:.3f} / 릿지 {_ridge:.3f} / 혼합(w={RIDGE_WEIGHT}) {_blend:.3f}")
    print(f"     **오차상관 ρ = {RHO:.3f}**  (GRU 계열끼리의 실측 ρ 는 0.85~0.92)")
    if RHO < 0.85:
        print("     -> 구조가 다른 오차다. 평균의 이득이 GRU 끼리보다 크다.")
    else:
        print("     -> 생각보다 닮았다. 릿지 피처를 더 줄이거나 w 를 낮추는 편이 낫다.")
    # 진단용으로만 출력한다 — val 로 w 를 고르면 P6 의 함정을 반복한다.
    _grid = np.linspace(0.0, 1.0, 21)
    _curve = [official_rmse(val_targets, (1 - w) * GRU_VAL + w * RIDGE_VAL)[0]
              for w in _grid]
    print(f"     (진단) val 최적 w = {_grid[int(np.argmin(_curve))]:.2f} "
          f"-> {min(_curve):.3f}. **여기에 맞추지 않는다.** 리더보드로 정할 것")
    del _gru, _ridge, _blend, _grid, _curve
else:
    RHO = float("nan")
    BLEND_VAL, BLEND_TEST = GRU_VAL, GRU_TEST
    print("[P1] 릿지 없음 -> GRU 평균 그대로")
'''

# ============================================================================
CELL_SHRINK = r'''
# ============================  [L5] 수축 보정  ==============================
# RMSE 최적 예측은 조건부 평균이다. 고분산 추정기는 평균 주위로 과하게 흔들리므로
# horizon 별 기후값 쪽으로 수축시키면 RMSE 가 내려간다.
# **적합은 train OOF 에서만 한다** — val 에 12개라도 적합하면 P6 의 함정을 반복한다.

def oof_predictions(settings, epochs, seed):
    """폴드별 홀드아웃 예측을 모아 train 전체 크기로 돌려준다."""
    configure(**settings)
    out = np.full((len(train_inputs), 12), np.nan)
    for train_rows, evaluate_rows in FOLDS:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        stats = fit_stats(train_rows)
        loader = make_batcher(train_inputs.iloc[train_rows], train_index_matrix[train_rows],
                              train_wind[train_rows], train_wind_valid[train_rows],
                              train_grid, stats, train_targets[train_rows],
                              shuffle=True, seed=seed, training=True)
        model = build_model(stats)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                      weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CV_EPOCHS)
        scaler = torch.amp.GradScaler(DEVICE.type, enabled=USE_AMP)
        for _ in range(epochs):
            model.train()
            for batch in loader:
                moved = {k: batch[k].to(DEVICE, non_blocking=PIN_MEMORY)
                         for k in BATCH_KEYS + ("target",)}
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(DEVICE.type, enabled=USE_AMP):
                    residual = model(moved["wind_seq"], moved["wind_stats"],
                                     moved["ch_seq"], moved["ballistic"],
                                     moved["gather_idx"], moved["flags"])
                loss = metric_loss(residual.float() + moved["last_wind"].unsqueeze(1),
                                   moved["target"])
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(optimizer); scaler.update()
            scheduler.step()
        loader.release()
        prediction, _ = predict_split(model, train_inputs.iloc[evaluate_rows],
                                      train_index_matrix[evaluate_rows],
                                      train_wind[evaluate_rows],
                                      train_wind_valid[evaluate_rows],
                                      train_grid, stats)
        out[evaluate_rows] = prediction
        del model
        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
    return out


if FIT_SHRINKAGE:
    _reference = MEMBERS[len(MEMBERS) // 2]
    print(f"[L5] OOF 적합 — {_reference['name']} 로 {len(FOLDS)}폴드", flush=True)
    _oof = oof_predictions(_reference["settings"], _reference["epochs"], _reference["seed"])
    _rows = np.isfinite(_oof).all(axis=1)
    _centered = _oof[_rows] - TRAIN_CLIMATOLOGY
    _target = train_targets[_rows] - TRAIN_CLIMATOLOGY
    _covariance = np.array([float(np.mean(_centered[:, h] * _target[:, h])) for h in range(12)])
    _variance = np.array([float(np.mean(_centered[:, h] ** 2)) for h in range(12)])
    LAM_SINGLE = _covariance / np.maximum(_variance, 1e-9)

    # OOF 는 멤버 1개의 예측이지만 보정 대상은 K개 평균이라 분산이 작다.
    #   Var(p_1)=V+σ², Var(p_K)=V+σ²/K, Cov(p,y)=C  ->  λ_K = C/(A − σ²(1−1/K))
    _k = max(len(TEST_PREDICTIONS), 1)
    _sigma2 = (np.var(np.stack(TEST_PREDICTIONS), axis=0, ddof=1).mean(axis=0)
               if _k > 1 else np.zeros(12))
    _denominator = np.maximum(_variance - _sigma2 * (1.0 - 1.0 / _k), 0.25 * _variance)
    LAM_H = np.clip(LAM_SINGLE * _variance / np.maximum(_denominator, 1e-9), *SHRINK_CLIP)
    print(f"[L5] 멤버1 λ = {np.round(np.clip(LAM_SINGLE, 0, 2), 3).tolist()}")
    print(f"[L5] 멤버{_k} λ = {np.round(LAM_H, 3).tolist()}   <- 실제로 쓰는 값")
    if float(np.mean(LAM_H <= SHRINK_CLIP[0] + 1e-9)) > 0.5:
        print("     ⚠️ λ 가 하한에 붙었다. 이 단계가 직전보다 나쁘면 이 항부터 뺄 것.")
else:
    LAM_H = np.ones(12, np.float64)
    print("[L5] 건너뜀")


def finalize(prediction):
    shrunk = TRAIN_CLIMATOLOGY[None, :] + LAM_H[None, :] * (prediction - TRAIN_CLIMATOLOGY)
    return np.clip(shrunk, 200.0, 1200.0)
'''

# ============================================================================
CELL_SUBMIT = r'''
# ============================  제출물 저장  ================================
FINAL_TEST = finalize(BLEND_TEST)
FINAL_VAL = finalize(BLEND_VAL)
_score, _horizon = official_rmse(val_targets, FINAL_VAL)

P3_PER_HORIZON = np.array([28.35, 44.22, 53.62, 59.68, 64.32, 68.00,
                           70.64, 72.72, 74.44, 76.22, 78.20, 80.02])
print(f"validation (참고용, 선택에 쓰지 않는다)  P3 {P3_PER_HORIZON.mean():.3f} -> "
      f"K1-{STAGE} {_score:.3f}")
print(pd.DataFrame({"horizon": HORIZONS, "P3": P3_PER_HORIZON,
                    "K1": _horizon.round(2),
                    "delta": (_horizon - P3_PER_HORIZON).round(2),
                    "persistence": VAL_PERSISTENCE.round(2),
                    "climatology": VAL_CLIMATOLOGY.round(2)}).to_string(index=False))

submission = pd.DataFrame(FINAL_TEST, columns=TARGET_COLUMNS)
submission.insert(0, "sample_id", test_inputs.sample_id.tolist())
assert submission.shape == (len(test_inputs), 13) and np.isfinite(FINAL_TEST).all()
submission.to_csv(SUBMISSION_DIR / "submission.csv", index=False)

torch.save({
    "code_version": "K1", "stage": STAGE,
    "members": MEMBER_STATES,
    "member_stats": MEMBER_STATS,
    "configs": [{"name": m["name"], "settings": m["settings"],
                 "seed": m["seed"], "epochs": m["epochs"]} for m in MEMBERS],
    "ridge": RIDGE, "ridge_weight": RIDGE_WEIGHT if USE_RIDGE else 0.0,
    "tau": {"mode": TAU_MODE, "a": TAU_A, "b": TAU_B, "iterations": TAU_ITERATIONS},
    "lam_h": np.asarray(LAM_H),
    "train_climatology": TRAIN_CLIMATOLOGY,
    "best_epoch": BEST_EPOCH, "epoch_policy": EPOCH_POLICY,
    "rho_gru_ridge": RHO,
    "ensemble": "equal weight mean + fixed ridge weight, 적합 파라미터 0개",
    "initialization": "random_from_scratch",
    "val_rmse": _score,
}, SUBMISSION_DIR / "model.pth")

print(f"\nsubmission.csv {submission.shape} / model.pth "
      f"GRU {len(MEMBER_STATES)}개 + 릿지 {'있음' if USE_RIDGE else '없음'} 저장")
print(submission[TARGET_COLUMNS].describe().loc[["mean", "std", "min", "max"]].round(1))
'''

# ============================================================================
CELL_REPRODUCE = r'''
# ======================  재현성 검증 (제출 전 필수)  =======================
# 규정: 제출 코드로 다시 추론한 결과가 실제 제출과 크게 다르면 불이익.
# model.pth 만으로 GRU 멤버와 릿지를 전부 복원해 submission.csv 를 다시 만든다.

blob = torch.load(SUBMISSION_DIR / "model.pth", map_location="cpu", weights_only=False)
TAU_MODE = blob["tau"]["mode"]
TAU_A, TAU_B = blob["tau"]["a"], blob["tau"]["b"]

reproduced = []
for config, state, stats in zip(blob["configs"], blob["members"], blob["member_stats"]):
    configure(**config["settings"])
    model = build_model(stats)
    model.load_state_dict(state)
    model.to(DEVICE)
    prediction, _ = predict_split(model, test_inputs, test_index_matrix, test_wind,
                                  test_wind_valid, test_grid, stats)
    reproduced.append(prediction)
    del model
    gc.collect()

gru_mean = np.mean(reproduced, axis=0)
weight = float(blob["ridge_weight"])
if weight > 0.0 and blob["ridge"] is not None:
    design = ridge_design(test_grid, test_index_matrix, test_wind, test_wind_valid)
    ridge_prediction = ridge_apply(blob["ridge"], design, test_wind[:, -1],
                                   blob["ridge"]["pick"])
    blended = (1.0 - weight) * gru_mean + weight * ridge_prediction
else:
    blended = gru_mean

reproduced_final = np.clip(
    blob["train_climatology"][None, :]
    + np.asarray(blob["lam_h"])[None, :]
    * (blended - blob["train_climatology"]), 200.0, 1200.0)
saved = pd.read_csv(SUBMISSION_DIR / "submission.csv")[TARGET_COLUMNS].to_numpy()
gap = float(np.sqrt(np.mean((reproduced_final - saved) ** 2)))
print(f"재현 오차 {gap:.6f} RMS km/s  ({'통과' if gap < 0.01 else '🔴 실패 — 원인 확인'})")
assert gap < 0.01, "model.pth 로 submission.csv 가 재현되지 않는다"
'''

# ============================================================================
CELL_CHECK = r'''
# ============================  제출 점검  ==================================
EXPECTED_TEST_ROWS = 3868

# 노트북 자신을 submission/code.ipynb 로 복사한다. **저장(Ctrl+S) 뒤** 실행할 것.
NOTEBOOK_NAME = "code_K1.ipynb"
if Path(NOTEBOOK_NAME).exists():
    shutil.copyfile(NOTEBOOK_NAME, SUBMISSION_DIR / "code.ipynb")
    print(f"{NOTEBOOK_NAME} -> submission/code.ipynb 복사")
else:
    print(f"⚠️ {NOTEBOOK_NAME} 없음 — 노트북을 submission/code.ipynb 로 직접 저장할 것")

ok = True
for name in ["code.ipynb", "model.pth", "submission.csv"]:
    path = SUBMISSION_DIR / name
    if path.exists():
        print(f"  {name:16s} {path.stat().st_size / 1024 ** 2:8.2f} MiB")
    else:
        print(f"  {name:16s} 없음")
        ok = False

check = pd.read_csv(SUBMISSION_DIR / "submission.csv")
rows_ok = len(check) == EXPECTED_TEST_ROWS
columns_ok = list(check.columns) == ["sample_id"] + TARGET_COLUMNS
ids_ok = check.sample_id.tolist() == test_inputs.sample_id.tolist()
missing = int(check.isna().sum().sum())
values = check[TARGET_COLUMNS].to_numpy()
range_ok = bool((values >= 200.0).all() and (values <= 1200.0).all())
ok = ok and rows_ok and columns_ok and ids_ok and missing == 0 and range_ok

print(f"\n행 수      {len(check)} / 규정 {EXPECTED_TEST_ROWS}   {'OK' if rows_ok else '🔴'}")
print(f"컬럼       {'OK' if columns_ok else '🔴'} / sample_id 순서 {'OK' if ids_ok else '🔴'}")
print(f"결측       {missing}")
print(f"값 범위    {values.min():.1f} ~ {values.max():.1f} km/s  "
      f"{'OK' if range_ok else '🔴 물리 범위(200~1200) 밖'}")
print(f"구성       GRU {len(MEMBER_STATES)}개 + 릿지 {USE_RIDGE} (w={RIDGE_WEIGHT}) "
      f"| STAGE {STAGE} | ρ={RHO:.3f}")
print("\n최종:", "통과" if ok else "실패 — 위 항목 확인")
'''


def main():
    notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
    cells = notebook["cells"][: KEEP_THROUGH + 1]
    cells[0] = md(HEADER)
    cells[15] = md("".join(cells[15]["source"]) + "\n" + LADDER_NOTE)
    cells += [
        md("## K1 — 릿지 멤버 · epoch 정정 · AR/경도미분 · τ 재설계 · 경계 선명도\n\n"
           "**제출할 때 바꾸는 것은 아래 셀의 `STAGE` 한 줄뿐이다.**\n\n"
           "| STAGE | 내용 | 확장 캐시 |\n|---|---|---|\n"
           "| A | [P0] epoch 정정한 GRU 앙상블 (P11-S1 의 정정판) | 불필요 |\n"
           "| **B** | **A + [P1] 릿지 50:50 — 최우선** | 불필요 |\n"
           "| C | B + [P2] AR · 경도 미분 · μ 보정 | 불필요 |\n"
           "| D | C + [P3] τ 면적 기반 | 불필요 |\n"
           "| E | D + [P4] 경계 선명도 · θ_b · 수축 | **필요** |\n"
           "| F | 최종 — 시드 5개로 확대 | 필요 |\n"),
        code(CELL_CONFIG),
        md("### K1 피처 레이어 — μ 보정 · 등각 경도 · **fine 격자 경도 미분** · 릿지용 요약"),
        code(CELL_FEATURES),
        md("### [P3] τ 재설계 — 소스 시각의 CH 면적으로 전달시간을 푼다"),
        code(CELL_TAU),
        md("### [P1] 릿지 회귀 — 저분산 추정기이자 **오차가 다른** 멤버"),
        code(CELL_RIDGE),
        md("### 학습기 · [P0] epoch 진단"),
        code(CELL_TRAINER),
        md("### 멤버 구성 — STAGE 가 전부 정한다"),
        code(CELL_MEMBERS),
        md("### GRU 멤버 학습 · 무적합 평균"),
        code(CELL_TRAIN),
        md("### [P1] 오차상관 측정 · 혼합"),
        code(CELL_BLEND),
        md("### [L5] 수축 보정 — train OOF 에서만 적합"),
        code(CELL_SHRINK),
        md("### 제출물 저장"),
        code(CELL_SUBMIT),
        md("### 재현성 검증"),
        code(CELL_REPRODUCE),
        md("## 제출 점검"),
        code(CELL_CHECK),
    ]
    notebook["cells"] = cells
    TARGET.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{TARGET.name} 생성: 셀 {len(cells)}개 (P9 0~{KEEP_THROUGH} + K1 신규)")


if __name__ == "__main__":
    main()
