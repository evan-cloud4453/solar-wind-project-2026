#!/usr/bin/env python
"""code_p9.ipynb -> code_p11.ipynb 생성기 (개발 도구, 제출물 아님).

P9 의 셀 0~16 을 **한 글자도 고치지 않고** 그대로 가져온 뒤, P11 레이어를
뒤에 붙여 함수를 재정의한다. 검증된 경로를 건드리지 않는 것이 요점이다.
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "Trial/P9/code_p9.ipynb"
TARGET = HERE / "code_p11.ipynb"
KEEP_THROUGH = 16          # P9 셀 0~16 (설정·데이터·CH추출·탄도·Dataset·모델·지표·CV도구)


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.strip("\n").splitlines(True)}


# ============================================================================
CELL_CONFIG = r'''
# ============================  P11 레이어 시작  ============================
# 위 셀(P9 원본)은 한 글자도 고치지 않았다. 아래에서 함수를 재정의해 덮어쓴다.
# P9 경로로 되돌리려면 이 셀부터 끝까지 실행하지 않으면 된다.

P11_VERSION = "p11a"

# ---- [L2] 기하 보정 --------------------------------------------------------
AREA_SOURCE     = "mu_approx"   # "raw" | "mu_approx"(재추출 불필요) | "true"(p11a 필요)
EQUAL_ANGLE_LON = True          # 경도 셀을 등각으로 (x 등간격 -> sinφ 등간격)
MU_FLOOR        = 0.30          # 1/μ 상한 클램프 (최대 3.33배). 잘라내지는 않는다

# ---- [L3] 물리 피처 (p11a 캐시 필요) ---------------------------------------
USE_DEPTH          = False      # 셀별 θ_b (최대 · 총합) — WSA 경계거리 항
USE_RATIO          = False      # 셀별 211/193 비 (온도 대리값)
USE_FRAME          = False      # 프레임 형태 스칼라
DEPTH_IN_BALLISTIC = False      # 탄도 소스 시각에서도 θ_b 를 뽑는다
FRAME_USE = ["big_frac_true", "n_components", "big_lat_deg", "big_lon_deg", "big_depth_deg"]

# ---- [L5] 수축 보정 --------------------------------------------------------
FIT_SHRINKAGE = False           # train OOF 로 horizon별 λ 적합 (폴드 수만큼 학습 추가)
SHRINK_CLIP   = (0.70, 1.00)
LAM_H = np.ones(12, np.float64)

# ---- 앙상블 ----------------------------------------------------------------
STAGE = "S1"                    # S1->S5 / S5M. **제출할 때 여기만 바꾼다** (아래 표 참고)
ENSEMBLE_SEEDS = (777, 778, 779)
ENSEMBLE_SEEDS_FINAL = (777, 778, 779, 780, 781)

# 멤버가 도는 epoch 수. i 번째 시드가 MEMBER_EPOCHS[i % len] 만큼 돈다.
# 🔴 S1(59.7332) 이 P3(58.8028) 에 진 뒤 이 값이 1순위 용의자다. **먼저 epoch 진단 셀을
#    돌려 보고 고칠 것.** P3 는 60 epoch + early stopping 으로 학습했다 (1 epoch 이 아니다).
MEMBER_EPOCHS = (1, 2)
MEMBER_CODES = ("A0", "F3")     # 평균에 넣을 사다리 config. 약한 계열은 빼면 된다

# ---- p11a 캐시 적재 --------------------------------------------------------
P11_FRAME_COLUMNS = ["cy", "cx", "radius", "total_px", "total_true",
                     "big_frac", "big_frac_true", "n_components",
                     "big_lat_deg", "big_lon_deg", "big_depth_deg"]
P11_KEYS = ("area", "area_true", "depth_max", "depth_sum", "ratio", "frame")


def load_p11(split, files):
    path = CACHE_ROOT / f"{P11_VERSION}_{split}.npz"
    if not path.exists():
        return None
    with np.load(path) as blob:
        if [str(n) for n in blob["files"]] != list(files):
            print(f"  ⚠️ {path.name}: 파일 목록이 다르다 -> 쓰지 않는다")
            return None
        return {k: np.asarray(blob[k], np.float32) for k in P11_KEYS}


P11 = {}
for _split, _files in (("train", train_files), ("validation", val_files), ("test", test_files)):
    _loaded = load_p11(_split, _files)
    if _loaded is not None:
        P11[_split] = _loaded
P11_AVAILABLE = len(P11) == 3

print(f"p11a 캐시: {'사용 가능' if P11_AVAILABLE else '없음'}")
if P11_AVAILABLE:
    # p11a 의 `area` 는 p7a 와 **같은 알고리즘·같은 파일 순서**로 낸 것이다.
    # 어긋나면 area_true·depth 도 다른 프레임을 가리키고 있다는 뜻이라 여기서 멈춘다.
    _gap = float(np.abs(P11["train"]["area"] - train_area).max())
    print(f"  p7a 대조: 면적 최대 오차 {_gap:.2e} (0 이어야 정상)")
    assert _gap < 1e-5, "p11a 가 p7a 와 다르다 — 파일 순서 / 원반 검출을 확인할 것"
    del _gap
    # 형태 피처는 cv2/scipy 가 있어야 나온다. 없으면 상수 열이 되므로 미리 뺀다.
    _frame = P11["train"]["frame"]
    _live = [c for c in FRAME_USE
             if np.isfinite(_frame[:, P11_FRAME_COLUMNS.index(c)]).any()
             and float(np.nanstd(_frame[:, P11_FRAME_COLUMNS.index(c)])) > 1e-6]
    if _live != FRAME_USE:
        print(f"  상수/결측 프레임 스칼라 제외: {[c for c in FRAME_USE if c not in _live]}")
        FRAME_USE = _live
    del _frame, _live
else:
    print("  -> [L3] 물리 피처를 끄고 [L2] 근사판만 쓴다. p11_extract.py 를 먼저 돌릴 것.")
    if AREA_SOURCE == "true":
        AREA_SOURCE = "mu_approx"
    USE_DEPTH = USE_RATIO = USE_FRAME = DEPTH_IN_BALLISTIC = False
'''

CELL_FEATURES = r'''
from collections import namedtuple

P11Grid = namedtuple("P11Grid", "seq cells")


def mu_weights(fine_lat=FINE_LAT, fine_lon=FINE_LON, mu_floor=None, side=512):
    """fine 셀별 평균 1/μ. 캐시된 픽셀면적에 곱하면 근사 진짜면적이 된다.

    fine 격자는 반지름 r_used = R * DISK_MARGIN 의 외접 사각형을 등분한 것이므로,
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


def lon_columns(fine_lon, grid_lon, equal_angle):
    """fine 경도 열 -> coarse 열 배정. 등각이면 sinφ 경계로 나눈다."""
    if not equal_angle:
        return np.minimum((np.arange(fine_lon) * grid_lon) // fine_lon, grid_lon - 1)
    centers = ((np.arange(fine_lon) + 0.5) / fine_lon) * 2.0 - 1.0      # x / r_used
    phi = np.degrees(np.arcsin(np.clip(centers * DISK_MARGIN, -1.0, 1.0)))
    return np.clip(((phi + 90.0) / 180.0 * grid_lon).astype(int), 0, grid_lon - 1)


def _reduce(block, axis, how):
    if how == "sum":
        return block.sum(axis)
    if how == "mean":
        return block.mean(axis)
    return block.max(axis)


def aggregate_fine(fine, grid_lat, grid_lon, fold, lon_map, how="sum"):
    """fine (n, FINE_CELLS) -> (n, USED_LAT * grid_lon).

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
        if how == "max":
            out = np.maximum(out, flipped)
        elif how == "mean":
            out = (out + flipped) / 2.0
        else:
            out = out + flipped
        out = out[:, : (grid_lat + 1) // 2, :]
    return np.ascontiguousarray(out.reshape(n, -1), dtype=np.float32)


def channel_fine(split, area_raw, name):
    """채널 이름 -> (fine 격자 (n, FINE_CELLS), 집계 방식)."""
    if name in LEVEL_NAMES:
        level = LEVEL_NAMES.index(name)
        if AREA_SOURCE == "true" and P11_AVAILABLE:
            return P11[split]["area_true"][:, level], "sum"
        fine = area_raw[:, level]
        if AREA_SOURCE == "mu_approx":
            fine = fine * MU_W
        return fine, "sum"
    if name == "depth_max":
        return P11[split]["depth_max"], "max"        # 깊이는 더하면 안 된다
    if name == "depth_mass":
        return P11[split]["depth_sum"], "sum"        # Σθ_b — 크기까지 반영한 양
    if name == "ratio":
        return P11[split]["ratio"], "mean"
    raise KeyError(name)


def frame_features(split):
    if not (USE_FRAME and P11_AVAILABLE and FRAME_USE):
        return None
    columns = [P11_FRAME_COLUMNS.index(c) for c in FRAME_USE]
    return np.nan_to_num(P11[split]["frame"][:, columns], nan=0.0).astype(np.float32)


def build_grid(split, area_raw):
    """-> P11Grid(seq=(n, D) 프레임 피처, cells=(n, Cb, N_CELLS) 탄도용)."""
    stacked, ballistic = [], []
    for name in CH_CHANNELS:
        fine, how = channel_fine(split, area_raw, name)
        coarse = aggregate_fine(fine, GRID_LAT, GRID_LON, FOLD_LATITUDE, LON_MAP, how)
        stacked.append(coarse)
        if name in BALLISTIC_CHANNELS:
            ballistic.append(coarse)
    stacked = np.stack(stacked, axis=1)
    sequence = stacked.reshape(len(stacked), -1)
    frame = frame_features(split)
    if frame is not None:
        sequence = np.concatenate([sequence, frame], axis=1)
    return P11Grid(seq=np.ascontiguousarray(sequence, np.float32),
                   cells=np.ascontiguousarray(np.stack(ballistic, axis=1), np.float32))


def configure(**overrides):
    """P9 의 configure 를 대체한다. ablation·멤버 전환은 이 함수로만 한다."""
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
    if USE_DEPTH and P11_AVAILABLE:
        CH_CHANNELS += ["depth_max", "depth_mass"]
    if USE_RATIO and P11_AVAILABLE:
        CH_CHANNELS += ["ratio"]
    BALLISTIC_CHANNELS = list(USE_LEVELS)
    if DEPTH_IN_BALLISTIC and USE_DEPTH and P11_AVAILABLE:
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


def ballistic_area(grid, indexes, index):
    """탄도 소스 시각의 셀 값. index (n, 12, K) -> (n, 12, K*D)."""
    columns, _ = ballistic_columns()
    sequence = grid.cells[indexes][:, :, :, columns]
    sequence = sequence.reshape(len(indexes), 20, -1)
    n_samples, n_horizon, n_offset = index.shape
    picked = pick_per_sample(sequence, index.reshape(n_samples, n_horizon * n_offset))
    return picked.reshape(n_samples, n_horizon, -1).astype(np.float32)


configure()
print(f"P11 격자 {GRID_LAT}x{GRID_LON} -> 셀 {N_CELLS} | 채널 {CH_CHANNELS}")
print(f"  ch_seq {CH_SEQ_DIM} / 탄도 {BALLISTIC_DIM} (채널 {BALLISTIC_CHANNELS})")
print(f"  면적 소스 {AREA_SOURCE} / 등각 경도 {EQUAL_ANGLE_LON} / LON_MAP {LON_MAP.tolist()}")

# --- 기하 보정이 실제로 무엇을 바꿨는지 한 번 본다 --------------------------
_raw = aggregate_fine(train_area[:, LEVEL_INDEX[0]], GRID_LAT, GRID_LON, FOLD_LATITUDE,
                      lon_columns(FINE_LON, GRID_LON, False), "sum")
_fix = aggregate_fine(train_area[:, LEVEL_INDEX[0]] * MU_W, GRID_LAT, GRID_LON,
                      FOLD_LATITUDE, lon_columns(FINE_LON, GRID_LON, True), "sum")
print(f"\n[L2] 셀별 총 CH 면적 원본 {_raw.sum(1).mean():.4f} -> 보정 {_fix.sum(1).mean():.4f}")
print(f"     프레임간 상대변동(std/mean) {_raw.sum(1).std() / _raw.sum(1).mean():.4f} -> "
      f"{_fix.sum(1).std() / _fix.sum(1).mean():.4f}  (작을수록 가짜 변조가 줄었다는 뜻)")
del _raw, _fix
'''

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


# ---- 제출 단계별 멤버 구성 -------------------------------------------------
GEOMETRY_OFF = dict(AREA_SOURCE="raw", EQUAL_ANGLE_LON=False)
GEOMETRY_ON = dict(AREA_SOURCE="true" if P11_AVAILABLE else "mu_approx",
                   EQUAL_ANGLE_LON=True)
PHYSICS_OFF = dict(USE_DEPTH=False, USE_RATIO=False, USE_FRAME=False,
                   DEPTH_IN_BALLISTIC=False)
PHYSICS_ON = dict(USE_DEPTH=True, USE_RATIO=False, USE_FRAME=True,
                  DEPTH_IN_BALLISTIC=True)
BALLISTIC_WIDE = dict(BALLISTIC_OFFSETS=(-8, -5, -2, 0, 2, 5))

# STAGE 하나가 기하·물리·탄도창·수축·시드 수를 전부 정한다. 따로 켤 스위치는 없다.
#   S1~S5 : 제출 사다리. 한 칸 올릴 때마다 레버가 하나씩 추가된다.
#   S5M   : 계획 §2 의 K=12 구성 — **구 기하 멤버와 새 기하 멤버를 같이 평균한다.**
#           S2 가 "기하 보정이 이득"이라고 말했더라도, 두 기하는 서로 다른 오차를 내므로
#           섞는 편이 평균의 이득이 크다. S5 와 둘 중 하나를 고르면 된다.
STAGE_TABLE = {
    "S1":  dict(geometry=GEOMETRY_OFF, physics=False, wide=False, shrink=False, seeds="normal"),
    "S2":  dict(geometry=GEOMETRY_ON,  physics=False, wide=False, shrink=False, seeds="normal"),
    "S3":  dict(geometry=GEOMETRY_ON,  physics=True,  wide=False, shrink=False, seeds="normal"),
    "S4":  dict(geometry=GEOMETRY_ON,  physics=True,  wide=True,  shrink=True,  seeds="normal"),
    "S5":  dict(geometry=GEOMETRY_ON,  physics=True,  wide=True,  shrink=True,  seeds="final"),
    "S5M": dict(geometry=GEOMETRY_ON,  physics=True,  wide=True,  shrink=True,  seeds="normal",
                mixed=True),
}


def member_groups(spec):
    """(이름표, 사다리 코드, 기하, 물리) 목록. 각 그룹이 시드 수만큼 멤버를 낸다."""
    if spec.get("mixed"):
        return [("A0", "A0", GEOMETRY_OFF, PHYSICS_OFF),
                ("F3", "F3", GEOMETRY_OFF, PHYSICS_OFF),
                ("L2", "F3", GEOMETRY_ON, PHYSICS_OFF),
                ("L3", "F3", GEOMETRY_ON, PHYSICS_ON)]
    groups = [(code, code, spec["geometry"], PHYSICS_OFF) for code in MEMBER_CODES]
    if spec["physics"]:
        groups.append(("L3", "F3", spec["geometry"], PHYSICS_ON))
    return groups


def members_for(stage):
    spec = STAGE_TABLE[stage]
    seeds = ENSEMBLE_SEEDS_FINAL if spec["seeds"] == "final" else ENSEMBLE_SEEDS
    ballistic = BALLISTIC_WIDE if spec["wide"] else {}
    members = []
    for label, code, geometry, physics in member_groups(spec):
        extra = {} if code == "A0" else ballistic      # A0 는 speeds 모드라 창이 없다
        for i, seed in enumerate(seeds):
            epochs = MEMBER_EPOCHS[i % len(MEMBER_EPOCHS)]
            members.append(dict(
                name=f"{label}_s{seed}_e{epochs}",
                settings={**config_at(code), **physics, **geometry, **extra},
                seed=seed, epochs=epochs))
    return members


STAGE_SPEC = STAGE_TABLE[STAGE]
if STAGE_SPEC["physics"] and not P11_AVAILABLE:
    print("⚠️ p11a 캐시가 없다 -> L3 멤버가 F3 와 같은 설정이 된다 (시드 다양성만 는다).")
FIT_SHRINKAGE = STAGE_SPEC["shrink"]           # [L5] 도 STAGE 가 켠다
MEMBERS = members_for(STAGE)

print(f"STAGE = {STAGE} | 멤버 {len(MEMBERS)}개 | "
      f"면적 {STAGE_SPEC['geometry']['AREA_SOURCE']} · "
      f"등각경도 {STAGE_SPEC['geometry']['EQUAL_ANGLE_LON']} · "
      f"물리 {STAGE_SPEC['physics']} · 탄도창확장 {STAGE_SPEC['wide']} · "
      f"수축 {FIT_SHRINKAGE}"
      + (" · **기하 혼합**" if STAGE_SPEC.get("mixed") else ""))
for m in MEMBERS:
    print(f"  {m['name']:<16s} seed {m['seed']} epoch {m['epochs']}")
'''

CELL_PROBE = r"""
# ==================  [진단] epoch 곡선 — 제출을 쓰지 않는 계측  ==============
# P11-S1 이 P3 에 진 뒤 1순위 용의자: **멤버가 1~2 epoch 라 덜 학습됐다.**
#   · A0 는 "P3 재현" 이라고 이름 붙였지만 P3 는 60 epoch + early stopping 으로 학습했다.
#   · 실측: A0(1 epoch) val 66.6 vs **P3 val 64.203**. 같은 피처인데 2.4 벌어져 있다.
# 이 셀은 제출을 한 장도 쓰지 않고 epoch 축을 훑는다. 멤버 하나가 epoch 당 ~2초다.
#
# ⚠️ val 은 좋은 계측기가 아니다 (CODE_REPORT §1.5). 다만 **같은 모델의 epoch 축**에서는
#    쓸 수 있는 유일한 신호이고, 최고 제출인 P3 자체가 val 로 epoch 를 골랐다.

RUN_EPOCH_PROBE = False        # True 로 바꿔 한 번만 돌린다 (config 2개 x 30 epoch ≈ 4분)
PROBE_CODES = ("A0", "F3")
PROBE_EPOCHS = 30
PROBE_SEED = 777

if RUN_EPOCH_PROBE:
    VERBOSE_MODEL = False
    PROBE_CURVES = {}
    for _code in PROBE_CODES:
        configure(**{**config_at(_code), **PHYSICS_OFF, **STAGE_SPEC["geometry"]})
        _stats = fit_stats(np.arange(len(train_inputs)))
        random.seed(PROBE_SEED); np.random.seed(PROBE_SEED); torch.manual_seed(PROBE_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(PROBE_SEED)
        _loader = make_batcher(train_inputs, train_index_matrix, train_wind, train_wind_valid,
                               train_grid, _stats, train_targets,
                               shuffle=True, seed=PROBE_SEED, training=True)
        _val_loader = make_batcher(val_inputs, val_index_matrix, val_wind, val_wind_valid,
                                   val_grid, _stats, val_targets, shuffle=False)
        _model = build_model(_stats)
        _optimizer = torch.optim.AdamW(_model.parameters(), lr=LEARNING_RATE,
                                       weight_decay=WEIGHT_DECAY)
        # 멤버 학습과 **같은 LR 궤적**이어야 epoch 번호를 그대로 옮겨 쓸 수 있다
        _scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(_optimizer, T_max=CV_EPOCHS)
        _scaler = torch.amp.GradScaler(DEVICE.type, enabled=USE_AMP)
        _curve = []
        for _epoch in range(PROBE_EPOCHS):
            _model.train()
            for _batch in _loader:
                _moved = {k: _batch[k].to(DEVICE, non_blocking=PIN_MEMORY)
                          for k in BATCH_KEYS + ("target",)}
                _optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(DEVICE.type, enabled=USE_AMP):
                    _residual = _model(_moved["wind_seq"], _moved["wind_stats"],
                                       _moved["ch_seq"], _moved["ballistic"],
                                       _moved["gather_idx"], _moved["flags"])
                _loss = metric_loss(_residual.float() + _moved["last_wind"].unsqueeze(1),
                                    _moved["target"])
                _scaler.scale(_loss).backward()
                _scaler.unscale_(_optimizer)
                nn.utils.clip_grad_norm_(_model.parameters(), GRAD_CLIP)
                _scaler.step(_optimizer); _scaler.update()
            _scheduler.step()
            _prediction, _ = predict_with(_model, _val_loader,
                                          _stats["clip_low"], _stats["clip_high"])
            _curve.append(official_rmse(val_targets, _prediction)[0])
            print(f"  {_code} epoch {_epoch + 1:2d}  val {_curve[-1]:7.3f}", flush=True)
        PROBE_CURVES[_code] = np.array(_curve)
        _loader.release(); _val_loader.release()
        del _model
        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
    VERBOSE_MODEL = True

    print("\nepoch 별 val (행 = config)")
    print(pd.DataFrame(PROBE_CURVES, index=np.arange(1, PROBE_EPOCHS + 1)).round(2).to_string())
    for _code, _curve in PROBE_CURVES.items():
        print(f"  {_code}: 최저 {_curve.min():.3f} @ epoch {int(_curve.argmin()) + 1} "
              f"| epoch 1 은 {_curve[0]:.3f}  (차이 {_curve[0] - _curve.min():+.3f})")
    print("\n기준선: P3 val 64.203 / P9(=F3, epoch 1) val 66.978")
    print("곡선이 64 근처까지 내려오면 '덜 학습됐다'가 원인이었다는 뜻이다.")
    print("-> MEMBER_EPOCHS 를 최저점 부근 값 2~3개로 바꾸고 (예: (10, 16, 24)) "
          "멤버 셀부터 다시 실행할 것.")
else:
    print("[진단] epoch 곡선 건너뜀 (RUN_EPOCH_PROBE=False)")
"""


CELL_TRAIN = r'''
def train_full(stats, epochs, seed):
    """train 전체로 고정 epoch 학습. LR 궤적은 P9 최종학습과 동일하게 T_max=CV_EPOCHS."""
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
    loader.release()
    return model


def predict_split(model, inputs, index_matrix, wind, valid, grid, stats, targets=None):
    batcher = make_batcher(inputs, index_matrix, wind, valid, grid, stats, targets,
                           shuffle=False)
    prediction, ids = predict_with(model, batcher, stats["clip_low"], stats["clip_high"])
    batcher.release()
    return prediction, ids


# ============================  멤버 학습  ==================================
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

VAL_MEAN = np.mean(VAL_PREDICTIONS, axis=0)
TEST_MEAN = np.mean(TEST_PREDICTIONS, axis=0)

# ---- 앙상블이 수학적으로 기대한 대로 동작했는지만 확인한다 -----------------
# (val 로 **모델을 고르지는 않는다**. 평균이 개별 멤버보다 나은 것은 항등식에 가깝고,
#  그게 안 나오면 파이프라인이 깨진 것이므로 이 확인은 선택과 무관하다.)
_member_scores = [official_rmse(val_targets, p)[0] for p in VAL_PREDICTIONS]
_ensemble_score, _ = official_rmse(val_targets, VAL_MEAN)
_pairs = [(a - b) ** 2 for i, a in enumerate(TEST_PREDICTIONS)
          for b in TEST_PREDICTIONS[i + 1:]]
_disagreement = float(np.sqrt(np.mean(_pairs))) if _pairs else 0.0
_mean_member = float(np.mean(_member_scores))
print(f"\n멤버 val 평균 {_mean_member:.3f} (최저 {min(_member_scores):.3f}) "
      f"-> 앙상블 {_ensemble_score:.3f}")
print(f"멤버간 test 예측 불일치 {_disagreement:.2f} RMS km/s "
      f"(클수록 평균의 이득이 크다)")
# 기준은 **최고 멤버**가 아니라 **멤버 평균**이다. 볼록성이 보장하는 것은
# "평균의 오차 <= 오차의 평균" 까지이고, 최고 멤버를 이기는 것은 보장되지 않는다.
# (최고 멤버를 기준으로 두면 val 로 멤버를 고르는 것과 같아진다 — P6 의 함정)
if len(MEMBERS) > 1 and _ensemble_score > _mean_member:
    print(f"🔴 앙상블이 멤버 평균보다 나쁘다 — 예측 정렬/통계/클립을 의심할 것")
assert len(MEMBERS) == 1 or _ensemble_score <= _mean_member + 0.5, \
    "앙상블이 멤버 평균보다 크게 나쁘다 — 파이프라인이 깨졌다"
'''

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
    _centered_prediction = _oof[_rows] - TRAIN_CLIMATOLOGY
    _centered_target = train_targets[_rows] - TRAIN_CLIMATOLOGY
    _covariance = np.array([float(np.mean(_centered_prediction[:, h] * _centered_target[:, h]))
                            for h in range(12)])
    _variance = np.array([float(np.mean(_centered_prediction[:, h] ** 2)) for h in range(12)])
    LAM_SINGLE = _covariance / np.maximum(_variance, 1e-9)          # 멤버 1개짜리 λ

    # ---- 멤버 1개 -> K개 평균으로 옮긴다 (이 보정이 없으면 반드시 과수축한다) ----
    # OOF 는 멤버 하나의 예측이지만, 실제로 보정할 대상은 K개 평균이라 분산이 작다.
    #   Var(p_1) = V + σ²,  Var(p_K) = V + σ²/K,  Cov(p, y) = C  (잡음은 y 와 무관)
    #   λ_1 = C / (V + σ²) = C / A        ->      λ_K = C / (A − σ²(1 − 1/K))
    # σ² 는 멤버간 예측 분산으로 잰다. 설정 차이까지 섞여 있어 σ² 는 과대추정되고,
    # 그 방향은 λ_K -> 1, 즉 **보정을 덜 하는 쪽**이라 안전하다.
    _k = len(TEST_PREDICTIONS)
    _sigma2 = (np.var(np.stack(TEST_PREDICTIONS), axis=0, ddof=1).mean(axis=0)
               if _k > 1 else np.zeros(12))
    _denominator = np.maximum(_variance - _sigma2 * (1.0 - 1.0 / _k), 0.25 * _variance)
    LAM_H = np.clip(LAM_SINGLE * _variance / np.maximum(_denominator, 1e-9), *SHRINK_CLIP)

    print(f"[L5] 멤버1 λ  = {np.round(np.clip(LAM_SINGLE, 0, 2), 3).tolist()}")
    print(f"[L5] 멤버{_k} λ = {np.round(LAM_H, 3).tolist()}   <- 실제로 쓰는 값")
    print(f"     멤버간 분산 σ = {np.sqrt(_sigma2).mean():.1f} km/s, "
          f"단일 예측 표준편차 = {np.sqrt(_variance).mean():.1f} km/s")
    if float(np.mean(LAM_H <= SHRINK_CLIP[0] + 1e-9)) > 0.5:
        print("     ⚠️ λ 가 하한에 붙었다 = 예측이 타깃 대비 과분산이라는 뜻이다. "
              "S4 가 S3 보다 나쁘면 이 항부터 뺄 것.")
    print("     1 에 가까우면 앙상블이 이미 분산을 잘 줄였다는 뜻이고, 그것도 정보다.")
else:
    LAM_H = np.ones(12, np.float64)
    print("[L5] 건너뜀 (FIT_SHRINKAGE=False)")


def finalize(prediction):
    """수축 보정 + 물리 범위 클립."""
    shrunk = TRAIN_CLIMATOLOGY[None, :] + LAM_H[None, :] * (prediction - TRAIN_CLIMATOLOGY)
    return np.clip(shrunk, 200.0, 1200.0)
'''

CELL_SUBMIT = r'''
# ============================  제출물 저장  ================================
FINAL_TEST = finalize(TEST_MEAN)
FINAL_VAL = finalize(VAL_MEAN)
_final_val_score, _final_val_horizon = official_rmse(val_targets, FINAL_VAL)

P3_PER_HORIZON = np.array([28.35, 44.22, 53.62, 59.68, 64.32, 68.00,
                           70.64, 72.72, 74.44, 76.22, 78.20, 80.02])
print(f"validation (참고용, 선택에 쓰지 않는다)  P3 {P3_PER_HORIZON.mean():.3f} -> "
      f"P11-{STAGE} {_final_val_score:.3f}")
print(pd.DataFrame({"horizon": HORIZONS, "P3": P3_PER_HORIZON,
                    "P11": _final_val_horizon.round(2),
                    "delta": (_final_val_horizon - P3_PER_HORIZON).round(2),
                    "persistence": VAL_PERSISTENCE.round(2),
                    "climatology": VAL_CLIMATOLOGY.round(2)}).to_string(index=False))

submission = pd.DataFrame(FINAL_TEST, columns=TARGET_COLUMNS)
submission.insert(0, "sample_id", test_inputs.sample_id.tolist())
assert submission.shape == (len(test_inputs), 13) and np.isfinite(FINAL_TEST).all()
submission.to_csv(SUBMISSION_DIR / "submission.csv", index=False)

torch.save({
    "code_version": "p11", "stage": STAGE,
    "members": MEMBER_STATES,
    "member_stats": MEMBER_STATS,
    "configs": [{"name": m["name"], "settings": m["settings"],
                 "seed": m["seed"], "epochs": m["epochs"]} for m in MEMBERS],
    "lam_h": np.asarray(LAM_H),
    "train_climatology": TRAIN_CLIMATOLOGY,
    "ensemble": "equal weight mean, 적합 파라미터 0개",
    "initialization": "random_from_scratch",
    "val_rmse": _final_val_score,
}, SUBMISSION_DIR / "model.pth")

print(f"\nsubmission.csv {submission.shape} / model.pth 멤버 {len(MEMBER_STATES)}개 저장")
print(submission[TARGET_COLUMNS].describe().loc[["mean", "std", "min", "max"]].round(1))
'''

CELL_REPRODUCE = r'''
# ======================  재현성 검증 (제출 전 필수)  =======================
# 규정: "제출된 코드로 다시 추론했을 때의 결과와 실제 제출 결과의 성능이 크게
# 차이 나는 경우 불이익". model.pth 만으로 submission.csv 가 재구성되는지 확인한다.

blob = torch.load(SUBMISSION_DIR / "model.pth", map_location="cpu", weights_only=False)
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

reproduced_mean = np.clip(
    blob["train_climatology"][None, :]
    + np.asarray(blob["lam_h"])[None, :]
    * (np.mean(reproduced, axis=0) - blob["train_climatology"]), 200.0, 1200.0)
saved = pd.read_csv(SUBMISSION_DIR / "submission.csv")[TARGET_COLUMNS].to_numpy()
gap = float(np.sqrt(np.mean((reproduced_mean - saved) ** 2)))
print(f"재현 오차 {gap:.6f} RMS km/s  ({'통과' if gap < 0.01 else '🔴 실패 — 원인 확인'})")
assert gap < 0.01, "model.pth 로 submission.csv 가 재현되지 않는다"
'''

CELL_CHECK = r"""
# ============================  제출 점검  ==================================
# P9 의 점검 셀은 단일 모델의 FINAL_STATS 를 참조한다. P11 은 멤버가 여러 개라
# 그 변수가 없으므로, 규정이 요구하는 것만 직접 확인한다.

EXPECTED_TEST_ROWS = 3868

# 노트북 자신을 submission/code.ipynb 로 복사한다. Jupyter 는 실행 중인 파일 경로를
# 알려주지 않으므로 이름으로 찾는다. **저장(Ctrl+S) 한 뒤** 이 셀을 실행할 것.
NOTEBOOK_NAME = "code_p11.ipynb"
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
print(f"멤버 수    {len(MEMBER_STATES)} (STAGE {STAGE})")
print("\n최종:", "통과" if ok else "실패 — 위 항목 확인")
"""


HEADER = """
# 태양풍 속도 예측 — P11

**P9 노트북의 셀 0~16 을 한 글자도 고치지 않고 그대로 쓰고**, 그 뒤에 P11 레이어를 붙여
필요한 함수만 재정의한다. 검증된 경로(데이터 로드 · CH 추출 · 탄도 정렬 · 모델 · 지표)는
건드리지 않았다.

## 무엇이 달라졌나 (기준: P3 = 58.80)

| 레버 | 내용 | 왜 |
|---|---|---|
| **L1** | 무적합 균등가중 앙상블 (설정 × 시드 × epoch) | epoch 1 에서 꺾이는 고분산 추정기라 평균의 이득이 크다. **적합 파라미터 0개** — P6 의 NNLS(val 로 84개 적합)와 다른 물건이다 |
| **L2** | μ 보정 진짜면적 + 등각 경도 셀 | 픽셀 면적은 림에서 최대 1/2.3 로 눌린다. 자전만으로 CH 시계열이 가짜로 흔들린다. **crop 은 하지 않는다** — P10 이 −2.98 을 잃은 요인이다 |
| **L3** | CH 경계거리 θ_b · 형태 | WSA 의 θ_b 항. 같은 면적이라도 큰 홀 하나와 작은 홀 여럿은 전혀 다른 바람을 낸다 |
| **L4** | 탄도창 확장 (−8 ~ +5) | 도착할 스트림의 τ 는 알 수 없으므로 창을 넓혀 head 가 고르게 둔다 |
| **L5** | horizon별 수축 보정 (12 파라미터, train OOF 적합) | RMSE 최적 예측은 조건부 평균이다 |

## 실행 순서

1. (선택) `python p11_extract.py` 를 백그라운드로 돌려 `work/cache/p11a_*.npz` 를 만든다.
   없으면 **L3 는 자동으로 꺼지고** L2 근사판만 쓴다 — 노트북은 그대로 끝까지 돈다.
2. 위에서부터 순서대로 실행한다. **바꿀 것은 P11 설정 셀의 `STAGE` 한 줄뿐이다.**
3. 마지막 두 셀(재현성 검증 · 제출 점검)이 통과해야 제출한다.

> **CV 사다리는 돌리지 않는다.** 계측기 감도가 실제 2.56 차이를 0.013 으로 읽는 수준이라
> 하루치 GPU 를 정당화하지 못한다. 판정은 리더보드 제출로 한다.
> 셀 16 은 [L5] 가 쓰는 `FOLDS` 정의 때문에 남겨 둔 것이고 `run_cv` 는 호출하지 않는다.

> **재현성.** `model.pth` 하나에 멤버 K개의 state_dict · 정규화 통계 · 설정이 전부 들어간다.
> "재현성 검증" 셀이 그 파일만으로 `submission.csv` 를 다시 만들어 RMS 차이를 잰다.
"""

LADDER_NOTE = """
> P11 에서는 이 계측기를 **돌리지 않는다.** [L5] 가 쓰는 `FOLDS` 와 학습 루프 정의만 가져간다.
"""


def main():
    notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
    cells = notebook["cells"][: KEEP_THROUGH + 1]
    cells[0] = md(HEADER)
    cells[15] = md("".join(cells[15]["source"]) + "\n" + LADDER_NOTE)
    cells += [
        md("## P11 — 물리 피처 확장 + 무적합 앙상블\n\n"
           "P9 셀 0~16 은 그대로 두고 여기서부터 함수를 재정의한다.\n"
           "**제출할 때 바꾸는 것은 아래 셀의 `STAGE` 한 줄뿐이다.**\n\n"
           "| STAGE | 내용 |\n|---|---|\n"
           "| S1 | 앙상블만 (구 기하) — 바닥 확보 |\n"
           "| S2 | + [L2] μ 보정 · 등각 경도 |\n"
           "| S3 | + [L3] θ_b · 형태 피처 |\n"
           "| S4 | + [L4] 탄도창 확장 + [L5] 수축 보정 |\n"
           "| S5 | 최종 — 시드 5개로 확대 (멤버 15) |\n"
           "| S5M | 최종 대안 — 구 기하 + 새 기하를 같이 평균 (멤버 12) |\n"),
        code(CELL_CONFIG),
        md("### P11 피처 레이어 — `aggregate` · `configure` · `flatten_ch` 재정의"),
        code(CELL_FEATURES),
        md("### 멤버 구성 — 단계별로 어떤 모델들을 평균할 것인가"),
        code(CELL_MEMBERS),
        md("### [진단] epoch 곡선 — 제출을 쓰지 않고 멤버 품질을 먼저 본다"),
        code(CELL_PROBE),
        md("### 멤버 학습 · 무적합 평균\n\n"
           "적합 파라미터 0개의 균등가중 평균이다. P6 의 NNLS(val 로 84개 적합)와 다르다."),
        code(CELL_TRAIN),
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
    print(f"{TARGET.name} 생성: 셀 {len(cells)}개 (P9 0~{KEEP_THROUGH} + P11 신규)")


if __name__ == "__main__":
    main()
