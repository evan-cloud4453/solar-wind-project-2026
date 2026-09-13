#!/usr/bin/env python
"""P3-renew 노트북 생성 — P3 원본을 패치해서 만든다.

P3-renew 는 새 모델이 아니다. **CH 피처를 어떻게 요약하는가** 만 바꾼 것이다.

  P3        원반을 3x5 격자로 나눠 셀별 코로나홀 면적 비율 15개  -> GRU
  P3-renew  격자를 버리고 **마스크 자체에서 뽑은 전 원반 스칼라** -> GRU
            기본값은 면적 1개. `CH_FEATURE_MODE="area_shape"` 면 마스크의
            1·2차 모멘트까지 5개.

모델·손실·증강·학습 루프·Dataset·탄도 정렬은 P3 그대로다. 그래서 이 파일은 P3
노트북을 다시 타이핑하지 않고 **원본 셀을 읽어 필요한 곳만 손댄다.**

  [교체] 제목
  [삽입] 설정 셀 뒤   — P3-renew 노브
  [교체] 3절 markdown — 격자 설명 -> 마스크 설명
  [교체] 3절 코드     — compute_ch_grid -> 마스크 피처 추출 (v1/v2 마스크 선택)
  [교체] 6절 markdown — CH 브랜치 설명 한 줄
  [패치] 학습 셀      — CONFIG 에 피처 모드·마스크 버전 기록 (한 줄 치환)
  [패치] 평가 셀      — 참고 점수에 P3 추가 (한 줄 치환)

Dataset·모델 셀은 `N_CELLS` 와 `CENTRAL_CELL` 을 참조하는데, 노브 셀에서 그 두
이름을 각각 "CH 피처 개수" 와 "탄도 정렬에 쓸 열 번호" 로 다시 정의하므로 원본
셀은 글자 하나 건드리지 않는다.

사용:
    python build_p3_renew_notebook.py            # -> code_p3_renew.ipynb
    python build_p3_renew_notebook.py --check    # 패치 지점만 확인하고 쓰지 않음
"""

import argparse
import json
import sys
from pathlib import Path

BASE = Path("Trial/P3/code_p3.ipynb")
OUTPUT = Path("code_p3_renew.ipynb")


def markdown_cell(text):
    return {"cell_type": "markdown", "metadata": {},
            "source": text.strip("\n").splitlines(keepends=True)}


def code_cell(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.strip("\n").splitlines(keepends=True)}


def source_of(cell):
    return "".join(cell["source"])


# =====================================================================
TITLE_MD = r"""
# 태양풍 속도 예측 — P3-renew (격자 제거 · 순수 코로나홀 마스크 피처)

**P3 와 모델이 같다.** 바뀐 것은 *코로나홀 마스크를 무엇으로 요약해서 GRU 에
넣는가* 뿐이다.

| | P3 | P3-renew |
|---|---|---|
| CH 피처 | 원반 3x5 격자의 셀별 면적 비율 **15개** | 마스크 전체의 스칼라 **1개**(기본) 또는 5개 |
| 탄도 정렬 | 중앙자오선 **셀** 시계열을 역산 인덱스에서 보간 | 전 원반 **면적** 시계열을 같은 방식으로 보간 |
| 마스크 규칙 | 193 AND 211 < 0.45 x 원반 중앙값 | 같음(`v1`) / `ch_mask_check` 수정본(`v2`) 선택 |
| 모델·손실·증강·Dataset·학습 루프 | — | **P3 와 동일** |

## 왜 격자를 버리나

격자 15개는 면적 1개를 **포함한다** (면적은 셀 면적들의 가중합이다). 그러므로
정보량으로는 renew 가 P3 의 부분집합이고, renew 가 이기는 경우는 하나뿐이다 —
**남은 14 자유도가 신호가 아니라 잡음이었을 때.** 격자를 넣어도 점수가 안 올랐다면
바로 그 상황을 의심하는 게 맞다. 이 노트북은 그 가설을 한 번의 학습으로 검정한다.

격자가 잡음이 되기 쉬운 이유도 분명하다. 셀 경계는 태양의 물리(자기 중립선·홀
경계)와 아무 상관 없는 **프레임 좌표 기준의 사각형**이고, 원반 검출이 몇 픽셀만
흔들려도 셀별 면적은 크게 요동친다. 반면 전 원반 면적은 그 흔들림에 훨씬 둔감하다.

> `CH_FEATURE_MODE = "area_shape"` 로 두면 면적에 **마스크의 1·2차 모멘트**
> (무게중심 동서·남북, 퍼짐)와 평균 깊이를 더한 5개를 쓴다. 격자처럼 공간을
> 미리 자르지 않고 마스크 자신이 위치를 말하게 하는 중간 단계다.
"""

KNOB_MD = r"""
## 0-b. P3-renew 설정 — CH 피처 요약 방식

여기 있는 값만 바꾸면 된다. 아래 셀들은 P3 원본 그대로다.

- `CH_FEATURE_MODE` — `"area"`(기본, 스칼라 1개) / `"area_shape"`(5개)
- `CH_MASK_VERSION` — `"v1"`(P3 와 동일한 마스크) / `"v2"`([ch_mask_check.ipynb](ch_mask_check.ipynb) 수정본)

**첫 실행은 `v1` 로 둔다.** P3(val 64.203, Public 58.8028) 와의 차이가 오직
"격자 -> 면적" 하나가 되어야 무엇이 점수를 움직였는지 말할 수 있다. v2 는 그
다음 칸이다 — 마스크가 고장나 있다는 진단이 맞다면, 면적 하나로 줄인 renew 가
그 고장에 P3 보다 더 민감하다(격자에선 림 고리가 가장자리 셀에 몰려 모델이
무시할 수 있었지만, 면적에 섞이면 분리할 방법이 없다).
"""

KNOB_CELL = r"""
# ===== P3-renew 노브 ==================================================
CH_FEATURE_MODE = "area"    # "area" = 면적 1개 | "area_shape" = 면적 + 마스크 모멘트 4개
CH_MASK_VERSION = "v1"      # "v1" = P3 와 동일 | "v2" = ch_mask_check.ipynb 의 수정 규칙

# --- v2 를 켤 때만 쓰인다. ch_mask_check.ipynb 셀 10 에서 고른 값을 옮겨 적는다 ---
V2_RATIO = 0.45             # 평탄화 후 "조용한 코로나 최빈값" 대비 비율
V2_CORE_FRACTION = 0.90     # 이 반지름 비율 안쪽만 코로나홀로 인정
V2_MIN_AREA = 0.0015        # 원반 넓이 대비 이보다 작은 조각은 버림 (scipy 필요)
V2_STACK_COUNT = 200        # 반경 프로파일에 쓸 train 프레임 수
RADIAL_BINS = 64            # 반경 방향 평탄화 구간 수

CH_FEATURE_NAMES = (["ch_area"] if CH_FEATURE_MODE == "area" else
                    ["ch_area", "ch_cx", "ch_cy", "ch_spread", "ch_depth"])
N_CH_FEATURES = len(CH_FEATURE_NAMES)

# 아래 두 이름은 P3 원본 셀(Dataset · 모델 · 탄도 정렬)이 그대로 참조한다.
# 셀을 고치지 않고 의미만 바꾼다 — "격자 셀 개수" -> "CH 피처 개수",
# "중앙자오선 셀 번호" -> "탄도 역산에 쓸 시계열의 열 번호".
N_CELLS = N_CH_FEATURES
CENTRAL_CELL = 0            # = ch_area. 전 원반 면적 시계열을 탄도 역산에 쓴다
CH_GRID = (1, 1)            # 격자 없음. 학습 셀의 CONFIG 기록 호환용으로만 남긴다

CH_RATIO = V2_RATIO if CH_MASK_VERSION == "v2" else CH_THRESHOLD_RATIO
EFFECTIVE_MARGIN = V2_CORE_FRACTION if CH_MASK_VERSION == "v2" else DISK_MARGIN
CH_CACHE_TAG = f"{CH_MASK_VERSION}_{CH_FEATURE_MODE}_r{CH_RATIO}_m{EFFECTIVE_MARGIN}_{IMAGE_SIZE}"

# P3 산출물을 덮어쓰지 않는다
OUTPUT_DIR = WORK_DIR / "outputs_p3_renew"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

assert CH_FEATURE_MODE in {"area", "area_shape"}, CH_FEATURE_MODE
assert CH_MASK_VERSION in {"v1", "v2"}, CH_MASK_VERSION
print(f"P3-renew — 마스크 {CH_MASK_VERSION} · 피처 {N_CH_FEATURES}개 {CH_FEATURE_NAMES}")
print(f"           임계비 {CH_RATIO} · 유효 반지름 x{EFFECTIVE_MARGIN}")
print(f"출력: {OUTPUT_DIR.resolve()}")
"""

CH_MD = r"""
## 3. 태양 원반 검출 · 코로나홀 마스크 피처 추출 — **P3-renew 핵심**

격자 binning 을 걷어내고, **마스크에서 직접** 프레임당 스칼라를 뽑는다.

1. **원반 검출** — `v1` 은 P3 그대로(밝은 픽셀 넓이 -> 반지름, x0.95).
   `v2` 는 림 기울기 최대점에 원을 최소제곱 적합하고 x0.90.
2. **코로나홀 마스크** — `v1` 은 P3 그대로(193 **AND** 211 이 원반 중앙값의
   `CH_THRESHOLD_RATIO` 미만). `v2` 는 반경 프로파일로 평탄화한 뒤 조용한 코로나
   **최빈값** 을 기준으로 삼고, 아주 작은 조각을 버린다.
3. **요약** — 격자 대신 마스크 전체에서:

   | 피처 | 정의 | `area` | `area_shape` |
   |---|---|:--:|:--:|
   | `ch_area` | 유효 원반 대비 코로나홀 픽셀 비율 | O | O |
   | `ch_cx` | 마스크 무게중심의 동서 위치 (원반 반지름으로 정규화, + = 서쪽 림) | | O |
   | `ch_cy` | 무게중심의 남북 위치 | | O |
   | `ch_spread` | 무게중심 기준 rms 거리 (홀이 뭉쳤나 흩어졌나) | | O |
   | `ch_depth` | 마스크 안 193 평균 밝기의 기준 대비 어두움 `1 - I/ref` | | O |

   모멘트는 격자와 달리 **경계가 없다.** 원반 검출이 몇 픽셀 흔들려도 값이
   튀지 않고, 홀이 셀 경계를 넘어갈 때 생기던 계단도 없다.

결과 `(n_images, N_CH_FEATURES)` 는 P3 의 `(n_images, 15)` 자리에 그대로 들어가므로
아래 Dataset · 모델 · 탄도 정렬 셀은 **P3 원본 그대로** 쓴다.
"""

CH_CELL = r"""
try:
    from scipy import ndimage as _ndimage
except ImportError:
    _ndimage = None


# ===== 1) 원반 기하 ===================================================
def detect_disk(image_array, sample_count=400):
    # P3 원본과 같은 넓이 역산. v2 에서는 림 적합의 씨앗으로만 쓴다.
    indexes = np.unique(np.linspace(0, len(image_array) - 1, sample_count).astype(int))
    mean_image = np.asarray(image_array[indexes], dtype=np.float64).mean(axis=(0, 1))
    # 배경은 어둡고 원반은 밝습니다. 단순 임계로 원반 픽셀을 잡습니다.
    mask = mean_image > mean_image.max() * 0.15
    ys, xs = np.nonzero(mask)
    center_y, center_x = float(ys.mean()), float(xs.mean())
    radius = float(np.sqrt(mask.sum() / np.pi))
    return center_y, center_x, radius, mean_image


def fit_circle(ys, xs):
    # Kasa 원 적합. x^2+y^2 = 2ax + 2by + c 를 선형 최소제곱으로 풉니다.
    design = np.stack([2 * xs, 2 * ys, np.ones_like(xs)], axis=1)
    target = xs ** 2 + ys ** 2
    (cx, cy, c), *_ = np.linalg.lstsq(design, target, rcond=None)
    return float(cy), float(cx), float(np.sqrt(max(c + cx ** 2 + cy ** 2, 1e-6)))


def fit_limb(mean_image, seed_y, seed_x, seed_r, n_rays=720, iterations=2):
    # 방향별로 밝기 기울기가 가장 급한 반지름 = 림. 그 점들에 원을 적합합니다.
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
        keep = usable & (depth > np.nanmedian(depth) * 0.25)
        deviation = np.abs(found - np.median(found[keep]))
        keep &= deviation < 4.0 * (np.median(deviation[keep]) + 1e-6)
        center_y, center_x, radius = fit_circle(
            center_y + found[keep] * np.sin(angles[keep]),
            center_x + found[keep] * np.cos(angles[keep]))
    return center_y, center_x, radius, int(keep.sum())


SEED_Y, SEED_X, SEED_R, MEAN_IMAGE = detect_disk(train_image_array)
if CH_MASK_VERSION == "v2":
    DISK_Y, DISK_X, DISK_R, FIT_RAYS = fit_limb(MEAN_IMAGE, SEED_Y, SEED_X, SEED_R)
    print(f"원반 적합(v2): center=({DISK_Y:.1f}, {DISK_X:.1f}) radius={DISK_R:.1f}px "
          f"({FIT_RAYS}/720 방향 채택)")
    print(f"    넓이 역산(v1) 반지름 {SEED_R:.1f}px 대비 {DISK_R / SEED_R - 1:+.1%}")
else:
    DISK_Y, DISK_X, DISK_R = SEED_Y, SEED_X, SEED_R
    print(f"원반 검출(v1): center=({DISK_Y:.1f}, {DISK_X:.1f}) radius={DISK_R:.1f}px")
EFFECTIVE_R = DISK_R * EFFECTIVE_MARGIN

grid_y, grid_x = np.mgrid[0:IMAGE_SIZE, 0:IMAGE_SIZE].astype(np.float64)
RADIUS_MAP = np.sqrt((grid_y - DISK_Y) ** 2 + (grid_x - DISK_X) ** 2)
DISK_MASK = RADIUS_MAP <= EFFECTIVE_R              # 코로나홀을 인정하는 유효 원반
DISK_PIXELS = float(DISK_MASK.sum())
X_NORM = ((grid_x - DISK_X) / DISK_R).astype(np.float32)   # 동서 (+ = 서쪽 림 방향)
Y_NORM = ((grid_y - DISK_Y) / DISK_R).astype(np.float32)   # 남북
X_ON_DISK, Y_ON_DISK = X_NORM[DISK_MASK], Y_NORM[DISK_MASK]
print(f"유효 반지름 {EFFECTIVE_R:.1f}px, 원반 픽셀 {int(DISK_PIXELS):,}개 "
      f"(프레임의 {DISK_PIXELS / IMAGE_SIZE ** 2:.0%})")


# ===== 2) v2 전용 — 반경 프로파일 (train 에서만 만든다) ================
PROFILE_MAP = None
if CH_MASK_VERSION == "v2":
    stack_index = np.unique(np.linspace(0, len(train_image_array) - 1,
                                        V2_STACK_COUNT).astype(int))
    stack = np.asarray(train_image_array[stack_index], dtype=np.float32)
    radial_bin = np.clip((RADIUS_MAP / DISK_R * RADIAL_BINS).astype(int), 0, RADIAL_BINS - 1)
    on_disk_full = RADIUS_MAP <= DISK_R
    per_frame = np.full((len(stack), len(CHANNELS), RADIAL_BINS), np.nan, dtype=np.float32)
    for bin_index in range(RADIAL_BINS):
        selector = on_disk_full & (radial_bin == bin_index)
        if selector.any():
            per_frame[:, :, bin_index] = np.median(stack[:, :, selector], axis=2)
    profile = np.nanmedian(per_frame, axis=0)                      # (2, RADIAL_BINS)
    for channel_index in range(len(CHANNELS)):
        row = profile[channel_index]
        bad = ~np.isfinite(row) | (row <= 0)
        if bad.any():
            row[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(~bad), row[~bad])
    # 시간중앙값이라 한 프레임의 홀이 자기 기준을 지우지 않는다 (극지방 홀 보호).
    PROFILE_MAP = profile[:, radial_bin].astype(np.float32)        # (2, H, W)
    del stack, per_frame
    print(f"반경 프로파일: {len(stack_index)}프레임 x {RADIAL_BINS}구간 -> 평탄화 준비 완료")
    if V2_MIN_AREA > 0 and _ndimage is None:
        print("  주의: scipy 가 없어 작은 조각 제거를 건너뜁니다 (V2_MIN_AREA 무시)")


# ===== 3) 코로나홀 마스크 =============================================
def quiet_level(values, bins=256, span=(0.0, 3.0)):
    # 평탄화한 원반 밝기의 히스토그램 봉우리 = 조용한 코로나 수준.
    # 중앙값과 달리 활동영역이 원반의 몇 %를 덮든 거의 움직이지 않는다.
    histogram, edges = np.histogram(values, bins=bins, range=span)
    smooth = np.convolve(histogram.astype(np.float64), np.ones(9) / 9.0, "same")
    centers = 0.5 * (edges[:-1] + edges[1:])
    return float(centers[int(np.argmax(smooth))])


def drop_small_components(mask):
    if _ndimage is None or V2_MIN_AREA <= 0:
        return mask
    limit = V2_MIN_AREA * DISK_PIXELS
    structure = np.ones((3, 3), dtype=bool)
    result = np.zeros_like(mask)
    for index in range(len(mask)):
        if not mask[index].any():
            continue
        labels, count = _ndimage.label(mask[index], structure=structure)
        if not count:
            continue
        sizes = np.bincount(labels.ravel())
        keep = sizes >= limit
        keep[0] = False
        result[index] = keep[labels]
    return result


def coronal_hole_masks(block):
    # block: (n, C, H, W) float32. 반환: 마스크 (n, H, W), 기준값 (n, C), 비교에 쓴 밝기
    if CH_MASK_VERSION == "v2":
        values = block / np.maximum(PROFILE_MAP, 1e-3)
        reference = np.empty((len(block), len(CHANNELS)), dtype=np.float32)
        for index in range(len(block)):
            for channel_index in range(len(CHANNELS)):
                reference[index, channel_index] = quiet_level(
                    values[index, channel_index][DISK_MASK])
    else:
        values = block
        reference = np.median(block[:, :, DISK_MASK], axis=2).astype(np.float32)  # (n, C)
    dark = values < (CH_RATIO * reference)[:, :, None, None]
    # 193 과 211 두 채널 모두에서 어두운 픽셀만 코로나홀로 인정 (P3 와 같은 규칙)
    mask = np.logical_and(dark[:, 0], dark[:, 1]) & DISK_MASK
    if CH_MASK_VERSION == "v2":
        mask = drop_small_components(mask)
    return mask, reference, values


# ===== 4) 마스크 -> 피처 ==============================================
def mask_features(mask, reference, values):
    on_disk = mask[:, DISK_MASK].astype(np.float32)                # (n, npix)
    pixel_count = on_disk.sum(axis=1)
    area = pixel_count / DISK_PIXELS
    if CH_FEATURE_MODE == "area":
        return area[:, None].astype(np.float32)

    safe_count = np.maximum(pixel_count, 1.0)
    center_x = on_disk @ X_ON_DISK / safe_count
    center_y = on_disk @ Y_ON_DISK / safe_count
    second_moment = on_disk @ (X_ON_DISK ** 2 + Y_ON_DISK ** 2) / safe_count
    spread = np.sqrt(np.maximum(second_moment - center_x ** 2 - center_y ** 2, 0.0))
    brightness = values[:, 0][:, DISK_MASK]                        # 193 채널
    depth = 1.0 - ((on_disk * brightness).sum(axis=1) / safe_count
                   / np.maximum(reference[:, 0], 1e-6))
    empty = pixel_count <= 0
    for feature in (center_x, center_y, spread, depth):
        feature[empty] = 0.0
    return np.stack([area, center_x, center_y, spread, depth], axis=1).astype(np.float32)


def compute_ch_features(image_array, chunk=256):
    result = np.zeros((len(image_array), N_CH_FEATURES), dtype=np.float32)
    for start in range(0, len(image_array), chunk):
        block = np.asarray(image_array[start:start + chunk], dtype=np.float32)
        mask, reference, values = coronal_hole_masks(block)
        result[start:start + chunk] = mask_features(mask, reference, values)
        if (start // chunk) % 20 == 0 or start + chunk >= len(image_array):
            print(f"  CH {min(start + chunk, len(image_array))}/{len(image_array)}", flush=True)
    return result


def cached_ch_features(split, image_array):
    # 파일명에 마스크 버전·피처 모드·임계비·여유반지름이 들어가므로 P3 캐시와 섞이지 않는다.
    path = CACHE_ROOT / f"chmask_{split}_{CH_CACHE_TAG}.npy"
    if path.exists():
        features = np.load(path)
        if features.shape == (len(image_array), N_CH_FEATURES):
            print(f"reusing CH cache: {path.name}")
            return features
    print(f"extracting CH features: {split}")
    features = compute_ch_features(image_array)
    np.save(path, features)
    print(f"created CH cache: {path.name}  shape={features.shape}")
    return features


train_ch = cached_ch_features("train", train_image_array)
val_ch = cached_ch_features("validation", val_image_array)
test_ch = cached_ch_features("test", test_image_array)
assert train_ch.shape[1] == N_CELLS

print()
print(pd.DataFrame(train_ch, columns=CH_FEATURE_NAMES).describe()
      .loc[["mean", "std", "min", "max"]].round(4))
legacy_path = CACHE_ROOT / f"ch_train_3x5_{CH_THRESHOLD_RATIO}_{IMAGE_SIZE}.npy"
if legacy_path.exists():
    legacy = np.load(legacy_path)
    if len(legacy) == len(train_ch):
        # 셀 평균은 셀 크기가 달라 면적과 정확히 같지 않다. 방향 확인용 참고값이다.
        correlation = np.corrcoef(legacy.mean(axis=1), train_ch[:, 0])[0, 1]
        print(f"\nP3 격자(3x5) 셀평균 vs renew 면적 상관: r={correlation:+.3f}")

figure, axes = plt.subplots(1, 3, figsize=(13, 4))
axes[0].imshow(MEAN_IMAGE, cmap="gray")
axes[0].add_patch(plt.Circle((DISK_X, DISK_Y), EFFECTIVE_R, fill=False,
                             color="red", linewidth=1.5))
if CH_MASK_VERSION == "v2":
    axes[0].add_patch(plt.Circle((SEED_X, SEED_Y), SEED_R * DISK_MARGIN, fill=False,
                                 color="cyan", linewidth=1.0, linestyle="--"))
    axes[0].set_title("train mean + disk (red = v2, cyan = v1)")
else:
    axes[0].set_title("train mean + detected disk")
sample_mask, _, _ = coronal_hole_masks(np.asarray(train_image_array[:1], dtype=np.float32))
axes[1].imshow(sample_mask[0], cmap="gray")
axes[1].set_title(f"CH mask ({CH_MASK_VERSION}) area={sample_mask[0].sum() / DISK_PIXELS:.2%}")
axes[2].plot(train_ch[:800, 0] * 100, linewidth=0.8)
axes[2].set_title("CH area [%] (first 800 frames)")
axes[2].set_xlabel("frame"); axes[2].grid(alpha=0.3)
for axis in axes[:2]:
    axis.set_xticks([]); axis.set_yticks([])
plt.tight_layout(); plt.show()
"""

MODEL_MD = r"""
## 6. 모델

- **CH 브랜치** — 마스크 피처 시퀀스 `(20, N_CELLS)` → GRU.
  `N_CELLS` 는 격자 셀이 아니라 **CH 피처 개수**다 (기본 1 = 면적).
  P3(15) 보다 입력이 좁아진 것 말고는 구조가 같다
- **Wind 브랜치** — P1 과 동일 (GRU + 통계)
- **CNN 브랜치** — `USE_CNN=True` 일 때만. 기본은 꺼짐
- **head** — horizon 12개에 **가중치를 공유**하고, horizon embedding 과
  해당 horizon 의 탄도 피처만 다르게 넣습니다. 파라미터가 12배 줄어 정규화 효과가 큽니다
"""

# --- 한 줄 치환 패치 ---------------------------------------------------
CONFIG_OLD = '"ch_threshold_ratio": CH_THRESHOLD_RATIO, "disk": [DISK_Y, DISK_X, DISK_R],'
CONFIG_NEW = ('"ch_threshold_ratio": CH_RATIO, "disk": [DISK_Y, DISK_X, DISK_R],\n'
              '          "ch_mask_version": CH_MASK_VERSION, "ch_feature_mode": CH_FEATURE_MODE,\n'
              '          "ch_features": CH_FEATURE_NAMES, "effective_margin": EFFECTIVE_MARGIN,')

REFERENCE_OLD = 'print("\\n[참고] P1 = 68.408 / P2a = 65.663")'
REFERENCE_NEW = 'print("\\n[참고] P1 = 68.408 / P2a = 65.663 / P3(3x5 격자) = 64.203")'


# =====================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    parser.add_argument("--check", action="store_true", help="패치 지점만 확인")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    notebook = json.loads(args.base.read_text(encoding="utf-8"))
    cells = notebook["cells"]

    # 패치 지점을 인덱스가 아니라 내용으로 찾는다. P3 가 수정돼도 조용히 어긋나지 않는다.
    def locate(needle, kind="code"):
        found = [i for i, c in enumerate(cells)
                 if c["cell_type"] == kind and needle in source_of(c)]
        if len(found) != 1:
            raise SystemExit(f"패치 지점 '{needle}' 가 {len(found)}개 — 중단")
        return found[0]

    config_at = locate("USE_BALLISTIC = True")
    ch_md_at = locate("코로나홀 격자 면적 추출", kind="markdown")
    ch_at = locate("def compute_ch_grid")
    model_md_at = locate("- **CH 브랜치**", kind="markdown")
    train_at = locate(CONFIG_OLD)
    validation_at = locate(REFERENCE_OLD)
    print(f"패치 지점 — 설정 {config_at} · CH markdown {ch_md_at} · CH 코드 {ch_at} · "
          f"모델 markdown {model_md_at} · 학습 {train_at} · 평가 {validation_at}")

    # 교체한 CH 셀이 아래 원본 셀들의 요구를 모두 채우는지 확인한다.
    for name in ("train_ch", "val_ch", "test_ch", "DISK_Y", "DISK_X", "DISK_R"):
        assert name in CH_CELL, f"CH 셀이 {name} 을 정의하지 않는다"
    for name in ("N_CELLS", "CENTRAL_CELL", "CH_GRID", "OUTPUT_DIR"):
        assert name in KNOB_CELL, f"노브 셀이 {name} 을 정의하지 않는다"
    if args.check:
        print("확인만 하고 종료합니다.")
        return

    patched = []
    for index, cell in enumerate(cells):
        if index == 0 and cell["cell_type"] == "markdown":
            patched.append(markdown_cell(TITLE_MD))
            continue
        if index == ch_md_at:
            patched.append(markdown_cell(CH_MD))
            continue
        if index == ch_at:
            patched.append(code_cell(CH_CELL))
            continue
        if index == model_md_at:
            patched.append(markdown_cell(MODEL_MD))
            continue
        if index == train_at:
            patched.append(code_cell(source_of(cell).replace(CONFIG_OLD, CONFIG_NEW)))
            continue
        if index == validation_at:
            patched.append(code_cell(source_of(cell).replace(REFERENCE_OLD, REFERENCE_NEW)))
            continue
        patched.append(cell)
        if index == config_at:
            patched += [markdown_cell(KNOB_MD), code_cell(KNOB_CELL)]

    notebook["cells"] = patched
    try:                                   # nbformat 4.5 는 셀 id 를 요구한다
        import nbformat
        notebook = nbformat.from_dict(notebook)
        _, notebook = nbformat.validator.normalize(notebook)
    except ImportError:
        pass
    args.out.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
                        encoding="utf-8")
    code_count = sum(1 for c in patched if c["cell_type"] == "code")
    print(f"wrote {args.out}  (셀 {len(patched)}개, 코드 {code_count}개)")


if __name__ == "__main__":
    main()
