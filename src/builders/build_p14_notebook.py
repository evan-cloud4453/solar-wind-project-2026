"""Build P14 from the validated P11 notebook.

P14 deliberately preserves P11's data loading, model, loss, and official-RMSE
calculation.  It adds independently switchable flare, CH-shape, rotation, and
intensity-depth features before P11's member-training cell.
"""

from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "code_p11.ipynb"
TARGET = HERE / "code_p14.ipynb"
TRIAL_TARGET = HERE / "Trial" / "P14" / "code_p14.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.strip("\n").splitlines(True)}


P14_CONFIG = r'''

# ============================ P14 독립 스위치 =============================
# 아래 네 줄만 True/False 로 바꾼 뒤, 이 셀부터 마지막까지 순서대로 실행한다.
# p14_extract.py가 만든 p14a 캐시가 없으면 캐시 의존 피처는 자동으로 꺼진다.
USE_FLARE = False             # 급격한 고휘도 EUV 사건: 면적·세기·6h 변화
USE_CH_SHAPE = False          # 연결성분·원형도·종횡비·theta_b 경계거리
USE_ROTATION = False          # 중앙 접근도·동서 불균형·6h 변화 (명시적 자전 동역학)
USE_INTENSITY_DEPTH = False   # dark 0.30/0.45/0.60 등고선 + CH 깊이

# 비교 조건도 고정할 수 있다. 같은 조합은 같은 seed/epoch로 비교해야 한다.
P14_SEED = 777
P14_EPOCHS = 1
P14_BASE_CODE = "F3"          # P9에서 채택된 넓은-경도 물리 피처 기준선

P14_EXTRA_KEYS = ("flare_area", "flare_power", "shape")
P14_SHAPE_COLUMNS = ("largest_frac", "component_count", "compactness", "elongation",
                     "boundary_density", "core_fraction")


def load_p14_extra(split, files):
    path = CACHE_ROOT / f"{P11_VERSION}_{split}.npz"
    if not path.exists():
        return None
    with np.load(path) as blob:
        if [str(name) for name in blob["files"]] != list(files):
            print(f"  ⚠️ {path.name}: 파일 목록이 다르다 -> P14 피처를 쓰지 않는다")
            return None
        if any(key not in blob for key in P14_EXTRA_KEYS):
            print(f"  ⚠️ {path.name}: P14 추가 키 없음 -> p14_extract.py를 실행할 것")
            return None
        return {key: np.asarray(blob[key], np.float32) for key in P14_EXTRA_KEYS}


P14 = {}
for _split, _files in (("train", train_files), ("validation", val_files), ("test", test_files)):
    _extra = load_p14_extra(_split, _files)
    if _extra is not None:
        P14[_split] = _extra
P14_AVAILABLE = len(P14) == 3

if not P14_AVAILABLE:
    print("P14 캐시 없음 -> flare/shape/intensity-depth 스위치를 자동으로 끈다")
    USE_FLARE = USE_CH_SHAPE = USE_INTENSITY_DEPTH = False
else:
    print("P14 캐시: 사용 가능 (flare · shape · contour/depth)")
'''


P14_FEATURES = r'''
# ============================ P14 피처 레이어 =============================
# P11 함수를 보존한 뒤 필요한 부분만 확장한다. flare는 탄도 채널에 넣지 않는다:
# 사건성 CME/ICME 신호는 CH-HSS와 다른 전달 물리를 가지므로 GRU 시계열 브랜치에서만
# horizon별로 사용하게 한다.

_p11_channel_fine = channel_fine
_p11_frame_features = frame_features
_p11_configure = configure
_p11_flatten_ch = flatten_ch
P14_ROTATION_DIM = 4


def channel_fine(split, area_raw, name):
    if name == "flare_area":
        return P14[split]["flare_area"], "sum"
    if name == "flare_power":
        return P14[split]["flare_power"], "sum"
    return _p11_channel_fine(split, area_raw, name)


def frame_features(split):
    pieces = []
    base_frame = _p11_frame_features(split)
    if base_frame is not None:
        pieces.append(base_frame)
    if USE_CH_SHAPE and P14_AVAILABLE:
        pieces.append(np.nan_to_num(P14[split]["shape"], nan=0.0,
                                    posinf=0.0, neginf=0.0).astype(np.float32))
    return (np.concatenate(pieces, axis=1).astype(np.float32)
            if pieces else None)


def configure(**overrides):
    """P11 설정 뒤에 P14 채널을 추가하고 입력 차원을 다시 계산한다."""
    _p11_configure(**overrides)
    global CH_CHANNELS, train_grid, val_grid, test_grid, CH_SEQ_DIM
    if USE_FLARE and P14_AVAILABLE:
        CH_CHANNELS = list(CH_CHANNELS) + ["flare_area", "flare_power"]
        train_grid = build_grid("train", train_area)
        val_grid = build_grid("validation", val_area)
        test_grid = build_grid("test", test_area)
    # rotation features are generated in flatten_ch because they require the ordered 20-frame window.
    CH_SEQ_DIM = int(train_grid.seq.shape[1]) + (P14_ROTATION_DIM if USE_ROTATION else 0)


def _rotation_features(cells):
    """(batch, 20, ballistic_channel, cell) -> 4 explicit rotation-dynamics features.

    `centrality` is high when CH area is near central meridian; `imbalance` retains
    the signed east/west state.  Their one-step changes tell the model whether the
    observed pattern is approaching or leaving disk center without hard-coding the
    image-axis convention.
    """
    area = cells[:, :, 0, :].reshape(len(cells), 20, USED_LAT, GRID_LON).sum(axis=2)
    columns = np.arange(GRID_LON, dtype=np.float32)
    centre = (GRID_LON - 1) / 2.0
    scale = max(centre, 1.0)
    signed = (columns - centre) / scale
    central = 1.0 - np.abs(signed)
    centrality = np.einsum("btl,l->bt", area, central)
    imbalance = np.einsum("btl,l->bt", area, signed)
    delta_centrality = np.diff(centrality, axis=1, prepend=centrality[:, :1])
    delta_imbalance = np.diff(imbalance, axis=1, prepend=imbalance[:, :1])
    return np.stack((centrality, imbalance, delta_centrality, delta_imbalance), axis=-1)


def flatten_ch(grid, indexes):
    sequence = _p11_flatten_ch(grid, indexes)
    if not USE_ROTATION:
        return sequence
    rotation = _rotation_features(grid.cells[indexes]).astype(np.float32)
    return np.concatenate((sequence, rotation), axis=-1).astype(np.float32)
'''


P14_MEMBERS = r'''
# ====================== P14 validation configuration =======================
# 이 셀을 실행하면 위의 네 토글이 하나의 기준 모델에 반영된다. 따라서 마지막
# "P14 validation" 출력이 바로 선택한 조합의 공식 validation RMSE다.

P14_SETTINGS = dict(config_at(P14_BASE_CODE))
P14_SETTINGS.update({
    # Shape에는 theta_b와 P11의 연결성분 스칼라를 같이 쓴다.
    "USE_DEPTH": bool(USE_CH_SHAPE or USE_INTENSITY_DEPTH),
    "USE_FRAME": bool(USE_CH_SHAPE),
    "DEPTH_IN_BALLISTIC": bool(USE_CH_SHAPE),
})
if USE_INTENSITY_DEPTH:
    # core(0.30) / body(0.45) / envelope(0.60): 면적 비가 어두움의 깊이를 표현한다.
    P14_SETTINGS["USE_LEVELS"] = ("dark0.30", "dark0.45", "dark0.60", "bright")

STAGE = "P14"
FIT_SHRINKAGE = False       # validation 조합 비교에서는 별도 OOF 보정을 넣지 않는다.
MEMBERS = [dict(name="P14_switches", settings=P14_SETTINGS,
                seed=P14_SEED, epochs=P14_EPOCHS)]

configure(**P14_SETTINGS)
print("\nP14 switches:", {
    "flare": USE_FLARE,
    "shape": USE_CH_SHAPE,
    "rotation": USE_ROTATION,
    "intensity_depth": USE_INTENSITY_DEPTH,
})
print(f"  base={P14_BASE_CODE}, seed={P14_SEED}, epochs={P14_EPOCHS} | "
      f"ch_seq={CH_SEQ_DIM}, ballistic={BALLISTIC_DIM}")
'''


HEADER = r'''# 태양풍 속도 예측 — P14: 물리 피처 토글 검증

P14는 P11의 학습·공식 RMSE 경로를 그대로 재사용하고 네 가지 물리 피처를 독립적으로
켤 수 있게 만든 실험 노트북입니다.

| 토글 | 피처 | 처리 방식 |
|---|---|---|
| `USE_FLARE` | 강한 EUV 고휘도 사건 | 193·211 동시 고휘도 면적/세기와 시간 변화 |
| `USE_CH_SHAPE` | 코로나홀 형태 | 최대 연결성분·원형도·종횡비·경계밀도·theta_b |
| `USE_ROTATION` | 자전 동역학 | 중앙 접근도·동서 불균형 및 6시간 변화 |
| `USE_INTENSITY_DEPTH` | 깊이/등고선 | core/body/envelope 면적과 경계거리 |

## 사용 순서

1. 먼저 `python p14_extract.py --workers 4`로 `work/cache/p14a_*.npz`를 만든다.
2. P14 설정 셀에서 네 토글과 `P14_EPOCHS`를 정한다.
3. 그 셀부터 마지막까지 실행한다. 마지막 제출 저장 셀이 출력하는 **`P14 validation`**이
   선택한 조합의 공식 validation score다.

공정한 비교를 위해 한 번에 토글 하나만 바꾸고, `P14_SEED`와 `P14_EPOCHS`는 유지한다.
플레어는 CME 방향을 직접 알 수 없는 약한 사건 피처이므로, 기본값은 모두 `False`다.
'''


def build() -> None:
    notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    cells[0] = md(HEADER)

    config = "".join(cells[18]["source"])
    config = config.replace('P11_VERSION = "p11a"', 'P11_VERSION = "p14a"', 1)
    cells[18]["source"] = (config + P14_CONFIG).splitlines(True)

    # P11 feature layer (cell 20) runs first; P14 overrides are inserted before member setup.
    cells.insert(21, md("### P14 피처 확장 — flare · shape · rotation · intensity depth"))
    cells.insert(22, code(P14_FEATURES))

    # Original P11 member setup moved from 22 to 24.  Keep its helpers, then override members.
    member_cell = 24
    cells[member_cell]["source"] = ("".join(cells[member_cell]["source"]) + P14_MEMBERS).splitlines(True)

    # P11's final text and artifact label should identify this notebook correctly.
    for cell in cells:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        source = source.replace('"code_version": "p11"', '"code_version": "p14"')
        source = source.replace('NOTEBOOK_NAME = "code_p11.ipynb"', 'NOTEBOOK_NAME = "code_p14.ipynb"')
        source = source.replace('P11-{STAGE}', 'P14-{STAGE}')
        if "P3_PER_HORIZON =" in source:
            source += r'''

print("\n" + "=" * 72)
print("P14 validation result (this switch combination)")
print({"flare": USE_FLARE, "shape": USE_CH_SHAPE,
       "rotation": USE_ROTATION, "intensity_depth": USE_INTENSITY_DEPTH,
       "seed": P14_SEED, "epochs": P14_EPOCHS})
print(f"OFFICIAL_VALIDATION_RMSE = {_final_val_score:.4f} km/s")
print("=" * 72)
'''
        cell["source"] = source.splitlines(True)

    TARGET.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    TRIAL_TARGET.parent.mkdir(parents=True, exist_ok=True)
    TRIAL_TARGET.write_text(TARGET.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"created {TARGET.name} and {TRIAL_TARGET.relative_to(HERE)} ({len(cells)} cells)")


if __name__ == "__main__":
    build()
