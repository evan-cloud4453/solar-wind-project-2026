"""Build P15: a single-model P3 extension with switchable physical features."""

from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "Trial" / "P3" / "code_p3.ipynb"
TARGET = HERE / "code_p15.ipynb"
TRIAL_TARGET = HERE / "Trial" / "P15" / "code_p15.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.strip("\n").splitlines(True)}


CONFIG = r'''

# ============================= P15 feature toggles ==========================
# P3의 단일 모델·3x5 CH 격자·탄도 정렬·early stopping은 유지한다.
# 아래 네 스위치만 True/False로 바꾸어 같은 seed/epoch에서 validation을 비교한다.
USE_FLARE = False
USE_CH_SHAPE = False
USE_ROTATION = False
USE_INTENSITY_DEPTH = False
P15_CACHE_VERSION = "p14a"
P15_REQUESTED_SWITCHES = {
    "flare": USE_FLARE,
    "shape": USE_CH_SHAPE,
    "rotation": USE_ROTATION,
    "intensity_depth": USE_INTENSITY_DEPTH,
}
'''


BOOTSTRAP = r'''
# ===================== P15: P14 물리 피처 캐시 준비 =========================
# work/cache/p14a_{train,validation,test}.npz 가 없으면 이 셀이 직접 만든다.
# 노트북과 같은 폴더에 p14_extract.py 와 p11_extract.py 가 함께 있어야 한다.
# 캐시가 이미 있으면 아무 일도 하지 않고 즉시 넘어간다.
import subprocess
import sys

P15_AUTO_EXTRACT = True     # False 로 두면 캐시가 없어도 추출하지 않는다
P15_EXTRACT_WORKERS = 4     # 운영진 공지: 워커 4개를 넘기지 말 것
P15_EXTRACT_SPLITS = ("train", "validation", "test")


def p15_cache_path(split):
    return CACHE_ROOT / f"{P15_CACHE_VERSION}_{split}.npz"


def p15_missing_splits():
    return [split for split in P15_EXTRACT_SPLITS if not p15_cache_path(split).exists()]


def p15_run_extract(splits):
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
               "--workers", str(P15_EXTRACT_WORKERS),
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


_p15_missing = p15_missing_splits()
if not _p15_missing:
    print("P14 캐시 확인 완료:")
    for split in P15_EXTRACT_SPLITS:
        _path = p15_cache_path(split)
        print(f"  {split:<10} {_path.resolve()} ({_path.stat().st_size / 1024 ** 2:.0f} MB)")
elif not P15_AUTO_EXTRACT:
    print("P14 캐시 없음:", ", ".join(_p15_missing), "| P15_AUTO_EXTRACT=False 이므로 건너뛴다")
else:
    print("P14 캐시 없음 ->", ", ".join(_p15_missing), "추출을 시작한다.")
    print("전체 split 은 수십 분에서 몇 시간까지 걸린다. 터미널을 쓸 수 있으면 대신")
    print("  nohup python p14_extract.py --workers 4 > extract.log 2>&1 &")
    print("로 돌리고 이 셀을 다시 실행하는 편이 낫다.", flush=True)
    p15_run_extract(_p15_missing)
    _p15_missing = p15_missing_splits()
    print("아직 없는 split:", ", ".join(_p15_missing) if _p15_missing else "없음")
'''


FEATURES = r'''
# ========================== P15: P3 physical features ======================
# p14a의 파일 순서는 시간 사슬 순서이고 P3 memmap은 파일명 정렬 순서다.
# 반드시 filename으로 재매핑한다. 배열 위치가 우연히 같다고 가정하지 않는다.

P15_CACHE_KEYS = ("area", "depth_max", "depth_sum", "flare_area", "flare_power", "shape")


def p15_load(split, image_index):
    path = CACHE_ROOT / f"{P15_CACHE_VERSION}_{split}.npz"
    if not path.exists():
        print(f"  ⚠️ 캐시 파일 없음: {path.resolve()}")
        return None
    with np.load(path) as blob:
        if any(key not in blob for key in P15_CACHE_KEYS) or "files" not in blob:
            print(f"  ⚠️ {path.name}: P14 키 없음 -> p14_extract.py를 실행할 것")
            return None
        source_index = {str(name): i for i, name in enumerate(blob["files"])}
        names = list(image_index)
        if set(names) != set(source_index):
            print(f"  ⚠️ {path.name}: filename 집합 불일치 -> P14 피처를 쓰지 않는다")
            return None
        order = np.asarray([source_index[name] for name in names], np.int64)
        return {key: np.asarray(blob[key][order], np.float32) for key in P15_CACHE_KEYS}


P15 = {
    "train": p15_load("train", train_image_index),
    "validation": p15_load("validation", val_image_index),
    "test": p15_load("test", test_image_index),
}
P15_AVAILABLE = all(value is not None for value in P15.values())
if not P15_AVAILABLE:
    print("P14 cache unavailable: flare/shape/intensity-depth disabled; rotation is still available.")
    print("  해결: 노트북 폴더에서 아래를 실행한 다음 위의 「P14 캐시 준비」 셀부터 다시 실행할 것")
    print("    python p14_extract.py --limit 30 --workers 2    # 스모크 테스트 (저장 안 됨)")
    print("    python p14_extract.py --workers 4               # 전체 (캐시 저장)")
    print(f"  기대 위치: {(CACHE_ROOT / (P15_CACHE_VERSION + '_train.npz')).resolve()}")
    USE_FLARE = USE_CH_SHAPE = USE_INTENSITY_DEPTH = False
else:
    print("P14 cache available: P3-compatible feature remapping complete.")


def p15_aggregate(fine, channels=1):
    """P14 12x30 fine grid -> P3 3x5 grid.  Fine values are disk fractions;
    their sum is intentionally kept and normalized later per feature column."""
    block = fine.reshape(len(fine), channels, 12, 30)
    return block.reshape(len(fine), channels, 3, 4, 5, 6).sum(axis=(3, 5)).reshape(len(fine), -1)


def p15_static(split, base_ch):
    pieces = [base_ch.astype(np.float32)]  # exact P3 grid: always first 15 columns
    if not P15_AVAILABLE:
        return np.concatenate(pieces, axis=1)
    cached = P15[split]
    if USE_FLARE:
        pieces += [p15_aggregate(cached["flare_area"]), p15_aggregate(cached["flare_power"])]
    if USE_CH_SHAPE:
        # theta_b max / mass are spatial shape; shape scalars discriminate compact vs fragmented holes.
        pieces += [p15_aggregate(cached["depth_max"]), p15_aggregate(cached["depth_sum"]),
                   cached["shape"]]
    if USE_INTENSITY_DEPTH:
        # core/body/envelope contours: dark 0.30 / 0.45 / 0.60.
        pieces += [p15_aggregate(cached["area"][:, :3], channels=3),
                   p15_aggregate(cached["depth_max"]), p15_aggregate(cached["depth_sum"])]
    return np.concatenate(pieces, axis=1).astype(np.float32)


P15_TRAIN_STATIC = p15_static("train", train_ch)
P15_VAL_STATIC = p15_static("validation", val_ch)
P15_TEST_STATIC = p15_static("test", test_ch)
P15_STATIC_MEAN = P15_TRAIN_STATIC.mean(axis=0).astype(np.float32)
P15_STATIC_STD = (P15_TRAIN_STATIC.std(axis=0) + 1e-8).astype(np.float32)
# 첫 15열은 P3가 쓰던 정규화를 byte-for-byte 유지한다. 추가 열만 별도 정규화한다.
P15_STATIC_MEAN[:N_CELLS] = CH_MEAN
P15_STATIC_STD[:N_CELLS] = CH_STD


def p15_rotation(raw_ch):
    """Explicit rotation state from the unchanged P3 3x5 CH grid.

    The signed east/west feature retains image-axis information without asserting
    which side is approaching.  Its 6h difference gives the model that dynamics.
    """
    area = raw_ch.reshape(len(raw_ch), 20, 3, 5).sum(axis=2)
    position = np.arange(5, dtype=np.float32) - 2.0
    centrality = 1.0 - np.abs(position) / 2.0
    core = np.einsum("btl,l->bt", area, centrality)
    side = np.einsum("btl,l->bt", area, position / 2.0)
    d_core = np.diff(core, axis=1, prepend=core[:, :1])
    d_side = np.diff(side, axis=1, prepend=side[:, :1])
    return np.stack((core, side, d_core, d_side), axis=-1).astype(np.float32)


if USE_ROTATION:
    _rotation_train = p15_rotation(train_ch[image_index_matrix(train_inputs, train_image_index)])
    P15_ROT_MEAN = _rotation_train.reshape(-1, 4).mean(axis=0).astype(np.float32)
    P15_ROT_STD = (_rotation_train.reshape(-1, 4).std(axis=0) + 1e-6).astype(np.float32)
    del _rotation_train
else:
    P15_ROT_MEAN = np.zeros(4, np.float32)
    P15_ROT_STD = np.ones(4, np.float32)

P15_CH_FEATURE_DIM = P15_TRAIN_STATIC.shape[1] + (4 if USE_ROTATION else 0)
P15_EXPECTED_DIM = (15 + (30 if USE_FLARE else 0) + (36 if USE_CH_SHAPE else 0)
                    + (75 if USE_INTENSITY_DEPTH else 0) + (4 if USE_ROTATION else 0))
assert P15_CH_FEATURE_DIM == P15_EXPECTED_DIM, (P15_CH_FEATURE_DIM, P15_EXPECTED_DIM)
P15_ACTIVE_SWITCHES = {
    "flare": USE_FLARE, "shape": USE_CH_SHAPE,
    "rotation": USE_ROTATION, "intensity_depth": USE_INTENSITY_DEPTH,
}
print("P15 requested switches:", P15_REQUESTED_SWITCHES)
print("P15 active switches:   ", P15_ACTIVE_SWITCHES)
if P15_REQUESTED_SWITCHES != P15_ACTIVE_SWITCHES:
    print("WARNING: requested cache-dependent features were disabled. "
          "Run p14_extract.py and rerun this notebook from the P15 feature cell.")
print(f"P15 CH input: base=15 + static={P15_TRAIN_STATIC.shape[1] - 15} + "
      f"rotation={4 if USE_ROTATION else 0} -> {P15_CH_FEATURE_DIM}")
if P15_TRAIN_STATIC.shape[1] > 15:
    print("P15 extra-feature std range:",
          f"{P15_STATIC_STD[15:].min():.3e} .. {P15_STATIC_STD[15:].max():.3e}")
'''


DATASET_PATCH_FROM = '''        # (n_samples, 20, N_CELLS) 시퀀스로 미리 펼쳐 둡니다.
        self.ch_seq = ((ch_grid[self.image_indexes] - CH_MEAN) / CH_STD).astype(np.float32)
        # 탄도 정렬 피처. 정규화는 train 통계(BALLISTIC_MEAN/STD)로 고정합니다.'''
DATASET_PATCH_TO = '''        # P15는 P3 원본 15개 CH 셀을 첫 열로 보존하고, 선택한 물리 피처만 뒤에 붙인다.
        if ch_grid is train_ch:
            static = P15_TRAIN_STATIC
        elif ch_grid is val_ch:
            static = P15_VAL_STATIC
        elif ch_grid is test_ch:
            static = P15_TEST_STATIC
        else:
            raise ValueError("unknown CH grid")
        raw_static = static[self.image_indexes]
        self.ch_seq = ((raw_static - P15_STATIC_MEAN) / P15_STATIC_STD).astype(np.float32)
        if USE_ROTATION:
            raw_rotation = p15_rotation(ch_grid[self.image_indexes])
            rotation = ((raw_rotation - P15_ROT_MEAN) / P15_ROT_STD).astype(np.float32)
            self.ch_seq = np.concatenate((self.ch_seq, rotation), axis=2).astype(np.float32)
        # 탄도 정렬 피처는 P3의 원래 중앙자오선 CH 피처를 그대로 쓴다.'''


HEADER = r'''# 태양풍 속도 예측 — P15: P3 단일 모델 + 물리 피처 토글

P15는 최고 Public 모델인 **P3의 단일 모델**을 그대로 기반으로 한다. 앙상블, P9 구조,
수축 보정은 사용하지 않는다. P3의 3×5 코로나홀 면적·탄도 정렬·60 epoch early stopping에
플레어·형태·자전·강도 깊이 피처만 보조 입력으로 추가한다.

플레어·형태·강도깊이 피처는 `work/cache/p14a_*.npz` 캐시가 있어야 켜진다. 캐시가 없으면
「P14 캐시 준비」 셀이 같은 폴더의 `p14_extract.py`(내부에서 `p11_extract.py`를 쓴다)를
직접 실행해 만든다. 터미널을 쓸 수 있으면 `python p14_extract.py --workers 4`를 먼저
돌려 두는 편이 빠르다.

설정 셀의 네 `USE_*` 값만 바꾼 뒤 전체 노트북을 실행하면 validation 셀에
`P15 OFFICIAL_VALIDATION_RMSE`가 출력된다.
'''


def find_cell(cells: list, anchor: str, label: str) -> int:
    """Locate the one P3 code cell holding an anchor.  Position must not be assumed."""
    matches = [i for i, cell in enumerate(cells)
               if cell["cell_type"] == "code" and anchor in "".join(cell["source"])]
    if len(matches) != 1:
        raise RuntimeError(f"{label} anchor matched {len(matches)} cells, expected 1")
    return matches[0]


def build() -> None:
    notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    cells[0] = md(HEADER)

    # P3 configuration remains intact, with four explicit switches appended.
    cells[2]["source"] = ("".join(cells[2]["source"]).replace("outputs_p3", "outputs_p15")
                            + CONFIG).splitlines(True)

    # Cache bootstrap first, then the feature build, after P3's train-only statistics setup.
    cells.insert(11, md("## P15 물리 피처 — P14 캐시 준비 (없으면 여기서 생성)"))
    cells.insert(12, code(BOOTSTRAP))
    cells.insert(13, md("## P15 물리 피처 — P3 파일 순서로 재매핑"))
    cells.insert(14, code(FEATURES))

    # Find the P3 Dataset/model cells by anchor so the inserts above cannot shift them out.
    dataset_cell = find_cell(cells, DATASET_PATCH_FROM, "P3 Dataset")
    dataset_source = "".join(cells[dataset_cell]["source"])
    dataset_source = dataset_source.replace(DATASET_PATCH_FROM, DATASET_PATCH_TO, 1)
    dataset_source = dataset_source.replace('(20, N_CELLS)', '(20, P15_CH_FEATURE_DIM)')
    cells[dataset_cell]["source"] = dataset_source.splitlines(True)

    model_cell = find_cell(cells, 'self.ch_gru = nn.GRU(N_CELLS, 64', "P3 model")
    model_source = "".join(cells[model_cell]["source"])
    model_source = model_source.replace('self.ch_gru = nn.GRU(N_CELLS, 64',
                                        'self.ch_gru = nn.GRU(P15_CH_FEATURE_DIM, 64', 1)
    cells[model_cell]["source"] = model_source.splitlines(True)

    # Add P15 metadata and an unambiguous final validation line.
    for cell in cells:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if '"initialization": "random_from_scratch"' in source:
            source = source.replace('"initialization": "random_from_scratch"',
                                    '"initialization": "random_from_scratch",\n'
                                    '          "p15_switches": {"flare": USE_FLARE, "shape": USE_CH_SHAPE,\n'
                                    '                           "rotation": USE_ROTATION, "intensity_depth": USE_INTENSITY_DEPTH},\n'
                                    '          "p15_ch_feature_dim": P15_CH_FEATURE_DIM')
        if 'print(f"공식 RMSE (mean of horizon RMSE)' in source:
            source += r'''

print("\nP15 single-model switches:", {"flare": USE_FLARE, "shape": USE_CH_SHAPE,
      "rotation": USE_ROTATION, "intensity_depth": USE_INTENSITY_DEPTH})
print(f"P15 OFFICIAL_VALIDATION_RMSE = {model_score:.4f} km/s")
'''
        cell["source"] = source.splitlines(True)

    TARGET.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    TRIAL_TARGET.parent.mkdir(parents=True, exist_ok=True)
    TRIAL_TARGET.write_text(TARGET.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"created {TARGET.name} and {TRIAL_TARGET.relative_to(HERE)} ({len(cells)} cells)")


if __name__ == "__main__":
    build()
