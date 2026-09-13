#!/usr/bin/env python
"""사슬 복원 -> timeline.parquet / windows.parquet

샘플은 6h stride 슬라이딩 윈도우이고 행은 셔플되어 있다. 이 스크립트는 프레임
후행관계로 원래의 연속 관측 구간(사슬)을 복원해 두 개의 표로 편다.

  timeline.parquet   고유 프레임 1개 = 1행   (split, chain, t, image, wind, ...)
  windows.parquet    원본 샘플 1개 = 1행     (split, sample_id, chain, t_start, t_end)

프레임 캐시 순서가 계보마다 둘로 갈려 있어 두 인덱스를 모두 실어 준다.

  chain_index  `[n for chain in chains for n in chain]` 순서.
               P9 `train_files` · `p11_extract.split_files()` 와 완전히 동일 →
               `p11a_{split}.npz` 행 인덱스에 그대로 맞는다.
  name_index   `sorted(unique(filenames))` 순서.
               P3 `prepare_image_memmap()` · `cached_ch_grid()` 와 동일 →
               `work/cache/{IMAGE_SIZE}px/{split}_images.npy`,
               `ch_{split}_*.npy` 행 인덱스에 그대로 맞는다.

주의 — split 경계
  사슬 복원은 split 내부에서만 한다. split 을 가로질러 프레임을 잇는 것은
  test 샘플 간 시계열 재구성 = target 역산에 해당해 규정상 실격이다.
  이 스크립트는 split 별로 독립 복원하며, 교차 조인을 만들지 않는다.

사용:
    python build_timeline.py                      # 서버: train/validation/test 전부
    python build_timeline.py --xlsx ~/INPUT.xlsx  # 로컬: train 만 (오프라인 검증용)
    python build_timeline.py --out work/cache
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

IMAGE_COLUMNS = [f"image_{i:02d}" for i in range(20)]
WIND_COLUMNS = [f"wind_{i:02d}" for i in range(20)]
TARGET_COLUMNS = [f"target_{i:02d}" for i in range(12)]
WINDOW = 20
HORIZONS = 12

DATA_ROOT_CANDIDATES = [
    Path(os.getenv("SW_DATA_ROOT", "")) if os.getenv("SW_DATA_ROOT") else None,
    Path("public_dataset/competition_dataset_6h"),
    Path("/home/jovyan/public_dataset/competition_dataset_6h"),
    Path("public/public_dataset/competition_dataset_6h"),
    Path("/home/jovyan/public/public_dataset/competition_dataset_6h"),
    Path("dataset"),
    Path("/home/jovyan/dataset"),
]


def find_data_root():
    for candidate in DATA_ROOT_CANDIDATES:
        if candidate is not None and (candidate / "train/inputs.csv").exists():
            return candidate
    return None


# ---- 사슬 복원 — p11_extract.reconstruct_frame_chains 와 동일 규약 ----------

def reconstruct_frame_chains(inputs):
    """후행관계로 연속 구간을 복원한다. 머리 정렬 순서라 실행 간 결정적이다."""
    images = inputs[IMAGE_COLUMNS].to_numpy()
    successor, predecessor, conflicts = {}, {}, 0
    for row in images:
        for current, following in zip(row[:-1], row[1:]):
            if successor.setdefault(current, following) != following:
                conflicts += 1
            if predecessor.setdefault(following, current) != current:
                conflicts += 1
    names = set(images.ravel().tolist())
    chains, visited = [], set()
    for head in sorted(names - set(predecessor)):
        chain, node = [], head
        while node is not None and node not in visited:
            visited.add(node)
            chain.append(node)
            node = successor.get(node)
        chains.append(chain)
    assert conflicts == 0, f"후행관계 충돌 {conflicts}건 — 슬라이딩 윈도우 가정이 깨졌다"
    assert not (names - visited), "복원되지 않은 프레임이 있다"
    return chains


def frame_wind_map(inputs):
    """image -> wind 전역 사상. 값이 갈리면 즉시 실패시킨다."""
    images = inputs[IMAGE_COLUMNS].to_numpy()
    winds = inputs[WIND_COLUMNS].to_numpy(np.float64)
    mapping, conflicts = {}, []
    for row_images, row_winds in zip(images, winds):
        for name, value in zip(row_images, row_winds):
            previous = mapping.setdefault(name, value)
            if previous != value:
                conflicts.append((name, previous, value))
    if conflicts:
        head = ", ".join(f"{n}:{a}!={b}" for n, a, b in conflicts[:5])
        raise AssertionError(f"image->wind 불일치 {len(conflicts)}건 ({head})")
    return mapping


def build_split(name, inputs, targets=None):
    """한 split 을 timeline / windows 두 프레임으로 편다."""
    chains = reconstruct_frame_chains(inputs)
    winds = frame_wind_map(inputs)

    # 프레임 캐시가 두 가지 순서로 존재한다. 둘 다 실어 준다.
    #   chain_index — 사슬 평탄화 순서.  P9 `train_files` · p11_extract.split_files()
    #   name_index  — 파일명 정렬 순서.  P3 prepare_image_memmap() · cached_ch_grid()
    name_order = {n: i for i, n in enumerate(sorted(winds))}

    position = {}          # image -> (chain_id, t)
    rows = []
    chain_index = 0
    for chain_id, chain in enumerate(chains):
        for t, image in enumerate(chain):
            position[image] = (chain_id, t)
            rows.append((name, chain_id, t, image, winds[image], len(chain),
                         chain_index, name_order[image]))
            chain_index += 1
    timeline = pd.DataFrame(rows, columns=[
        "split", "chain", "t", "image", "wind", "chain_len",
        "chain_index", "name_index"])

    integral = bool(np.all(np.equal(np.mod(timeline["wind"].to_numpy(), 1), 0)))
    timeline["wind"] = timeline["wind"].astype("int32" if integral else "float32")
    for column, dtype in (("chain", "int16"), ("t", "int32"), ("chain_len", "int32"),
                          ("chain_index", "int32"), ("name_index", "int32")):
        timeline[column] = timeline[column].astype(dtype)

    # ---- 윈도우 위치 — 20 프레임이 실제로 사슬 위 연속인지까지 확인한다
    images = inputs[IMAGE_COLUMNS].to_numpy()
    chain_of = np.empty(len(inputs), np.int32)
    t_start = np.empty(len(inputs), np.int32)
    for row_index, row_images in enumerate(images):
        located = [position[n] for n in row_images]
        chain_ids = {c for c, _ in located}
        steps = [t for _, t in located]
        assert len(chain_ids) == 1, f"{name} row {row_index}: 윈도우가 사슬을 가로지른다"
        assert steps == list(range(steps[0], steps[0] + WINDOW)), \
            f"{name} row {row_index}: 프레임이 사슬 위에서 연속이 아니다"
        chain_of[row_index] = chain_ids.pop()
        t_start[row_index] = steps[0]

    windows = pd.DataFrame({
        "split": name,
        "sample_id": inputs["sample_id"].to_numpy() if "sample_id" in inputs
                     else np.arange(len(inputs)),
        "row": np.arange(len(inputs), dtype=np.int32),
        "chain": chain_of.astype("int16"),
        "t_start": t_start,
        "t_end": (t_start + WINDOW - 1).astype("int32"),
    })

    audit = audit_split(name, timeline, windows, chains, inputs, targets)
    return timeline, windows, audit


# ---- 감사 ------------------------------------------------------------------

def audit_split(name, timeline, windows, chains, inputs, targets):
    lengths = np.array([len(c) for c in chains], np.int64)
    expected = int(np.maximum(lengths - (WINDOW - 1), 0).sum())
    assert expected == len(inputs), \
        f"{name}: Σ(사슬길이-19)={expected} 인데 행은 {len(inputs)} — 윈도우가 stride 1 이 아니다"

    # 순서 타당성 — 사슬 인접 스텝이 무작위 쌍보다 훨씬 붙어 있어야 한다
    steps, wind_all = [], timeline["wind"].to_numpy(np.float64)
    for _, group in timeline.groupby("chain", sort=True):
        series = group.sort_values("t")["wind"].to_numpy(np.float64)
        if len(series) > 1:
            steps.append(np.abs(np.diff(series)))
    steps = np.concatenate(steps) if steps else np.array([np.nan])
    rng = np.random.default_rng(0)
    random_gap = np.abs(rng.choice(wind_all, 50_000) - rng.choice(wind_all, 50_000))

    audit = {
        "split": name,
        "rows": len(inputs),
        "frames": len(timeline),
        "reuse": len(inputs) * WINDOW / len(timeline),
        "chains": len(chains),
        "steps_days": len(timeline) * 6 / 24,
        "rotations": len(timeline) * 6 / 24 / 27.2753,
        "chain_min": int(lengths.min()),
        "chain_max": int(lengths.max()),
        "adjacent_step": float(steps.mean()),
        "random_gap": float(random_gap.mean()),
        "target_checked": 0,
        "target_match": float("nan"),
    }

    # ---- target 정합 — `_19` = T0 이고 오프셋이 없다면 target_j == wind[t_end+1+j]
    if targets is not None:
        wind_grid = {}
        for chain_id, group in timeline.groupby("chain", sort=True):
            wind_grid[int(chain_id)] = group.sort_values("t")["wind"].to_numpy(np.float64)
        checked = matched = 0
        values = targets.to_numpy(np.float64)
        for chain_id, t_end, row in zip(windows["chain"], windows["t_end"], windows["row"]):
            series = wind_grid[int(chain_id)]
            for horizon in range(HORIZONS):
                index = int(t_end) + 1 + horizon
                if index < len(series):
                    checked += 1
                    matched += int(series[index] == values[row, horizon])
        audit["target_checked"] = checked
        audit["target_match"] = matched / checked if checked else float("nan")

    return audit


def report(audit):
    print(f"\n[{audit['split']}]")
    print(f"  행 {audit['rows']:,} · 고유 프레임 {audit['frames']:,} "
          f"· 재사용 {audit['reuse']:.2f}회/장")
    print(f"  사슬 {audit['chains']}개 (최단 {audit['chain_min']} · 최장 {audit['chain_max']}) "
          f"· 총 {audit['steps_days']:,.0f}일 = 자전 {audit['rotations']:.1f}회전")
    print(f"  순서 검사 — 사슬 인접 |Δwind| {audit['adjacent_step']:.1f} km/s "
          f"vs 무작위 쌍 {audit['random_gap']:.1f} km/s")
    if audit["target_checked"]:
        print(f"  target 정합 — {audit['target_checked']:,}쌍 중 "
              f"{audit['target_match']:.4%} 일치  (target_j == wind[t_end+1+j])")


def write_table(frame, path):
    try:
        frame.to_parquet(path, index=False)
        return path
    except Exception as error:                       # pyarrow 부재 등
        fallback = path.with_suffix(".csv")
        frame.to_csv(fallback, index=False)
        print(f"  ! parquet 실패({type(error).__name__}) — CSV 로 저장: {fallback}")
        return fallback


def main():
    # Windows 콘솔은 기본이 cp949 라 한글·em dash 에서 죽는다. 서버(UTF-8)는 무영향.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=None,
                        help="대회 데이터 루트 (미지정 시 표준 후보 경로 탐색)")
    parser.add_argument("--xlsx", type=Path, default=None,
                        help="오프라인 검증용 train inputs 사본 (INPUT.xlsx)")
    parser.add_argument("--out", type=Path, default=Path("work/cache"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    sources = {}
    if args.xlsx is not None:
        sources["train"] = (pd.read_excel(args.xlsx), None)
        print(f"source: {args.xlsx}  (train 만, 오프라인 모드)")
    else:
        root = args.data_root or find_data_root()
        if root is None:
            searched = "\n".join(f"  - {c}" for c in DATA_ROOT_CANDIDATES if c)
            sys.exit("데이터 경로를 찾지 못했습니다:\n" + searched +
                     "\n로컬이면 --xlsx 로 INPUT.xlsx 를 주세요.")
        print(f"source: {root.resolve()}")
        for split in ("train", "validation", "test"):
            inputs_path = root / split / "inputs.csv"
            if not inputs_path.exists():
                print(f"  ! {split}/inputs.csv 없음 — 건너뜀")
                continue
            targets_path = root / split / "targets.csv"
            targets = (pd.read_csv(targets_path)[TARGET_COLUMNS]
                       if targets_path.exists() else None)
            sources[split] = (pd.read_csv(inputs_path), targets)

    timelines, windows_all, audits = [], [], []
    for split, (inputs, targets) in sources.items():
        timeline, windows, audit = build_split(split, inputs, targets)
        timelines.append(timeline)
        windows_all.append(windows)
        audits.append(audit)

    # split 간 파일명 충돌 — 있으면 image 만으로 키를 잡으면 안 된다는 뜻이다.
    # 충돌 여부만 보고할 뿐, 어떤 조인도 만들지 않는다.
    if len(timelines) > 1:
        names = [set(t["image"]) for t in timelines]
        collisions = sum(len(a & b) for i, a in enumerate(names) for b in names[i + 1:])
        print(f"\nsplit 간 파일명 충돌: {collisions:,}건"
              + ("  → 프레임 키는 반드시 (split, image) 로 잡을 것" if collisions
                 else "  → image 단독 키 안전"))

    timeline = pd.concat(timelines, ignore_index=True)
    windows = pd.concat(windows_all, ignore_index=True)
    for audit in audits:
        report(audit)

    timeline_path = write_table(timeline, args.out / "timeline.parquet")
    windows_path = write_table(windows, args.out / "windows.parquet")
    print(f"\nwrote {timeline_path}  ({len(timeline):,} rows)")
    print(f"wrote {windows_path}  ({len(windows):,} rows)")


if __name__ == "__main__":
    main()
