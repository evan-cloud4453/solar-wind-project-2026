#!/usr/bin/env python
"""사슬 기반 교차검증 · 윈도우 서브샘플링 — timeline.parquet 위에 얹는 계측 도구.

`build_timeline.py` 가 만든 두 표를 읽어 다음을 제공한다.

  chain_folds()       사슬을 통째로 폴드에 배정한 반복 grouped CV
  stride_windows()    상관거리만큼 띄운 윈도우 부분집합 (에폭 정의 교정용)
  paired_summary()    같은 폴드·시드에서 두 설정을 대응비교

--- 왜 "시간 연속 블록" 이 아니라 반복 grouped CV 인가 -----------------------

P9 는 `FOLD_MODE="block"` 을 "시간 연속 블록" 이라 부르며 사슬 인덱스를 순서대로
잘랐다. 그 전제는 사슬 인덱스가 시간 순서라는 것인데, 실측하면 성립하지 않는다.

  · 사슬은 `sorted(names - predecessors)` 로 만들어진다. 따라서 P9 의 점검식
    `heads == sorted(heads)` 는 **구성상 항상 True** 다. 경고가 뜰 수 없다.
  · 파일명이 시간을 인코딩하는지 직접 재면, 사슬 내부에서 파일명 id 가
    증가하는 비율이 **49.7%** 다. 무작위(50%)와 구분되지 않는다.

즉 파일명은 익명화되어 있고 **28개 사슬의 실제 시간 순서는 주어진 데이터로 복원
불가능하다.** 그러므로 `block` 은 사실상 무작위 그룹 분할이고, `forward`(전진 검증)
는 의미가 없다.

여기서는 그 사실을 받아들이고 대신 **배정 자체를 여러 번 바꿔 평균낸다.**
사슬 길이가 33~980으로 극단적으로 불균등해서 단일 배정의 분산이 크기 때문이다.
사슬을 통째로 배정하므로 윈도우 중첩 누수는 구조적으로 0이고, embargo 가 필요 없다.
"""

from pathlib import Path

import numpy as np
import pandas as pd

WINDOW = 20

# wind 자기상관이 1/e 로 떨어지는 지연 = 9스텝(2.2일). 이보다 가까운 두 윈도우는
# 사실상 같은 표본이다. stride_windows() 의 기본 간격 근거.
DECORRELATION_STEPS = 9


def load_timeline(cache_root=Path("work/cache"), split="train"):
    """timeline / windows 를 읽어 한 split 만 돌려준다. parquet 없으면 csv 폴백."""
    cache_root = Path(cache_root)
    tables = []
    for stem in ("timeline", "windows"):
        parquet, csv = cache_root / f"{stem}.parquet", cache_root / f"{stem}.csv"
        if parquet.exists():
            frame = pd.read_parquet(parquet)
        elif csv.exists():
            frame = pd.read_csv(csv)
        else:
            raise FileNotFoundError(
                f"{parquet} 도 {csv} 도 없습니다. 먼저 `python build_timeline.py` 를 돌리세요.")
        tables.append(frame[frame["split"] == split].reset_index(drop=True))
    return tables[0], tables[1]


def chain_sample_counts(windows):
    """사슬별 윈도우 수."""
    return windows.groupby("chain").size().sort_index()


def assign_chains(counts, n_folds, rng):
    """사슬을 무작위 순서로 훑으며 가장 가벼운 폴드에 넣는다 (그리디 균형)."""
    chains = counts.index.to_numpy()
    order = rng.permutation(len(chains))
    assignment = {}
    load = np.zeros(n_folds, np.int64)
    for position in order:
        fold = int(np.argmin(load))
        assignment[int(chains[position])] = fold
        load[fold] += int(counts.iloc[position])
    return assignment, load


def chain_folds(windows, n_folds=5, n_repeats=3, seed=0):
    """사슬 통째 배정 grouped CV 를 n_repeats 번 서로 다른 배정으로 만든다.

    반환: [{repeat, fold, train, evaluate, chains}] — train/evaluate 는 windows 의
    행 인덱스(= 원본 inputs.csv 행 번호와 동일한 `row` 값) 배열이다.
    """
    counts = chain_sample_counts(windows)
    if n_folds > len(counts):
        raise ValueError(f"폴드 {n_folds}개 > 사슬 {len(counts)}개")
    chain_of = windows["chain"].to_numpy()
    rows = windows["row"].to_numpy()
    rng = np.random.default_rng(seed)

    folds = []
    for repeat in range(n_repeats):
        assignment, _ = assign_chains(counts, n_folds, rng)
        fold_of_row = np.array([assignment[int(c)] for c in chain_of])
        for fold in range(n_folds):
            held = fold_of_row == fold
            train_rows, evaluate_rows = rows[~held], rows[held]
            # 사슬을 통째로 배정했으므로 프레임 공유가 있을 수 없다. 확인만 한다.
            assert not (set(chain_of[~held]) & set(chain_of[held])), "폴드가 사슬을 쪼갰다"
            folds.append({
                "repeat": repeat, "fold": fold,
                "train": train_rows, "evaluate": evaluate_rows,
                "chains": sorted(c for c, f in assignment.items() if f == fold),
            })
    return folds


def stride_windows(windows, stride=DECORRELATION_STEPS, offset=0, rows=None):
    """사슬 안에서 `stride` 스텝마다 하나씩만 남긴 윈도우 행 번호.

    에폭마다 offset 을 바꾸면 전체 윈도우를 결국 다 보되, 한 에폭 안에서는
    거의 중복이 없는 표본만 본다. offset 은 stride 로 모듈러한다.
    """
    frame = windows if rows is None else windows[windows["row"].isin(rows)]
    keep = ((frame["t_start"].to_numpy() - int(offset)) % int(stride)) == 0
    return frame["row"].to_numpy()[keep]


def paired_summary(scores_a, scores_b=None, labels=("A", "B")):
    """폴드별 점수 배열을 요약한다. 둘 주면 대응비교(paired) 로 낸다."""
    a = np.asarray(scores_a, float)
    if scores_b is None:
        return {"mean": float(a.mean()), "se": float(a.std(ddof=1) / np.sqrt(len(a))),
                "n": len(a)}
    b = np.asarray(scores_b, float)
    if len(a) != len(b):
        raise ValueError("대응비교는 같은 폴드 구성에서 나온 같은 길이여야 한다")
    difference = b - a
    se = float(difference.std(ddof=1) / np.sqrt(len(difference)))
    return {
        labels[0]: float(a.mean()), labels[1]: float(b.mean()),
        "delta": float(difference.mean()), "se": se,
        "t": float(difference.mean() / se) if se > 0 else float("nan"),
        "n": len(difference),
    }


def describe_folds(folds, windows):
    """폴드 구성 요약표."""
    rows = []
    for entry in folds:
        rows.append({
            "repeat": entry["repeat"], "fold": entry["fold"],
            "chains": len(entry["chains"]),
            "train": len(entry["train"]), "evaluate": len(entry["evaluate"]),
            "eval_frac": len(entry["evaluate"]) / len(windows),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    timeline, windows = load_timeline()
    print(f"train 윈도우 {len(windows):,} · 사슬 {windows['chain'].nunique()}개")
    print("\n사슬별 윈도우 수:", chain_sample_counts(windows).to_numpy().tolist())

    folds = chain_folds(windows, n_folds=5, n_repeats=3, seed=0)
    table = describe_folds(folds, windows)
    print("\n폴드 구성:")
    print(table.to_string(index=False,
                          formatters={"eval_frac": "{:.1%}".format}))
    print(f"\n평가 비율 범위 {table.eval_frac.min():.1%} ~ {table.eval_frac.max():.1%}"
          f"  (그리디 균형 후)")

    subset = stride_windows(windows, stride=DECORRELATION_STEPS, offset=0)
    print(f"\nstride={DECORRELATION_STEPS} 서브샘플: {len(subset):,} 윈도우"
          f" ({len(subset)/len(windows):.1%})")
    covered = set()
    for offset in range(DECORRELATION_STEPS):
        covered |= set(stride_windows(windows, DECORRELATION_STEPS, offset).tolist())
    print(f"offset 0~{DECORRELATION_STEPS-1} 합집합: {len(covered):,} "
          f"({len(covered)/len(windows):.1%}) — 전체를 덮는가: {len(covered) == len(windows)}")
