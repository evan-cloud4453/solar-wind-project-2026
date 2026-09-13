#!/usr/bin/env python
"""P18 노트북 생성 — P3 원본을 패치해서 만든다.

P18 은 새 모델이 아니다. **P3 의 계측기만 바꾼 것**이다.

  P3   epoch·체크포인트를 official validation 최저점으로 고른다
       (60 epoch 중 최저 = val 1,199샘플 = 자전 11.2회전 위에서 60번 고르기)
  P18  epoch 을 train 사슬 CV 로 고르고, val 은 마지막에 딱 한 번 읽는다
       (28개 사슬 = 자전 93회전)

모델·피처·손실·증강·전처리는 P3 그대로다. 그래서 이 파일은 P3 노트북을 다시
타이핑하지 않고 **원본 셀을 읽어 세 군데만 손댄다.**

  [삽입] 설정 셀 뒤     — P18 노브
  [삽입] 데이터 로드 뒤 — 사슬 복원 · 폴드 구성
  [교체] 학습 셀        — CV 로 epoch 결정 후 전체 재학습

사용:
    python build_p18_notebook.py            # -> code_p18.ipynb
    python build_p18_notebook.py --check    # 패치 지점만 확인하고 쓰지 않음
"""

import argparse
import json
import sys
from pathlib import Path

BASE = Path("Trial/P3/code_p3.ipynb")
OUTPUT = Path("code_p18.ipynb")


def markdown_cell(text):
    return {"cell_type": "markdown", "metadata": {},
            "source": text.strip("\n").splitlines(keepends=True)}


def code_cell(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.strip("\n").splitlines(keepends=True)}


def source_of(cell):
    return "".join(cell["source"])


# =====================================================================
TITLE_MD = """
# 태양풍 속도 예측 — P18 (P3 + 사슬 CV 계측기)

**P3 와 모델이 같다.** 바뀐 것은 *무엇을 보고 epoch 을 고르는가* 뿐이다.

| | P3 | P18 |
|---|---|---|
| epoch 선택 | official validation 최저점 (60 중 1) | **train 사슬 CV 평활 곡선의 최저점** |
| validation | early stopping · 체크포인트 선택에 사용 | **마지막에 1회 읽기만** |
| LR 스케줄 | `ReduceLROnPlateau` (val 에 의존) | **`CosineAnnealingLR`** (데이터 비의존) |
| 모델·피처·손실·증강 | — | **P3 와 동일** |

스케줄러를 바꾼 건 선택이 아니라 필연이다. `ReduceLROnPlateau` 는 LR 궤적이
평가 지표에 의존해서, CV 에서 고른 epoch 을 전체 재학습에 옮길 수 없다.
"""

CONFIG_MD = """
## 0-b. P18 설정 — 계측기

사슬은 파일명만으로 복원되는 **연속 관측 구간**이다. train 은 28개 사슬,
총 자전 93회전. 사슬을 통째로 폴드에 배정하므로 윈도우 중첩 누수가 구조적으로 0이고
embargo 가 필요 없다.
"""

CONFIG_CELL = """
# ===== P18 계측기 노브 ================================================
P18_N_FOLDS = 5           # 사슬 grouped CV 폴드 수
P18_N_REPEATS = 3         # 배정을 몇 번 다시 뽑을지. 사슬 길이가 33~980 로
                          # 불균등해서 단일 배정의 분산이 크다
P18_CV_SEEDS = (777,)     # 시드를 늘리면 (777, 778, 779) 처럼. 비용은 배수로 증가
P18_CV_EPOCHS = EPOCHS    # CV 곡선을 어디까지 볼지. P3 의 EPOCHS(60) 과 동일하게
P18_EPOCH_SMOOTH = 3      # epoch 축 이동평균. argmin 이 단발 노이즈를 집는 것 방지
P18_FOLD_SEED = 0         # 사슬 -> 폴드 배정 난수

# 상관거리 서브샘플링 (다음 사다리 칸). 0 = 끔 = P3 와 동일한 에폭 정의.
# wind 자기상관 1/e 감쇠가 9스텝(2.2일)이라, 9로 두면 한 에폭이 거의 중복 없는
# 표본만 보게 된다. rung 0 에서는 반드시 0 으로 둔다 — 한 번에 한 가지만 바꾼다.
P18_STRIDE = 0

# P3 산출물을 덮어쓰지 않는다
OUTPUT_DIR = WORK_DIR / "outputs_p18"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"P18 계측기 — 폴드 {P18_N_FOLDS} x 반복 {P18_N_REPEATS} x 시드 {len(P18_CV_SEEDS)}"
      f" = {P18_N_FOLDS * P18_N_REPEATS * len(P18_CV_SEEDS)} 회 학습")
print(f"           CV epoch {P18_CV_EPOCHS} · 평활창 {P18_EPOCH_SMOOTH}"
      f" · stride {P18_STRIDE or '끔'}")
print(f"출력: {OUTPUT_DIR.resolve()}")
"""

CHAIN_MD = """
## 1-b. 사슬 복원 · 폴드 구성

샘플은 6h stride 슬라이딩 윈도우이고 행은 셔플되어 있다. 프레임 후행관계로
원래의 연속 구간을 복원한다. `build_timeline.py` 와 같은 규약이며, 제출 노트북이
외부 파일에 의존하지 않도록 여기에 그대로 넣는다.

> **사슬 인덱스는 시간 순서가 아니다.** 사슬은 `sorted(머리 파일명)` 으로 열거되는데
> 파일명은 익명화되어 있어 시간을 담지 않는다 (아래 진단이 실측치를 찍는다).
> 그래서 P9 의 `FOLD_MODE="block"`(시간 연속 블록)·`"forward"`(전진 검증)가 표방한
> 시간 성질은 성립하지 않는다. 여기서는 그 사실을 인정하고 **배정을 여러 번 다시
> 뽑아 평균내는** 쪽으로 간다.
"""

CHAIN_CELL = """
def reconstruct_frame_chains(inputs):
    # 후행관계로 연속 구간을 복원한다. 머리 정렬 순서라 실행 간 결정적이다.
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
            visited.add(node); chain.append(node); node = successor.get(node)
        chains.append(chain)
    assert conflicts == 0 and not (names - visited), "사슬 복원 실패"
    return chains


def window_positions(inputs, chains):
    # 각 샘플이 어느 사슬의 몇 번째 스텝에서 시작하는지
    position = {name: (c, o) for c, chain in enumerate(chains)
                for o, name in enumerate(chain)}
    first = inputs[IMAGE_COLUMNS[0]].to_numpy()
    chain_id = np.array([position[n][0] for n in first], np.int64)
    t_start = np.array([position[n][1] for n in first], np.int64)
    return chain_id, t_start


TRAIN_CHAINS = reconstruct_frame_chains(train_inputs)
TRAIN_CHAIN_ID, TRAIN_T_START = window_positions(train_inputs, TRAIN_CHAINS)
CHAIN_LENGTHS = np.array([len(c) for c in TRAIN_CHAINS], np.int64)

# --- 무결성: stride 1 슬라이딩 윈도우가 맞는지 --------------------------
expected_windows = int(np.maximum(CHAIN_LENGTHS - 19, 0).sum())
assert expected_windows == len(train_inputs), (
    f"Σ(사슬길이-19)={expected_windows} != 행 {len(train_inputs)}")
unique_frames = int(CHAIN_LENGTHS.sum())

print(f"사슬 {len(TRAIN_CHAINS)}개 · 고유 프레임 {unique_frames:,} "
      f"· 재사용 {len(train_inputs) * 20 / unique_frames:.2f}회/장")
print(f"사슬 길이 {CHAIN_LENGTHS.min()}~{CHAIN_LENGTHS.max()} "
      f"· 총 {unique_frames * 6 / 24:,.0f}일 = 자전 {unique_frames * 6 / 24 / 27.2753:.1f}회전")

# --- 진단: 파일명이 시간을 인코딩하는가 ---------------------------------
increasing = total = 0
for chain in TRAIN_CHAINS:
    ids = np.array([int(n[6:12]) for n in chain])
    increasing += int((np.diff(ids) > 0).sum()); total += len(ids) - 1
monotone_ratio = increasing / total
print(f"\\n[진단] 사슬 내 파일명 id 증가 비율 {monotone_ratio:.1%} "
      f"(시간 인코딩이면 100%, 무작위면 ~50%)")
if monotone_ratio < 0.9:
    print("       -> 파일명에 시간 정보 없음. 사슬 간 시간 순서는 복원 불가.")
    print("          '시간 연속 블록' · '전진 검증' 은 성립하지 않는다.")


def chain_folds(chain_id, n_folds=P18_N_FOLDS, n_repeats=P18_N_REPEATS,
                seed=P18_FOLD_SEED):
    # 사슬을 무작위 순서로 훑으며 가장 가벼운 폴드에 넣는다 (그리디 균형).
    # 반복마다 순서를 다시 뽑아 배정 분산을 평균낸다.
    counts = np.bincount(chain_id, minlength=len(TRAIN_CHAINS))
    rows = np.arange(len(chain_id))
    generator = np.random.default_rng(seed)
    folds = []
    for repeat in range(n_repeats):
        assignment = np.zeros(len(counts), np.int64)
        load = np.zeros(n_folds, np.int64)
        for chain in generator.permutation(len(counts)):
            fold = int(np.argmin(load))
            assignment[chain] = fold
            load[fold] += counts[chain]
        fold_of_row = assignment[chain_id]
        for fold in range(n_folds):
            held = fold_of_row == fold
            assert held.any() and (~held).any(), "빈 폴드"
            folds.append({"repeat": repeat, "fold": fold,
                          "train": rows[~held], "evaluate": rows[held]})
    return folds


P18_FOLDS = chain_folds(TRAIN_CHAIN_ID)
fractions = [len(f["evaluate"]) / len(train_inputs) for f in P18_FOLDS]
print(f"\\n폴드 {len(P18_FOLDS)}개 · 평가 비율 {min(fractions):.1%}~{max(fractions):.1%}")
print("  (학습, 평가):", [(len(f["train"]), len(f["evaluate"])) for f in P18_FOLDS[:5]], "...")
"""

TRAIN_CELL = """
# ===== P18 학습 — 사슬 CV 로 epoch 을 정하고 그 epoch 으로 전체 재학습 =====
#
# val 은 이 셀에서 한 번도 읽지 않는다. 아래 9번 셀이 처음이자 마지막으로 읽는다.

# 폴드 평가용 데이터셋 — 증강이 꺼진 같은 train 데이터
train_eval_dataset = SolarWindDataset(train_image_array, train_image_index, train_inputs,
                                      train_wind, train_wind_valid, train_ch,
                                      train_targets, training=False)

_WORKER_SEED = SEED


def p18_seed_worker(worker_id):
    worker_seed = (_WORKER_SEED + worker_id) % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def subset_loader(dataset, rows, shuffle, seed, stride_offset=None):
    if stride_offset is not None:
        rows = rows[((TRAIN_T_START[rows] - stride_offset) % P18_STRIDE) == 0]
    subset = torch.utils.data.Subset(dataset, rows.tolist())
    options = dict(dataset=subset, batch_size=BATCH_SIZE, shuffle=shuffle,
                   num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=False,
                   worker_init_fn=p18_seed_worker,
                   generator=torch.Generator().manual_seed(seed))
    # stride 를 쓰면 에폭마다 loader 를 다시 만드므로 워커를 붙잡아 두지 않는다
    if NUM_WORKERS > 0 and stride_offset is None:
        options.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(**options)


@torch.no_grad()
def predict_with(network, loader):
    network.eval()
    predictions = []
    for batch in loader:
        moved = {key: batch[key].to(DEVICE, non_blocking=PIN_MEMORY)
                 for key in ("images", "wind_seq", "wind_stats", "ch_seq",
                             "ballistic", "last_wind")}
        with torch.amp.autocast(DEVICE.type, enabled=USE_AMP):
            residual = network(moved["images"], moved["wind_seq"], moved["wind_stats"],
                               moved["ch_seq"], moved["ballistic"])
        prediction = (residual.float() + moved["last_wind"].unsqueeze(1)
                      ).clamp(CLIP_LOW, CLIP_HIGH)
        predictions.append(prediction.cpu().numpy())
    return np.concatenate(predictions).astype(np.float64)


def fit(train_rows, evaluate_rows, seed, epochs, t_max=None, verbose=False):
    # evaluate_rows=None 이면 곡선을 재지 않고 학습만 한다 (최종 전체 학습).
    # t_max 를 CV 와 같게 두면 LR 궤적이 CV 와 동일해져 epoch 이 그대로 옮겨진다.
    global _WORKER_SEED
    _WORKER_SEED = seed
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    network = build_model()
    optimizer = torch.optim.AdamW(network.parameters(), lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=t_max or epochs)
    scaler = torch.amp.GradScaler(DEVICE.type, enabled=USE_AMP)

    fixed_loader = None if P18_STRIDE else subset_loader(
        train_dataset, train_rows, True, seed)
    evaluate_loader = None if evaluate_rows is None else subset_loader(
        train_eval_dataset, evaluate_rows, False, seed)
    evaluate_targets = None if evaluate_rows is None else train_targets[evaluate_rows]

    curve = np.zeros((epochs, 12), np.float64)
    for epoch in range(epochs):
        loader = fixed_loader if fixed_loader is not None else subset_loader(
            train_dataset, train_rows, True, seed, epoch % P18_STRIDE)
        network.train()
        for batch in loader:
            images = batch["images"].to(DEVICE, non_blocking=PIN_MEMORY)
            wind_seq = batch["wind_seq"].to(DEVICE, non_blocking=PIN_MEMORY)
            wind_stats = batch["wind_stats"].to(DEVICE, non_blocking=PIN_MEMORY)
            ch_seq = batch["ch_seq"].to(DEVICE, non_blocking=PIN_MEMORY)
            ballistic = batch["ballistic"].to(DEVICE, non_blocking=PIN_MEMORY)
            last_wind = batch["last_wind"].to(DEVICE, non_blocking=PIN_MEMORY)
            target = batch["target"].to(DEVICE, non_blocking=PIN_MEMORY)
            if USE_CNN and AUGMENT:
                images = augment_batch(images)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(DEVICE.type, enabled=USE_AMP):
                residual = network(images, wind_seq, wind_stats, ch_seq, ballistic)
            loss = metric_loss(residual.float() + last_wind.unsqueeze(1), target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(network.parameters(), GRAD_CLIP)
            scaler.step(optimizer); scaler.update()
        scheduler.step()
        if evaluate_loader is not None:
            curve[epoch] = official_rmse(
                evaluate_targets, predict_with(network, evaluate_loader))[1]
            if verbose:
                print(f"    epoch {epoch + 1:03d} rmse {curve[epoch].mean():7.3f}",
                      flush=True)
    del fixed_loader, evaluate_loader
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return network, curve


def smooth_curve(curve, window=P18_EPOCH_SMOOTH):
    # epoch 축 이동평균. argmin 이 단발 노이즈를 집는 것을 막는다.
    if window <= 1:
        return curve
    left = window // 2
    padded = np.pad(curve, ((left, window - 1 - left), (0, 0)), mode="edge")
    return np.stack([padded[i:i + window].mean(axis=0) for i in range(len(curve))])


# --- CV 실행 -----------------------------------------------------------
runs = []
started_all = time.perf_counter()
for entry in P18_FOLDS:
    for seed in P18_CV_SEEDS:
        started = time.perf_counter()
        _, curve = fit(entry["train"], entry["evaluate"], seed, P18_CV_EPOCHS)
        runs.append(curve)
        elapsed = time.perf_counter() - started
        remaining = (len(P18_FOLDS) * len(P18_CV_SEEDS) - len(runs)) * elapsed
        print(f"repeat {entry['repeat']} fold {entry['fold']} seed {seed} "
              f"| best {smooth_curve(curve).mean(axis=1).min():7.3f} "
              f"| {elapsed:6.1f}s | 남은 예상 {remaining / 60:5.1f}분", flush=True)

CV_CURVES = np.stack(runs)                                   # (runs, epochs, 12)
np.save(OUTPUT_DIR / "cv_curves.npy", CV_CURVES)

per_run = np.stack([smooth_curve(c).mean(axis=1) for c in CV_CURVES])   # (runs, epochs)
mean_curve = per_run.mean(axis=0)
P18_BEST_EPOCH = int(np.argmin(mean_curve)) + 1
CV_SCORE = float(mean_curve.min())
CV_SE = float(per_run[:, P18_BEST_EPOCH - 1].std(ddof=1) / np.sqrt(len(per_run)))

print(f"\\n[CV] 기준 epoch = {P18_BEST_EPOCH} / {P18_CV_EPOCHS}")
print(f"[CV] 공식 RMSE   = {CV_SCORE:.3f} +- {CV_SE:.3f} km/s  ({len(per_run)}회 평균)")
print(f"[CV] 폴드별 최저 epoch 분포: "
      f"{np.bincount(per_run.argmin(axis=1) + 1, minlength=P18_CV_EPOCHS + 1).nonzero()[0].tolist()}")
print(f"총 CV 시간 {(time.perf_counter() - started_all) / 60:.1f}분")

# --- 전체 재학습 -------------------------------------------------------
# t_max 를 CV 와 같게 둔다. 그래야 epoch k 에서의 LR 이 CV 때와 같다.
checkpoint_path = OUTPUT_DIR / "best_model.pth"
CONFIG = {"image_size": IMAGE_SIZE, "channels": list(CHANNELS), "use_cnn": USE_CNN,
          "use_ch": USE_CH, "use_ballistic": USE_BALLISTIC, "ch_grid": list(CH_GRID),
          "ch_threshold_ratio": CH_THRESHOLD_RATIO, "disk": [DISK_Y, DISK_X, DISK_R],
          "transit_speeds": list(TRANSIT_SPEEDS), "seed": SEED,
          "image_mean": IMAGE_MEAN.tolist(), "image_std": IMAGE_STD.tolist(),
          "wind_mean": WIND_MEAN, "wind_std": WIND_STD, "diff_std": DIFF_STD,
          "ch_mean": CH_MEAN.tolist(), "ch_std": CH_STD.tolist(),
          "stats_mean": STATS_MEAN.tolist(), "stats_std": STATS_STD.tolist(),
          "residual_mean": RESIDUAL_MEAN.tolist(), "residual_std": RESIDUAL_STD.tolist(),
          "clip_low": CLIP_LOW, "clip_high": CLIP_HIGH,
          "initialization": "random_from_scratch",
          "selector": "chain_cv", "cv_folds": P18_N_FOLDS, "cv_repeats": P18_N_REPEATS,
          "cv_seeds": list(P18_CV_SEEDS), "cv_epochs": P18_CV_EPOCHS,
          "stride": P18_STRIDE, "scheduler": "cosine"}

model, _ = fit(np.arange(len(train_inputs)), None, SEED,
               epochs=P18_BEST_EPOCH, t_max=P18_CV_EPOCHS)
torch.save({"model_state_dict": model.state_dict(), "epoch": P18_BEST_EPOCH,
            "cv_official_rmse": CV_SCORE, "cv_se": CV_SE, **CONFIG}, checkpoint_path)
print(f"\\n전체 재학습 완료 ({P18_BEST_EPOCH} epoch) -> {checkpoint_path}")

history_frame = pd.DataFrame({"epoch": np.arange(1, P18_CV_EPOCHS + 1),
                              "cv_rmse": mean_curve})
history_frame.to_csv(OUTPUT_DIR / "history.csv", index=False)

figure, axis = plt.subplots(figsize=(7, 4))
for row in per_run:
    axis.plot(np.arange(1, P18_CV_EPOCHS + 1), row, color="gray", alpha=0.25, linewidth=0.8)
axis.plot(np.arange(1, P18_CV_EPOCHS + 1), mean_curve, color="C0", linewidth=2,
          label="CV mean (smoothed)")
axis.axvline(P18_BEST_EPOCH, color="k", linestyle=":", label=f"epoch {P18_BEST_EPOCH}")
axis.axhline(persistence_score, color="gray", linestyle="--", label="persistence")
axis.set_xlabel("epoch"); axis.set_ylabel("official RMSE (km/s)")
axis.grid(alpha=0.3); axis.legend()
plt.tight_layout(); plt.savefig(OUTPUT_DIR / "cv_curve.png", dpi=140); plt.show()
"""


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
    load_at = locate("train_inputs = pd.read_csv")
    train_at = locate("for epoch in range(1, EPOCHS + 1)")
    print(f"패치 지점 — 설정 셀 {config_at} · 데이터 로드 {load_at} · 학습 셀 {train_at}")

    # 학습 셀이 참조하는 것들이 교체 후에도 살아 있는지 확인
    for name in ("checkpoint_path", "CONFIG", "model"):
        assert name in TRAIN_CELL, f"교체 셀이 {name} 을 정의하지 않는다"
    if args.check:
        print("확인만 하고 종료합니다.")
        return

    patched = []
    for index, cell in enumerate(cells):
        if index == train_at:
            patched.append(code_cell(TRAIN_CELL))
            continue
        if index == 0 and cell["cell_type"] == "markdown":
            patched.append(markdown_cell(TITLE_MD))
            continue
        patched.append(cell)
        if index == config_at:
            patched += [markdown_cell(CONFIG_MD), code_cell(CONFIG_CELL)]
        if index == load_at:
            patched += [markdown_cell(CHAIN_MD), code_cell(CHAIN_CELL)]

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
