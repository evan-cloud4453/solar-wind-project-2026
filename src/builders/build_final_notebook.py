#!/usr/bin/env python
"""제출 노트북 생성 — SWEEP 이 고른 설정 하나를 자기 완결 노트북으로 뽑는다.

    python build_final_notebook.py work/sweep/final/01_ab12cd34/config.json

가 `code_final.ipynb` 를 만든다. 그 노트북은

  * SWEEP 과 **같은 엔진 코드**(build_sweep_notebook.ENGINE_SRC)를 셀로 품고,
  * `model.pth`(시드 앙상블 state_dict 묶음)를 읽어 test 를 추론하고,
  * `submission/` 에 3종(`code.ipynb`·`model.pth`·`submission.csv`)을 채운다.

`RETRAIN = True` 로 두면 같은 시드·epoch 으로 처음부터 다시 학습한다 — 규정의
"제출 코드로 결과가 재현되어야 한다" 를 문자 그대로 만족시키는 경로다.
"""

import argparse
import json
import pprint
import sys
from pathlib import Path

from build_sweep_notebook import ENGINE_SRC, code_cell, markdown_cell

TITLE_MD = """
# 태양풍 속도 예측 — 최종 제출 노트북

SWEEP(`code_sweep.ipynb`) 이 고른 설정 **하나**를 재현한다. 탐색 코드는 들어 있지 않다.

| | |
|---|---|
| 설정 | `{label}` |
| 사슬 CV | {cv} |
| official validation (시드 앙상블) | {val} |
| 시드 | {seeds} |
| epoch | {epochs} (cosine T_max={t_max}) |
| 출처 | `{source}` |

## 실행 순서

1. 위에서부터 전부 실행한다. 기본값(`RETRAIN = False`)이면 `model.pth` 를 읽어
   추론만 하므로 몇 분이면 끝난다.
2. 마지막 셀이 `submission/` 에 3종을 채우고 점검표를 찍는다.
3. **노트북을 저장(Ctrl+S)한 뒤** 마지막 셀을 한 번 더 실행하면 `submission/code.ipynb`
   가 최신본으로 갱신된다 (실행 중인 노트북 파일은 저장 전까지 디스크에 없다).

> `RETRAIN = True` 로 두면 같은 시드·epoch 으로 train 전체를 다시 학습해 `model.pth`
> 를 새로 쓴다. 규정상 "제출 코드로 재현" 을 문자 그대로 보이고 싶을 때 쓴다.
> validation 은 어느 경로에서도 학습에 쓰이지 않는다.
"""

KNOB_TMPL = '''
# ===== 이 노트북이 재현하는 설정 =======================================
from pathlib import Path

CONFIG = {config}

SOURCE_DIR = Path(r"{source}")      # SWEEP 이 만든 후보 폴더 (model.pth 가 여기 있다)
MODEL_PATH = SOURCE_DIR / "model.pth"
SEEDS = {seeds}
EPOCHS = {epochs}
T_MAX = {t_max}
IMAGE_SIZE = 128

RETRAIN = False        # True 로 두면 model.pth 를 무시하고 처음부터 다시 학습한다
SUBMISSION_DIR = Path("submission")
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
'''

SETUP_SRC = """
set_log(Path("work") / "final_notebook.log")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")
if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True
    log(f"PyTorch {torch.__version__} | GPU {torch.cuda.get_device_name(0)}")
else:
    log(f"PyTorch {torch.__version__} | CPU")

DATA_ROOT = find_data_root()
CACHE_ROOT = Path("work/cache")
log(f"데이터: {DATA_ROOT.resolve()}")

TABLES = load_tables(DATA_ROOT)
IMAGE_ARRAYS, INDEX_MATRICES = {}, {}
for split in SPLITS:
    array, index = prepare_image_memmap(split, TABLES[split]["inputs"], DATA_ROOT,
                                        CACHE_ROOT, IMAGE_SIZE)
    IMAGE_ARRAYS[split] = array
    INDEX_MATRICES[split] = image_index_matrix(TABLES[split]["inputs"], index)

DISK_Y, DISK_X, DISK_R, _ = detect_disk(IMAGE_ARRAYS["train"], IMAGE_SIZE)
log(f"원반: center=({DISK_Y:.1f}, {DISK_X:.1f}) radius={DISK_R:.1f}px")

STORES = {split: extract_ch_store(split, IMAGE_ARRAYS[split], CH_SPEC,
                                  (DISK_Y, DISK_X, DISK_R), CACHE_ROOT, IMAGE_SIZE)
          for split in SPLITS}

MEAN_SPEED = float(TABLES["train"]["wind"].mean())
ALL_TRAIN_ROWS = np.arange(len(TABLES["train"]["inputs"]))
TRAIN_CHAINS = reconstruct_frame_chains(TABLES["train"]["inputs"])
_, TRAIN_T_START = window_positions(TABLES["train"]["inputs"], TRAIN_CHAINS)
VAL_TARGETS = TABLES["validation"]["targets"].astype(np.float64)
VAL_PERSISTENCE = np.repeat(TABLES["validation"]["wind"][:, -1:], 12, axis=1).astype(np.float64)

PACKS = {split: build_pack(CONFIG, split, TABLES, STORES, INDEX_MATRICES, DEVICE, MEAN_SPEED)
         for split in SPLITS}
log(f"피처 폭 — CH {PACKS['train']['ch_dim']} / 탄도 {PACKS['train']['bal_dim']}")
"""

INFER_SRC = """
# ===== 모델 확보: model.pth 로드 또는 재학습 ============================
if RETRAIN or not MODEL_PATH.exists():
    log(f"재학습 — 시드 {SEEDS} x epoch {EPOCHS} (T_max={T_MAX})")
    STATES, STATS = [], None
    for seed in SEEDS:
        started = time.perf_counter()
        model, STATS, _ = run_training(CONFIG, PACKS["train"], ALL_TRAIN_ROWS, None, seed,
                                       EPOCHS, t_start=TRAIN_T_START, record_curve=False,
                                       t_max=T_MAX)
        STATES.append({k: v.detach().cpu() for k, v in model.state_dict().items()})
        log(f"  seed {seed} 완료 {hms(time.perf_counter() - started)}")
        del model
        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
    PAYLOAD = {"ensemble": STATES, "config": CONFIG, "stats": stats_to_cpu(STATS),
               "seeds": list(SEEDS), "epochs": EPOCHS, "t_max": T_MAX, "engine": "sweep1"}
    torch.save(PAYLOAD, SUBMISSION_DIR / "model.pth")
    MODEL_PATH = SUBMISSION_DIR / "model.pth"
    log(f"저장: {MODEL_PATH}")
else:
    PAYLOAD = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    assert PAYLOAD["config"] == CONFIG, "model.pth 의 설정이 이 노트북의 CONFIG 와 다릅니다"
    STATES, STATS = PAYLOAD["ensemble"], stats_from_cpu(PAYLOAD["stats"], DEVICE)
    log(f"로드: {MODEL_PATH} (시드 {len(STATES)}개)")

# ===== 추론 — 시드별 예측을 균등 평균 ==================================
val_predictions, test_predictions = [], []
for state in STATES:
    model = SweepNet(CONFIG, PACKS["train"]["ch_dim"], PACKS["train"]["bal_dim"]).to(DEVICE)
    model.load_state_dict({k: v.to(DEVICE) for k, v in state.items()})
    val_predictions.append(predict_rows(CONFIG, model, PACKS["validation"], STATS,
                                        np.arange(PACKS["validation"]["n"])))
    test_predictions.append(predict_rows(CONFIG, model, PACKS["test"], STATS,
                                         np.arange(PACKS["test"]["n"])))
    del model

VAL_PREDICTION = np.mean(val_predictions, axis=0)
TEST_PREDICTION = np.mean(test_predictions, axis=0)
val_score, val_per_horizon = official_rmse(VAL_TARGETS, VAL_PREDICTION)
persistence_score = official_rmse(VAL_TARGETS, VAL_PERSISTENCE)[0]

log("")
log(f"official validation RMSE (시드 {len(STATES)}개 앙상블) : {val_score:8.3f} km/s")
log(f"persistence 기준선                                   : {persistence_score:8.3f} km/s")
log(f"[참고] P1 68.408 / P3 64.203 (Public 58.8028)")
log("horizon별 RMSE: " + " ".join(f"{v:.1f}" for v in val_per_horizon))
if val_score >= persistence_score:
    log(">>> 경고: persistence 미달. 제출하지 마세요.")
"""

SUBMIT_SRC = '''
# ===== 제출물 3종 =======================================================
submission = pd.DataFrame(TEST_PREDICTION, columns=TARGET_COLUMNS)
submission.insert(0, "sample_id", TABLES["test"]["inputs"].sample_id.to_numpy())
assert np.isfinite(TEST_PREDICTION).all()
submission.to_csv(SUBMISSION_DIR / "submission.csv", index=False)

if MODEL_PATH.resolve() != (SUBMISSION_DIR / "model.pth").resolve():
    shutil.copyfile(MODEL_PATH, SUBMISSION_DIR / "model.pth")

# SWEEP 이 만든 원본 예측과 같은지 확인한다 (재현성 점검)
reference = SOURCE_DIR / "submission.csv"
if reference.exists():
    original = pd.read_csv(reference)
    same_ids = original.sample_id.tolist() == submission.sample_id.tolist()
    gap = float(np.abs(original[TARGET_COLUMNS].to_numpy()
                       - submission[TARGET_COLUMNS].to_numpy()).max()) if same_ids else float("nan")
    log(f"SWEEP 산출물과의 최대 차이: {gap:.6f} km/s  (0 에 가까우면 완전 재현)")

# 실행 중인 노트북 파일을 제출 폴더로 복사한다. **저장(Ctrl+S) 후** 실행할 것.
for candidate in ("code_final.ipynb", "code.ipynb"):
    if Path(candidate).exists():
        shutil.copyfile(candidate, SUBMISSION_DIR / "code.ipynb")
        log(f"code.ipynb 복사: {candidate} -> {SUBMISSION_DIR / 'code.ipynb'}")
        break
else:
    log("!! 이 노트북 파일을 찾지 못했습니다. 저장(Ctrl+S) 후 이 셀을 다시 실행하세요.")

print("=== 제출 점검 ===")
ok = True
for name in ("code.ipynb", "model.pth", "submission.csv"):
    path = SUBMISSION_DIR / name
    if path.exists():
        print(f"  [O] {name:16s} {path.stat().st_size / 1024 ** 2:8.2f} MiB")
    else:
        print(f"  [X] {name:16s} 없음")
        ok = False

check = pd.read_csv(SUBMISSION_DIR / "submission.csv")
test_ids = TABLES["test_ids"]
print(f"  행 수      : {len(check):,} (기대 {len(test_ids):,})",
      "OK" if len(check) == len(test_ids) else "<-- 불일치")
print(f"  컬럼       : {check.columns.tolist() == ['sample_id'] + TARGET_COLUMNS}")
print(f"  결측       : {int(check[TARGET_COLUMNS].isna().sum().sum())}")
print(f"  sample_id  : 유일={check.sample_id.is_unique}, "
      f"test_ids 일치={sorted(check.sample_id) == sorted(test_ids.sample_id)}")
print(f"  값 범위    : [{check[TARGET_COLUMNS].to_numpy().min():.1f}, "
      f"{check[TARGET_COLUMNS].to_numpy().max():.1f}] km/s")
print(f"  val RMSE   : {val_score:.3f} (persistence {persistence_score:.3f})")
print("\\n제출 준비 완료" if ok else "\\n빠진 파일이 있습니다")
'''


def build(config_path, output):
    payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
    config = payload["config"]
    source = Path(config_path).parent
    seeds = payload.get("seeds", [777])
    epochs = int(payload.get("epochs", 60))
    t_max = payload.get("t_max") or config.get("epochs", epochs)

    header = TITLE_MD.format(
        label=payload.get("label", "-"),
        cv=f"{payload.get('cv', float('nan')):.3f}" if payload.get("cv") else "-",
        val=f"{payload.get('val_official_rmse', float('nan')):.3f}",
        seeds=seeds, epochs=epochs, t_max=t_max, source=source.as_posix())
    # json.dumps 는 null/true/false 를 쓴다 -> 파이썬 리터럴로 찍는다
    knobs = KNOB_TMPL.format(
        config=pprint.pformat(config, width=86, sort_dicts=True),
        source=str(source), seeds=list(seeds), epochs=epochs, t_max=t_max)

    cells = [
        markdown_cell(header),
        markdown_cell("## 0. 설정"),
        code_cell(knobs),
        markdown_cell("## 1. 엔진 — SWEEP 과 같은 코드"),
        code_cell(ENGINE_SRC),
        markdown_cell("## 2. 데이터 · 코로나홀 피처"),
        code_cell(SETUP_SRC),
        markdown_cell("## 3. 모델 · 추론"),
        code_cell(INFER_SRC),
        markdown_cell("## 4. 제출물 생성 · 점검"),
        code_cell(SUBMIT_SRC),
    ]
    notebook = {"cells": cells,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                            "name": "python3"},
                             "language_info": {"name": "python", "version": "3.11"}},
                "nbformat": 4, "nbformat_minor": 5}
    try:
        import nbformat
        notebook = nbformat.from_dict(notebook)
        _, notebook = nbformat.validator.normalize(notebook)
    except ImportError:
        pass
    Path(output).write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
                            encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path, help="work/sweep/final/<폴더>/config.json")
    parser.add_argument("--out", type=Path, default=Path("code_final.ipynb"))
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    build(args.config, args.out)
    print(f"wrote {args.out}  <- {args.config}")
    print("노트북을 열어 위에서부터 실행하면 submission/ 에 3종이 채워집니다.")


if __name__ == "__main__":
    main()
