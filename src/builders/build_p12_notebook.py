"""Build the self-contained P12 ensemble notebook.

P12 reproduces the three already-submitted, complementary members inside one
notebook, then averages their predictions.  The source cells are copied into
isolated namespaces so their globals and model classes cannot overwrite one
another.  This file is development tooling; the generated code_p12.ipynb is
the submission artifact.
"""

from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf


HERE = Path(__file__).resolve().parent
TARGET = HERE / "code_p12.ipynb"
SOURCES = {
    "P3": HERE / "Trial" / "P3" / "New" / "code_p3_fix.ipynb",
    "P9": HERE / "Trial" / "P9" / "code_p9.ipynb",
    "N1": HERE / "Trial" / "N1" / "code_N1.ipynb",
}


def source_cells(notebook: Path, indices: list[int]) -> str:
    """Concatenate selected code cells verbatim, with an origin comment."""
    payload = json.loads(notebook.read_text(encoding="utf-8"))
    cells = payload["cells"]
    pieces: list[str] = []
    for index in indices:
        cell = cells[index]
        assert cell["cell_type"] == "code", (notebook, index)
        pieces.append(f"# ===== copied from {notebook.name}, cell {index} =====\n")
        pieces.append("".join(cell["source"]))
        pieces.append("\n")
    return "\n".join(pieces)


def exec_cell(namespace: str, script: str) -> str:
    """Use repr rather than a hand-written triple quoted literal safely."""
    return f"{namespace} = {{}}\nexec({script!r}, {namespace})"


P3_SCRIPT = source_cells(SOURCES["P3"], [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22])
P9_BASE_SCRIPT = source_cells(SOURCES["P9"], [2, 4, 6, 8, 10, 12, 14, 16])
N1_SCRIPT = source_cells(SOURCES["N1"], [1, 3, 5, 6, 7, 10, 12, 14, 16])


P9_TRAIN = r'''
# Published P9 member = adopted P7/F3 configuration, trained for its selected
# one epoch.  Do not switch this to G3: P9's own paired CV chose F3.
P9_SETTINGS = dict(
    CH_GRID=(6, 3), FOLD_LATITUDE=True,
    USE_LEVELS=("dark0.45", "bright"),
    TRANSIT_SPEEDS=(315.0, 345.0, 385.0, 435.0, 500.0, 600.0),
    BALLISTIC_SOURCE="window", BALLISTIC_LAT="profile", BALLISTIC_LON="all",
    BALLISTIC_OFFSETS=(-4, -2, 0, 2),
    ADAPTIVE_AREA=False, ADAPTIVE_GATHER=False, USE_TRANSIT_FLAGS=False,
    USE_CH_GATHER=True, USE_BALLISTIC_WINDOW=True, CH_BIDIRECTIONAL=True,
)
P9_FINAL_EPOCHS = 1
configure(**P9_SETTINGS)
FINAL_STATS = fit_stats(np.arange(len(train_inputs)))
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

final_loader = make_batcher(
    train_inputs, train_index_matrix, train_wind, train_wind_valid,
    train_grid, FINAL_STATS, train_targets, shuffle=True, seed=SEED, training=True,
)
model = build_model(FINAL_STATS)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CV_EPOCHS)
scaler = torch.amp.GradScaler(DEVICE.type, enabled=USE_AMP)
for epoch in range(P9_FINAL_EPOCHS):
    model.train()
    for batch in final_loader:
        moved = {k: batch[k].to(DEVICE, non_blocking=PIN_MEMORY)
                 for k in BATCH_KEYS + ("target",)}
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(DEVICE.type, enabled=USE_AMP):
            residual = model(moved["wind_seq"], moved["wind_stats"], moved["ch_seq"],
                             moved["ballistic"], moved["gather_idx"], moved["flags"])
        prediction = residual.float() + moved["last_wind"].unsqueeze(1)
        loss = metric_loss(prediction, moved["target"])
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer); scaler.update()
    scheduler.step()
    print(f"P9/F3 epoch {epoch + 1}/{P9_FINAL_EPOCHS} complete", flush=True)
final_loader.release()

def p9_predict(inputs, index_matrix, wind, valid, grid, targets=None):
    batcher = make_batcher(inputs, index_matrix, wind, valid, grid, FINAL_STATS,
                           targets, shuffle=False)
    prediction, ids = predict_with(model, batcher, FINAL_STATS["clip_low"],
                                   FINAL_STATS["clip_high"])
    batcher.release()
    return prediction, ids

val_prediction, val_ids = p9_predict(
    val_inputs, val_index_matrix, val_wind, val_wind_valid, val_grid, val_targets,
)
test_prediction, test_ids = p9_predict(
    test_inputs, test_index_matrix, test_wind, test_wind_valid, test_grid,
)
assert val_ids == val_inputs.sample_id.tolist()
assert test_ids == test_inputs.sample_id.tolist()
'''


INTRO = r'''# 태양풍 속도 예측 — P12: P3 + P9(F3) + N1 재현 앙상블

`RUN_P11.md`의 사후분석을 반영한 제출 노트북입니다.

- P11의 실패 원인은 평균 자체가 아니라, P3 대체 멤버를 1–2 epoch만 학습해 P3 품질을 재현하지 못한 점입니다.
- P12는 기존 제출본 세 개를 **각각의 원래 학습 레시피**로 다시 학습합니다: P3의 validation early stopping, P9의 채택 F3 1 epoch, N1의 validation best checkpoint.
- 사전에 고정한 동일 가중치 평균만 사용합니다. validation 점수로 멤버나 가중치를 고르지 않습니다.

예상 public RMSE는 기존 제출본 오차 상관으로 역산한 56.72 (P3 대비 −2.08)입니다. 이는 추정치이며, 새 학습 실행의 난수/하드웨어 차이에 따라 달라질 수 있습니다.
'''


SAVE = r'''# ========================== P12 equal-weight ensemble ==========================
from pathlib import Path
import gc
import numpy as np
import pandas as pd
import torch

P3_VAL = np.asarray(P3["validation_prediction"], dtype=np.float32)
P3_TEST = np.asarray(P3["test_prediction"], dtype=np.float32)
P9_VAL = np.asarray(P9["val_prediction"], dtype=np.float32)
P9_TEST = np.asarray(P9["test_prediction"], dtype=np.float32)
N1_VAL = np.asarray(N1["validation_prediction"], dtype=np.float32)
N1_TEST = np.asarray(N1["test_prediction"], dtype=np.float32)
VAL_TARGETS = np.asarray(P3["val_targets"], dtype=np.float32)
SAMPLE_IDS = P3["test_inputs"].sample_id.tolist()

assert P3_TEST.shape == P9_TEST.shape == N1_TEST.shape == (len(SAMPLE_IDS), 12)
assert P3_VAL.shape == P9_VAL.shape == N1_VAL.shape == VAL_TARGETS.shape
assert P3["test_inputs"].sample_id.tolist() == P9["test_inputs"].sample_id.tolist()
assert P3["test_inputs"].sample_id.tolist() == N1["test_inputs"].sample_id.tolist()

# Fixed before inspecting this run: no fitted blend weights, no validation member selection.
MEMBER_NAMES = ("P3", "P9_F3", "N1")
WEIGHTS = np.array((1 / 3, 1 / 3, 1 / 3), dtype=np.float32)
VAL_STACK = np.stack((P3_VAL, P9_VAL, N1_VAL))
TEST_STACK = np.stack((P3_TEST, P9_TEST, N1_TEST))
FINAL_VAL = np.tensordot(WEIGHTS, VAL_STACK, axes=(0, 0))
FINAL_TEST = np.tensordot(WEIGHTS, TEST_STACK, axes=(0, 0))

def official_rmse(y_true, y_pred):
    per_horizon = np.sqrt(np.mean((y_pred - y_true) ** 2, axis=0))
    return float(per_horizon.mean()), per_horizon

for name, pred in zip(MEMBER_NAMES, VAL_STACK):
    print(f"{name:7s} validation RMSE: {official_rmse(VAL_TARGETS, pred)[0]:.3f}")
print(f"P12    validation RMSE: {official_rmse(VAL_TARGETS, FINAL_VAL)[0]:.3f}")
pairwise = {
    f"{MEMBER_NAMES[i]}__{MEMBER_NAMES[j]}": float(np.sqrt(np.mean((TEST_STACK[i] - TEST_STACK[j]) ** 2)))
    for i in range(3) for j in range(i + 1, 3)
}
print("test prediction disagreement (RMS km/s):", {k: round(v, 2) for k, v in pairwise.items()})

SUBMISSION_DIR = Path("submission")
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
TARGET_COLUMNS = [f"target_{i:02d}" for i in range(12)]
submission = pd.DataFrame(FINAL_TEST, columns=TARGET_COLUMNS)
submission.insert(0, "sample_id", SAMPLE_IDS)
assert submission.shape == (len(SAMPLE_IDS), 13)
assert submission.sample_id.is_unique and np.isfinite(FINAL_TEST).all()
submission.to_csv(SUBMISSION_DIR / "submission.csv", index=False)

def cpu_state(model):
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}

# State dicts make all three trained members inspectable/reloadable from one artifact.
torch.save({
    "code_version": "P12",
    "ensemble": "fixed equal-weight mean; no fitted blend parameters",
    "weights": WEIGHTS,
    "members": {
        "P3": {"state_dict": cpu_state(P3["model"]),
               "best_epoch": int(P3["checkpoint"]["epoch"]),
               "recipe": "P3 original: AdamW + ReduceLROnPlateau + validation early stopping"},
        "P9_F3": {"state_dict": cpu_state(P9["model"]),
                  "epochs": int(P9["P9_FINAL_EPOCHS"]),
                  "settings": P9["P9_SETTINGS"],
                  "recipe": "P9 adopted F3, fixed one epoch"},
        "N1": {"state_dict": cpu_state(N1["model"]),
               "best_epoch": int(N1["checkpoint"]["epoch"]),
               "recipe": "N1 SpeedNet original: Adam + validation best checkpoint"},
    },
    "test_sample_ids": SAMPLE_IDS,
}, SUBMISSION_DIR / "model.pth")
print(f"saved {SUBMISSION_DIR / 'submission.csv'}: {submission.shape}")
print(f"saved {SUBMISSION_DIR / 'model.pth'}: three member state dicts")
'''


CHECK = r'''# ================================ Reproducibility check ================================
# Use the in-notebook preprocessing and the three saved states to regenerate test predictions.
blob = torch.load(SUBMISSION_DIR / "model.pth", map_location="cpu", weights_only=False)
assert tuple(blob["members"]) == MEMBER_NAMES
assert np.allclose(blob["weights"], WEIGHTS)

def reload_predict_p3():
    ns = P3
    model = ns["build_model"]()
    model.load_state_dict(blob["members"]["P3"]["state_dict"])
    model.eval()
    ns["model"] = model
    dataset = ns["SolarWindDataset"](ns["test_image_array"], ns["test_image_index"], ns["test_inputs"],
                                      ns["test_wind"], ns["test_wind_valid"], ns["test_ch"], targets=None)
    loader = ns["make_loader"](dataset, shuffle=False)
    prediction, ids = ns["predict"](loader)
    assert ids == SAMPLE_IDS
    del model, dataset, loader
    return prediction

def reload_predict_p9():
    ns = P9
    ns["configure"](**ns["P9_SETTINGS"])
    model = ns["build_model"](ns["FINAL_STATS"])
    model.load_state_dict(blob["members"]["P9_F3"]["state_dict"])
    ns["model"] = model
    prediction, ids = ns["p9_predict"](
        ns["test_inputs"], ns["test_index_matrix"], ns["test_wind"], ns["test_wind_valid"], ns["test_grid"],
    )
    assert ids == SAMPLE_IDS
    return prediction

def reload_predict_n1():
    ns = N1
    model = ns["SpeedNet"]().to(ns["DEVICE"])
    model.load_state_dict(blob["members"]["N1"]["state_dict"])
    ns["model"] = model
    dataset = ns["SpeedNetDataset"](ns["test_image_array"], ns["test_ch_features"],
                                     ns["test_image_index"], ns["test_inputs"], ns["test_index"],
                                     ns["LOG_MEAN"], ns["LOG_STD"], targets=None)
    loader = ns["make_loader"](dataset, shuffle=False)
    prediction, ids = ns["predict"](loader)
    assert ids == SAMPLE_IDS
    del dataset, loader
    return prediction

reloaded = np.tensordot(WEIGHTS, np.stack((reload_predict_p3(), reload_predict_p9(),
                                            reload_predict_n1())), axes=(0, 0))
saved = pd.read_csv(SUBMISSION_DIR / "submission.csv")[TARGET_COLUMNS].to_numpy(np.float32)
gap = float(np.sqrt(np.mean((reloaded - saved) ** 2)))
print(f"reproduction RMS error: {gap:.6f}")
assert gap < 0.01, "saved model states do not reproduce submission.csv"
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
'''


FINAL_CHECK = r'''# ================================ Submission check ================================
import shutil

EXPECTED_TEST_ROWS = 3868
assert submission.shape == (EXPECTED_TEST_ROWS, 13), submission.shape
assert submission.columns.tolist() == ["sample_id"] + TARGET_COLUMNS
assert np.isfinite(submission[TARGET_COLUMNS].to_numpy()).all()
assert (SUBMISSION_DIR / "model.pth").is_file()

# Save this notebook as code_p12.ipynb before running this cell; it is then copied to the required name.
notebook_here = Path("code_p12.ipynb")
if notebook_here.exists():
    shutil.copyfile(notebook_here, SUBMISSION_DIR / "code.ipynb")
    print("code_p12.ipynb -> submission/code.ipynb")
else:
    print("Save this notebook as code_p12.ipynb, then copy it to submission/code.ipynb.")
for path in (SUBMISSION_DIR / "code.ipynb", SUBMISSION_DIR / "model.pth", SUBMISSION_DIR / "submission.csv"):
    print(path, "OK" if path.exists() else "MISSING")
'''


def build() -> None:
    nb = nbf.v4.new_notebook()
    nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata.language_info = {"name": "python", "version": "3"}
    nb.cells = [
        nbf.v4.new_markdown_cell(INTRO),
        nbf.v4.new_markdown_cell("## 1. P3 — 원래 early-stopping 레시피 재현"),
        nbf.v4.new_code_cell(exec_cell("P3", P3_SCRIPT)),
        nbf.v4.new_markdown_cell("## 2. P9(F3) — 채택된 1 epoch 멤버 재현"),
        nbf.v4.new_code_cell(exec_cell("P9", P9_BASE_SCRIPT + "\n" + P9_TRAIN)),
        nbf.v4.new_markdown_cell("## 3. N1 — 원래 SpeedNet best-checkpoint 레시피 재현"),
        nbf.v4.new_code_cell(exec_cell("N1", N1_SCRIPT)),
        nbf.v4.new_markdown_cell("## 4. 사전 고정한 P3 + P9 + N1 동일 가중치 평균"),
        nbf.v4.new_code_cell(SAVE),
        nbf.v4.new_markdown_cell("## 5. 저장 상태만으로 재추론 검증"),
        nbf.v4.new_code_cell(CHECK),
        nbf.v4.new_markdown_cell("## 6. 제출 점검"),
        nbf.v4.new_code_cell(FINAL_CHECK),
    ]
    nbf.write(nb, TARGET)
    print(f"created {TARGET} ({len(nb.cells)} cells)")


if __name__ == "__main__":
    build()
