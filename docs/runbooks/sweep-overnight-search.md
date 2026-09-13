# SWEEP — 6시간 무인 탐색, 아침에 순위표

밤새 돌려 놓고 자면, 아침에 **어떤 조합이 가장 좋았는지**와 **바로 제출할 수 있는
후보 폴더**가 나와 있게 하는 노트북이다. 새 모델이 아니라 **P3 파이프라인의 탐색기**다.

| | P3 | SWEEP |
|---|---|---|
| 설정 | 노트북에 상수로 박힌 하나 | **27개 노브 x 후보값** 을 예산 안에서 훑는다 |
| 1회 학습 | DataLoader + 워커 4개 | **입력 전체를 GPU 상주** — DataLoader 없음 |
| 판정 | official validation 최저점 | **사슬 CV** + **official val** 을 나란히 |
| 산출물 | submission 1개 | 순위표 · 노브별 효과표 · 제출 후보 폴더 여러 개 |

---

## 0. 올릴 파일

대회 서버 작업 디렉터리(=`public_dataset/` 이 보이는 곳)에 둔다.

| 파일 | 역할 | 제출물 |
|---|---|---|
| `code_sweep.ipynb` | 밤새 도는 탐색기 | 아니오 |
| `build_final_notebook.py` | 아침에 제출 노트북을 뽑는 생성기 | 아니오 |
| `build_sweep_notebook.py` | 위 둘의 원본 생성기 (엔진 소스가 여기 있다) | 아니오 |

`build_final_notebook.py` 는 `build_sweep_notebook.py` 를 import 하므로 **둘을 같은
폴더에** 둔다. 이미지 캐시(`work/cache/128px/*.npy`)는 **P3 것을 그대로 재사용**한다.

---

## 1. 재우기 전에 (5분)

```bash
python build_sweep_notebook.py
```

`code_sweep.ipynb` 를 열고 **셀 0(노브)** 만 확인한 뒤 위에서부터 전부 실행한다.

```python
BUDGET_HOURS = 6.0     # 이 시간이 지나면 어느 단계든 멈추고 보고서를 쓴다
RESUME = True          # 중간에 죽어도 다시 돌리면 이어서 한다
```

**첫 5분에 이 세 줄이 나오는지만 보고 자면 된다.**

```
PyTorch 2.5.1+cu124 | GPU NVIDIA A100-SXM4-40GB
[기준] P3 사슬 CV = 6x.xxx +- x.xxx km/s (epoch xx, 15회, xxs)
[예산] A 에 3h2xm (약 1xx개 설정) · B 에 1h5xm (약 2x개 설정)
```

세 번째 줄이 **예산이 실제로 몇 개의 설정을 훑을지**를 말해 준다.
`약 20개 설정` 처럼 작게 나오면 GPU 가 느린 것이니 노브를 줄인다 (§5).

터미널을 쓸 수 있으면 브라우저와 무관하게 도는 이쪽이 더 안전하다.

```bash
nohup jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 code_sweep.ipynb > sweep.out 2>&1 &
```

어느 쪽이든 **모든 로그가 `work/sweep/sweep.log` 에 남는다.** 브라우저가 끊겨도,
노트북 출력이 날아가도 결과는 파일로 남는다.

---

## 2. 밤새 무슨 일이 일어나나

| 단계 | 예산 | 하는 일 |
|---|---|---|
| 0. 준비 | ~5분 | CH 피처를 **임계 5 x 격자 8 x 밝기레벨** 한 번에 추출 → `work/cache/sweepch_*.npz` |
| A-1 사다리 | | 기준(P3)에서 **한 번에 한 노브만** 바꿔 66개 설정. "무엇이 몇 km/s 짜리인가" 표가 여기서 나온다 |
| A-1b 탐욕 | | 사다리에서 이긴 값을 전부 켠 조합. 이기면 탐색 중심이 하나 늘고, 지면 "노브끼리 간섭한다"는 결론이 남는다 |
| A-2 조합 | 합쳐서 62% | 기준·탐욕 조합 주변을 무작위로 흔든다 (평균 12개 노브씩) |
| B 정밀 | 38% | 상위 후보를 **폴드 5 x 시드 2** 로 다시 재고, **train 전체 학습 → official val** 까지 읽는다. 기준과 대응비교(paired) |
| C 제출물 | 예약분 | 상위 4개(+기준)를 **train 전체 x 시드 5 앙상블** 로 만들어 `submission.csv` 생성. 시간이 남으면 1위의 시드를 12개까지 늘린다 |
| D 보고서 | ~1분 | `REPORT.md` · `results.csv` · `leaderboard.png` |

**C 단계 몫은 처음부터 예산에서 떼어 둔다.** 탐색이 길어져도 제출 후보는 반드시 나온다.
설정 하나가 예외를 던져도 그 설정만 건너뛰고 계속 간다(`[!]` 로 로그에 남는다).

---

## 3. 아침에 (10분)

```bash
cat work/sweep/REPORT.md
```

읽는 순서는 이렇다.

1. **1절 제출 후보 표** — `CV` 와 `val(시드앙상블)` 두 열을 같이 본다.
2. **2절 대응비교 표** — `판정` 열이 `**개선**` 인 것만 진짜 후보다.
   `차이없음` 은 dCV 가 2se 안이라는 뜻이고, **그건 이겼다는 뜻이 아니다.**
3. **3절 노브별 효과** — 다음 실험을 어디에 걸지 정할 때 쓴다. 오늘 제출과는 무관.

그리고 고른다.

* **CV·val 이 함께 기준을 이긴 후보가 있으면** 그걸 고른다.
* **없으면 P3 기준의 시드 앙상블 폴더**(라벨이 `P3 기준`)를 고른다.
  시드 앙상블은 탐색으로 얻은 이득이 아니라 **분산 감소**라서, 현재 최고 제출(P3
  단일 모델, Public 58.8028)보다 나빠질 이유가 구조적으로 없다.

제출 노트북을 뽑는다.

```bash
python build_final_notebook.py work/sweep/final/01_ab12cd34/config.json
```

`code_final.ipynb` 를 열어 위에서부터 실행하면 끝이다. 이 노트북은

* `model.pth`(시드별 state_dict 묶음)를 읽어 test 를 추론하고,
* **SWEEP 이 만든 submission.csv 와 최대 차이를 찍어 재현을 확인하고**,
* `submission/` 에 `code.ipynb` · `model.pth` · `submission.csv` 3종을 채운다.

> `submission/code.ipynb` 는 **디스크에 저장된 노트북 파일**을 복사한다.
> 마지막 셀을 실행하기 전에 반드시 **Ctrl+S** 로 저장할 것.

`RETRAIN = True` 로 두면 같은 시드·epoch 으로 처음부터 다시 학습한다.
로컬 점검에서는 이 경로가 원래 예측을 **오차 0.000000 으로** 재현했다.

---

## 4. 이 결과를 어떻게 믿을 것인가

**이 노트북은 1등 하나를 골라 주지 않는다.** [CODE_REPORT §1.5](../code-report.md) 가
실측한 대로 로컬 지표의 분해능이 우리가 구분해야 할 차이보다 크기 때문이다.
val 이 1위로 뽑은 P6 은 Public 4위였고, CV 는 P9↔P10 의 2.56 km/s 차이를 0.013 으로,
**부호까지 반대로** 읽었다. 그래서 설계를 이렇게 했다.

| 위험 | 대응 |
|---|---|
| 설정 200개를 훑으면 1위 CV 는 운이 섞인다 | B 단계에서 **같은 폴드·시드**로 다시 재고 대응비교 표준오차를 찍는다 |
| val 과적합 (P6 의 실패) | val 은 B·C 에서 **읽기만** 한다. 폴드는 전부 train 안에서 자른다 |
| 앙상블 가중치를 val 로 적합 (P6 의 실패) | **균등 평균만** 쓴다. NNLS 없음 |
| 폴드 통계 누수 | 정규화 통계를 **폴드마다 다시** 뽑는다 (P18 이 남겨 둔 근사를 여기서 없앴다) |
| 윈도우 중첩 누수 | 사슬을 통째로 폴드에 배정 — 무작위 행 분할이면 99.8%가 겹친다 |
| 시드 편차 0.78~0.90 km/s | 그보다 작은 차이는 순위로 읽지 않는다. 제출본은 시드 앙상블로 만든다 |

남은 한계도 분명하다.

* **epoch 을 CV 곡선 최저점으로 고른다.** 설정 수만큼의 선택 편향이 남는다. B 가 그
  일부를 걷어내지만 전부는 아니다.
* **CH 추출은 고정 원반**(train 평균 영상)이다. P7 이 측정한 프레임별 반지름 변동
  6.4% 는 반영하지 않았다 — P3 와 같은 조건을 유지하기 위해서다.
* **3D CNN 영상 브랜치는 공간에서 뺐다.** 1회 학습이 100배 비싸져 6시간 예산을
  통째로 먹는다. P2/P6 에서 이미 과적합 주범으로 지목된 브랜치이기도 하다.

---

## 5. GPU 가 느리거나 시간이 다를 때

셀 0 만 고친다.

```python
BUDGET_HOURS = 4.0            # 예산 자체를 줄인다
A_FOLDS, A_SEEDS = 2, 1       # 훑기를 더 싸게 (분해능은 더 나빠진다)
B_TOP_K = 6                   # 정밀 재측정 목표 개수
C_SEEDS = 3                   # 최종 앙상블 시드 수
MAX_RANDOM_CONFIGS = 150      # 사다리만 돌리려면 RUN_RANDOM = False
A_BUDGET_FRACTION = 0.5       # 훑기 대신 정밀 재측정에 더 쓰고 싶을 때 낮춘다
```

중간에 죽었으면 **그냥 다시 위에서부터 실행한다.** `RESUME = True` 면
`results.csv` 에 있는 설정과 이미 만들어진 후보 폴더는 건너뛴다.
CH 추출과 이미지 캐시도 재사용되므로 준비 단계는 1분 안에 끝난다.

---

## 6. 탐색 공간에 무엇이 들어 있나

전부 P4~P18 에서 실제로 시험됐거나 [CODE_REPORT](../code-report.md) 가 "재보자" 라고
남긴 것들이다. 근거 없는 값은 넣지 않았다.

| 묶음 | 노브 | 후보값 | 출처 |
|---|---|---|---|
| 코로나홀 | `ch_threshold` | 0.35 ~ 0.55 | P5 가 "임계가 지배 변수" 라고 확인 |
| | `ch_grid` | (1,1) ~ (6,5) | P7 의 축 교정 지적(위도 세분) · P3-renew 의 면적 1개 |
| | `ch_mode` | grid / area / area_shape / grid_shape | P3-renew 의 모멘트 피처 |
| | `use_bright` | 활성영역 면적 추가 | P7 EDA: bright≥1.6 이 corr +0.128 (반대 부호) |
| 탄도 정렬 | `transit_speeds` | P3 3속도 / 385 단일 / P6 5속도 등 | P7 경험적 385 km/s · P6 5속도 |
| | `ballistic_lon` | central / all | P9 R1 (P10 이 밴드를 좁혔다가 2.56 잃음) |
| | `tau_mode` | fixed / adaptive | P9 R2 샘플별 전달시간 |
| | `clip_flags` | 관측창 이탈 플래그 | P9 R2 |
| 모델 | `gru_hidden/layers/bidir`, `wind_hidden`, `head_width`, `horizon_embed` | | P7 의 양방향 GRU · 용량 축소 계보 |
| 정규화 | `dropout`, `ch_dropout`, `aug_ch_noise`, `weight_decay` | | P3-fix 의 두 줄 변경 |
| 학습 | `lr`, `batch_size`, `epochs`, `loss`, `loss_horizon_weight`, `target`, `ema`, `stride` | | P6 EMA · P7 장기 horizon 표적 · P18 stride 9 |
| 대조군 | CH 브랜치 끔 / 탄도 끔 | | "코로나홀이 실제로 일을 하는가" |

`stride = 9` 는 P18 이 남긴 다음 칸이다. wind 자기상관 1/e 감쇠가 9스텝(2.2일)이라,
켜면 1 epoch 이 거의 중복 없는 표본만 보게 된다.
