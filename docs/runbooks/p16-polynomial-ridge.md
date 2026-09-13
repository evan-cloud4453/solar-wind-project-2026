# P16 — P3 피처 + 다항 릿지회귀

P16 은 P3 의 피처는 그대로 두고 학습기만 GRU → 다항 릿지회귀로 바꾼 대조군이다.
3×5 코로나홀 격자, 탄도 정렬, 결측 처리, 공식 지표는 P3 와 같다. `USE_*` 토글은
[P15](p15-single-model-features.md) 와 이름·의미가 같으므로 같은 조합끼리 직접 비교할 수 있다.

제출 후보가 아니라 **피처 선별 도구**로 쓰는 게 맞다. 20 프레임 시계열을 요약통계
몇 개로 접어 버리므로 GRU 가 쓰는 시간 구조를 상당히 버린다.

## 왜 쓰는가

| 결과 | 해석 |
|---|---|
| P16 이 persistence 미달 | 그 피처 조합에 선형·2차 범위의 신호가 없다 |
| P16 은 이기는데 P15 가 못 이김 | 피처는 멀쩡하다. 학습 쪽(용량·정규화·epoch) 문제 |
| 둘 다 이김 | 피처가 실제로 일한다. GRU 로 확장 |

릿지는 정규방정식 한 번이라 조합당 수십 초다. P15 의 60 epoch GRU 를 조합마다
돌리기 전에 여기서 먼저 훑는 게 싸다.

## 1. 피처 캐시

P15 와 같은 `work/cache/p14a_*.npz` 를 쓴다. 이미 만들어 뒀다면 그대로 재사용된다.
없으면 노트북의 「P14 캐시 준비」 셀이 만든다. 자세한 건 [RUN_P15.md](p15-single-model-features.md) 참고.

```bash
python p14_extract.py --workers 4
```

## 2. 실행

[code_p16.ipynb](../../experiments/p16-polynomial-ridge/code_p16.ipynb)를 위에서부터 실행한다. 설정 셀에서 바꾸는 값은 다음과 같다.

```python
USE_FLARE = False
USE_CH_SHAPE = False
USE_ROTATION = False
USE_INTENSITY_DEPTH = False
P16_DEGREE = 2
```

마지막 validation 셀의 `P16 OFFICIAL_VALIDATION_RMSE` 가 비교 점수다.

## 3. 설계행렬 구조

다항 전개를 전 피처에 걸면 항이 폭발한다 (토글 전부 켜면 요약 후 640 열, 2 차만
205,000 항). 그래서 두 층으로 나눈다.

* **core (18열, 다항 전개)** — 바람 통계 9 + CH 총면적 last/mean/slope 3 +
  중앙자오선 last/mean/slope 3 + horizon 별 탄도 3. 상호작용이 물리적으로 말이 되는
  것만 넣었다. 토글과 무관하게 고정이라 조합을 바꿔도 비교축이 흔들리지 않는다.
* **linear (나머지, 1차항만)** — 선택된 CH·물리 피처의 20 프레임 요약
  (`last`, `mean`, `mean4`, `slope`).

`P16_DEGREE = 2` 기준으로 core 는 18 → 189 항이 된다. 3 으로 올리면 1,329 항이라
표본 대비 급격히 넓어지므로 `n/D` 경고를 보고 판단할 것.

horizon 12 개는 각각 별도 회귀를 푼다 (탄도 피처가 horizon 마다 다르다). 목표는
P3 와 같은 잔차 `target − 마지막 관측 풍속`이다.

## 4. 읽을 때 주의

* **alpha 를 validation 으로 고른다.** 따라서 출력되는 validation RMSE 는 낙관적으로
  편향된다. 조합 간 상대 비교에는 문제없지만 절대 성능은 test 로만 판단할 것.
  12 horizon 이 alpha 하나를 공유하게 해서 편향을 그나마 줄여 뒀다.
* **alpha 가 격자 끝에서 골라지면** 경고가 찍힌다. `P16_ALPHAS` 범위를 넓혀야 한다.
* **`n/D < 5` 경고**가 뜨면 `P16_DEGREE` 를 낮추거나 `P16_SUMMARIES` 를 줄인다.
* **분산 0 인 열은 자동으로 버린다.** 플레어처럼 대부분 0 인 피처를 켜면 실제 폭이
  기대보다 작게 찍히는데 정상이다.
* 상위 계수 표(`work/outputs_p16/top_terms.csv`)는 설계행렬이 열별로 표준화되어
  있어 계수 크기를 그대로 비교할 수 있다. 어떤 피처가 일하는지 여기서 먼저 본다.

## 5. 권장 순서

P15 와 같은 순서로 돌려서 두 결과를 나란히 놓는다.

1. 모두 `False`: 기준선
2. `USE_CH_SHAPE=True`
3. `USE_INTENSITY_DEPTH=True`
4. `USE_ROTATION=True`
5. `USE_FLARE=True`
6. 단독으로 개선된 것만 조합

`P16_DEGREE`, `P16_ALPHAS`, `P16_SUMMARIES` 는 조합 간 고정한다.
