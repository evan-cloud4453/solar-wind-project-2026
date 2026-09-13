# P14 실행 안내

P14는 플레어·코로나홀 형태·자전·강도 깊이를 독립적으로 켜고, 같은 학습 조건에서 공식 validation RMSE를 비교하는 노트북이다.

## 1. 추가 피처 캐시 생성

대회 데이터가 보이는 작업 디렉터리에서 먼저 실행한다.

```bash
python p14_extract.py --limit 30 --workers 2
python p14_extract.py --workers 4
```

결과는 `work/cache/p14a_{train,validation,test}.npz`에 저장된다. 이 캐시에는 기존 P11의 면적·경계거리·비율 피처와 P14의 플레어·형태 피처가 함께 들어 있다.

## 2. 토글 조합 평가

[code_p14.ipynb](../../experiments/p14-feature-toggles/code_p14.ipynb)의 P14 설정 셀에서 다음만 변경한다.

```python
USE_FLARE = False
USE_CH_SHAPE = False
USE_ROTATION = False
USE_INTENSITY_DEPTH = False
P14_SEED = 777
P14_EPOCHS = 1
```

설정 셀부터 마지막까지 다시 실행한다. 마지막 셀의 다음 줄이 비교할 값이다.

```text
OFFICIAL_VALIDATION_RMSE = ... km/s
```

`P14_SEED`, `P14_EPOCHS`, 기준 모델(`P14_BASE_CODE`)은 조합 간 동일하게 유지한다. 토글은 한 번에 하나씩 켜서 비교하는 것을 권장한다.

권장 순서:

1. 전부 `False`인 기준선
2. `USE_CH_SHAPE=True`
3. 기준선 + `USE_INTENSITY_DEPTH=True`
4. 기준선 + `USE_ROTATION=True`
5. 기준선 + `USE_FLARE=True`
6. 단독에서 개선된 피처들만 함께 켠 조합

플레어는 지구 방향 CME 여부를 직접 알려 주지 못하므로, 단독 결과가 불안정하거나 악화되면 최종 모델에서 제외한다.
