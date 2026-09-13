# P15 — P3 단일 모델 물리 피처 실험

P15는 P3의 단일 CH-GRU 모델을 유지한다. 앙상블은 사용하지 않으며, P3의 3×5 CH 격자와 탄도 정렬은 변경하지 않는다.

## 1. 피처 캐시

플레어·형태·강도깊이 토글은 `work/cache/p14a_{train,validation,test}.npz`가 있어야 켜진다.
이 캐시는 `p14_extract.py`가 만들고, `p14_extract.py`는 같은 폴더의 `p11_extract.py`를
import 한다. **두 파일 모두 노트북과 같은 폴더에 있어야 한다.**

터미널을 쓸 수 있으면 이쪽이 빠르다.

```bash
python p14_extract.py --limit 30 --workers 2
```

```bash
nohup python p14_extract.py --workers 4 > extract.log 2>&1 &
```

터미널이 없으면 노트북의 「P14 캐시 준비」 셀이 같은 일을 대신한다. 캐시가 이미 있으면
경로와 크기만 찍고 넘어가고, 없으면 `p14_extract.py`를 자식 프로세스로 돌리며 진행 로그를
셀 출력에 그대로 흘린다. 데이터 경로는 노트북이 찾은 `DATA_ROOT`를 `SW_DATA_ROOT`로
넘기므로 자식이 다시 추측하지 않는다.

캐시가 없는 상태로 진행하면 아래가 찍히고 세 토글이 자동으로 꺼진다. 자전 피처는 기존 P3
격자만으로 만들어지므로 캐시 없이도 비교할 수 있다.

```text
P14 cache unavailable: flare/shape/intensity-depth disabled; rotation is still available.
```

## 2. 실행

[code_p15.ipynb](../../experiments/p15-single-model-features/code_p15.ipynb)를 위에서부터 실행한다. 설정 셀에서 아래 네 토글을 바꾼다.

```python
USE_FLARE = False
USE_CH_SHAPE = False
USE_ROTATION = False
USE_INTENSITY_DEPTH = False
```

마지막 validation 셀에 표시되는 `P15 OFFICIAL_VALIDATION_RMSE`가 비교 점수다. 비교할 때는 `SEED`, `EPOCHS`, early-stopping 설정을 바꾸지 않는다.

권장 비교 순서:

1. 모두 `False`: P3 직접 기준선
2. Shape만 `True`
3. Intensity Depth만 `True`
4. Rotation만 `True`
5. Flare만 `True`
6. 단독으로 개선된 기능들만 조합

토글을 바꾼 뒤에는 설정 셀부터 다시 실행해야 한다. `P15 requested switches`와
`P15 active switches`가 다르게 찍히면 캐시가 없어서 꺼진 것이므로, 위의 캐시 절을 먼저 마친다.
