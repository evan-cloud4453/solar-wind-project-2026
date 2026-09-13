# src — 노트북 바깥의 도구들

대회 제출물은 **`code.ipynb` 하나만으로 실행 가능해야 한다**는 규정이 있어서, 제출 노트북은
전부 자기 완결이다. 여기 있는 것은 그 바깥에서 도는 오프라인 도구다.

## `extract/` — 물리 피처 1회 추출

10,139장의 원본 영상에서 피처를 뽑아 `work/cache/*.npz` 로 굽는다. 40~60분 걸리므로
학습과 분리했다. 캐시가 없으면 해당 토글이 자동으로 꺼지고 노트북은 그대로 끝까지 돈다.

| 파일 | 뽑는 것 |
|---|---|
| `p11_extract.py` | 코로나홀 면적 · 경계거리 θ_b · 형태 스칼라. 백엔드 `cv2` / `scipy` / `numpy` 3종 지원 |
| `p14_extract.py` | 위에 더해 플레어 · CH 형태 · 강도 깊이. `p11_extract` 를 import 한다 |

```bash
python src/extract/p11_extract.py --limit 30 --workers 2   # 스모크 테스트, 1분
python src/extract/p14_extract.py --workers 4              # 전체
```

> **워커를 4개 넘기지 말 것.** 다중 프로세스로 대회 VM 이 내려간 전례가 있어 운영진 공지가 있었다.

## `analysis/` — 측정 도구

| 파일 | 하는 일 |
|---|---|
| `eda_missing_and_order.py` | **데이터 무결성 감사.** 결측·sentinel·런 길이 분포·인덱스 방향·이미지↔wind 정렬을 전 split 에서 검사한다. 출력은 `print` 요약뿐이라 원본을 반출하지 않는다 |
| `build_timeline.py` | 파일명 사슬로 관측 시간축을 복원해 `timeline.parquet` / `windows.parquet` 생성 |
| `chain_cv.py` | 사슬을 통째로 폴드에 배정하는 grouped CV. 윈도우 중첩 누수가 구조적으로 0이 된다 |
| `make_portfolio_figures.py` · `make_portfolio_figures_2.py` | 최상위 README 의 그림 15장 생성 (대회 데이터 불필요) |

`eda_missing_and_order.py` 가 닫은 두 가설과 그 방법은
[`../docs/code-report.md` §5.6](../docs/code-report.md) 에 있다.

## `builders/` — 노트북 생성기

**제출물이 아니다.** P18 · P3-renew · SWEEP · Final1 등은 원본 노트북을 읽어
**필요한 셀만 패치해서** 새 노트북을 만든다.

```python
# build_p18_notebook.py 는 Trial/P3/code_p3.ipynb 를 읽어 세 군데만 고친다
```

이렇게 만든 이유는 하나다 — **"이 버전의 모델은 P3 와 같다"는 주장을 코드로 검증 가능하게
하기 위해서**다. 나머지 셀이 원본 객체 그대로이므로, 차이가 정말 세 군데뿐인지
`python -c` 한 줄로 확인할 수 있다.

| 생성기 | 만드는 노트북 | 원본 |
|---|---|---|
| `build_p11_notebook.py` | P11 | P9 |
| `build_p12_notebook.py` | P12 (3멤버 앙상블) | P3-fix · P9 · N1 을 격리된 네임스페이스로 합침 |
| `build_p14/15/16/17_notebook.py` | P14 ~ P17 | P3 계열 |
| `build_p18_notebook.py` | P18 | P3 (**세 군데만** 패치) |
| `build_p3_renew_notebook.py` | P3-renew | P3 (**여섯 군데만** 패치) |
| `build_sweep_notebook.py` | SWEEP 탐색기 | 엔진 소스가 이 파일 안에 있다 |
| `build_final_notebook.py` | SWEEP 결과로부터 제출 노트북 | `build_sweep_notebook` 를 import |
| `build_final1_notebook.py` | Final1 단순화 노선 | — |
| `build_k1_notebook.py` · `build_ch_check_notebook.py` | 보조 실험 · CH 마스크 진단 | — |

> 노트북을 손으로 고친 뒤에는 해당 생성기를 다시 돌리지 말 것 — **덮어쓴다.**
