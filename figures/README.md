# figures

`01`~`08` 은 [`../src/analysis/make_portfolio_figures.py`](../src/analysis/make_portfolio_figures.py) 가
리더보드·제출 기록에서 직접 그린다. `09`~`15` 는 두 번째 스크립트가 **실행 기록이 남은 노트북과
대회 서버 캡처에서 꺼내 온 것**이다 — 즉 실제 실행 산출물이지 재현 그림이 아니다.

두 스크립트 모두 대회 데이터 없이 돈다.
**★ 표시가 최상위 [README](../README.md) 에 실린 10장**이고, 나머지는 여기에만 있는 EDA 자료다.

| | 그림 | 내용 | 출처 |
|:--:|---|---|---|
| ★ | `01-final-leaderboard.png` | 본선 최종 public 리더보드 33팀과 우리 위치 | 리더보드 |
| ★ | `02-submissions-and-inversion.png` | 제출 7회의 Public 점수 · val↔Public 순위 역전 | 제출 기록 |
| ★ | `03-leaderboard-drift.png` | 점수는 고정, 순위는 2위 → 22위 → 31위 | 리더보드 |
| ★ | `04-resolution-gap.png` | 계측 노이즈 바닥 ±3 km/s 가 33팀 중 30팀을 덮는다 | 리더보드 + 시드 편차 실측 |
| ★ | `05-horizon-error.png` | horizon 별 RMSE 와 persistence 대비 개선폭 | P5-T1 실행 출력 |
| ★ | `06-literature-comparison.png` | Son et al. 2023 대비 단기 우세 · 장기 열세 | 문헌 + P6 val |
| ★ | `07-p3-pipeline.png` | 최고 제출 P3 의 전체 구조 | 설계도 |
| ★ | `08-ideas-kept-and-dropped.png` | 살아남은 아이디어 13개 · 접은 아이디어 10개 | 계보 |
| | `09-eda-horizon-baselines.png` | persistence · climatology · 모델의 horizon 별 RMSE | `eda_p7_executed.ipynb` |
| | `10-eda-disk-geometry.png` | 프레임별 원반 반지름 · 중심 · 밝기 변동 (반지름 std 6.4 %) | `eda_p7_executed.ipynb` |
| ★ | `11-eda-lag-correlation.png` | CH 면적↔타깃 지연 상관, horizon×윈도우 상관 히트맵 | `eda_p7_executed.ipynb` |
| | `12-eda-longitude-latitude.png` | 경도 밴드별 지연 · 위도 밴드별 신호 세기 — 격자 축 교정의 근거 | `eda_p7_executed.ipynb` |
| | `13-eda-ch-area-timeseries.png` | CH 면적 시계열과 Hampel 이상치 (오염 1.47 %) | `eda_p7_executed.ipynb` |
| | `14-paired-cv-comparison.png` | 평활 CV 곡선 · horizon 별 대응 차이 | `code_p9.ipynb` |
| ★ | `15-ch-mask-limb-comparison.png` | 임계·림보정 두 설정에서의 코로나홀 마스크 | P5 서버 실행 캡처 |

```bash
python src/analysis/make_portfolio_figures.py
python src/analysis/make_portfolio_figures_2.py
```
