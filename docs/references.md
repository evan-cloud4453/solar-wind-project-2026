# 참고 문헌 — 무엇을 가져오고 무엇을 왜 버렸나

> **논문 PDF 원문은 저장소에 포함하지 않았다.** 유료 저널 저작물이기 때문이다.
> 아래는 서지사항과, 이 프로젝트가 각 논문에서 **실제로 무엇을 구현했고 무엇을 왜 배제했는지**의 매핑이다.

---

## 1. Son et al. (2023)

*Three-day Forecasting of Solar Wind Speed Using SDO/AIA Extreme-ultraviolet Images by a Deep-learning Model*

**이 대회의 원본 과제다.** 입력(SDO/AIA EUV 영상)·출력(3일 예보)·기간 구성이 사실상 동일하다.

| 가져온 것 | 어디에 |
|---|---|
| horizon 별 RMSE 벤치마크 (6 h 37.4 → 72 h 68.2 km/s) | P7 의 표적 설정 기준 |

**이 벤치마크가 전략을 바꿨다.** 우리 모델은 6 h 에서 28.3 으로 **9.1 앞서고**,
72 h 에서 74.4 로 **6.2 뒤졌다.** 지표가 12개 horizon 평균이므로 승부처는 장기다 —
36 h 이상 7개를 각 4 km/s 줄이면 전체가 2.33 내려간다. 6 h 를 1 깎아 봐야 0.08 이다.

---

## 2. Collin et al. (2025) — *Space Weather*

*Forecasting High-Speed Solar Wind Streams From Solar Images*

**P3~P7 의 뼈대.** 코로나홀 면적을 격자 binning 한 소수의 물리 피처 + 다항 회귀로
최신 DNN 과 MHD 시뮬레이션을 모두 이겼다 (timeline RMSE 68.1).

| 가져온 것 | 어디에 |
|---|---|
| CH 면적 격자 binning | P3 `CH_GRID` |
| 중앙자오선 CH 면적이 최중요라는 결론 | P3 탄도 정렬의 대상 |
| 곡률(림) 보정이 불필요하다는 지적 | P4 의 실패를 사후 확인 → `LIMB_CORRECT=False` 확정 |
| 위도 4~6 × 경도 3 격자 권장 | P7 에서 축 교정 — **우리 격자는 축이 반대였다** |

| 못 가져온 것 | 이유 |
|---|---|
| 27일 전 태양풍 | 입력이 120 h 뿐 |
| 흑점수 `N_SS`, 지구 헬리오위도 α | **외부 데이터 = 실격** |
| **Box-Cox 분포 변환** | 논문 자체가 명시한다 — HSS peak 향상은 **timeline 성능 저하의 대가**(68.1 → 75.1). 우리 지표가 timeline RMSE 이므로 **쓰면 손해** |

---

## 3. Abraham-Alowonle et al. (2026) — *Earth and Space Science*

*Deep Learning-Based Prediction of High-Speed Solar Wind Streams: Spatio-Temporal Dependencies in Coronal Hole Dynamics* (article no. e2025EA004523)

**P10 과 N1 의 출처.** 아키텍처·알고리즘·전처리만 가져와 대회 출력(12 horizon)에 맞췄다.

| 가져온 것 | 어디에 |
|---|---|
| Stonyhurst 일면좌표 투영 | P10 `stonyhurst_sampler` (±60°) · N1 (±75°) |
| 로그 + 데이터셋 고정 상수 표준화 | `LOG_MEAN` / `LOG_STD` 를 train 에서 실측해 대체 |
| 동적 임계 CH 이진맵 | P10 · N1 |
| SpeedNet — Conv블록 4개 → LSTM(100) → FFNN(200) | P10 (32/64/128/128) · N1 (16/32/64/128, 논문 표기 그대로) |
| Table 4 하이퍼파라미터 | `PAPER_FAITHFUL` 스위치로 재현 가능 |
| 중앙자오선 ±10° (IG attribution 결론) | P10 `MERIDIAN_HALF_WIDTH_DEG` — **이 데이터에서는 역효과였다** |
| Threat Score · Grad-CAM | 진단 지표 |

| 못 가져온 것 | 이유 |
|---|---|
| 171 Å / 304 Å | 대회 데이터에 없음 |
| ICME 날짜 제거 (Richardson & Cane 목록) | **외부 데이터 = 실격** |
| 27일 persistence 기준선 | 입력이 120 h 뿐 |
| 태양주기 위상별 개별 모델 | 대회 분할이 위상이 아니라 **연중 월** 기준 |
| B0 (자전축 기울기 ±7.25°) 보정 | 천체력 = **외부 데이터**. `B0 = 0` 고정 |

> 논문에서 이긴 쪽은 SpeedNet-EUV 가 아니라 **SpeedNet-BM** 이다
> (RMSE 71.4 vs 75.9 · r 0.68 vs 0.65). 두 변형을 모두 구현해 나란히 측정했고,
> **이 데이터에서는 둘 다 맵 없는 격자 피처 대조군에 졌다.**

---

## 4. Upendran et al. (2020) — *Space Weather*

*Solar Wind Prediction Using Deep Learning*

AIA EUV → 태양풍 속도 LSTM, corr 0.55. **211 Å 이 더 좋은 채널**이라는 결과를
채널 선택의 근거로 썼다. 우리 EDA 도 같은 방향이었다 — `211 ≈ (193 AND 211) ≫ 193`.

---

## 5. Milošić et al. (2023)

CATCH 카탈로그 통계에 기반한 **동적 임계 코로나홀 검출**. P10 · N1 의 임계 결정에 썼다.

---

## 6. 비교 대상으로만 참조

| 논문 | 성능 | 비고 |
|---|---|---|
| **Raju & Das (2021)** — *CNN-Based Deep Learning Model for Solar Wind Forecasting* (arXiv:2108.09114) | RMSE 76.3 | CNN |
| **Aneesh et al. (2024)** | 6시간 단위 RMSE 33.7 | CNN-LSTM |
| **Ahn et al. (2025)** — *ApJ* 987, 179 | — | 태양풍 예측 관련 국내 후속 연구 |

---

## 규정상 처음부터 배제한 것

이 대회는 **외부 데이터와 pretrained weight 를 전면 금지**한다
([`competition/final-round-rules.md`](competition/final-round-rules.md) §5).
따라서 위 논문들이 쓴 다음 자원은 처음부터 선택지가 아니었다.

- 흑점수 · 천체력(B0 · 지구 헬리오위도) · ICME 목록 · 27일 전 태양풍 관측
- ImageNet 등으로 사전학습된 백본 가중치 — 아키텍처 구조 재사용만 허용된다
- test 샘플 간 시계열 재구성 — Data Leakage 로 실격 사유

---

<sub>
서지 정보는 프로젝트 진행 중 참조한 원문 기준이다. 정확한 권·호·DOI 는 각 저널에서 확인할 것.
</sub>
