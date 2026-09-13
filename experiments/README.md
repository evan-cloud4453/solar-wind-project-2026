# experiments — 버전별 인덱스

한 폴더 = 한 버전이다. 실행 기록이 남아 있는 버전은 노트북 안에 출력이 그대로 들어 있고,
제출한 버전은 `model.pth` 와 `submission.csv` 가 함께 있다.

계보의 서사와 각 버전의 설계 근거는 [`../docs/code-report.md`](../docs/code-report.md) 에 있다.

## 제출한 버전 (7회)

| 폴더 | 한 줄 요약 | val | **Public** | 산출물 |
|---|---|---:|---:|---|
| [`p01-fundamentals`](p01-fundamentals/) | 잔차 예측 · wind GRU · 지표 정렬 · 128 px | 68.408 | 62.3393 | 노트북 |
| [`p03-coronal-hole-grid`](p03-coronal-hole-grid/) | **CH 3×5 격자 + 탄도 정렬, CNN 끔** | 64.203 | **58.8028** ★ | 노트북 · 가중치 · CSV |
| [`p06-nnls-ensemble`](p06-nnls-ensemble/) | 512 px 추출 · 다중 임계 · 7모델 NNLS · EMA | **63.357** | 60.1955 | trial1 / trial2 |
| [`p09-paired-cv`](p09-paired-cv/) | 시드×폴드 대응비교 계측기 + 적응형 τ | 66.978 | 59.2246 | 노트북 · 가중치 · CSV |
| [`n1-baseline-speednet`](n1-baseline-speednet/) | baseline 에 SpeedNet 만 최소 이식 (별도 갈래) | — | 59.7313 | 노트북 · 가중치 · CSV |
| [`p10-speednet-stonyhurst`](p10-speednet-stonyhurst/) | Stonyhurst 투영 + SpeedNet CNN-LSTM | 66.772 | 61.7853 | 노트북 · 가중치 · CSV |
| [`p11-physics-ensemble`](p11-physics-ensemble/) | 무적합 균등가중 앙상블 (S1 단계) | 65.622 | 59.7332 | 노트북 |

★ 최고 제출. **val 최저(P6)와 Public 최저(P3)가 다른 모델이다.**

## 제출하지 않은 버전

| 폴더 | 왜 만들었나 | 결과 |
|---|---|---|
| [`00-baseline`](00-baseline/) | 주최측 제공 3D CNN + Inception + LSTM, 64 px | 이후 전 버전의 골격 |
| [`p02-regularization`](p02-regularization/) | P1 의 best epoch 2 진단 → 증강·용량축소·정규화 강화 | val 65.663. 여전히 과적합 → CNN 포기 결정 |
| [`p03-fix-regularization`](p03-fix-regularization/) | P3 에서 CH 곱셈 노이즈 제거 + CH 전용 dropout 분리 | 두 줄 차이인데 test 예측이 16.6 RMS 다르다 |
| [`p04-limb-correction`](p04-limb-correction/) | 림 밝아짐 평탄화 + 12 % 백분위 임계 | **val 67.988 — 3.8 퇴보.** 림 보정은 이후 영구 폐기 |
| [`p05-threshold-sweep`](p05-threshold-sweep/) | P4 의 실패를 "림 보정" / "임계값" 두 조각으로 분리 | **임계가 지배 변수** 확인 (상관 0.107~0.339) |
| [`p07-cv-instrument`](p07-cv-instrument/) | val 을 버리고 train 사슬 CV 로 · 장기 horizon 표적화 | EDA 는 남았고 학습 기록은 없다 |
| [`p12-three-member-ensemble`](p12-three-member-ensemble/) | P3 · P9 · N1 을 **각자의 원래 레시피로** 재학습해 평균 | 오차상관 역산 기대값 56.72. 시간 부족으로 미제출 |
| [`p14-feature-toggles`](p14-feature-toggles/) | 플레어 · CH 형태 · 자전 · 강도깊이를 하나씩 켜고 비교 | 토글 비교 프레임 |
| [`p15-single-model-features`](p15-single-model-features/) | P3 단일 모델을 유지한 채 위 토글만 비교 | 앙상블 없는 순수 비교축 |
| [`p16-polynomial-ridge`](p16-polynomial-ridge/) | 같은 피처 · 학습기만 GRU → 다항 릿지 | **피처 선별용 대조군.** 조합당 수십 초 |
| [`p17-unet-aux-supervision`](p17-unet-aux-supervision/) | CH 마스크를 입력이 아니라 **라벨**로 주고 U-Net 이 표현을 학습 | 영상 브랜치 4연패에 대한 대응책 |
| [`p18-chain-cv`](p18-chain-cv/) | P3 를 글자 그대로 두고 **epoch 선택 기준만** 사슬 CV 로 | 무작위 분할이면 eval 프레임 99.8 % 가 train 에도 등장 |
| [`p3-renew-mask-only`](p3-renew-mask-only/) | 격자 15개를 버리고 마스크 면적 1개(또는 모멘트 5개)만 | "남은 14 자유도가 잡음인가"를 학습 1회로 검정 |
| [`sweep-overnight-search`](sweep-overnight-search/) | 27개 노브를 6시간 예산 안에서 무인 탐색 | 사다리 → 탐욕 → 무작위 → 정밀 재측정 → 제출 후보 |
| [`final1-simplified`](final1-simplified/) | 전부 접고 네 문장만 남긴 단순화 노선 | 원반 내부 · 193∪211 · 극지 가중↓ · 프레임 CNN |

## 폴더 안에 무엇이 있나

| 파일 | 뜻 |
|---|---|
| `code_*.ipynb` | 그 버전의 노트북. 출력이 들어 있으면 실행 기록이 남은 것이다 |
| `model.pth` | 제출한 학습 가중치. 메타데이터(채택 설정 · CV · val)가 함께 저장돼 있다 |
| `submission.csv` | 실제로 Public 점수를 받은 예측 (3,868행 × 12 horizon) |
| `output-*.png` | 대회 서버 실행 화면 캡처 (P5) |

> **대회 데이터셋은 포함하지 않았다.** 노트북은 `SW_DATA_ROOT` 또는
> `public_dataset/competition_dataset_6h` 에서 데이터를 찾는다. 자세한 건
> 최상위 [README 의 재현 방법](../README.md#10-재현-방법) 참고.
