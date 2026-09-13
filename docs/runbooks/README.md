# runbooks — 버전별 실행 안내

대회 서버에서 "무엇을 어떤 순서로 치는가"만 적은 문서들이다. 전략과 근거는
[`../code-report.md`](../code-report.md) 와 [`../plan-p11.md`](../plan-p11.md) 에 있다.

경로가 저장소 구조에 맞게 수정된 것 외에는 대회 당시 원문 그대로다.

| 문서 | 대상 | 이 문서에만 있는 것 |
|---|---|---|
| [p11-physics-ensemble.md](p11-physics-ensemble.md) | P11 | 5단계 제출 사다리(S1→S5M)와 정지 규칙. **§7 에 S1 실패 사후분석**이 붙어 있다 |
| [p14-feature-toggles.md](p14-feature-toggles.md) | P14 | 플레어·형태·자전·강도깊이 토글을 하나씩 켜는 비교 순서 |
| [p15-single-model-features.md](p15-single-model-features.md) | P15 | 캐시가 없을 때 토글이 자동으로 꺼지는 동작과 그 확인법 |
| [p16-polynomial-ridge.md](p16-polynomial-ridge.md) | P16 | 다항 전개의 항 폭발을 막는 core(18열)/linear 2층 설계행렬 |
| [p17-unet-aux.md](p17-unet-aux.md) | P17 | **판정 근거가 되는 대조군 실험 4개.** RTX 4060 기준 VRAM·처리량 실측표 |
| [p18-chain-cv.md](p18-chain-cv.md) | P18 | 무작위 분할 99.8 % vs 사슬 배정 0 % 누수 측정. **부록에 P9 폴드 설계의 오류 실측** |
| [p3-renew-mask-only.md](p3-renew-mask-only.md) | P3-renew | 격자가 잡음이 되기 쉬운 이유 3가지와 `area_shape` 모멘트 피처 정의 |
| [sweep-overnight-search.md](sweep-overnight-search.md) | SWEEP | 6시간 예산 배분(A/B/C/D 단계)과 위험별 대응표. 탐색 공간 27개 노브의 출처 |
| [final1-simplified.md](final1-simplified.md) | Final1 | 전부 접고 남긴 네 문장. **알고 쓰는 한계 4가지**를 명시 |

## 읽는 순서 추천

1. **[p18-chain-cv.md](p18-chain-cv.md)** — 이 대회 데이터의 누수 구조가 가장 압축적으로 설명돼 있다.
2. **[p11-physics-ensemble.md](p11-physics-ensemble.md) §7** — 제출 하나가 어떻게 사후분석으로 이어졌는지.
3. **[sweep-overnight-search.md](sweep-overnight-search.md) §4** — "이 결과를 어떻게 믿을 것인가".
