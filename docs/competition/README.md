# 대회 자료

**제3회 한국천문연구원(KASI) × KAIST 천문우주 AI 경진대회**
주제: *태양 관측 영상을 활용한 지구 도달 태양풍 속도 예측*

| 파일 | 내용 |
|---|---|
| [`final-round-rules.md`](final-round-rules.md) | 본선 문제·데이터 구조·채점 기준·운영 규정·제출 방법 (공지 전문) |
| [`final-round-briefing-stt.md`](final-round-briefing-stt.md) | 본선 설명회 음성 전사본 (2026-08-14) |
| `briefing-preliminary.pdf` | 예선 설명회 자료 |
| `briefing-final.pdf` | 본선 설명회 자료 |
| `poster-a.jpg` · `poster-b.jpg` | 대회 포스터 |

## 대회 진행

| 라운드 | 방식 | 팀 시온 결과 |
|---|---|---|
| **예선** | AI 퀴즈(40점) + 우주 퀴즈(60점) | AI **1위** 40점 · SPACE 6위 58점 · **총점 98점 종합 5위** |
| **본선** | 태양풍 속도 12-horizon 회귀, Public/Private 각 12.5 % | Public **58.8028**, 최종 **31위 / 33팀** |

## 규정 요약 — 설계를 제한한 네 줄

1. **외부 데이터 사용 금지** — 흑점수·천체력·ICME 목록 전부 배제됐다.
2. **pretrained weight 사용 금지** — 아키텍처 구조 재사용은 허용, 가중치는 처음부터 학습해야 한다.
3. **validation 을 학습에 사용 금지** — 검증·모델 선택·하이퍼파라미터 조정 용도로만.
4. **재현성** — 제출한 `code.ipynb` + `model.pth` 로 결과가 재현돼야 하며, 크게 어긋나면 불이익.

전문은 [`final-round-rules.md`](final-round-rules.md) 참고.

---

> 이 폴더의 자료는 **주최측 저작물**이며, 대회 기록 보존 목적으로만 포함했다.
> 저작권은 한국천문연구원 · KAIST 에 있다.
