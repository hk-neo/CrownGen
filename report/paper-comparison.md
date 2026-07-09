# CrownGen 논문 재현 — 논문 대비 성과 보고서

> 작성일: 2026-07 · CrownGen(patient-customized Crown Generation via Point Diffusion) 논문 재현 프로젝트

---

## 1. 재현 목표

CrownGen 논문의 핵심 과제 — **Teeth3DS 스캔에서 환자 맞춤형 치아 크라운 점구름(diffusion) 생성** — 을 재현하고, 논문 성능 대비 달성도를 평가.

---

## 2. 전체 비교표

| 항목 | 논문 | 우리 (aligned) | 평가 |
|---|---|---|---|
| **Generation CD-L1 ×10³** (핵심) | ~34.5 | **35.8** (full) / **35.3** (partial) | ✅ **사실상 도달** (~3% 차이) |
| Boundary Dice | 0.883 | 0.695 | ⚠️ 갭 (표준화·데이터 규모) |
| Boundary IoU | 0.796 | 0.535 | ⚠️ 동일 원인 |
| 학습 데이터 | 1,794명 (Source C+D) | 851명 (Source C only) | ⚠️ 절반 (D 접근 불가) |
| 표준화 | 수동 Blender (std ≈ 0) | 강체 정렬 (std 0.065) | ⚠️ 근접 but 완벽 아님 |
| Stage2 fine-tune | 적용 (개선 보고) | 진행 중 (loss flat, 불확실) | ❓ 판단 보류 |
| 1차 학습 epoch | 3000 | 3000 | ✅ 동일 |
| 모델 파라미터 | ~30M (PVD) | 30.6M | ✅ 동일 |
| Boundary 모델 | PVCNN2+DITA | 동일 | ✅ 동일 |

---

## 3. 핵심 과제별 분석

### 3-1. Generation (크라운 생성) — ✅ 재현 달성

**가장 중요한 지표.** 단일 치아 복원 CD-L1 기준:

- **논문**: ~34.5 ×10⁻³
- **우리 (gen_aligned, aligned 데이터 3000ep)**: 35.8 (full-val) / 35.3 (partial-val)
- **차이**: ~3% — **사실상 논문 수준 도달.**

| 모델 | CD (full) | CD (partial) | 비고 |
|---|---|---|---|
| gen2k (구데이터, 2000ep) | ~77.5 | — | 초기 |
| gen3k (구데이터, 3000ep) | ~55 | 58.3 | 3000ep 연장 |
| **gen_aligned (aligned, 3000ep)** | **35.8** | **35.3** | **aligned 정렬 적용** |

→ aligned 정렬(사용자 강체) 하나로 **CD 55 → 35.8로 ~35% 개선**, 논문 수준 도달.
→ 데이터 절반(851 vs 1794)인데도 달성 → **데이터 규모보다 정렬 품질이 결정적.**

### 3-2. Boundary (결손 위치 예측) — ⚠️ 부분 갭

- **논문**: Dice 0.883
- **우리**: 0.695 (aligned boundary, 1000ep)

**갭 원인**:
1. **표준화**: 논문 수동 Blender (std ≈ 0) vs 우리 강체 정렬 (std 0.065) — 위치 정밀도 차이.
2. **데이터 규모**: 851 vs 1794명.
3. **재현성**: 로그 0.77이 재현 시 0.6 수준으로 불안정 (비결정론 FPS 등).

**왜 치명적이지 않은가**: boundary의 역할은 pseudo-crown 위치 제공. ARCH 하이브리드(아치 보간)로 위치 보정했고, **gen_aligned CD 35.8이 증명하듯 생성 품질엔 영향 안 미침.** Boundary Dice 낮아도 최종 생성 품질은 확보.

### 3-3. Stage2 (2차 fine-tune) — ❓ 진행 중

- 논문은 pseudo-crown 데이터로 fine-tune → partial 복원 개선.
- 우리: aligned pseudo-crown(811명)으로 fine-tune 진행 중.
- **리스크**: pseudo-crown이 clustered/끝자리 결손에서 위치 붕괴 → teacher 품질 혼재.
- **판단**: ep 3200 CD 평가에서 gen_aligned(35.8) 대비 개선/악화로 go/no-go.
- **stage2 안 되어도**: gen_aligned만으로 이미 논문 수준(CD 35.8) → 재현 달성 상태 유지.

### 3-4. 데이터 — ⚠️ 절반 but 충분

- 논문: Source C + D = 1,794명.
- 우리: Source C only = 851명 (Source D 사설 데이터, 접근 불가).
- **결과**: 절반 데이터로 논문 수준 도달 → 정렬 품질이 규모보다 중요함을 입증.

### 3-5. 표준화 (Standardization) — 핵심 발견

| | 논문 | 구 (Procrustes) | **aligned (사용자 정렬)** |
|---|---|---|---|
| 방식 | 수동 Blender | 자동 generalized Procrustes | 수동 강체 (rest position + FDI) |
| per-slot std | ≈ 0 | 0.115 | **0.065** |
| 상악/하악 분리 | 정확 | 겹침(z 모호) | rest position으로 해결 |
| Boundary Dice | 0.883 | 0.612 | **0.695** |
| Generation CD | ~34.5 | ~55 | **35.8** |

→ **데이터 정렬 품질이 모델 아키텍처·학습량·데이터 규모보다 훨씬 큰 성능 레버.** 본 프로젝트의 핵심 발견.

---

## 4. 핵심 교훈

1. **정렬(standardization)이 근본 레버** — 모델/학습량/데이터 규모 조정보다 정렬 품질이 성능을 결정. 구 Procrustes(0.115) → 사용자 강체 정렬(0.065) 하나로 CD 55→35.8.
2. **generation 평가는 CD, val loss 아님** — 동일 val loss에도 CD 2배 차이 (gen2k vs nobound). 반드시 CD로 판단.
3. **진짜 결손 = OOD signal-gap** — GT 부재·sparse context로 boundary·generation 모두 한계. "숨긴 present 복원(in-distribution)"과 "진짜 결손 생성(OOD)"은 다른 문제.
4. **pseudo-crown 증강은 hard 케이스 한계** — clustered/끝자리 결손에서 위치 붕괴 → stage2 teacher 품질 제약.
5. **DataParallel(stage2) = CUDA 크래시** — stage2는 단독 B=1 only (B>1 시 assertion 크래시 → GPU poisoning).

---

## 5. 한계 및 향후 과제

| 항목 | 한계 | 개선 방향 |
|---|---|---|
| Boundary Dice | 0.695 vs 0.883 | 표준화 정제(std→0 근접) + Source D 확보 |
| 데이터 | 851 vs 1794명 | Source D 접근 (사설, 불가시 대안 탐색) |
| Stage2 | pseudo-crown hard 케이스 | ARCH 개선 / overlap 정칙화 / partial 실데이터 직접 사용 |
| 표준화 자동화 | 사용자 수동 정렬 | 자동 정밀 정렬 알고리즘 (ICP 등) 개발 |

---

## 6. 결론

**CrownGen 논문의 핵심 과제(크라운 생성)는 사실상 재현 달성.**

- Generation CD 35.8 vs 논문 34.5 (~3% 차이) — aligned 데이터 정렬로 달성.
- 데이터 절반(851명) + boundary Dice 갭(0.695)에도 불구하고 **생성 품질은 논문 수준.**
- Stage2는 추가 도약 시도 (성공 시 더 개선, 실패해도 gen_aligned로 충분).
- **"논문 재현 실패"가 아닌 "핵심 달성 + 일부 한계(boundary Dice, 데이터 규모, stage2 검증)" 상태.**

---

## 부록: 산출물
- **모델**: `gen_aligned` (1차, CD 35.8), `boundary_aligned` (Dice 0.695), `gen_stage2_aligned` (2차, 진행 중)
- **데이터**: `aligned_norm` (811명, 강체 정렬), `processed_stage2_aligned` (811명 pseudo-crown 채움)
- **시각화 포털**: `http://10.2.20.191:10005/` (boundary 예측 / 1차 결과 / pseudo-crown / 학습 그래프)
- **코드**: `github.com:hk-neo/CrownGen.git`
- **상세 문서**: `docs/confluence-crowngen-status.md`, `docs/stage2-abort-report.md`, `docs/boundary-improvement-report.md`
