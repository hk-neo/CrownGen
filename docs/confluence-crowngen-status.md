# CrownGen 재현 — 현재 상태 및 pseudo-crown 위치 이슈 정리

> 작성일: 2026-07 / CrownGen(patient-customized Crown Generation via Point Diffusion) 논문 재현 프로젝트

---

## 1. 프로젝트 개요
- **목표**: Teeth3DS 스캔에서 환자 맞춤형 치아 크라운 점구름(diffusion) 생성 재현.
- **Pipeline**: 원본 스캔 → 전처리(PCA + 정렬) → **boundary**(결손 치아 위치/윤곽 예측, PVCNN2+DITA) → **generation 1차**(크라운 점구름 생성, PVD) → **pseudo-crown**(결손 자리 채우기) → **generation 2차**(fine-tune).
- **평가**: boundary는 Dice/IoU, generation은 **Chamfer Distance(CD-L1, ×10³)** (val loss는 신뢰 불가 — 동일 loss에도 CD 큰 차이).

---

## 2. 핵심 전환: aligned 데이터 (사용자 강체 정렬)

| | 구 pipeline (자동 Procrustes) | **aligned (사용자 강체 정렬)** |
|---|---|---|
| 정렬 방식 | 자동 generalized Procrustes | **수동 강체**(rest position + FDI 배치) |
| per-slot 위치 변동(std) | 0.115 | **0.065** (~44%↓, z는 0.074→0.027) |
| Boundary Dice | 0.612 | **0.695** |
| Generation 1차 CD | ~55 (gen3k) | **35.8** (gen_aligned, 논문 ~34.5 도달) |

- **결론**: **데이터 정렬(standardization) 품질이 모델/학습량 조정보다 훨씬 큰 레버.** 논문이 수동 Blender 정렬로 얻은 이점을 사용자 강체 정렬로 재현.
- 상악/하악을 실제처럼 벌린 rest position 정합이 z 모호성(두 턱 겹침)을 해결 → boundary·generation 모두 개선.

---

## 3. 현재 이슈: pseudo-crown 위치 (결손 채운 크라운이 기존 치아 위에)

### 3-1. pseudo-crown 이란
- 부분무치아(결손) 환자의 **빈 자리**를 generation 모델(gen_aligned)로 채운 가짜 크라운.
- **2차 학습(stage2)의 teacher 데이터**로 쓰임 — 완전치열을 흉내낸 확장 데이터셋 구축용.
- 생성 시 위치는 **ARCH 하이브리드**(내부 결손=아치 보간, 끝자리 결솅=boundary 예측) + 치아 크기(h,r)=boundary.

### 3-2. 문제
- 결손이 **clustered(인접 다수) / 끝자리**인 hard 케이스에서, pseudo-crown이 **인접 present 치아 위에 겹쳐** 생성됨.
- 측정(pseudo-crown 중심 → 가까운 present 치아 거리; **<0.08 = 겹침**, 정상 간격 ~0.16):

| 환자 | 결손 패턴 | 겹침 여부 |
|---|---|---|
| 00OMSZGW | 단일 결손 | 0.20 / 0.27 ✅ 양호 |
| **0132CR0A** | clustered 5결손 | **0.056~0.067 (4/5 겹침)** ❌ |
| 013475VT | 부분 | **0.042** 등 일부 겹침 ❌ |

→ "결손 자리가 아닌데 크라운이 덮여 있다"고 보이는 현상. 단일 결손은 깔끔하지만 clustered/끝자리는 위치 붕괴.

### 3-3. 원인 (가설)
1. **ARCH 보간 한계**: clustered 결손은 양옆 present가 멀어, 선형 보간이 일부 슬롯을 present에 가깝게 배치.
2. **생성 모델 drift (~0.05)**: bound(지시 위치)에서 크라운이 이웃 쪽으로 약간 밀림.
3. **근본(signal-gap)**: 진짜 결손 치아는 **GT도 없고 주변 context도 sparse** → boundary·generation 모두 **OOD**. 정렬 개선으로 in-distribution(숨긴 present 복원)은 좋아졌지만, 진짜 결손 자체는 여전히 학습 신호가 없는 hard 문제.

---

## 4. 시사점 및 방향

- **gen_aligned(1차)만으로 이미 논문 수준(CD 35.8 ≈ 논문 34.5).** 단일 치아 복원 기준으로는 재현 목표 달성.
- **stage2(pseudo-crown fine-tune) 리스크**: pseudo-crown teacher 품질이 케이스별 혼재(겹침 hard 케이스 존재) → 과거 구 데이터에서 CD 55→66으로 **악화**한 전례. aligned라도 hard 케이스 잡음이 남아 비슷한 위험.
- **옵션**:
  - **(A) gen_aligned를 최종 모델로 채택, stage2 스킵** — 이미 논문 수준이고 리스크 회피.
  - **(B) 겹침 hard 케이스 보정(ARCH 개선 / overlap 정칙화) 후 stage2 재시도** — 추가 공수, 추가 도약 가능.
  - **(C) 2차 학습을 partial 실데이터(stage2_train) 직접 사용** — pseudo 우회 (단 결손 슬롯 GT 부재 한계).

---

## 5. 핵심 교훈
1. **데이터 정렬(standardization)이 근본 레버** — 모델 아키텍처/학습량/마스킹 전략보다 정렬 품질이 성능을 크게 좌우.
2. **generation 평가는 CD, val loss 아님** — 동일 val loss에도 CD 2배 차이.
3. **진짜 결손 = OOD signal-gap** — GT 부재·sparse context로 boundary·generation 모두 한계; "숨긴 present 복원(in-distribution)"과 "진짜 결손 생성(OOD)"은 다른 문제.
4. **pseudo-crown 증강은 hard 케이스 한계** — clustered/끝자리 결손 위치 정확도에 근본적 제약.

---

## 부록: 산출물
- 모델: `gen_aligned`(1차, CD 35.8), `boundary_aligned`(Dice 0.695).
- 데이터: `aligned_norm`(811명, 강체 정렬), `processed_stage2_aligned`(811명 pseudo-crown 채움).
- 시각화 포털: `http://10.2.20.191:10005/` (aligned boundary 예측 / 1차 결과 / pseudo-crown 뷰어).
- 코드: `github.com:hk-neo/CrownGen.git`.
