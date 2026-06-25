# Boundary 모델 약점 분석 및 개선 리포트
> 작성일: 2026-06-25 · CrownGen boundary 예측 모델 (`boundary_official_long.pt`) 개선 근거

## 1. 현상 요약
Pseudo-crown(3000ep generation 모델이 빈 자리에 채운 가짜 크라운)의 **위치가 많이 틀림** (인접 치아와 겹침, 아치 간격 무시 등).

**진단 결과 (측정 기반):**
- 생성(디퓨전)은 **정상** — 28개 pseudo 크라운 전부 boundary 예측 중심(cx,cy,cz)에 충실 (drift < 0.1, 이탈 0/28).
- 인접 치아와 **겹치는** boundary 예측 다수 (예: `0132CR0A` FDI 16-17 이 0.039 거리로 거의 겹침; `013Z9SM2` FDI 17↔15 0.027).
- → **위치 오류의 원인은 boundary 모델**. 생성은 지시받은 자리에 정확히 찍을 뿐.

## 2. 원인 분석

> **두 문제를 분리해야 한다** (논의 중 도출된 핵심 통찰):
> - **문제 A (원래 관찰) — pseudo-crown 위치 틀림**: 원인은 **C1(OOD)**. boundary는 학습 분포(labeled 완전치열 + 시뮬레이션 마스킹)에서 GT에 **근접**(Dice 0.6~0.77) → 모델 자체는 정상. 하지만 **실제 partial은 학습 못 본 분포** → 무너짐.
> - **문제 B (별개) — 논문 Dice 격차 (0.883 vs 0.77)**: 표준화/재현성(C0). pseudo-crown 위치 오류의 원인이 아님.
>
> **데이터 규모 논거 정정**: 논문 "1,794명(C+D)"은 generation stage2 기준. **Source D가 무라벨 원본이면 boundary(치아별 라벨 + GT bound 필요)는 C만 사용했을 것** → boundary에 대한 "데이터 2배" 논거는 성립 안 함. (논문 텍스트로 확증은 어려웠으나 합리적 추론.) 즉 boundary의 논문 격차는 **데이터 규모보다 표준화·재현성**이 주요.

### C1 [pseudo-crown 위치 오류의 직접 원인 · 확인됨] 학습/추론 분포 불일치 = OOD
- **학습**: `stage1_train` **완전치열(28개) 319명 + val 40명** (부분무치아 **0명**).
  - `make_exist_mask`: present 치아 중 **1~6개 랜덤 마스킹**, GT = 그 치아의 실제 bound → "거의 다 있는 아치에서 몇 개 숨긴" 케이스만 학습.
- **추론**(pseudo-crown): 진짜 부분무치아 (결손 2~6+, sparse context).
- **근거(결정적)**: boundary는 **labeled(학습 분포)에서는 GT에 근접**(0.6~0.77) → 모델 정상. partial에서만 붕괴 = **OOD**. 진짜 결손 슬롯은 GT가 없어 학습 타겟 자체가 불가 → 현실 결손 패턴을 한 번도 안 배움.

### C0 [논문 Dice 격차의 원인 · 문제 B · 별개] 표준화 + 재현성
- **★표준화**: 논문 수동 Blender(슬롯 분산 ~0) vs 우리 자동 Procrustes(**0.12**). 측정: 같은 슬롯 GT 위치 std **xy 0.096**(치아 반지름의 64%), z 0.079 → 위치 대응 흐릿 → Dice 천장 하향. 단, **학습 분포 내에선 근접**하므로 pseudo-crown 위치 오류의 직접 원인은 아님.
- **재현성**: logged 0.77이 현재 0.6으로 재현 안 됨 → 성능 상한·불안정 (C3 참조).
- **데이터**: boundary가 C-only라면 규모 차이 작음(정정). Source D가 라벨 있어야 의미.

### C2 official `mask_mode`의 attention 고립
- official: `allowed[q,k] = (q==k)` → missing↔missing, present↔present 분리.
- 빠진 치아가 present(context)를 **inter-tooth attention으로 직접 참조 못 함**.
- 위치 추론은 주변 context에 의존하는데, attention이 이를 차단 → per-slot FC + 점 집계에만 의존 → 위치 정확도 제한.
- (반면 `context` 모드는 missing→present 허용해 위치엔 유리하지만, 현재 context ckpt는 재현 불량 ~0.24 — `boundary-checkpoint-dice-discrepancy` 참조.)

### C3 체크포인트 재현성 discrepancy (성능 상한 + 불안정)
- official: 로그 best 0.771 → 현재 측정 **~0.6** Dice. context/voxatt는 더 심하게 붕괴(0.24 / 0.08).
- 추정 원인: forward 비결정론(FPS·ball_query 샘플링), 또는 ckpt 저장 시점/상태, 또는 PyTorch·CUDA 버전 drift로 인한 voxel/attention 경로 변화.
- → 재학습 전에 이것부터 잡아야 "개선이 진짜 개선인지" 판별 가능.

### C4 per-slot 독립 회귀, 아치 기하 제약 없음
- 각 슬롯 cylinder를 독립 FC로 예측. 인접 치아와 **겹치지 않으라는 정칙화/제약 없음** → 16-17 겹침 같은 기하적 오류 허용.

### C5 데이터 한계
- Source C 단일 851명; boundary 학습은 완전치열 319명. 표현력·일반화 제한.

## 3. 개선 방법 (우선순위 + 공수/기대효과)

### G0 [선순위 · 진단] 재현성 확보
- 결정론 eval(`torch.manual_seed` + `cudnn.deterministic=True`, DataLoader `generator` 고정)로 official 0.77 재현 시도.
- 안 되면 비결정 원인(FPS 등) 특정.
- **공수 낮음 / 효과**: 이후 재학습이 의미 있는지의 전제.

### G1 [핵심] 학습 분포를 추론에 맞게 — C1 해결
- (a) `max_missing` 상향 (6 → 8~10) + (b) **cluster/현실적 마스킹** (인접 다수, 후방 어금니 등 실제 결손 패턴) → sparse-context 노출.
- (c) 학습 데이터에 **부분무치아 환자 포함** (present 치아 마스킹은 가능) → 더 다양한 sparse 패턴.
- **공수 중간 / 효과 큼** (가장 직접적 원인 제거).

### G2 [핵심] `mask_mode='context'`로 위치 정확도↑ — C2 해결
- missing→present attention 허용 → 빠진 치아가 주변 context로 위치 추론.
- 단, **G0(재현성) 먼저 확보** 후 context 모드 재학습 (현재 context ckpt 불량).
- **공수 중간 / 효과 큼**.

### G3 아치 기하 정칙화 / 후보정 — C4 해결
- 학습 시 **overlap 패널티** (인접 cylinder 중심 간 최소 거리 제약) 추가.
- 또는 추론 후 **arch-curve 스냅/보간** (인접 present 치아로 보간해 겹침 제거) — 빠른 임시방편.
- **공수 중간(정칙화)·낮음(후보정) / 효과 중간** (겹침 직접 제거).

### G4 데이터·증강 강화 — C5 완화
- scale/mirror 외 증강(회전·노이즈), 또는 pseudo-crown 반복 확보.
- **공수 낮음 / 효과 한계** (보조).

## 4. 권장 진행 순서

**문제 A(pseudo-crown 위치 오류) 해결이 1순위** — 원인이 OOD(C1)이므로:
```
G1 partial-realistic boundary 재학습 (현실적/강도 높은 마스킹 + 부분무치아 포함)
  → (보조) G2 context 모드, G3 겹침 정칙화/후보정
  → 평가(pseudo 재생성으로 겹침 감소 확인) → 2차 학습
```
- **가장 직접적 한 방 = G1**: boundary가 partial 분포를 학습하면 OOD 해소. 모델은 labeled에서 이미 GT에 근접하므로 **partial만 가르치면 됨**.
- **문제 B(논문 Dice 천장 0.883)는 별개**: 표준화 정제 + 재현성(G0) + 데이터. pseudo-crown 해결(A) 후 검토.

> 우선순위 변경: 이전(표준화 우선) → **G1(partial 학습) 우선**. boundary가 라벨된 분포에선 괜찮으므로, 표준화는 Dice 천장(논문 따라잡기)용이지 pseudo-crown 위치 오류의 원인이 아님.

## 5. 평가 방법 (개선 확인)
- 기존 eval_dice_iou(val, 시뮬레이션 마스킹) + **새 벤치마크**: 부분무치아 val 케이스에서 boundary 예측의 인접-치아 겹침 비율·아치 간격 적합도 측정.
- 최종: pseudo-crown 재생성 → 위 뷰어(`:8770`)로 위치/겹침 시각적 확인.
```
```
