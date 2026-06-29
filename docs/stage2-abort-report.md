# Stage2 Fine-tune 중단 사유 리포트
> 작성일: 2026-06-29 · generation stage2(ARCH pseudo-crown 데이터 fine-tune) 중단 결정과 근거

## 1. 결론 (TL;DR)
generation **stage2 fine-tune를 중단**한다. pseudo-crown(ARCH 하이브리드로 빈 자리를 채운 가짜 크라운) 데이터로 gen3k를 fine-tune했더니, **CD 기준 gen3k(stage1)보다 양쪽 평가(full·partial) 모두 악화**되었다. pseudo-crown 증강이 모델을 개선이 아닌 **퇴보**시켰다.

대신, **aligned 데이터(사용자 강체 정렬)로 전체 pipeline을 재진행**한다 — boundary에서 aligned가 이미 **Dice +0.08 개선(0.612→0.695)**을 입증했기 때문.

## 2. stage2 설정 (중단 시점)
- **입력 모델**: gen3k(stage1, 3000ep, CD~55) resume → 가중치+EMA 로드.
- **데이터**: `Data/processed_stage2` (811명, ARCH 하이브리드 pseudo-crown으로 빈 자리 채운 완전치열 확장).
- **구간**: `--epochs 3800` (= gen3k ep3000 이후 +800ep, cosine 800). B=1, lr 4e-5.
- **중단 지점**: **ep 3327/3800** (stage2 327ep 진행).

## 3. 중단 근거 — 측정 데이터
generation은 **반드시 CD로 평가**해야 (val loss 동일해도 CD 차이 큼, [[boundary-conditioning-helps-generation]]).

| 평가 셋 | gen3k (stage1) | stage2 (ep~3270) | 판정 |
|---|---|---|---|
| full-val (stage1_val, 4환자) | CD×10³ ≈ **55** | **66.8** | ❌ 악화 |
| partial-val (stage2_val, 12환자) | CD×10³ = **58.3** | **62.9** | ❌ 악화 |

→ stage2가 full뿐 아니라 **진짜 타겟인 partial 복원에서도 gen3k보다 나쁨**. 일시적이 아니라 양쪽 다 일관되게 악화 → pseudo-crown fine-tune이 모델을 깎고 있다는 명확한 신호.

## 4. 원인 분석
1. **pseudo-crown의 '위치'는 ARCH가 잡았지만 '형상(quality)'은 여전히 불완전.** ARCH 하이브리드는 빈 자리 크라운이 인접 치아와 겹치는 위치 문제(겹침 26%→11%)는 해결했으나, 생성된 크라운 점구름 자체의 품질/형상은 gen3k(진짜 완전치열 학습 모델) 수준이 아님.
2. **stage2는 pseudo-crown을 'teacher'로 fine-tune**하는 구조 → teacher 품질이 모델 품질의 상한. pseudo-crown이 진짜 치아보다 노이즈가 많으면 fine-tune이 모델을 진짜 분포에서 멀어지게 함.
3. **boundary도 구(Procrustes) 데이터 기반**이어서 pseudo-crown 위치 정확도에 한계가 있었고, ARCH 보간이 보완했지만 근본(clean teacher)은 아니었음.
4. 327ep로 이르지만, **초기부터 gen3k 밑에서 시작해 회복 기미 없음** → 데이터 품질 문제, epoch 늘려도 한계.

## 5. 대조 — aligned 데이터는 효과 확실
같은 기간 aligned 데이터로 boundary를 재학습·검증한 결과, **확실한 개선** 확인:

- aligned boundary Dice **0.695** vs 현 official **0.612** (native 2×2 cross-eval, [[eval_boundary_dice]]).
- aligned_norm per-slot std **0.065** vs processed_norm2 **0.115** (위치 변동 ~44% 감소; z 특히 0.074→0.027로 rest position 정합이 상/하악 분리 해결).

→ **데이터 정렬(standardization) 품질이 boundary 정확도의 유효한 레버**임이 입증됨. 논문 수동 Blender 정제에 더 가까운 방향.

## 6. 결정 및 다음 단계
**stage2(구 데이터) 중단 → aligned 데이터로 전체 pipeline 재진행.**

```
[완료] aligned_norm 구축 (811명, rest position 강체 정렬, std 0.065)
[진행] aligned boundary 재학습 연장 (boundary_aligned_ext, 1500ep warm restart)
[예정] aligned 1차 generation 학습 (3000ep) → gen_aligned
[예정] aligned boundary로 pseudo-crown 재생성 (위치 더 정확 → cleaner teacher)
[예정] aligned stage2 fine-tune → 최종 크라운 품질 판정
```

기대: aligned의 정밀 정렬 → boundary 위치 정확도↑ → pseudo-crown teacher 품질↑ → stage2가 gen3k를 개선하는 방향으로 작동 (구 데이터에서 실패한 원인 해소).

## 7. 핵심 교훈
- **generation 평가는 CD, loss 아님** (이번에도 val loss는 비슷했으나 CD는 명확히 악화).
- **pseudo-crown 증강은 teacher 품질이 전제** — 위치(ARCH)만 잡고 형상 품질이 안 받쳐면 stage2가 퇴보함.
- **데이터 정렬(standardization)이 근본 레버** — 모델/학습량 조정보다 aligned 데이터가 boundary·전체 품질에 더 큰 영향.
- 관련: [[g1-boundary-result]], [[boundary-weakness-root-cause]], [[stage2-run-and-gotchas]], `PROJECT_STATUS.md`
