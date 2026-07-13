# DPSR fine-tune on Teeth3DS (CrownGen)

**Date:** 2026-07-14
**Author:** Claude (brainstorming with hk.sim)
**Status:** design approved, ready for plan

## Motivation

CrownGen 논문에서 DPSR(Shape as Points)로 만든 메시 품질이 인상적. 우리는 이전에
SAP pre-trained 가중치(`ours_noise_005.pt`)를 그대로 적용해 봄 → **치아는 OOD라 GT도
망가짐** (`9da9d47` 에서 회귀). 결론: CrownGen의 DPSR 우위는 **치아 특화 학습 가중치**
덕분. 따라서 우리 데이터로 fine-tune해서 논문 수준에 근접한지 확인한다.

## Goals / Non-Goals

**Goals**
- 우리 811명 GT 크라운 표면 점(17,800 teeth)으로 SAP(`Encode2Points`) fine-tune
- 표준 Poisson+Taubin 대비 메시 품질(Chamfer-L2 · normal consistency · watertight
  여부) 개선 정량 평가
- fine-tuned 모델로 새 viewer 갤러리 추가 (3-panel: standard Poisson · SAP-pre ·
  SAP-fine)

**Non-Goals**
- generation 모델 재학습 / boundary 재학습 — 현재 모델 그대로 사용
- 실시간 인터랙티브 학습 / GAN / 다른 메소드 — 이번엔 SAP 만
- 학습된 모델 외부 배포 / API화

## Decision: 작업 격리 방식

- **branch**: `mesh/dpsr-finetune` 에서 작업 (main은 깨끗하게 유지)
- **viewer port**: 새 갤러리는 **8778** 포트로 띄움 (mesh_demo는 8777 그대로).
  롤백/독립 배포를 위함.
- 기존 main의 viewer · viewer·포털 변경 **금지** (충돌 방지).
- merge는 viewer 검증 후 사용자 승인 시.

## §1. Data Pipeline

### Input
- 851 환자 GT 크라운 표면 점 (`Data/aligned_norm/{pid}.npz` — 811명 실재 + 40 val sparse)
  - 실재 학습 캐시: stage1_train 319 + stage2_train 포함 환자 = 약 750명
  - val: stage1_val의 16명
- per-tooth `{jaw}_{fdi}_pc: (1024, 3)` keys
- 대상 환자: 811 전체 (stage1_train/stage1_val 분할은 SAP 학습용으로 재정의)

### PSR GT 캐시
SAP 학습은 "완전 mesh → PSR volume GT" 쌍이 필요. 우리엔 mesh가 없으니:
```
GT PC (1024,3)
  → Open3D Poisson(depth=9, normals estimated)
  → orient normals outward
  → DPSR(points, normals, res=64³, σ=2)
  → ★ cache: 64³ float16 PSR volume
```

### 저장 포맷
- 위치: `runs2/sap_cache/{pid}_FDI{fdi}.npz`
- 내용: `psr_vol (64,64,64) float16` + `pc (1024,3) float32`
- 디스크: ~9 GB (17,800 teeth × 524 KB)

### 캐시 스크립트
`scripts/build_sap_cache.py`:
1. `Data/aligned_norm/{pid}.npz` 로드
2. for each present tooth: pc 추출 → Poisson mesh → normals orient → GPU DPSR
3. `.npz` 로 저장 (있으면 skip; resume 가능)

## §2. Training

| 항목 | 값 |
|---|---|
| 모델 init | `runs2/dpsr_weights/ours_noise_005.pt` (SAP pre-trained) |
| Loss w_psr | 1.0 (PSR MSE) |
| Loss w_reg_point | 10.0 (Chamfer L2 reg — points vs refined) |
| Loss w_normals | 5.0 (refined normal vs GT normal L1) |
| Optimizer | Adam lr=1e-4, wd=1e-5 |
| Scheduler | CosineAnnealingLR (T_max=50*steps_per_epoch) |
| Batch size | 8 |
| Epochs | **50** |
| Steps/epoch | ~2,225 (17,800/8) |
| Total steps | ~111,250 |
| 1 step time | ~0.5s |
| **총 학습 시간** | ~15h (RTX 4090) |
| GPU memory | 6–10 GB |
| Gradient clip | 1.0 |
| TF32 | enabled |
| bf16 mixed precision | **off** (DPSR FFT 커널 호환성) |
| Checkpoint | `runs2/sap_finetuned_e{}.pt` every 10 ep + `sap_finetuned_best.pt` |
| Best metric | val Chamfer-L2 minimize |

### Train/Val split
- **stage1_train: 319명** (확정)
- **stage1_val: 40명** (확정)
- val set: stage1_val 12명 + stage1_val 외 4명 (총 16 val 환자)

## §3. Evaluation

### 비교 대상 (3-way)
1. **표준 Poisson + Taubin** — 현재 viewer (`runs2/mesh_demo/` 결과 그대로)
2. **SAP pre-trained (frozen)** — `runs2/dpsr_weights/ours_noise_005.pt` 그대로
3. **SAP fine-tuned** — `runs2/sap_finetuned_best.pt`

### 케이스
- 환자: stage1_val의 6명 (`runs2/mesh_demo/` 와 동일 인덱스)
- teeth per patient: 1~3 random, 마스크 동일 (재현성 위해 동일 seed)

### 메트릭
- **Chamfer-L2**: mesh 표면 30K uniform sample ↔ GT PC 1024. **단위 mm²**
- **Normal Consistency**: nearest neighbor로 mesh vertex normal · GT normal; `mean(1 - cos)`
- **Edge length L2**: smoothness proxy. mean edge length.
- **Watertight rate**: 100% closed일 때 1, 아니면 0 (connected components == 1)
- **Mesh quality score**: 위 가중 평균 (chamfer ≤ 0.05mm², normal consistency ≥ 0.95)
  일 때 통과.

### 결과 비교 차트
- `runs2/viz/mesh_sap_compare/charts/sap_eval.json` + bar chart PNG
  (matplotlib, "표준 Poisson vs SAP-pre vs SAP-fine" 메트릭 막대그래프)

## §4. 출력 (산출물)

| 산출물 | 경로 |
|---|---|
| 캐시 | `runs2/sap_cache/{pid}_FDI{fdi}.npz` (~9 GB) |
| ckpt | `runs2/sap_finetuned_e{10,20,...,50}.pt` + `_best.pt` |
| 평가 JSON | `runs2/sap_eval.json` |
| 평가 PLY | `runs2/mesh_sap_compare/{pid}_FDI{fdi}_{gt|gen}__{method}.ply` |
| 차트 PNG | `runs2/viz/mesh_sap_compare/charts/comparison.png` |
| Viewer | `runs2/viz/mesh_sap_compare/index.html` (+ three.min.js, OrbitControls.js) |
| 정적 서버 | `python -m http.server 8778 --directory runs2/viz/mesh_sap_compare` |

> viewer는 main의 mesh_demo(8777)와 독립적. merge 전까지 포털(10005)은 건드리지 않음.

## §5. Code 변경 요약

| 파일 | 상태 | 설명 |
|---|---|---|
| `scripts/build_sap_cache.py` | 신규 | PSR GT 캐시 생성 (~1.5h) |
| `crowngen/external/mesh_recon/src/data/tooth_dataset.py` | 신규 | 우리 cache에 맞는 dataset + PSR Field |
| `scripts/train_sap.py` | 신규 | Trainer wrap + fine-tune 50 ep (~15h) |
| `scripts/eval_sap.py` | 신규 | 6명 × 3치아 3-way 비교 + chart PNG |
| `scripts/ply_to_sap_compare_js.py` | 신규 | `mesh_sap_compare/*.ply` → 3-panel viewer `data.js` |
| `runs2/viz/mesh_sap_compare/index.html` `app.js` `style.css` | 신규 | 3-panel three.js viewer |
| 기존 `gen_mesh.py`, `gen_mesh_dpsr.py`, `mesh_demo/` | **변경 없음** | main 보호 |

## §6. Step Order (실행 순서)

1. **branch 생성**: `git checkout -b mesh/dpsr-finetune`
2. **PSR GT 캐시**: `python scripts/build_sap_cache.py` (background, ~1.5h)
3. **dataset 구현 + 1 step smoke**: step time < 1s 확인
4. **train**: `python scripts/train_sap.py` (background, ~15h)
5. **eval**: `python scripts/eval_sap.py` + chart
6. **viewer 띄우기**: `python -m http.server 8778 --directory runs2/viz/mesh_sap_compare`
7. **사용자 리뷰**: viewer 확인 + 평가 결과
8. **merge 여부 결정**: 통과 → main merge + 포털(10005) 카드 추가 / 실패 → 폐기

## §7. Failure Modes / Risks

| 위험 | 대응 |
|---|---|
| 캐시 생성 중 OOM (GPU DPSR 메모리) | chunk 단위로 batch 4 teeth, 부족 시 CPU DPSR fallback |
| 50 ep 후 val Chamfer 악화 | lr scheduler 재조정 (lr=5e-5로 재시작) 또는 조기 종료 |
| viewer 3-panel 동일 인덱스 정렬 실패 | seed=11 고정, 동일 환자/치아 |
| 디스크 ~9GB 부담 | 이전 stage2 ckpt 백업 후 삭제 후 캐시 생성 (사용자 확인) |
| generation 모델 변경 (main 머지 충돌) | viewer/포털은 사용자 승인 시에만 머지 |

## §8. Definition of Done

- [ ] `runs2/sap_finetuned_best.pt` 존재, val Chamfer-L2 개선 or 동등
- [ ] `runs2/viz/mesh_sap_compare/` 갤러리 viewer 3-panel 정상 동작 (port 8778)
- [ ] 비교 차트 PNG 생성 (3-way bar chart)
- [ ] 모든 결과 `mesh/dpsr-finetune` 브랜치 커밋 + push
- [ ] main 머지 사용자 승인
- [ ] (merge 시) 포털 `~/docker/crowngen-viz/index.html` 카드 추가
