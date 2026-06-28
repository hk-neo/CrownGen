# CrownGen

> **CrownGen: Patient-customized Crown Generation via Point Diffusion Model** — 논문 재현 프로젝트.
> Teeth3DS(Source C) 단일 소스, 자동 PCA + Procrustes 표준화, boundary 조건부 점구름 디퓨전으로 환자 맞춤형 치아 크라운 생성.

경계(boundary) 예측 모델이 빈 자리의 실린더 윤곽을 예측하고, generation 디퓨전 모델이 그 조건하에 크라운 점구름을 합성합니다. 치아 간 DITA(Displacement-aware Inter-Tooth Attention) 적용.

---

## 📊 결과 요약

| 모델 | 메트릭 | ours | 논문 |
|---|---|---|---|
| Boundary (official) | Dice / IoU | **0.770 / 0.635** | 0.883 / 0.796 |
| Generation stage1 (gen3k, 3000ep) | CD-L1 ×10⁻³ | **~55** (2000ep=51.6) | 34.5 |

**핵심 발견**: boundary 조건 유무가 generation 품질을 ~3.5배 결정 (CD 71 vs 254, 12/12, t=−5.67) — val loss는 같아도. generation은 **반드시 CD로 평가**해야.

---

## 🔧 파이프라인

```
Teeth3DS 스캔(OBJ+JSON 라벨)
  │ preprocess_teeth3ds.py        → Data/processed        (PCA 정준좌표, 표면 샘플링)
  │ procrustes_standardize.py     → Data/processed_norm2  (FDI Procrustes, 851명 학습용)
  │ train_boundary_official.py    → boundary 모델 (PVCNN2+DITA, 결손 실린더 예측)
  │ gen_train.py (stage1)         → gen 디퓨전 (완전치열 400명) → gen2k/gen3k
  │ gen_stage2_pseudo.py          → Data/processed_stage2 (부분무치아 빈자리 pseudo-crown 채움, 811명)
  │ gen_train.py --stage2         → stage2 fine-tune
  │ gen_sample.py                 → 크라운 생성 + CD 평가 / Poisson 메시
```

---

## 📁 저장소 구조

```
crowngen/                 # 모델 소스 (external: 공식 코드 포팅, CUDA 커널+torch fallback)
  external/               # gen_diffusion, gen_autoencoder, gen_modules, pvcnn, modules(DITA)
  data/, losses/, models/
scripts/                  # 전처리·학습·샘플링·시각화 스크립트
  train_boundary_official.py, gen_train.py, gen_sample.py, gen_stage2_pseudo.py
  train_boundary_g1.py    # G1: partial-realistic boundary 재학습 (개선 중)
  viz_*                   # 웹 뷰어 데이터 생성
runs2/viz/                # 인터랙티브 3D 뷰어 (Three.js) — index.html/app.js/data.js
docs/                     # 설계·리포트 문서 (boundary-improvement-report.md 등)
PROJECT_STATUS.md         # 전체 상태 문서
```

---

## ⚠️ 데이터 & 가중치 (저장소에 미포함)

| 자원 | 크기 | 비고 |
|---|---|---|
| `Data/` (Teeth3DS 처리본) | **~39 GB** | 원본은 Teeth3DS(비영리 학술)에서 확보 필요 |
| 가중치 `*.pt` | **~11 GB** (73개) | 학습으로 재생성 가능 |

- 두 자원 모두 **`.gitignore`로 제외** — 이 repo는 **코드·문서·뷰어(3.9 MB)** 만 포함.
- 가중치가 필요하면: 위 파이프라인대로 학습, 또는 Git LFS로 별도 tracked 가능.

---

## 🛠️ 환경
- Python 3.10+, PyTorch 2.x + CUDA 12.8 (nvcc), Open3D 0.19
- GPU: ~24 GB VRAM (generation은 B=1 기준; boundary/추론은 더 가벼움)
- 상세: `PROJECT_STATUS.md` §9

## ▶️ 실행 (요약)
```bash
# 1) 전처리
python scripts/preprocess_teeth3ds.py
python scripts/procrustes_standardize.py
# 2) boundary 학습 (1000ep) → runs2/boundary_official_long.pt
python scripts/train_boundary_official.py --epochs 1000
# 3) generation 1단계 (논문 3000ep) → gen3k
python scripts/gen_train.py --epochs 3000 --batch_size 1
# 4) pseudo-crown 증식 → 2단계 fine-tune
#    ARCH 하이브리드 위치(내부 결손=아치 보간, 끝자리=boundary) + G1 boundary
python scripts/gen_stage2_pseudo.py --gen_ckpt runs2/gen3k_ep3000.pt \
  --bound_ckpt runs2/boundary_g1_best.pt --arch_pos
# 주의: resume가 gen3k(ep3000)이라 start_ep=3000. stage2 2400ep = --epochs 5400 (3001~5400).
# (빈 루프 주의: --epochs 3000으로 하면 학습 안 됨)
python scripts/gen_train.py --stage2 1 --data_dir Data/processed_stage2 \
  --resume runs2/gen3k_last.pt --epochs 5400 --batch_size 1 --tag gen_stage2
# 5) 평가/샘플링
python scripts/gen_sample.py --ckpt runs2/gen_stage2_last.pt
```

## 🌐 시각화 뷰어
정적 Three.js 3D 뷰어 (데이터는 `scripts/viz_*_data.py`로 생성):
- **boundary** 런별 실린더 예측 비교 · **gen_compare** boundary 조건 효과(bound vs nobound)
- **gen_progression** 에폭별 크라운 진화 · **gen_2k_3k** 1차 연장(3000ep) 효과
```bash
python -m http.server 8766 --directory runs2/viz/web   # 각 뷰어 디렉토리
```

---

## 📉 논문 대비 한계
- **데이터**: 논문 C+D(1,794명) vs ours C만(851명) — Source D(사설) 접근 불가.
- **표준화**: 논문 수동 Blender(슬롯 분산 ~0) vs ours 자동 Procrustes(0.12) → boundary 위치 정확도 하한.
- **Stage1 에폭**: 3000ep(논문 스펙) 도달. Stage2는 진행 중.
- 상세 한계/개선: `docs/boundary-improvement-report.md`, `PROJECT_STATUS.md`

## 🧪 현재 진행 상태 (2026-06)
- ✅ Boundary(official, Dice 0.77), Generation stage1(gen3k 3000ep, CD ~55)
- ⏳ **G1**: partial-realistic boundary 재학습(부분무치아 포함 + 현실적 마스킹) — pseudo-crown 위치 정확도 개선 목표
- ⏸ Stage2 fine-tune (G1 이후)

---

## 📄 문서
- `PROJECT_STATUS.md` — 전체 상태·아키텍처·설정
- `docs/boundary-improvement-report.md` — boundary 약점 분석 + 개선 리포트
- `docs/superpowers/specs,plans/` — 뷰어 설계·구현 계획

## 참고
- 논문: *CrownGen: Patient-customized Crown Generation via Point Diffusion Model*
- 공식 코드: https://github.com/baejustin/CrownGen
