# CrownGen 프로젝트 상태 문서
> 최종 업데이트: 2026-06-22

## 1. 프로젝트 개요
CrownGen 논문 재현 — 치아 크라운(point cloud) 생성 diffusion 모델.
- 논문: "CrownGen: Patient-customized Crown Generation via Point Diffusion Model"
- 공식 코드: https://github.com/baejustin/CrownGen (1저자 Bae, Juyoung)
- 우리 환경: Source C (Teeth3DS) 데이터만 사용, 자동 Procrustes 표준화

## 2. 모델별 상태

| 모델 | 상태 | 위치 | 비고 |
|------|------|------|------|
| Boundary (Dice 0.77) | ✅ 완료 | runs2/boundary_official_long.pt | 논문 0.883, 1000ep |
| Generation 2000ep (CD 51.6×10⁻³) | ✅ 완료 | runs2/gen2k_last.pt | 논문 34.5×10⁻³ |
| Generation 3000ep (ep2106 중단) | ⏸ 중단 | runs2/gen3k_last.pt | resume 가능 |
| Stage2 pseudo-crown (ep2062 중단) | ⏸ 중단 | runs2/gen_stage2_last.pt | resume 가능 |
| Pseudo-crown 데이터셋 (811명) | ✅ 구축 완료 | Data/processed_stage2/ | 403 완전치열 + 408 pseudo 채움 |
| 크라운 .ply / .obj 메시 | ✅ | runs2/ply/, runs2/mesh3/ | Poisson consistent_d8 |

## 3. 데이터 파이프라인

### 원본 → 처리 과정
```
Data/SourceC_Teeth3DS/Teeth3DS_full/  (원본 OBJ + JSON 라벨)
  ↓ preprocess_teeth3ds.py (표면 샘플링, PCA 정준 좌표계, min-enclosing-circle, z-분리)
Data/processed/  (정규화 전)
  ↓ procrustes_standardize.py (FDI-correspondence generalized Procrustes, 슬롯 분산 0.37→0.12)
Data/processed_norm2/  (학습용, 851명)
  ↓ gen_stage2_pseudo.py (gen2k 모델로 빈 자리 채우기)
Data/processed_stage2/  (811명 = 403 완전치열 + 408 pseudo-crown)
```

### 데이터 규모
- 전체: 851명 (Source C 단일 소스, 논문은 C+D 1,794명)
- 완전 치열(28개): 443명 (논문 stage1: 420명과 비슷)
- 부분 무치아: 408명 (논문 stage2: 1,364명의 30%)
- stage2 확장: 811명 (403 + 408 pseudo-crown)

## 4. 포팅한 아키텍처 (crowngen/external/)

공식 1저자 코드를 CUDA 커널 포함해 그대로 포팅. 순수 torch fallback도 구비.

### functional.py → cuda_func/ (CUDA 커널 우선, torch fallback)
- avg_voxelize, trilinear_devoxelize, ball_query, grouping, furthest_point_sample, nearest_neighbor_interpolate
- CUDA 12.8 + nvcc 설치됨 (/usr/local/cuda-12.8)
- 컴파일 불가 시 functional_torch.py 로 자동 fallback

### modules.py (boundary 모듈)
- SharedMLP, SE3d, BallQuery, Voxelization, PVConv (Conv3d voxel branch)
- PointNetSAModule (FPS + ball query + MLP)
- IntertoothAttentionBlock / RPEAttention / RPENet (DITA, per-tooth RPE 어텐션)
- key_mask 옵션: 'official'(atlas) | 'context'(missing→present 허용)

### pvcnn.py (boundary 모델)
- PVCNN2 + BoundEncoder (3 SA blocks + DITA + regressor)
- sa_blocks = [((16,2,32),(128,0.1,32,(32,64))), ((32,3,16),(64,0.2,32,(64,128))), ((64,3,8),(16,0.4,32,(128,256)))]
- 학습: 1000ep, lr 3e-4 cosine→3e-6, dropout 0.3, weight_decay 1e-4
- 결과: **Dice 0.77 / IoU 0.635** (논문 0.883/0.796)

### gen_modules.py (generation 모듈)
- GenPVConv (temb 3-tuple, o_mask 없음)
- GenPointNetSAModule, GenPointNetFPModule (U-Net decoder)
- Transformer (CLIP-ViT, boundary conditioning용, 4 layers 8 heads)

### gen_autoencoder.py (generation 모델)
- PVCNN2Base: 4 SA(encoder) + 4 FP(decoder) + FactorizedAttention(치아 간 DITA)
- sa_blocks = [((32,2,32),(512,0.1,32,(32,64))), ((64,3,16),(256,0.2,32,(64,128))), ((128,3,8),(64,0.4,32,(128,256))), (None,(16,0.8,32,(256,256,512)))]
- fp_blocks = [((256,256),(256,3,8)), ((256,256),(256,3,8)), ((256,128),(128,2,16)), ((128,128,64),(64,2,32))]
- fdi_embedding(28,8), bound_embedding(5→64) + Transformer, time embedding
- forward(xt, t, x0, l_mask, o_mask, bound): 타겟은 noise, 컨텍스트는 clean

### gen_diffusion.py (PVD-style diffusion)
- GaussianDiffusion: linear β 1e-4→2e-2, T=1000, eps-prediction, fixedsmall variance
- pred_xstart clamp(-1,1) for stability
- p_sample_loop: 타겟 순수 noise 시작 → 컨텍스트 clean 유지 → 1000스텝 역샘플링
- GenModel: diffusion + PVCNN2 wrapper, EMA 지원

### ema.py
- EMA decay 0.995, 추론 시 EMA 가중치 사용

## 5. 학습 설정

### Boundary (train_boundary_official.py)
- lr 3e-4 cosine→3e-6, 1000ep, Adam wd 1e-4, B=4
- voxel_attention off (메모리), mask_mode='official'
- 결과: Dice 0.770 (ep1000 best)

### Generation Stage 1 (gen_train.py)
- lr 4e-5 cosine, 2000ep (논문 3000ep), Adam, B=4
- EMA decay 0.995, n_points=1024, mask_range=(1,6)
- CUDA ops (53s/ep at B=4 on RTX PRO 6000 Blackwell)
- 결과: CD-L1 = 0.0516 (×10³ = 51.6), 논문 34.5

### Generation Stage 2 (gen_train.py --stage2)
- gen2k(2000ep)에서 resume, processed_stage2(811명)로 fine-tune
- lr 1e-5, ep2000→3000 (ep2062에서 중단, resume 가능)

### 데이터 증강
- random_shuffle_points, bilateral_mirror(x→-x, Procrustes 프레임), isotropic_scale [0.95,1.05]

## 6. 스크립트 목록

| 스크립트 | 용도 |
|----------|------|
| scripts/preprocess_teeth3ds.py | 원본 → 처리된 .npz (PCA 표준화, 표면 샘플링, 실린더) |
| scripts/procrustes_standardize.py | FDI-correspondence Procrustes 정렬 (슬롯 분산 감소) |
| scripts/train_boundary_official.py | Boundary PVCNN2 학습 + Dice/IoU 평가 |
| scripts/gen_train.py | Generation diffusion 학습 (--stage2, --resume 지원) |
| scripts/gen_sample.py | 크라운 샘플링 + CD + PNG |
| scripts/gen_progression.py | 에폭별 snapshot으로 크라운 진화 .ply 생성 |
| scripts/gen_stage2_pseudo.py | gen2k 모델로 부분무치아 빈 자리 채우기 |
| scripts/viz_boundary.py | Boundary 예측 시각화 (GT vs pred 실린더) |
| scripts/sanity_boundary.py | Boundary 빠른 sanity 학습 + Dice |

## 7. 논문과의 차이점 (한계)

| 항목 | 논문 | 우리 | 영향 |
|------|------|------|------|
| 데이터 | C+D (1,794명) | C만 (851명) | Source D 사설 1,000건 없음 |
| 표준화 | 수동 Blender + 이상아치 재배치 | 자동 PCA + Procrustes | 슬롯 분산 0.12 vs ~0 |
| Stage1 에폭 | 3,000 | 2,000 | val loss 평탄, CD는 개선 중 |
| Stage2 데이터 | 1,364 부분무치아 | 408 (30%) | 확장 효과 제한 |
| 메시 재구성 | 학습 가능 DPSR | 표준 Open3D Poisson | 메시 품질 다소 열위 |
| Boundary Dice | 0.883 | 0.770 | 표준화 한계 |
| Generation CD | 34.5×10⁻³ | 51.6×10⁻³ | 데이터+표준화+에폭 한계 |

## 8. 다음 단계 (resume 시)

1. **Stage 2 재개**: `--resume runs2/gen_stage2_last.pt --epochs 3000 --stage2 1 --data_dir Data/processed_stage2`
2. **3000ep 재개**: `--resume runs2/gen3k_last.pt --epochs 3000`
3. **Source D 요청**: 1저자(baejustin)에게 비영리 학술 목적 데이터 요청
4. **DPSR 포팅**: cg_generation_module/mesh_recon/ (점→학습 가능한 메시)
5. **더 긴 학습**: GPU 확보 시 3000+ep + stage2 조합

## 9. 환경 정보
- GPU: NVIDIA RTX PRO 6000 Blackwell × 2장
- CUDA: 12.8 + nvcc (/usr/local/cuda-12.8)
- Python 3.10, PyTorch 2.11+cu128, Open3D 0.19.0
- OS: Linux (Ubuntu 22.04), timezone KST (+0900)
- 서버: /root/Projects/CrownGen
