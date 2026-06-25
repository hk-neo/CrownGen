# CrownGen: 논문 분석 및 구현 계획서

> **논문**: CrownGen: Patient-customized Crown Generation via Point Diffusion Model
> **저자**: Juyoung Bae, Moo Hyun Son, Jiale Peng, Wanting Qu, Wener Chen, Zelin Qiu, Kaixin Li, Xiaojuan Chen, Yifan Lin*, Hao Chen*
> **소속**: HKUST (Hong Kong University of Science and Technology), HKU (University of Hong Kong), Delun Dental Hospital
> **arXiv**: https://arxiv.org/abs/2512.21890v2
> **작성일**: 2026-06-09

---

## 1. 논문 요약

### 1.1 연구 배경 및 동기

디지털 치과 보철(Digital Prosthodontics)에서 크라운 설계는 여전히 숙련된 기공사가 수작업으로 진행하는 노동 집약적 병목 공정입니다. CAD/CAM 기술이 제작을 자동화했지만, 설계 단계는 여전히:
- 기공사가 일반적인 템플릿 라이브러리에서 수동 선택 후 환자 맞춤형 수정 필요
- 크라운 1개당 설계 시간이 1시간 이상 소요
- 다수 치아 복원 시 시간이 선형적으로 증가

기존 AI 기반 크라운 생성 방법의 근본적 한계:
1. **단일 크라운만 생성 가능**: 치과 아치를 단일 기하학적 입력으로 처리하는 아키텍처
2. **지대치(Prepared Abutment) 의존성**: 지대치 기하학이 필수적인 위치 신호로 작용 → 임플란트 지지 보철이나 브릿지 pontic에서 사용 불가

### 1.2 CrownGen의 핵심 기여

CrownGen은 **Denoising Diffusion Probabilistic Model (DDPM)** 과 **치아 수준 포인트 클라우드 표현**을 결합한 생성 프레임워크입니다.

| 기여 | 설명 |
|------|------|
| **다수 크라운 동시 생성** | 최대 6개의 결손치에 대해 단일 추론으로 크라운 생성 가능 |
| **치아 수준 객체 표현** | 치열을 단일 표면이 아닌 개별 치아 객체들의 집합으로 모델링 |
| **DITA (Distance-weighted Inter-Tooth Attention)** | 인접/대합 치아의 형태학적 영향을 거리 가중치로 명시적 모델링 |
| **Pseudo-crown 데이터 확장** | 불완전한 임상 데이터를 활용한 자가 부트스트래핑 학습 전략 |

### 1.3 시스템 아키텍처 개요

```
입력: 컨텍스트 치아(Y) + 결손치 FDI 라벨
          ↓
[Boundary Prediction Module] → 실린더 경계(B) 예측 (위치, 방향, 스케일)
          ↓
[Diffusion-based Generative Module] → 포인트 클라우드 크라운(X) 생성
          ↓                                    ↑
    DDPM 역확산 과정 (T=1000)         DITA 레이어 (치아 간 관계 학습)
          ↓
[DPSR Mesh Reconstruction] → Watertight 메쉬 변환
          ↓
출력: CAD/CAM 워크플로우용 3D 메쉬
```

---

## 2. 상세 기술 분석

### 2.1 데이터 파이프라인

#### 2.1.1 데이터 소스

| 소스 | 유형 | 스캔 수 | 역할 | 스캐너 |
|------|------|---------|------|--------|
| Source A | 공개 (Wang et al.) | 453 | 외부 테스트 | iTero |
| Source B | 공개 (Li et al.) | 43 | 외부 테스트 | TRIOS 3 |
| Source C | 공개 (ToothFairy) | 794 | 모델 학습 | iTero, TRIOS 3, Primescan |
| Source D | 사설 (덕륜치과) | 1000 | 모델 학습 | Aoralscan 3 |
| Source E | 사설 (덕륜치과) | 23 | 임상 평가 | TRIOS 3 |

**총 데이터**: 2,310스캔 → 큐레이션 후 개발 코호트 1,794스캔 (완전 치열 430 + 부분 무치아 1,364)

#### 2.1.2 전처리 파이프라인

1. **치아 분할 (Tooth Segmentation)**: Point Transformer 기반 분할 네트워크 → 치과의 검수
2. **교합 정렬 (Occlusal Alignment)**: Midline-Canine-Molar (MCM) 관계 기반 최적 교합 등록
3. **좌표계 표준화**: 상악 4전치 중심 → 원점, 안면 방향 → -Y축, 시상면 → YZ 평면, 교합면 → XY 평면
4. **품질 필터링**: 6개 이상 결손/기형 제외, 심한 부정교합 제외, 경도 부정교합은 디지털 시뮬레이션으로 교정

### 2.2 Boundary Prediction Module

#### 2.2.1 목적
각 결손치에 대한 **원통형 경계(Cylindrical Bound)** 예측 → 공간적 사전 정보(Spatial Prior) 제공

#### 2.2.2 원통형 파라미터 (5개 스칼라)

```
B = (cx, cy, cz, r, h)
```
- `(cx, cy)`: XY 평면 투영 최소 외접원의 중심
- `r`: 최소 외접원 반지름
- `h = zmax - zmin`: 치아 높이
- `cz = (zmin + zmax) / 2`: 축 방향 중심

#### 2.2.3 아키텍처
- 메인 디노이징 네트워크의 **인코더 부분만 사용** (3개 SA 블록)
- PVC (Point-Voxel Convolution) 연산자 + DITA 레이어
- 입력: 512 포인트/치아, 타겟 치아는 영벡터로 마스킹
- 손실 함수: **Smooth L1 Loss**

```python
# Smooth L1 Loss
def smooth_l1(x):
    return 0.5 * x^2 if |x| < 1 else |x| - 0.5

L_bound = (1/|X|) * Σ smooth_l1(B_pred_i - B_gt_i)
```

#### 2.2.4 학습 설정
- 에폭: 1000
- 옵티마이저: Adam
- 학습률: 3×10⁻⁴ → 3×10⁻⁶ (Cosine Annealing)
- 드롭아웃: 0.3

#### 2.2.5 성능
- 전체 평균 Dice: **0.883 ± 0.041**
- 전체 평균 IoU: **0.796 ± 0.061**
- 유형별 범위: 중절치 Dice 0.897 ~ 제2대구치 0.859

### 2.3 Diffusion-based Generative Module

#### 2.3.1 DDPM 수식 체계

**전방 확산 (Forward Diffusion)**:
```
q(X_t | X_{t-1}) = N(√α_t · X_{t-1}, β_t · I)
```
- T = 1000 타임스텝
- β_t: 선형 스케줄 (β_min=10⁻⁴, β_max=2×10⁻²)

**역확산 (Reverse Diffusion)**:
```
p_θ(X_{0:T} | Y, B) = p(X_T) · Π p_θ(X_{t-1} | X_t, Y, B)
```

**훈련 목적 함수**:
```
L = E[||ε - ε_θ(√ᾱ_t · X_0 + √(1-ᾱ_t) · ε, Y, B, t)||²]
```

**샘플링 (추론)**:
```
X_{t-1} = (1/√α_t)(X_t - β_t/√(1-ᾱ_t) · ε_θ(X_t, Y, B, t)) + √β̃_t · η
```

#### 2.3.2 Denoising Network 아키텍처

```
U-Net 구조 (PointNet++ 기반, PVC 연산자 대체)

인코더:                    디코더:
  SA Block 1 (1024pts)  ←→  FP Block 1
  SA Block 2              ←→  FP Block 2
  SA Block 3              ←→  FP Block 3
  SA Block 4 ( latent)   ←→  FP Block 4

모든 SA/FP 블록 사이 + Bottleneck에 DITA 레이어 삽입
```

**입력 표현** (치아당 1024 포인트):
- 3D 좌표 (x, y, z)
- Binary indicator (0=컨텍스트, 1=타겟)
- 8차원 FDI 임베딩 (치아 식별 정보)
- 실린더 경계 정보 (Boundary 모듈 출력)

#### 2.3.3 DITA (Distance-weighted Inter-Tooth Attention)

**핵심 아이디어**: 치아 간의 거리 정보를 활용한 상대적 위치 인코딩(RPE)

**치아 인덱싱**: 지그재그 FDI 순서
```
17, 47, 16, 46, ..., 11, 41, 21, 31, ..., 27, 37
```
→ 인덱스 차이 Δij = i - j 가 해부학적 거리의 효과적 프록시

**상대적 위치 인코딩**:
```python
r_ij = [log(1 + max(Δij, 0)),
        log(1 + max(-Δij, 0)),
        1_{Δij=0}]  ∈ R³
```

**어텐션 계산**:
```python
e_ij = (1/√F) · q_i^T · k_j + q_i^T · p^K_ij + p^Q_ij^T · k_j
α_ij = softmax(e_ij)
z_i^out = z_i^in + Σ_j α_ij · (v_j + p^V_ij)
```

**효과**: 인접 치아 및 대합치에 높은 어텐션 가중치 부여 → 형태학적 영향을 해부학적으로 학습

### 2.4 2단계 학습 전략 (Pseudo-crown Training)

```
Stage 1: 420개 완전 치열 스캔으로 초기 모델 학습 (3000 에폭)
           ↓
Stage 2: 초기 모델로 1,364개 부분 무치아 스캔에 "가짜 크라운" 생성
           ↓
         확장된 데이터셋으로 파인튜닝 (2400 에폭)
```

**핵심 통찰**: 가짜 크라운의 품질이 완벽하지 않아도, 학습 신호는 고품질 자연 치아가 지배하므로 견고함

### 2.5 Mesh Reconstruction (DPSR)

- **Differentiable Poisson Surface Reconstruction** 기반
- 포인트 클라우드 → 업샘플링 + 법선 벡터 예측 → Poisson 솔버 → 3D indicator function → Marching Cubes
- Watertight manifold 메쉬 출력

---

## 3. 성능 평가 결과

### 3.1 정량 평가 (496 외부 테스트 스캔)

#### 3.1.1 Point Cloud 레벨 성능 (모든 치아 유형)

| 결손 수 | CD L1 (↓) | EMD (↓) | F1@0.3mm (↑) | F1@0.5mm (↑) | F1@1.0mm (↑) |
|---------|-----------|---------|--------------|--------------|--------------|
| 1개 | **34.532** | **0.471** | **0.471** | **0.790** | **0.980** |
| 2개 | **35.111** | **0.462** | **0.462** | **0.781** | **0.979** |
| 3개 | **34.924** | **0.465** | **0.465** | **0.784** | **0.979** |
| 4개 | **34.645** | **0.470** | **0.470** | **0.788** | **0.980** |
| 5개 | **34.479** | **0.474** | **0.474** | **0.791** | **0.980** |
| 6개 | **34.338** | **0.476** | **0.476** | **0.793** | **0.981** |

**핵심 발견**: 결손치 수가 증가해도 성능이 **안정적으로 유지**되며, 일부 메트릭에서는 오히려 향상

#### 3.1.2 경쟁 방법 대비 성능 우위

| 방법 | 1개 CD | 6개 CD | 성능 저하율 |
|------|--------|--------|-------------|
| **CrownGen** | 34.5 | 34.3 | **~0%** |
| PointSea | 38.8 | 54.3 | ~40% ↓ |
| AdaPoinTr | 41.4 | 59.9 | ~45% ↓ |
| ProxyFormer | 40.0 | 66.8 | ~67% ↓ |

→ 기존 방법들은 다수 치아 복원 시 성능이 **급격히 붕괴**, CrownGen은 안정적

#### 3.1.3 Mesh 레벨 성능 (Ablation Study)

| 변형 | ASD (↓) | NC (↑) | 비고 |
|------|---------|--------|------|
| **CrownGen (Full)** | **0.267** | **0.925** | 전체 모델 |
| w/o Boundary | 0.411 | 0.887 | ASD +35%, NC -4.3% |
| w/o DITA | 0.333 | 0.915 | ASD +19.8%, NC -1.1% |
| w/o Data Expansion | 0.438 | 0.884 | ASD +39%, NC -4.6% |

### 3.2 임상 평가 (26 복원 사례, 23 환자)

#### 3.2.1 설계 시간
- CrownGen 보조: **740 ± 131초** (평균)
- 수동 전문가: **900 ± 180초** (평균)
- **17.78% 감소** (p < 0.01)

#### 3.2.2 임상 품질 평가 (3점 Likert 척도)

| 항목 | CrownGen | 수동 | p-value |
|------|----------|------|---------|
| 종합 | 2.938 | 2.928 | 0.425 |
| 교합 | 2.942 | 2.904 | 0.161 |
| 인접 접촉 | 2.942 | 2.923 | 0.327 |
| 치열궁 정렬 | 2.942 | 2.942 | 동일 |
| 크라운 형태 | 2.923 | 2.942 | 0.327 |

#### 3.2.3 비열등성 (Non-inferiority) 분석
- NI 마진: -0.10 점 (5% 포인트)
- **모든 항목에서 NI 확립** → CrownGen은 숙련 기공사의 수동 워크플로우에 대해 통계적으로 비열등
- 임상 허용 가능(점수 3) 비율: CrownGen 95.2% vs 수동 94.2%

#### 3.2.4 평가자 간 일치도
- Gwet's AC2: **0.947** (우수한 일치)
- Overall Percentage Agreement: 89.4%

---

## 4. 핵심 혁신 포인트 분석

### 4.1 치아 수준 표현 (Tooth-Level Representation)

```
기존: 치과 아치 전체 → 단일 포인트 클라우드 → 단일 출력
CrownGen: 개별 치아별 포인트 클라우드 → 객체 집합 → 다수 출력
```

**장점**:
- 컨텍스트/타겟 치아를 동적으로 구분 가능
- 결함 치아를 입력에서 선택적 제외 가능
- 생성 복잡도가 결손치 수에 독립적

### 4.2 DITA의 해부학적 의미

어텐션 히트맵 분석 결과:
- 근/원심 인접 치아에 높은 어텐션 (인접 접촉 형태 결정)
- 대합 치아에 높은 어텐션 (교합 관계 결정)
- 원거리 치아는 낮은 어텐션 → 임상적으로 타당

### 4.3 확장성 (Scalability)

- 추론 시간: **~85초/패스** (RTX 4090, 크라운 수 무관)
- 단일 통합 모델로 1~6개 크라운 모두 처리 (경쟁 방법은 각각 별도 모델 필요)

---

## 5. 한계점 및 향후 연구 방향

### 5.1 현재 한계

| 한계 | 설명 |
|------|------|
| 치경부(Cervical) 적응 | Margin line 형태가 매우 다양하여 상용 CAD 툴에 의존 |
| 치아 분할 의존성 | 독립적인 분할 단계 필요 (단, 성숙 기술) |
| 교정 환자 제외 | 심한 부정교합 케이스는 학습에서 제외 |
| 제3대구치 미포함 | 28개 치아 기준 (사랑니 제외) |
| 추론 속도 | Diffusion 모델 특성상 85초 소요 |
| 6개 이상 결손 | 가대치(removable partial denture)는 설계 우선순위가 다름 |

### 5.2 향후 개선 방향

1. **가속화**: DDIM, DPM-Solver 등 샘플링 가속 기법 적용
2. **End-to-End**: 분할 → 경계 예측 → 생성을 통합 파이프라인으로
3. **Margin Line 모델링**: 준비된 치아/임플란트 지대치 기하학 통합
4. **더 큰 모델**: Transformer 기반 백본 확장
5. **임상 통합**: ExoCAD, 3Shape 등 상용 CAD 플랫폼 플러그인 개발

---

## 6. 구현 계획

### 6.1 프로젝트 개요

```
프로젝트명: CrownGen Implementation
목표: CrownGen 논문의 핵심 파이프라인을 PyTorch로 구현
예상 기간: 12~16주
개발 환경: Python 3.11, PyTorch 2.5+, CUDA 11.8+
하드웨어: NVIDIA RTX 4090 (또는 A100) 이상 권장
```

### 6.2 모듈별 구현 계획

#### Phase 1: 기반 인프라 (1~2주)

```
📁 crowngen/
├── configs/           # 학습/추론 설정 (YAML)
├── data/
│   ├── dataset.py     # 포인트 클라우드 데이터셋 클래스
│   ├── preprocessing/ # 전처리 유틸리티
│   └── augmentation.py # 데이터 증강
├── models/
│   ├── pvc.py         # Point-Voxel Convolution
│   ├── pointnet2.py   # PointNet++ SA/FP 모듈
│   ├── dita.py        # Distance-weighted Inter-Tooth Attention
│   ├── denoise_net.py # U-Net Denoising Network
│   ├── boundary.py    # Boundary Prediction Module
│   ├── diffusion.py   # DDPM Forward/Reverse 프로세스
│   └── dpsr.py        # Differentiable Poisson Surface Reconstruction
├── losses/
│   ├── smooth_l1.py
│   ├── chamfer.py     # Chamfer Distance
│   └── emd.py         # Earth Mover's Distance
├── metrics/
│   └── evaluation.py  # F1, ASD, NC 등 평가 메트릭
├── train/
│   ├── train_boundary.py
│   ├── train_diffusion_stage1.py
│   ├── train_diffusion_stage2.py
│   └── train_dpsr.py
├── inference/
│   ├── pipeline.py    # End-to-end 추론
│   └── visualize.py   # 결과 시각화
└── utils/
    ├── fdi.py         # FDI 인덱싱 유틸리티
    ├── point_cloud.py # 포인트 클라우드 I/O
    └── mesh.py        # 메쉬 변환 유틸리티
```

**핵심 구현 항목**:

1. **데이터 로더**: STL/OBJ 메쉬 → 균일 샘플링(1024점) + FDI 라벨 + 컨텍스트/타겟 마스킹
2. **데이터 증강**: 포인트 셔플링, 좌우 반전(FDI 리매핑), 등방성 스케일링[0.95, 1.05]
3. **평가 메트릭**: CD-L1, EMD, F1@0.3/0.5/1.0mm, ASD, NC

#### Phase 2: 핵심 모델 구현 (3~6주)

##### 2.1 PVC (Point-Voxel Convolution) 연산자
```python
# 핵심 구조
class PVC(nn.Module):
    """Point-Voxel Convolution: 포인트 특징 + 복셀 특징 결합"""
    def __init__(self, in_channels, out_channels, voxel_resolution):
        self.point_branch = PointBranch(in_channels, out_channels)
        self.voxel_branch = VoxelBranch(in_channels, out_channels, voxel_resolution)
        self.fusion = nn.Linear(out_channels * 2, out_channels)

    def forward(self, xyz, features):
        point_feat = self.point_branch(xyz, features)
        voxel_feat = self.voxel_branch(xyz, features)
        return self.fusion(torch.cat([point_feat, voxel_feat], dim=-1))
```

##### 2.2 PointNet++ SA/FP 모듈
```python
class SetAbstraction(nn.Module):
    """계층적 다운샘플링 + 로컬 특징 추출"""
    # ball query 기반 로컬 그룹핑
    # PVC 연산자 적용
    # DITA 레이어 (옵션)

class FeaturePropagation(nn.Module):
    """계층적 업샘플링 + 특징 융합 (Skip Connection 포함)"""
    # 보간(interpolation) 기반 업샘플링
    # Skip connection from SA blocks
    # PVC 연산자 적용
    # DITA 레이어 (옵션)
```

##### 2.3 DITA 레이어
```python
class DITA(nn.Module):
    """Distance-weighted Inter-Tooth Attention"""
    def __init__(self, dim, num_heads, max tooth_count=28):
        self.num_heads = num_heads
        # 상대적 위치 인코딩: 3차원 → Q/K/V 편향
        self.rpe_mlp = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, num_heads * dim * 3)
        )

    def compute_rpe(self, tooth_indices):
        """
        지그재그 FDI 순서 인덱스로 상대적 위치 벡터 계산
        r_ij = [log(1+max(Δ,0)), log(1+max(-Δ,0)), 1_{Δ=0}]
        """
        delta = tooth_indices[:, None] - tooth_indices[None, :]
        r = torch.stack([
            torch.log1p(delta.clamp(min=0).float()),
            torch.log1p((-delta).clamp(min=0).float()),
            (delta == 0).float()
        ], dim=-1)
        return r

    def forward(self, z, tooth_indices):
        # Q, K, V 프로젝션
        # RPE 편향 계산
        # 어텐션: e_ij = (1/√F)·q_i·k_j + q_i·p^K + p^Q·k_j
        # 잔차 연결: z_i^out = z_i^in + Σ α_ij·(v_j + p^V)
```

##### 2.4 DDPM 프레임워크
```python
class GaussianDiffusion(nn.Module):
    """DDPM Forward/Reverse 프로세스"""
    def __init__(self, T=1000, beta_min=1e-4, beta_max=2e-2):
        # 선형 분산 스케줄
        # 미리 계산: α_t, ᾱ_t, β̃_t

    def forward_process(self, x_0, t, noise):
        """q(X_t | X_0) = N(√ᾱ_t·X_0, (1-ᾱ_t)·I)"""
        return sqrt_alphas_bar[t] * x_0 + sqrt_one_minus_alphas_bar[t] * noise

    def training_loss(self, model, x_0, context, boundary):
        """L = E[||ε - ε_θ(X_t, Y, B, t)||²]"""
        t = torch.randint(1, T, (B,))
        noise = torch.randn_like(x_0)
        x_t = self.forward_process(x_0, t, noise)
        noise_pred = model(x_t, context, boundary, t)
        return F.mse_loss(noise, noise_pred)  # 타겟 치아에만 적용

    @torch.no_grad()
    def sample(self, model, context, boundary, shape):
        """X_T ~ N(0,I) → 반복 디노이징 → X_0"""
        x = torch.randn(shape)  # 경계 내에서 초기화
        for t in reversed(range(1, T+1)):
            noise_pred = model(x, context, boundary, t)
            x = (1/√α_t) * (x - β_t/√(1-ᾱ_t) * noise_pred)
            if t > 1:
                x += √β̃_t * torch.randn_like(x)
        return x
```

##### 2.5 Boundary Prediction Module
```python
class BoundaryPredictor(nn.Module):
    """경량 인코더 + 실린더 파라미터 회귀"""
    def __init__(self):
        self.encoder = nn.ModuleList([
            SA_Block(512, pvc=True, dita=True) for _ in range(3)
        ])
        self.regressor = nn.Linear(latent_dim, 5)  # cx, cy, cz, r, h

    def forward(self, context_teeth, target_mask):
        # 컨텍스트 특징 추출
        # 타겟 위치에 대한 실린더 파라미터 회귀
        return boundary_params  # (|X|, 5)
```

#### Phase 3: 학습 파이프라인 (7~10주)

##### 3.1 학습 스케줄

```
Step 1: DPSR Mesh Reconstruction 모델 독립 학습
        - 데이터: 정상 치아 메쉬
        - 손실: MSE on indicator grid
        - 에폭: 500~

Step 2: Boundary Prediction Module 학습
        - 데이터: 완전 치열 스캔 (1~6개 랜덤 마스킹)
        - 손실: Smooth L1
        - 에폭: 1000
        - LR: 3e-4 → 3e-6 (Cosine Annealing)

Step 3: Generative Module Stage 1
        - 데이터: 완전 치열 스캔만
        - 에폭: 3000
        - LR: 4e-5, 1500 에폭에서 0.4 감쇠

Step 4: Pseudo-crown 생성
        - Stage 1 모델로 부분 무치아 스캔에 추론
        - 가짜 크라운으로 빈 공간 채우기

Step 5: Generative Module Stage 2
        - 데이터: 확장된 전체 데이터셋
        - 에폭: 2400
        - LR: 2e-5, 800 에폭마다 0.45 감쇠
```

##### 3.2 학습 설정

```yaml
# configs/train_diffusion.yaml
model:
  point_per_tooth: 1024
  max_teeth: 28
  sa_levels: 4
  dita_heads: 8
  pvc_dropout: 0.1

diffusion:
  timesteps: 1000
  beta_min: 1.0e-4
  beta_max: 2.0e-2
  schedule: linear

training:
  stage1:
    epochs: 3000
    batch_size: 32
    lr: 4.0e-5
    lr_decay_epoch: 1500
    lr_decay_factor: 0.4
    optimizer: adam

  stage2:
    epochs: 2400
    batch_size: 32
    lr: 2.0e-5
    lr_step_size: 800
    lr_decay_factor: 0.45
    optimizer: adam

augmentation:
  random_shuffle: true
  bilateral_mirror: true
  isotropic_scale: [0.95, 1.05]
  mask_range: [1, 6]  # 1~6개 랜덤 마스킹
```

#### Phase 4: 평가 및 최적화 (11~14주)

##### 4.1 평가 메트릭 구현

```python
def chamfer_distance_l1(pred, gt):
    """양방향 최근접 이웃 거리의 L1 평균"""

def earth_mover_distance(pred, gt):
    """최적 수송 기반 전역 기하학 대응"""

def f1_score(pred, gt, threshold):
    """Precision/Recall 기반 재구성 품질"""

def average_surface_distance(pred_mesh, gt_mesh):
    """메쉬 표면 간 평균 거리"""

def normal_consistency(pred_mesh, gt_mesh):
    """대응점 법선 벡터 코사인 유사도"""
```

##### 4.2 벤치마크 프로토콜

```
Protocol 1: 포인트 클라우드 레벨
  - 26,288 테스트 시나리오 (496 스캔 × 다양한 결손 조합)
  - 메트릭: CD-L1, EMD, F1@0.3/0.5/1.0mm

Protocol 2: 메쉬 레벨 (CrownGen만)
  - 6,944 시나리오 → 16,368 개별 크라운
  - 메트릭: ASD, NC

Protocol 3: 임상 평가 (선택)
  - 숙련 치과의사 2인 판독
  - 4항목 3점 Likert 척도
```

##### 4.3 최적화 전략

1. **추론 가속**: DDIM Sampler (10~50 스텝), DPM-Solver++
2. **메모리 최적화**: Gradient Checkpointing, Mixed Precision (FP16/BF16)
3. **분산 학습**: DDP (Distributed Data Parallel)
4. **ONNX Export**: 상용 CAD 플러그인 통합 준비

#### Phase 5: 통합 및 배포 (15~16주)

```
[입력] STL/OBJ 메쉬 (환자 스캔)
    ↓
[치아 분할] 외부 모델 또는 수동
    ↓
[전처리] 교합 정렬, 좌표계 표준화
    ↓
[Boundary Prediction] 실린더 경계 예측
    ↓
[Diffusion Generation] 크라운 포인트 클라우드 생성
    ↓
[DPSR Reconstruction] Watertight 메쉬 변환
    ↓
[후처리] 상용 CAD에서 margin line 적응
    ↓
[출력] CAD/CAM ready 크라운 메쉬
```

### 6.3 의존성 라이브러리

```
# requirements.txt
torch>=2.5.1
numpy>=1.26.4
pandas>=2.2.3
scipy>=1.12.0
open3d>=0.18.0          # 포인트 클라우드/메쉬 처리
trimesh>=4.0.0          # 메쉬 I/O
pytorch3d>=0.7.5        # Chamfer Distance, EMD
polyscope>=2.2.1        # 3D 시각화
matplotlib>=3.8.3
seaborn>=0.13.2
tqdm>=4.65.0
pyyaml>=6.0
wandb>=0.16.0           # 실험 추적 (선택)
```

### 6.4 마일스톤

| 주차 | 마일스톤 | 산출물 |
|------|----------|--------|
| 1~2 | 프로젝트 설정, 데이터 로더, 증강 | 데이터 파이프라인 |
| 3~4 | PVC, PointNet++ SA/FP 구현 | 기본 모델 컴포넌트 |
| 5~6 | DITA 레이어, Boundary 모듈 | 핵심 모델 완성 |
| 7~8 | DDPM 프레임워크, U-Net 통합 | 전체 생성 모델 |
| 9~10 | Stage 1 학습, Pseudo-crown 생성 | 1차 학습 완료 |
| 11~12 | Stage 2 학습, DPSR 구현 | 전체 학습 완료 |
| 13~14 | 평가 메트릭, 벤치마크 | 정량 평가 결과 |
| 15~16 | 통합 파이프라인, 최적화 | 배포 가능 프로토타입 |

### 6.5 리스크 및 대응

| 리스크 | 확률 | 영향 | 대응 방안 |
|--------|------|------|-----------|
| 학습 데이터 확보 어려움 | 높음 | 높음 | 공개 데이터셋(Source A, B, C)으로 시작, 합성 데이터 활용 |
| GPU 메모리 부족 | 중간 | 중간 | Gradient Checkpointing, 작은 배치, Mixed Precision |
| DDPM 수렴 어려움 | 중간 | 높음 | 하이퍼파라미터 그리드 서치, EMA 적용 |
| DPSR 품질 저하 | 낮음 | 중간 | 사전 학습된 DPSR 사용, 후처리 smoothing |
| 평가 지표 불일치 | 낮음 | 낮음 | 논문의 정확한 평가 프로토콜 준수 |

---

## 7. 참고 문헌 (논문 내 핵심 인용)

1. **DDPM**: Ho, J., Jain, A. & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models. NeurIPS.
2. **Point-Voxel CNN**: Liu, Z. et al. (2019). Point-Voxel CNN for Efficient 3D Deep Learning. NeurIPS.
3. **PointNet++**: Qi, C.R. et al. (2017). PointNet++: Deep Hierarchical Feature Learning on Point Sets. NeurIPS.
4. **DPSR**: Williams, F. et al. (2021). Neural Fields as Learnable Kernels for 3D Reconstruction. CVPR.
5. **PointSea**: Du, Y. et al. (2023). PointSea: Point Cloud Completion via Multi-modal Learning. ICCV.
6. **AdaPoinTr**: Yu, X. et al. (2023). AdaPoinTr: Adaptive Point Cloud Completion. CVPR.
7. **ProxyFormer**: Huang, T. et al. (2023). ProxyFormer for Point Cloud Completion. AAAI.
8. **Relative PE**: Shaw, P. et al. (2018). Self-Attention with Relative Position Representations. NAACL.

---

## 부록: 다운로드한 자료 목록

```
Docs/
├── CrownGen_Paper.pdf          # 논문 PDF (6.3MB)
├── CrownGen_Paper.html         # 논문 HTML (949KB)
├── CrownGen_Abstract.html      # 초록 페이지
├── source.tar.gz               # LaTeX 소스 압축 (5.3MB)
├── source/                     # LaTeX 소스 파일
│   ├── sn_article.tex          # 메인 TeX 파일
│   ├── sn-jnl.cls             # Springer Nature 저널 클래스
│   ├── sn-basic.bst           # 참고문헌 스타일
│   ├── sn-bibliography.bib    # 참고문헌 데이터
│   └── figures_folder/        # 고품질 PDF 피겨
├── images/                     # 피겨 모음
│   ├── fig_overall.pdf         # Figure 7: 전체 아키텍처
│   ├── fig_ddpm.pdf           # Figure 2: 성능 비교
│   ├── fig_dita.pdf           # Figure 9: DITA 어텐션 시각화
│   ├── fig_ablation_metric.pdf # Figure 3: Ablation Study
│   ├── fig_pointcloud_all.pdf  # 포인트 클라우드 평가
│   ├── fig_pointcloud_toothgroup.pdf # 기능 그룹별 평가
│   ├── fig_reader_results.pdf  # Figure 5: 임상 결과
│   ├── fig_reader_workflow.pdf # Figure 4: 워크플로우
│   ├── fig_meshrecon_visual.pdf # 메쉬 재구성 시각화
│   ├── fig_postprocess.pdf     # Figure 10: 후처리 결과
│   └── fig_irr.pdf            # Figure 6: 평가자 간 일치도
└── CrownGen_분석및구현계획서.md  # 본 문서
```
