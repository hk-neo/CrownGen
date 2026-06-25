"""
CrownGen 데이터 증강 모듈.

논문에 명시된 3가지 증강:
1. 포인트 셔플링: 각 치아 내 포인트 순서를 무작위로 섞음
2. 좌우 반전: 치열 전체를 시상면 기준 반전 + FDI 인덱스 리매핑
3. 등방성 스케일링: [0.95, 1.05] 범위에서 균일 스케일 적용
"""

import torch
import random
from typing import Tuple, Dict
from .fdi import MIRROR_REMAP, ALL_28_FDI


def random_shuffle_points(points: torch.Tensor) -> torch.Tensor:
    """치아 내 포인트 순서를 무작위로 섞음.

    Args:
        points: (N, 3) 또는 (28, N, 3) 포인트 클라우드

    Returns:
        동일 형태의 셔플된 포인트 클라우드
    """
    if points.dim() == 2:
        perm = torch.randperm(points.shape[0])
        return points[perm]
    elif points.dim() == 3:
        # (28, N, 3) — 각 치아마다 독립적으로 셔플
        B = points.shape[0]
        for i in range(B):
            perm = torch.randperm(points.shape[1])
            points[i] = points[i, perm]
        return points
    return points


def bilateral_mirror(
    tooth_points: torch.Tensor,
    fdi_labels: torch.Tensor,
    p: float = 0.5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """치열 전체를 시상면 기준으로 좌우 반전.

    확률 p로 반전 적용. 반전 시:
    - x 좌표 반전 (x → -x)
    - FDI 라벨 리매핑 (좌측 ↔ 우측)
    - 치아 슬롯 위치 교환

    Args:
        tooth_points: (28, N, 3) 전체 치열 포인트 클라우드
        fdi_labels: (28,) FDI 라벨
        p: 반전 적용 확률

    Returns:
        (반전된 포인트, 반전된 FDI 라벨)
    """
    if random.random() > p:
        return tooth_points, fdi_labels

    # x 좌표 반전
    mirrored_points = tooth_points.clone()
    mirrored_points[:, :, 0] = -mirrored_points[:, :, 0]

    # FDI 리매핑 및 슬롯 교환
    mirrored_fdi = fdi_labels.clone()
    remap_indices = {}

    for i in range(len(fdi_labels)):
        fdi = fdi_labels[i].item()
        if fdi > 0 and fdi in MIRROR_REMAP:
            mirrored_fdi[i] = MIRROR_REMAP[fdi]
            mirror_fdi = MIRROR_REMAP[fdi]
            # 대응하는 치아의 인덱스 찾기
            for j in range(len(fdi_labels)):
                if fdi_labels[j].item() == mirror_fdi:
                    remap_indices[i] = j
                    break

    # 치아 슬롯 교환 (포인트 클라우드와 FDI 라벨 모두)
    for i, j in remap_indices.items():
        if i != j:
            mirrored_points[[i, j]] = mirrored_points[[j, i]]
            mirrored_fdi[[i, j]] = mirrored_fdi[[j, i]]

    return mirrored_points, mirrored_fdi


def isotropic_scaling(
    tooth_points: torch.Tensor,
    scale_range: Tuple[float, float] = (0.95, 1.05)
) -> torch.Tensor:
    """치열 전체에 등방성 스케일링 적용.

    Args:
        tooth_points: (28, N, 3) 또는 (N, 3) 포인트 클라우드
        scale_range: (min_scale, max_scale) 범위

    Returns:
        스케일된 포인트 클라우드
    """
    scale = random.uniform(scale_range[0], scale_range[1])
    return tooth_points * scale


def random_mask_teeth(
    n_teeth: int = 28,
    mask_range: Tuple[int, int] = (1, 6)
) -> torch.Tensor:
    """1~6개 치아를 무작위로 마스킹하여 타겟으로 지정.

    Args:
        n_teeth: 전체 치아 수
        mask_range: (min_mask, max_mask) 마스킹할 치아 수 범위

    Returns:
        (n_teeth,) 이진 마스크 (1=타겟/마스킹됨, 0=컨텍스트)
    """
    k = random.randint(mask_range[0], mask_range[1])
    k = min(k, n_teeth)
    mask = torch.zeros(n_teeth, dtype=torch.long)
    indices = torch.randperm(n_teeth)[:k]
    mask[indices] = 1
    return mask


class CrownGenAugmentation:
    """CrownGen 학습용 종합 데이터 증강.

    순서:
    1. 포인트 셔플링 (항상 적용)
    2. 좌우 반전 (50% 확률)
    3. 등방성 스케일링 (항상 적용)
    4. 랜덤 치아 마스킹
    """

    def __init__(
        self,
        shuffle: bool = True,
        mirror: bool = True,
        scale_range: Tuple[float, float] = (0.95, 1.05),
        mask_range: Tuple[int, int] = (1, 6)
    ):
        self.shuffle = shuffle
        self.mirror = mirror
        self.scale_range = scale_range
        self.mask_range = mask_range

    def __call__(
        self,
        tooth_points: torch.Tensor,
        fdi_labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """증강 적용.

        Args:
            tooth_points: (28, N, 3)
            fdi_labels: (28,)

        Returns:
            augmented_points: (28, N, 3)
            augmented_fdi: (28,)
            target_mask: (28,) 1=타겟, 0=컨텍스트
        """
        # 1. 포인트 셔플링
        if self.shuffle:
            tooth_points = random_shuffle_points(tooth_points)

        # 2. 좌우 반전
        if self.mirror:
            tooth_points, fdi_labels = bilateral_mirror(tooth_points, fdi_labels)

        # 3. 등방성 스케일링
        tooth_points = isotropic_scaling(tooth_points, self.scale_range)

        # 4. 랜덤 치아 마스킹
        target_mask = random_mask_teeth(28, self.mask_range)

        return tooth_points, fdi_labels, target_mask
