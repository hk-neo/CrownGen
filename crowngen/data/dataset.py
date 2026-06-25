"""
CrownGen 데이터셋 및 콜레이터.

Teeth3DS+ 처리된 .npz 파일을 로드하여 CrownGen 학습에 필요한
텐서 형식으로 변환합니다.

출력 텐서:
  - tooth_points: (28, N, 3) — 전체 치아 포인트 클라우드
  - fdi_labels: (28,) — FDI 치아 번호
  - target_mask: (28,) — 1=타겟, 0=컨텍스트
  - boundaries: (28, 5) — 실린더 경계 파라미터
  - tooth_valid: (28,) — 1=치아 존재, 0=결손
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, List, Dict, Any

from .fdi import ZIGZAG_FDI_ORDER, FDI_TO_ZIGZAG, ALL_28_FDI
from .augmentation import CrownGenAugmentation


class CrownGenDataset(Dataset):
    """CrownGen 학습용 데이터셋.

    처리된 .npz 파일에서 치아별 포인트 클라우드, FDI 라벨,
    실린더 경계를 로드합니다.

    Args:
        data_dir: 처리된 .npz 파일 디렉토리
        split_file: train_val_split.json 경로
        split_name: 사용할 분할 ('stage1_train', 'stage1_val', etc.)
        n_points: 치아당 포인트 수 (1024 또는 512)
        mask_range: 마스킹할 치아 수 범위
        augment: 데이터 증강 여부
        is_boundary: Boundary 예측용 (512 포인트, 타겟 영벡터)
    """

    # 28개 치아의 고정 슬롯 순서 (지그재그)
    SLOT_FDI = torch.tensor(ZIGZAG_FDI_ORDER, dtype=torch.long)

    def __init__(
        self,
        data_dir: str,
        split_file: str,
        split_name: str = 'stage1_train',
        n_points: int = 1024,
        mask_range: tuple = (1, 6),
        augment: bool = True,
        is_boundary: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.n_points = n_points
        self.is_boundary = is_boundary
        self.mask_range = mask_range

        # 분할 파일에서 환자 ID 로드
        with open(split_file) as f:
            splits = json.load(f)
        self.patient_ids = splits[split_name]

        # 존재하는 파일만 필터링
        self.valid_patients = []
        for pid in self.patient_ids:
            npz_path = self.data_dir / f"{pid}.npz"
            if npz_path.exists():
                self.valid_patients.append(pid)

        print(f"Dataset [{split_name}]: {len(self.valid_patients)}/{len(self.patient_ids)} patients loaded")

        # 증강
        self.augment = CrownGenAugmentation(
            shuffle=augment,
            mirror=augment,
            mask_range=mask_range
        ) if augment else None

    def __len__(self) -> int:
        return len(self.valid_patients)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pid = self.valid_patients[idx]
        npz_path = self.data_dir / f"{pid}.npz"
        data = np.load(npz_path)

        # 치아 데이터를 28개 슬롯에 배정
        tooth_points = torch.zeros(28, self.n_points, 3, dtype=torch.float32)
        boundaries = torch.zeros(28, 5, dtype=torch.float32)
        fdi_labels = torch.zeros(28, dtype=torch.long)
        tooth_valid = torch.zeros(28, dtype=torch.long)

        for slot_idx, fdi in enumerate(ZIGZAG_FDI_ORDER):
            # 상악 치아
            jaw = 'upper' if fdi in [x for x in range(11, 18)] + [x for x in range(21, 28)] else 'lower'
            pc_key = f"{jaw}_{fdi}_pc"
            bound_key = f"{jaw}_{fdi}_bound"

            if pc_key in data:
                pc = torch.from_numpy(data[pc_key]).float()
                # 포인트 수 조정
                if pc.shape[0] >= self.n_points:
                    indices = torch.randperm(pc.shape[0])[:self.n_points]
                    pc = pc[indices]
                else:
                    # 부족하면 반복 + 노이즈
                    n_extra = self.n_points - pc.shape[0]
                    extra_idx = torch.randint(0, pc.shape[0], (n_extra,))
                    extra = pc[extra_idx] + torch.randn(n_extra, 3) * 0.01
                    pc = torch.cat([pc, extra], dim=0)

                tooth_points[slot_idx] = pc
                fdi_labels[slot_idx] = fdi
                tooth_valid[slot_idx] = 1

                if bound_key in data:
                    boundaries[slot_idx] = torch.from_numpy(data[bound_key]).float()

        # 데이터 증강
        target_mask = torch.zeros(28, dtype=torch.long)
        if self.augment is not None:
            tooth_points, fdi_labels, target_mask = self.augment(tooth_points, fdi_labels)
        else:
            # 증강 없이 기본 마스킹
            k = np.random.randint(self.mask_range[0], self.mask_range[1] + 1)
            valid_indices = torch.where(tooth_valid == 1)[0]
            if len(valid_indices) > 0:
                k = min(k, len(valid_indices))
                mask_idx = valid_indices[torch.randperm(len(valid_indices))[:k]]
                target_mask[mask_idx] = 1

        # Boundary 모듈용: 타겟 치아의 포인트를 영벡터로
        if self.is_boundary:
            for i in range(28):
                if target_mask[i] == 1:
                    tooth_points[i] = 0.0

        # 타겟 치아의 포인트에 노이즈 추가 (DDPM 학습 시)
        # (여기서는 깨끗한 데이터 반환, diffusion 모듈에서 노이즈 추가)

        return {
            'tooth_points': tooth_points,       # (28, N, 3)
            'fdi_labels': fdi_labels,           # (28,)
            'target_mask': target_mask,          # (28,) 1=타겟
            'context_mask': 1 - target_mask,     # (28,) 1=컨텍스트
            'boundaries': boundaries,            # (28, 5)
            'tooth_valid': tooth_valid,          # (28,) 1=존재
        }


def crown_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """CrownGen 커스텀 콜레이터.

    배치 차원을 추가하고 텐서를 스택합니다.

    Args:
        batch: 리스트 of __getitem__ 출력

    Returns:
        배치된 딕셔너리, 모든 값은 (B, 28, ...) 형태
    """
    keys = batch[0].keys()
    result = {}
    for key in keys:
        result[key] = torch.stack([item[key] for item in batch], dim=0)
    return result
