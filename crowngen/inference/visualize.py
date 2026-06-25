"""
CrownGen 결과 시각화 모듈.

생성된 크라운 포인트 클라우드를 시각화합니다:
  - 개별 치아 포인트 클라우드 (matplotlib)
  - 전체 치열 아치 (컨텍스트 + 생성 크라운)
  - Boundary 실린더 오버레이
  - 비교 시각화 (GT vs 생성)

사용법:
  python -m crowngen.inference.visualize \
      --patient_id SAMPLE001 \
      --result_dir results/ \
      --data_dir Data/processed/ \
      --output_dir visualizations/
"""

import argparse
import json
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from crowngen.data.fdi import ZIGZAG_FDI_ORDER, FDI_TO_ZIGZAG


# 치아별 색상 매핑
TOOTH_COLORS = {
    'context': '#4A90D9',   # 파란색 - 컨텍스트
    'target': '#E74C3C',    # 빨간색 - 생성된 크라운
    'gt': '#2ECC71',        # 초록색 - 정답
    'boundary': '#F39C12',  # 주황색 - 경계 실린더
}


def plot_single_tooth(
    points: np.ndarray,
    title: str = '',
    save_path: Optional[str] = None,
    color: str = '#4A90D9',
    point_size: float = 0.5,
):
    """단일 치아 포인트 클라우드 시각화.

    Args:
        points: (N, 3) 포인트 클라우드
        title: 플롯 제목
        save_path: 저장 경로 (None이면 화면에 표시)
        color: 포인트 색상
        point_size: 포인트 크기
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               c=color, s=point_size, alpha=0.6)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)

    # 동일 스케일
    max_range = np.array([
        points[:, 0].max() - points[:, 0].min(),
        points[:, 1].max() - points[:, 1].min(),
        points[:, 2].max() - points[:, 2].min(),
    ]).max() / 2.0
    mid = points.mean(axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_full_arch(
    tooth_points_dict: Dict[int, np.ndarray],
    target_fdis: List[int],
    boundaries_dict: Optional[Dict[int, np.ndarray]] = None,
    title: str = '',
    save_path: Optional[str] = None,
    point_size: float = 0.3,
):
    """전체 치열 아치 시각화 (컨텍스트 + 생성 크라운).

    Args:
        tooth_points_dict: {fdi: (N, 3)} 치아별 포인트 클라우드
        target_fdis: 생성된 크라운의 FDI 번호 리스트
        boundaries_dict: {fdi: (5,)} 실린더 경계 (선택)
        title: 플롯 제목
        save_path: 저장 경로
        point_size: 포인트 크기
    """
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    target_set = set(target_fdis)

    for fdi, points in tooth_points_dict.items():
        if fdi in target_set:
            color = TOOTH_COLORS['target']
            alpha = 0.8
            label = f'Generated Crown ({fdi})'
        else:
            color = TOOTH_COLORS['context']
            alpha = 0.4
            label = None

        ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                   c=color, s=point_size, alpha=alpha, label=label)

    # Boundary 실린더 표시
    if boundaries_dict:
        for fdi, bnd in boundaries_dict.items():
            if fdi in target_set:
                cx, cy, cz, r, h = bnd
                # 실린더 원 (XY 평면)
                theta = np.linspace(0, 2 * np.pi, 32)
                circle_x = cx + r * np.cos(theta)
                circle_y = cy + r * np.sin(theta)
                # 상/하 원
                for z_off in [cz - h/2, cz + h/2]:
                    ax.plot(circle_x, circle_y,
                            np.full_like(theta, z_off),
                            color=TOOTH_COLORS['boundary'],
                            linewidth=1.5, alpha=0.7)
                # 수직선
                for angle in [0, np.pi/2, np.pi, 3*np.pi/2]:
                    ax.plot([cx + r*np.cos(angle)] * 2,
                            [cy + r*np.sin(angle)] * 2,
                            [cz - h/2, cz + h/2],
                            color=TOOTH_COLORS['boundary'],
                            linewidth=1, alpha=0.5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)

    if target_fdis:
        ax.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_comparison(
    generated: Dict[int, np.ndarray],
    gt: Dict[int, np.ndarray],
    save_dir: str,
    patient_id: str,
):
    """GT vs 생성 크라운 비교 시각화.

    Args:
        generated: {fdi: (N, 3)} 생성된 크라운
        gt: {fdi: (N, 3)} 정답 크라운
        save_dir: 저장 디렉토리
        patient_id: 환자 ID
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for fdi in generated:
        if fdi not in gt:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(14, 6), subplot_kw={'projection': '3d'})

        gen_pts = generated[fdi]
        gt_pts = gt[fdi]

        # GT
        axes[0].scatter(gt_pts[:, 0], gt_pts[:, 1], gt_pts[:, 2],
                        c=TOOTH_COLORS['gt'], s=0.5, alpha=0.6)
        axes[0].set_title(f'GT — FDI {fdi}')

        # Generated
        axes[1].scatter(gen_pts[:, 0], gen_pts[:, 1], gen_pts[:, 2],
                        c=TOOTH_COLORS['target'], s=0.5, alpha=0.6)
        axes[1].set_title(f'Generated — FDI {fdi}')

        # 동일 스케일
        all_pts = np.vstack([gen_pts, gt_pts])
        max_range = np.array([
            all_pts[:, 0].ptp(), all_pts[:, 1].ptp(), all_pts[:, 2].ptp()
        ]).max() / 2.0
        mid = all_pts.mean(axis=0)
        for ax in axes:
            ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
            ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
            ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

        plt.suptitle(f'Patient {patient_id} — FDI {fdi}')
        plt.tight_layout()
        plt.savefig(save_dir / f'{patient_id}_FDI{fdi}_comparison.png',
                    dpi=150, bbox_inches='tight')
        plt.close()


def plot_metrics_bar(
    metrics: Dict[str, Dict[str, float]],
    save_path: str,
    title: str = 'CrownGen Evaluation Metrics',
):
    """메트릭 바 차트 시각화.

    Args:
        metrics: {metric_name: {tooth_type: value}} 중첩 딕셔너리
        save_path: 저장 경로
        title: 차트 제목
    """
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]

    for ax, (metric_name, values) in zip(axes, metrics.items()):
        types = list(values.keys())
        vals = list(values.values())

        bars = ax.bar(types, vals, color='#4A90D9', alpha=0.8)
        ax.set_title(metric_name)
        ax.set_ylabel('Value')

        # 값 표시
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description='CrownGen 결과 시각화')
    parser.add_argument('--patient_id', type=str, required=True)
    parser.add_argument('--result_dir', type=str, default='results/')
    parser.add_argument('--data_dir', type=str, default='Data/processed/')
    parser.add_argument('--output_dir', type=str, default='visualizations/')
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result_dir = Path(args.result_dir)

    # 결과 로드
    results_file = result_dir / f'{args.patient_id}_results.json'
    if not results_file.exists():
        print(f"결과 파일 없음: {results_file}")
        return

    with open(results_file) as f:
        results = json.load(f)

    print(f"환자: {args.patient_id}")
    print(f"생성 크라운: {list(results.keys())}")

    # 각 크라운 시각화
    for fdi_str, info in results.items():
        crown_file = result_dir / f'{args.patient_id}_{fdi_str}_crown.npy'
        if crown_file.exists():
            points = np.load(crown_file)
            plot_single_tooth(
                points,
                title=f'Patient {args.patient_id} — FDI {fdi_str}',
                save_path=str(output_dir / f'{args.patient_id}_FDI{fdi_str}_crown.png'),
                color=TOOTH_COLORS['target'],
            )
            print(f"  FDI {fdi_str}: {points.shape[0]} points")

    print(f"\n시각화 저장: {output_dir}")


if __name__ == '__main__':
    main()
