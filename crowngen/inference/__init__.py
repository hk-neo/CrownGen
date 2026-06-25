"""CrownGen 추론 모듈."""

from .pipeline import CrownGenPipeline, load_patient_data, save_results

try:
    from .visualize import (
        plot_single_tooth,
        plot_full_arch,
        plot_comparison,
    )
except ImportError:
    pass  # matplotlib 미설치 시 시각화 생략
